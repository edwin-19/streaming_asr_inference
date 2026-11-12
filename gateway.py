
import asyncio
import json
import logging
import os
import time
import uuid

import numpy as np
import ray
import uvicorn

from ray import serve
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from fireredvad.core.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)

from fireredvad.stream_vad import (
    FireRedStreamVad,
    FireRedStreamVadConfig,
)


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("websocket_gateway")

app = FastAPI()


VAD_MODEL_DIR = os.getenv(
    "VAD_MODEL_DIR",
    "/mnt/data/workspace/freelance/voice/diar/"
    "pretrained_models/FireRedVAD/Stream-VAD",
)

# 20 frames ~= 200ms when frame shift is 10ms.
STREAM_CHUNK_FRAMES = int(
    os.getenv("STREAM_CHUNK_FRAMES", "20")
)

# CPU temporarily. Explicitly set VAD_USE_GPU=true
# only when you want to test GPU VAD again.
VAD_USE_GPU = (
    os.getenv("VAD_USE_GPU", "false").lower() == "true"
)

# Limits simultaneous CPU VAD inference calls.
VAD_MAX_CONCURRENT_INFERENCE = int(
    os.getenv("VAD_MAX_CONCURRENT_INFERENCE", "4")
)

# Maximum concurrent WebSocket sessions.
MAX_ACTIVE_SESSIONS = int(
    os.getenv("MAX_ACTIVE_SESSIONS", "50")
)

SESSION_ACQUIRE_TIMEOUT = float(
    os.getenv("SESSION_ACQUIRE_TIMEOUT", "0.5")
)

RAY_ADDRESS = os.getenv("RAY_ADDRESS", "auto")
RAY_NAMESPACE = os.getenv("RAY_NAMESPACE", "default")
SERVE_APP_NAME = os.getenv("SERVE_APP_NAME", "default")

HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("GATEWAY_PORT", "8000"))

WS_PING_INTERVAL = float(
    os.getenv("WS_PING_INTERVAL", "20")
)

WS_PING_TIMEOUT = float(
    os.getenv("WS_PING_TIMEOUT", "120")
)


# ============================================================
# Ray Serve connection
# ============================================================

ray.init(
    address=RAY_ADDRESS,
    namespace=RAY_NAMESPACE,
)

ASR_HANDLE = serve.get_deployment_handle(
    "ASRRouter",
    app_name=SERVE_APP_NAME,
)

logger.info(
    "Connected to Ray Serve | app=%s",
    SERVE_APP_NAME,
)


# ============================================================
# Application lifecycle
# ============================================================

@app.on_event("startup")
async def startup():

    app.state.session_semaphore = asyncio.Semaphore(
        MAX_ACTIVE_SESSIONS
    )

    app.state.vad_semaphore = asyncio.Semaphore(
        VAD_MAX_CONCURRENT_INFERENCE
    )

    app.state.active_sessions = set()

    logger.info(
        "Gateway initialized | "
        "vad_device=%s | "
        "vad_concurrency=%s | "
        "max_sessions=%s | "
        "vad_chunk_frames=%s",
        "GPU" if VAD_USE_GPU else "CPU",
        VAD_MAX_CONCURRENT_INFERENCE,
        MAX_ACTIVE_SESSIONS,
        STREAM_CHUNK_FRAMES,
    )


# ============================================================
# VAD
# ============================================================

def create_vad():

    config = FireRedStreamVadConfig(
        use_gpu=VAD_USE_GPU,
        smooth_window_size=5,
        speech_threshold=0.3,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=40,
        chunk_max_frame=30000,
    )

    return FireRedStreamVad.from_pretrained(
        VAD_MODEL_DIR,
        config,
    )


# ============================================================
# Helpers
# ============================================================

def frame_to_seconds(frame: int) -> float:

    return (
        frame
        * FRAME_SHIFT_SAMPLE
        / SAMPLE_RATE
    )


def frame_to_samples(frame: int) -> int:

    return frame * FRAME_SHIFT_SAMPLE


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:

    return audio.astype(np.float32) / 32768.0


def is_disconnect(exc: Exception) -> bool:

    return (
        isinstance(exc, WebSocketDisconnect)
        or type(exc).__name__ in {
            "WebSocketDisconnect",
            "WebSocketDisconnected",
        }
    )


# ============================================================
# Audio Session
# ============================================================

class AudioSession:

    def __init__(
        self,
        vad_model,
        vad_semaphore,
    ):

        self.session_id = (
            f"sess_{uuid.uuid4().hex[:12]}"
        )

        # Independent FireRedVAD instance.
        self.vad = vad_model

        # Shared CPU inference limiter.
        self.vad_semaphore = vad_semaphore

        # Direct Ray actor handle.
        self.asr_worker = None
        self.asr_session_id = None

        # Pending VAD audio, stored as raw PCM16 bytes.
        self.audio_buffer = bytearray()

        # Full audio history for ASR boundary extraction.
        self.audio_history = bytearray()

        self.samples_received = 0
        self.vad_samples_processed = 0

        self.speech_active = False
        self.speech_start_frame = None

        self.asr_audio_cursor = 0
        self.last_partial_text = ""

        self.finalized = False
        self.closed = False

        self.vad_calls = 0
        self.vad_processing_time = 0.0

        self.started_at = time.perf_counter()

        self.vad.reset()

    # --------------------------------------------------------
    # Initialize ASR session
    # --------------------------------------------------------

    async def init_asr(self):

        result = await ASR_HANDLE.start_session.remote(
            self.session_id
        )

        self.asr_session_id = result["session_id"]
        self.asr_worker = result["worker_handle"]

        logger.info(
            "[%s] ASR assigned | worker=%s",
            self.session_id,
            result["worker_id"],
        )

    # --------------------------------------------------------
    # Incremental ASR
    # --------------------------------------------------------

    async def feed_asr_until(
        self,
        end_sample: int,
        websocket: WebSocket,
    ):

        if (
            self.asr_worker is None
            or self.asr_session_id is None
            or self.finalized
        ):
            return

        end_sample = min(
            end_sample,
            self.samples_received,
        )

        start_sample = self.asr_audio_cursor

        if end_sample <= start_sample:
            return

        audio = np.frombuffer(
            self.audio_history,
            dtype=np.int16,
            count=end_sample - start_sample,
            offset=start_sample * 2,
        )

        if len(audio) == 0:
            return

        pcm16k = pcm_to_float32(audio)

        started = time.perf_counter()

        result = await self.asr_worker.process_audio.remote(
            self.asr_session_id,
            pcm16k,
        )

        remote_elapsed = (
            time.perf_counter() - started
        )

        # Advance only after successful processing.
        self.asr_audio_cursor = end_sample

        timing = result.get("timing", {})

        logger.info(
            "[%s] ASR | "
            "audio=%.2fs | "
            "remote=%.3fs | "
            "inference=%.3fs | "
            "semaphore_wait=%.3fs | "
            "decoded=%s | worker=%s",
            self.session_id,
            len(pcm16k) / SAMPLE_RATE,
            remote_elapsed,
            timing.get("inference", 0.0),
            timing.get("semaphore_wait", 0.0),
            result["decoded"],
            result["worker_id"],
        )

        if not result["decoded"]:
            return

        payload = result["transcript"]

        self.last_partial_text = payload["text"]

        await websocket.send_json({
            "type": "transcript",
            **payload,
            "audio_end_sec": (
                end_sample / SAMPLE_RATE
            ),
            "decode_wall_sec": remote_elapsed,
            "is_final": False,
        })

    # --------------------------------------------------------
    # Final ASR
    # --------------------------------------------------------

    async def finish_asr(self, websocket: WebSocket):

        if (
            self.asr_worker is None
            or self.asr_session_id is None
            or self.finalized
        ):
            return

        worker = self.asr_worker
        session_id = self.asr_session_id

        started = time.perf_counter()

        result = await worker.finish_session.remote(
            session_id
        )

        elapsed = time.perf_counter() - started

        # Worker has removed its streaming session.
        self.finalized = True

        # Release router bookkeeping.
        try:

            await ASR_HANDLE.release_session.remote(
                self.session_id
            )

        finally:

            self.asr_worker = None
            self.asr_session_id = None

        logger.info(
            "[%s] ASR finalized | "
            "elapsed=%.3fs | "
            "worker=%s",
            self.session_id,
            elapsed,
            result["worker_id"],
        )

        await websocket.send_json({
            "type": "transcript",
            **result["transcript"],
            "audio_end_sec": (
                self.samples_received / SAMPLE_RATE
            ),
            "decode_wall_sec": elapsed,
            "is_final": True,
        })

    # --------------------------------------------------------
    # Cancel / cleanup remote ASR session
    # --------------------------------------------------------

    async def close_asr(self):

        session_id = self.asr_session_id
        worker = self.asr_worker

        if session_id is None:
            return

        try:

            if worker is not None:

                await worker.cancel_session.remote(
                    session_id
                )

        except Exception:

            logger.exception(
                "[%s] Failed to cancel ASR worker session",
                self.session_id,
            )

        finally:

            try:

                await ASR_HANDLE.release_session.remote(
                    self.session_id
                )

            except Exception:

                logger.exception(
                    "[%s] Failed to release router session",
                    self.session_id,
                )

            self.asr_worker = None
            self.asr_session_id = None

    # --------------------------------------------------------
    # Internal VAD inference
    # --------------------------------------------------------

    async def detect_vad(self, audio_chunk):

        started = time.perf_counter()

        async with self.vad_semaphore:

            results = await asyncio.to_thread(
                self.vad.detect_chunk,
                audio_chunk,
            )

        elapsed = time.perf_counter() - started

        self.vad_calls += 1
        self.vad_processing_time += elapsed

        logger.debug(
            "[%s] VAD | elapsed=%.2fms | "
            "audio=%.3fs",
            self.session_id,
            elapsed * 1000,
            len(audio_chunk) / SAMPLE_RATE,
        )

        return results

    # --------------------------------------------------------
    # VAD processing
    # --------------------------------------------------------

    async def process_audio(
        self,
        audio: np.ndarray,
    ):

        if self.closed:
            raise RuntimeError("Session already closed")

        # Store complete audio for ASR boundary extraction.
        raw_audio = audio.tobytes()

        self.audio_history.extend(raw_audio)
        self.audio_buffer.extend(raw_audio)

        self.samples_received += len(audio)

        chunk_length_samples = (
            FRAME_LENGTH_SAMPLE
            + (
                STREAM_CHUNK_FRAMES - 1
            ) * FRAME_SHIFT_SAMPLE
        )

        chunk_shift_samples = (
            STREAM_CHUNK_FRAMES
            * FRAME_SHIFT_SAMPLE
        )

        chunk_length_bytes = (
            chunk_length_samples * 2
        )

        chunk_shift_bytes = (
            chunk_shift_samples * 2
        )

        batches = []

        while len(self.audio_buffer) >= chunk_length_bytes:

            # Copy the chunk before removing it from buffer.
            chunk_bytes = bytes(
                self.audio_buffer[:chunk_length_bytes]
            )

            chunk = np.frombuffer(
                chunk_bytes,
                dtype=np.int16,
            )

            chunk_start = self.vad_samples_processed

            results = await self.detect_vad(
                chunk
            )

            self.vad_samples_processed += (
                chunk_shift_samples
            )

            chunk_end = min(
                chunk_start + chunk_length_samples,
                self.samples_received,
            )

            batches.append({
                "results": results,
                "processed_end_sample": chunk_end,
            })

            del self.audio_buffer[
                :chunk_shift_bytes
            ]

        return batches

    # --------------------------------------------------------
    # VAD flush
    # --------------------------------------------------------

    async def flush(self):

        if self.closed:
            return []

        silence_samples = (
            FRAME_LENGTH_SAMPLE
            + 50 * FRAME_SHIFT_SAMPLE
        )

        silence = np.zeros(
            silence_samples,
            dtype=np.int16,
        )

        self.audio_buffer.extend(
            silence.tobytes()
        )

        chunk_length_samples = (
            FRAME_LENGTH_SAMPLE
            + (
                STREAM_CHUNK_FRAMES - 1
            ) * FRAME_SHIFT_SAMPLE
        )

        chunk_shift_samples = (
            STREAM_CHUNK_FRAMES
            * FRAME_SHIFT_SAMPLE
        )

        chunk_length_bytes = (
            chunk_length_samples * 2
        )

        chunk_shift_bytes = (
            chunk_shift_samples * 2
        )

        batches = []

        while len(self.audio_buffer) >= chunk_length_bytes:

            chunk_bytes = bytes(
                self.audio_buffer[:chunk_length_bytes]
            )

            chunk = np.frombuffer(
                chunk_bytes,
                dtype=np.int16,
            )

            chunk_start = self.vad_samples_processed

            results = await self.detect_vad(
                chunk
            )

            self.vad_samples_processed += (
                chunk_shift_samples
            )

            batches.append({
                "results": results,
                "processed_end_sample": min(
                    chunk_start + chunk_length_samples,
                    self.samples_received,
                ),
            })

            del self.audio_buffer[
                :chunk_shift_bytes
            ]

        return batches

    # --------------------------------------------------------
    # Local cleanup
    # --------------------------------------------------------

    def reset(self):

        if self.closed:
            return

        try:
            self.vad.reset()

        except Exception:
            logger.exception(
                "[%s] VAD reset failed",
                self.session_id,
            )

        self.audio_buffer.clear()
        self.audio_history.clear()

        self.samples_received = 0
        self.vad_samples_processed = 0

        self.speech_active = False
        self.speech_start_frame = None

        self.asr_audio_cursor = 0
        self.last_partial_text = ""

        self.finalized = True
        self.closed = True

        elapsed = (
            time.perf_counter() - self.started_at
        )

        logger.info(
            "[%s] Local session cleanup | "
            "duration=%.2fs | "
            "vad_calls=%s | "
            "vad_total=%.3fs | "
            "vad_avg=%.2fms",
            self.session_id,
            elapsed,
            self.vad_calls,
            self.vad_processing_time,
            (
                self.vad_processing_time
                / self.vad_calls
                * 1000
                if self.vad_calls else 0
            ),
        )


# ============================================================
# VAD result handler
# ============================================================

async def handle_vad_results(
    session: AudioSession,
    batches,
    websocket: WebSocket,
):

    for batch in batches:

        results = batch["results"]

        processed_end_sample = (
            batch["processed_end_sample"]
        )

        for result in results:

            # --------------------------------------------
            # Speech start
            # --------------------------------------------

            if result.is_speech_start:

                start_frame = int(
                    result.speech_start_frame
                )

                session.speech_active = True
                session.speech_start_frame = start_frame

                start_sample = min(
                    frame_to_samples(start_frame),
                    session.samples_received,
                )

                session.asr_audio_cursor = max(
                    session.asr_audio_cursor,
                    start_sample,
                )

                await websocket.send_json({
                    "type": "speech_start",
                    "frame": start_frame,
                    "timestamp": frame_to_seconds(
                        start_frame
                    ),
                })

            # --------------------------------------------
            # Speech end
            # --------------------------------------------

            if result.is_speech_end:

                end_frame = int(
                    result.speech_end_frame
                )

                end_sample = min(
                    frame_to_samples(end_frame)
                    + FRAME_LENGTH_SAMPLE,
                    session.samples_received,
                )

                if session.speech_active:

                    await session.feed_asr_until(
                        end_sample,
                        websocket,
                    )

                session.speech_active = False

                await websocket.send_json({
                    "type": "speech_end",
                    "frame": end_frame,
                    "timestamp": frame_to_seconds(
                        end_frame
                    ),
                })

        # --------------------------------------------
        # Feed audio while speech is active
        # --------------------------------------------

        if session.speech_active:

            await session.feed_asr_until(
                processed_end_sample,
                websocket,
            )


# ============================================================
# WebSocket endpoint
# ============================================================

@app.websocket("/transcribe")
async def vad_websocket(websocket: WebSocket):

    await websocket.accept()

    session = None
    acquired = False

    try:

        # ----------------------------------------------------
        # Admission control
        # ----------------------------------------------------

        try:

            await asyncio.wait_for(
                app.state.session_semaphore.acquire(),
                timeout=SESSION_ACQUIRE_TIMEOUT,
            )

            acquired = True

        except asyncio.TimeoutError:

            await websocket.send_json({
                "type": "error",
                "message": "Server is currently at capacity",
            })

            await websocket.close(code=1013)

            return

        # ----------------------------------------------------
        # Create per-session VAD
        # ----------------------------------------------------

        vad_started = time.perf_counter()

        vad_model = await asyncio.to_thread(
            create_vad
        )

        vad_elapsed = (
            time.perf_counter() - vad_started
        )

        logger.info(
            "VAD created | device=%s | initialization=%.3fs",
            "GPU" if VAD_USE_GPU else "CPU",
            vad_elapsed,
        )

        session = AudioSession(
            vad_model,
            app.state.vad_semaphore,
        )

        app.state.active_sessions.add(
            session.session_id
        )

        # ----------------------------------------------------
        # Initialize ASR
        # ----------------------------------------------------

        await session.init_asr()

        logger.info(
            "WebSocket session started | "
            "session=%s | "
            "active=%s",
            session.session_id,
            len(app.state.active_sessions),
        )

        await websocket.send_json({
            "type": "session_started",
            "session_id": session.session_id,
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
        })

        # ----------------------------------------------------
        # Receive loop
        # ----------------------------------------------------

        while True:

            message = await websocket.receive()

            # =================================================
            # Binary PCM audio
            # =================================================

            if message.get("bytes") is not None:

                raw_bytes = message["bytes"]

                if len(raw_bytes) % 2 != 0:
                    raw_bytes = raw_bytes[:-1]

                if not raw_bytes:
                    continue

                audio = np.frombuffer(
                    raw_bytes,
                    dtype=np.int16,
                )

                batches = await session.process_audio(
                    audio
                )

                await handle_vad_results(
                    session,
                    batches,
                    websocket,
                )

            # =================================================
            # JSON commands
            # =================================================

            elif message.get("text") is not None:

                try:

                    command = json.loads(
                        message["text"]
                    )

                except json.JSONDecodeError:

                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format",
                    })

                    continue

                command_type = command.get("type")

                # --------------------------------------------
                # End of audio
                # --------------------------------------------

                if command_type == "end_of_audio":

                    logger.info(
                        "[%s] End of audio | duration=%.2fs",
                        session.session_id,
                        session.samples_received / SAMPLE_RATE,
                    )

                    flush_batches = await session.flush()

                    await handle_vad_results(
                        session,
                        flush_batches,
                        websocket,
                    )

                    if session.speech_active:

                        await session.feed_asr_until(
                            session.samples_received,
                            websocket,
                        )

                        session.speech_active = False

                    await session.finish_asr(
                        websocket
                    )

                    await websocket.send_json({
                        "type": "completed",
                        "duration": (
                            session.samples_received
                            / SAMPLE_RATE
                        ),
                    })

                    break

                # --------------------------------------------
                # Cancel
                # --------------------------------------------

                elif command_type == "cancel":

                    await websocket.send_json({
                        "type": "cancelled",
                        "session_id": session.session_id,
                    })

                    break

                else:

                    await websocket.send_json({
                        "type": "error",
                        "message": (
                            f"Unknown command: {command_type}"
                        ),
                    })

    except Exception as exc:

        if is_disconnect(exc):

            logger.info(
                "[%s] WebSocket disconnected",
                session.session_id if session else "unknown",
            )

        else:

            logger.exception(
                "[%s] WebSocket session failed",
                session.session_id if session else "unknown",
            )

            try:

                await websocket.send_json({
                    "type": "error",
                    "message": "Streaming ASR processing failed",
                })

            except Exception:
                pass

    finally:

        if session is not None:

            await session.close_asr()

            session.reset()

            app.state.active_sessions.discard(
                session.session_id
            )

            logger.info(
                "[%s] Session closed | active=%s",
                session.session_id,
                len(app.state.active_sessions),
            )

        if acquired:

            app.state.session_semaphore.release()


# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():

    active_sessions = len(
        app.state.active_sessions
    )

    try:

        result = await ASR_HANDLE.health.remote()

        return {
            "status": "ok",
            "vad": {
                "device": (
                    "GPU" if VAD_USE_GPU else "CPU"
                ),
                "mode": "per-session",
                "active_sessions": active_sessions,
                "max_active_sessions": MAX_ACTIVE_SESSIONS,
                "max_concurrent_inference": (
                    VAD_MAX_CONCURRENT_INFERENCE
                ),
                "chunk_frames": STREAM_CHUNK_FRAMES,
            },
            "asr": result,
        }

    except Exception:

        logger.exception(
            "ASR health check failed"
        )

        return {
            "status": "error",
            "vad": {
                "device": (
                    "GPU" if VAD_USE_GPU else "CPU"
                ),
                "active_sessions": active_sessions,
            },
            "asr": "unavailable",
        }


# ============================================================
# Standalone Uvicorn entrypoint
# ============================================================

# if __name__ == "__main__":

#     uvicorn.run(
#         app,
#         host=HOST,
#         port=PORT,
#         ws_ping_interval=WS_PING_INTERVAL,
#         ws_ping_timeout=WS_PING_TIMEOUT,
#         ws="websockets",
#         log_level="info",
#         workers=1,
#     )
