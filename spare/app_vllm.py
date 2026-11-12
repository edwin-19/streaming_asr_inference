
import asyncio
import json
import logging
import uuid
import time

import numpy as np
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
from qwen_asr.inference.qwen3_asr import Qwen3ASRModel


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI()

VAD_MODEL_DIR = (
    "/mnt/data/workspace/freelance/voice/diar/"
    "pretrained_models/FireRedVAD/Stream-VAD"
)

ASR_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B-hf"

STREAM_CHUNK_FRAMES = 10

ASR_CHUNK_SIZE_SECONDS = 2.0
ASR_UNFIXED_CHUNK_NUM = 2
ASR_UNFIXED_TOKEN_NUM = 5
ASR_MAX_NEW_TOKENS = 256
ASR_MAX_MODEL_LEN = 8192
ASR_GPU_MEMORY_UTILIZATION = 0.90


# ============================================================
# Model Loading
# ============================================================

def create_vad():

    config = FireRedStreamVadConfig(
        use_gpu=True,
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


logger.info("Loading Qwen3-ASR vLLM model...")

GLOBAL_ASR_MODEL = Qwen3ASRModel.LLM(
    model=ASR_MODEL_NAME,
    trust_remote_code=True,
    max_model_len=ASR_MAX_MODEL_LEN,
    gpu_memory_utilization=ASR_GPU_MEMORY_UTILIZATION,
    max_inference_batch_size=-1,
    max_new_tokens=ASR_MAX_NEW_TOKENS,
    # Speculative decoding (DFlash)
    speculative_config={
        "model": "mattymchen/Qwen3-ASR-1.7B-speculator.dflash2",
        "method": "dflash",
        "num_speculative_tokens": 7,
    },
)

logger.info("Qwen3-ASR vLLM loaded.")

# Shared vLLM engine lock.
GLOBAL_ASR_LOCK = asyncio.Lock()


# ============================================================
# Helpers
# ============================================================

def is_websocket_disconnect(exc: Exception) -> bool:
    return (
        isinstance(exc, WebSocketDisconnect)
        or type(exc).__name__ in {
            "WebSocketDisconnect",
            "WebSocketDisconnected",
        }
    )


def frame_to_seconds(frame: int) -> float:
    return (
        frame * FRAME_SHIFT_SAMPLE
    ) / SAMPLE_RATE


def frame_to_samples(frame: int) -> int:
    return frame * FRAME_SHIFT_SAMPLE


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32768.0


def state_to_dict(state) -> dict:
    return {
        "text": getattr(state, "text", "") or "",
        "language": getattr(state, "language", "") or "",
    }


def log_disconnect(session_id: str, exc: Exception):
    """
    Log WebSocket disconnect details.

    WebSocketDisconnect normally exposes:
        code
        reason
    """

    code = getattr(exc, "code", None)
    reason = getattr(exc, "reason", None)

    logger.warning(
        "[%s] WebSocket disconnected | type=%s code=%s reason=%r",
        session_id,
        type(exc).__name__,
        code,
        reason,
    )


# ============================================================
# Audio Session
# ============================================================

class AudioSession:

    def __init__(
        self,
        vad_model: FireRedStreamVad,
        asr_model: Qwen3ASRModel,
    ):

        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"

        self.vad = vad_model
        self.asr = asr_model

        # ----------------------------------------------------
        # VAD state
        # ----------------------------------------------------

        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.audio_history = np.empty(0, dtype=np.int16)

        self.samples_received = 0
        self.vad_samples_processed = 0

        self.speech_active = False
        self.speech_start_frame = None

        # ----------------------------------------------------
        # ASR state
        # ----------------------------------------------------

        self.streaming_state = None

        self.asr_audio_cursor = 0
        self.last_partial_text = ""

        self.finalized = False

        self.asr_lock = asyncio.Lock()

        self.vad.reset()

        self.init_asr_state()

    # --------------------------------------------------------
    # Qwen initialization
    # --------------------------------------------------------

    def init_asr_state(self):

        self.streaming_state = self.asr.init_streaming_state(
            context="",
            language=None,
            unfixed_chunk_num=ASR_UNFIXED_CHUNK_NUM,
            unfixed_token_num=ASR_UNFIXED_TOKEN_NUM,
            chunk_size_sec=ASR_CHUNK_SIZE_SECONDS,
        )

        self.asr_audio_cursor = 0
        self.last_partial_text = ""
        self.finalized = False

    # --------------------------------------------------------
    # vLLM Streaming ASR
    # --------------------------------------------------------

    async def feed_asr_until(
        self,
        end_sample: int,
        websocket: WebSocket,
    ):

        if self.streaming_state is None or self.finalized:
            return

        async with self.asr_lock:

            if self.streaming_state is None or self.finalized:
                return

            end_sample = min(
                end_sample,
                self.samples_received,
            )

            start_sample = self.asr_audio_cursor

            if end_sample <= start_sample:
                return

            audio = self.audio_history[
                start_sample:end_sample
            ]

            if len(audio) == 0:
                return

            pcm16k = pcm_to_float32(audio)

            old_chunk_id = getattr(
                self.streaming_state,
                "chunk_id",
                0,
            )

            old_text = getattr(
                self.streaming_state,
                "text",
                "",
            ) or ""

            loop = asyncio.get_running_loop()

            # ------------------------------------------------
            # Measure lock wait separately from inference
            # ------------------------------------------------

            lock_wait_start = loop.time()

            async with GLOBAL_ASR_LOCK:

                lock_wait_sec = (
                    loop.time() - lock_wait_start
                )

                inference_start = loop.time()

                self.streaming_state = await asyncio.to_thread(
                    self.asr.streaming_transcribe,
                    pcm16k,
                    self.streaming_state,
                )

                inference_sec = (
                    loop.time() - inference_start
                )

            elapsed = lock_wait_sec + inference_sec

            self.asr_audio_cursor = end_sample

            new_chunk_id = getattr(
                self.streaming_state,
                "chunk_id",
                old_chunk_id,
            )

            payload = state_to_dict(
                self.streaming_state
            )

            text = payload["text"]

            decoded_chunk = new_chunk_id != old_chunk_id

            logger.info(
                "[%s] ASR | audio=%.2fs | lock_wait=%.3fs "
                "| inference=%.3fs | total=%.3fs "
                "| chunk_id=%s->%s",
                self.session_id,
                len(pcm16k) / SAMPLE_RATE,
                lock_wait_sec,
                inference_sec,
                elapsed,
                old_chunk_id,
                new_chunk_id,
            )

            if not decoded_chunk or not text or text == old_text:
                return

            self.last_partial_text = text

        # Send outside the session ASR lock.
        await websocket.send_json({
            "type": "transcript",
            **payload,
            "audio_end_sec": end_sample / SAMPLE_RATE,
            "decode_wall_sec": elapsed,
            "is_final": False,
        })

    # --------------------------------------------------------
    # Final ASR
    # --------------------------------------------------------

    async def finish_asr(
        self,
        websocket: WebSocket,
    ):

        if self.streaming_state is None or self.finalized:
            return

        async with self.asr_lock:

            if self.streaming_state is None or self.finalized:
                return

            old_chunk_id = getattr(
                self.streaming_state,
                "chunk_id",
                0,
            )

            loop = asyncio.get_running_loop()

            lock_wait_start = loop.time()

            async with GLOBAL_ASR_LOCK:

                lock_wait_sec = (
                    loop.time() - lock_wait_start
                )

                inference_start = loop.time()

                self.streaming_state = await asyncio.to_thread(
                    self.asr.finish_streaming_transcribe,
                    self.streaming_state,
                )

                inference_sec = (
                    loop.time() - inference_start
                )

            elapsed = lock_wait_sec + inference_sec

            new_chunk_id = getattr(
                self.streaming_state,
                "chunk_id",
                old_chunk_id,
            )

            self.finalized = True

            logger.info(
                "[%s] ASR FINAL | audio=%.2fs "
                "| lock_wait=%.3fs | inference=%.3fs "
                "| total=%.3fs | chunk_id=%s->%s",
                self.session_id,
                self.samples_received / SAMPLE_RATE,
                lock_wait_sec,
                inference_sec,
                elapsed,
                old_chunk_id,
                new_chunk_id,
            )

        payload = state_to_dict(
            self.streaming_state
        )

        await websocket.send_json({
            "type": "transcript",
            **payload,
            "audio_end_sec": self.samples_received / SAMPLE_RATE,
            "decode_wall_sec": (
                elapsed
                if new_chunk_id != old_chunk_id
                else 0.0
            ),
            "is_final": True,
        })

    # --------------------------------------------------------
    # VAD audio processing
    # --------------------------------------------------------

    async def process_audio(
        self,
        audio: np.ndarray,
    ):

        self.audio_history = np.concatenate(
            (self.audio_history, audio)
        )

        self.audio_buffer = np.concatenate(
            (self.audio_buffer, audio)
        )

        self.samples_received += len(audio)

        chunk_length = (
            FRAME_LENGTH_SAMPLE
            + (STREAM_CHUNK_FRAMES - 1) * FRAME_SHIFT_SAMPLE
        )

        chunk_shift = (
            STREAM_CHUNK_FRAMES * FRAME_SHIFT_SAMPLE
        )

        batches = []

        while len(self.audio_buffer) >= chunk_length:

            chunk = self.audio_buffer[:chunk_length]

            chunk_start = self.vad_samples_processed

            vad_started = time.perf_counter()

            chunk_results = await asyncio.to_thread(
                self.vad.detect_chunk,
                chunk,
            )

            vad_elapsed = (
                time.perf_counter() - vad_started
            )

            self.vad_samples_processed += chunk_shift

            chunk_end = min(
                chunk_start + chunk_length,
                self.samples_received,
            )

            logger.debug(
                "[%s] VAD | processing=%.4fs | audio=%.3fs",
                self.session_id,
                vad_elapsed,
                len(chunk) / SAMPLE_RATE,
            )

            batches.append({
                "results": chunk_results,
                "processed_end_sample": chunk_end,
            })

            self.audio_buffer = self.audio_buffer[chunk_shift:]

        return batches

    # --------------------------------------------------------
    # VAD flush
    # --------------------------------------------------------

    async def flush(self):

        silence_samples = (
            FRAME_LENGTH_SAMPLE
            + 50 * FRAME_SHIFT_SAMPLE
        )

        silence = np.zeros(
            silence_samples,
            dtype=np.int16,
        )

        self.audio_buffer = np.concatenate(
            (self.audio_buffer, silence)
        )

        chunk_length = (
            FRAME_LENGTH_SAMPLE
            + (STREAM_CHUNK_FRAMES - 1) * FRAME_SHIFT_SAMPLE
        )

        chunk_shift = (
            STREAM_CHUNK_FRAMES * FRAME_SHIFT_SAMPLE
        )

        batches = []

        while len(self.audio_buffer) >= chunk_length:

            chunk = self.audio_buffer[:chunk_length]

            chunk_start = self.vad_samples_processed

            vad_started = time.perf_counter()

            chunk_results = await asyncio.to_thread(
                self.vad.detect_chunk,
                chunk,
            )

            vad_elapsed = (
                time.perf_counter() - vad_started
            )

            self.vad_samples_processed += chunk_shift

            batches.append({
                "results": chunk_results,
                "processed_end_sample": min(
                    chunk_start + chunk_length,
                    self.samples_received,
                ),
            })

            logger.debug(
                "[%s] VAD FLUSH | processing=%.4fs",
                self.session_id,
                vad_elapsed,
            )

            self.audio_buffer = self.audio_buffer[chunk_shift:]

        return batches

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def reset(self):

        self.vad.reset()

        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.audio_history = np.empty(0, dtype=np.int16)

        self.samples_received = 0
        self.vad_samples_processed = 0

        self.speech_active = False
        self.speech_start_frame = None

        self.streaming_state = None

        self.asr_audio_cursor = 0
        self.last_partial_text = ""

        self.finalized = True


# ============================================================
# VAD Result Handler
# ============================================================

async def handle_vad_results(
    session: AudioSession,
    batches,
    websocket: WebSocket,
):

    for batch in batches:

        results = batch["results"]

        processed_end_sample = batch["processed_end_sample"]

        for result in results:

            # ------------------------------------------------
            # Speech start
            # ------------------------------------------------

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
                    "timestamp": frame_to_seconds(start_frame),
                })

            # ------------------------------------------------
            # Speech end
            # ------------------------------------------------

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
                    "timestamp": frame_to_seconds(end_frame),
                })

        # ----------------------------------------------------
        # Feed active speech audio
        # ----------------------------------------------------

        if session.speech_active:

            await session.feed_asr_until(
                processed_end_sample,
                websocket,
            )


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/transcribe")
async def vad_websocket(websocket: WebSocket):

    await websocket.accept()

    session = None

    try:

        vad_started = time.perf_counter()

        vad_model = await asyncio.to_thread(
            create_vad
        )

        logger.info(
            "VAD created | initialization=%.3fs",
            time.perf_counter() - vad_started,
        )

        session = AudioSession(
            vad_model,
            GLOBAL_ASR_MODEL,
        )

        logger.info(
            "New WebSocket session: %s",
            session.session_id,
        )

        await websocket.send_json({
            "type": "session_started",
            "session_id": session.session_id,
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
        })

        while True:

            try:
                message = await websocket.receive()

            except Exception as exc:

                if is_websocket_disconnect(exc):
                    log_disconnect(session.session_id, exc)
                    break

                raise

            # ------------------------------------------------
            # Binary PCM audio
            # ------------------------------------------------

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

                receive_started = time.perf_counter()

                batches = await session.process_audio(
                    audio
                )

                vad_processing_elapsed = (
                    time.perf_counter() - receive_started
                )

                logger.debug(
                    "[%s] Audio received | bytes=%d "
                    "| samples=%d | total_audio=%.2fs "
                    "| VAD processing=%.4fs",
                    session.session_id,
                    len(raw_bytes),
                    len(audio),
                    session.samples_received / SAMPLE_RATE,
                    vad_processing_elapsed,
                )

                await handle_vad_results(
                    session,
                    batches,
                    websocket,
                )

            # ------------------------------------------------
            # JSON commands
            # ------------------------------------------------

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
                        "[%s] Received end_of_audio | audio=%.2fs",
                        session.session_id,
                        session.samples_received / SAMPLE_RATE,
                    )

                    flush_started = time.perf_counter()

                    flush_batches = await session.flush()

                    logger.info(
                        "[%s] VAD flush completed | elapsed=%.3fs",
                        session.session_id,
                        time.perf_counter() - flush_started,
                    )

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
                            session.samples_received / SAMPLE_RATE
                        ),
                    })

                    logger.info(
                        "[%s] Session completed | audio=%.2fs",
                        session.session_id,
                        session.samples_received / SAMPLE_RATE,
                    )

                    break

                # --------------------------------------------
                # Cancel
                # --------------------------------------------

                elif command_type == "cancel":

                    logger.info(
                        "[%s] Received cancel command",
                        session.session_id,
                    )

                    await websocket.send_json({
                        "type": "cancelled",
                        "session_id": session.session_id,
                    })

                    break

    except Exception as exc:

        if is_websocket_disconnect(exc):

            if session is not None:
                log_disconnect(session.session_id, exc)
            else:
                log_disconnect("unknown", exc)

        else:

            logger.exception(
                "Session failed: %s",
                session.session_id if session else "unknown",
            )

            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Streaming ASR processing failed",
                })

            except Exception:
                logger.warning(
                    "[%s] Could not send error message to client",
                    session.session_id if session else "unknown",
                )

    finally:

        if session is not None:

            session.reset()

            logger.info(
                "Session closed: %s",
                session.session_id,
            )


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "vad": "per-session",
        "asr": "vllm_streaming",
        "asr_model": ASR_MODEL_NAME,
    }

