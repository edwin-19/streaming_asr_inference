import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from ray import serve

from fireredvad.core.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig


# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("websocket_gateway")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    vad_model_dir: str = os.getenv(
        "VAD_MODEL_DIR",
        "/mnt/data/workspace/freelance/voice/diar/"
        "pretrained_models/FireRedVAD/Stream-VAD",
    )
    vad_use_gpu: bool = os.getenv("VAD_USE_GPU", "false").lower() == "true"
    stream_chunk_frames: int = env_int("STREAM_CHUNK_FRAMES", 20)
    vad_max_concurrent_inference: int = env_int(
        "VAD_MAX_CONCURRENT_INFERENCE", 20
    )
    max_active_sessions: int = env_int("MAX_ACTIVE_SESSIONS", 1000)
    session_acquire_timeout: float = env_float("SESSION_ACQUIRE_TIMEOUT", 0.5)
    max_audio_message_bytes: int = env_int(
        "MAX_AUDIO_MESSAGE_BYTES", SAMPLE_RATE * 2
    )
    max_session_duration_seconds: int = env_int(
        "MAX_SESSION_DURATION_SECONDS", 3600
    )
    max_buffer_seconds: float = env_float("MAX_BUFFER_SECONDS", 30.0)
    audio_preroll_seconds: float = env_float("AUDIO_PREROLL_SECONDS", 0.5)
    asr_update_interval_seconds: float = env_float(
        "ASR_UPDATE_INTERVAL_SECONDS", 0.5
    )
    asr_call_timeout_seconds: float = env_float("ASR_CALL_TIMEOUT_SECONDS", 30.0)
    cleanup_timeout_seconds: float = env_float("CLEANUP_TIMEOUT_SECONDS", 5.0)
    max_audio_queue_chunks: int = env_int("MAX_AUDIO_QUEUE_CHUNKS", 10)
    audio_queue_put_timeout: float = env_float("AUDIO_QUEUE_PUT_TIMEOUT", 1.0)
    ray_address: str = os.getenv("RAY_ADDRESS", "auto")
    ray_namespace: str = os.getenv("RAY_NAMESPACE", "default")
    serve_app_name: str = os.getenv("SERVE_APP_NAME", "default")
    asr_deployment_name: str = os.getenv("ASR_DEPLOYMENT_NAME", "ASRRouter")
    gateway_host: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
    gateway_port: int = env_int("GATEWAY_PORT", 8000)


CFG = Settings()


# ============================================================================
# Prometheus metrics
# ============================================================================

WS_ACTIVE_SESSIONS = Gauge("ws_active_sessions", "Active WebSocket sessions")
WS_SESSIONS_TOTAL = Counter("ws_sessions_total", "Total WebSocket sessions")
WS_SESSION_ERRORS = Counter(
    "ws_session_errors_total", "WebSocket session errors", ["kind"]
)
WS_AUDIO_BYTES = Counter(
    "ws_audio_received_bytes_total", "Total PCM bytes received"
)
WS_SESSION_DURATION = Histogram(
    "ws_session_duration_seconds", "WebSocket session duration"
)
VAD_INFERENCE = Histogram(
    "vad_inference_seconds", "FireRedVAD inference wall time"
)
VAD_ACTIVE = Gauge("vad_active_inferences", "Active VAD inference calls")
ASR_REMOTE_LATENCY = Histogram(
    "asr_remote_latency_seconds", "ASR Ray remote call wall time", ["operation"]
)
ASR_AUDIO_SECONDS = Counter(
    "asr_audio_submitted_seconds_total", "Audio seconds submitted to ASR"
)
AUDIO_BUFFER_BYTES = Gauge(
    "audio_buffer_bytes", "Current rolling audio buffer bytes", ["session_id"]
)
AUDIO_QUEUE_DEPTH = Gauge(
    "audio_queue_depth", "Current per-session incoming audio queue depth"
)


# ============================================================================
# FireRedVAD factory
# ============================================================================

def create_vad() -> FireRedStreamVad:
    vad_config = FireRedStreamVadConfig(
        use_gpu=CFG.vad_use_gpu,
        smooth_window_size=5,
        speech_threshold=0.3,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=40,
        chunk_max_frame=30000,
    )
    return FireRedStreamVad.from_pretrained(CFG.vad_model_dir, vad_config)


# ============================================================================
# Helpers
# ============================================================================

def frame_to_seconds(frame: int) -> float:
    # FireRed Stream-VAD frame indices are one-based.
    return max(0, frame - 1) * FRAME_SHIFT_SAMPLE / SAMPLE_RATE


def frame_to_samples(frame: int) -> int:
    # FireRed Stream-VAD frame indices are one-based.
    return max(0, frame - 1) * FRAME_SHIFT_SAMPLE


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32, copy=False) / 32768.0


def is_disconnect(exc: BaseException) -> bool:
    return isinstance(exc, WebSocketDisconnect) or type(exc).__name__ in {
        "WebSocketDisconnect",
        "WebSocketDisconnected",
    }


async def await_with_timeout(awaitable, timeout: float, operation: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Timed out during {operation} after {timeout}s") from exc


# ============================================================================
# Application lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Connecting to Ray | address=%s namespace=%s", CFG.ray_address, CFG.ray_namespace)
    ray.init(address=CFG.ray_address, namespace=CFG.ray_namespace, ignore_reinit_error=True)

    application.state.asr_handle = serve.get_deployment_handle(
        CFG.asr_deployment_name,
        app_name=CFG.serve_app_name,
    )
    application.state.session_semaphore = asyncio.Semaphore(CFG.max_active_sessions)
    application.state.vad_semaphore = asyncio.Semaphore(CFG.vad_max_concurrent_inference)
    application.state.active_sessions = set()
    application.state.ready = True

    logger.info(
        "Gateway ready | VAD=%s | sessions=%s | VAD concurrency=%s | chunk_frames=%s",
        "GPU" if CFG.vad_use_gpu else "CPU",
        CFG.max_active_sessions,
        CFG.vad_max_concurrent_inference,
        CFG.stream_chunk_frames,
    )
    try:
        yield
    finally:
        application.state.ready = False
        logger.info("Gateway shutting down | active_sessions=%s", len(application.state.active_sessions))
        # Do not call ray.shutdown(): this process may share the Ray connection.


app = FastAPI(lifespan=lifespan)


# ============================================================================
# Audio session
# ============================================================================

class AudioSession:
    def __init__(self, vad_model: FireRedStreamVad, vad_semaphore: asyncio.Semaphore, asr_handle):
        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self.vad = vad_model
        self.vad_semaphore = vad_semaphore
        self.asr_handle = asr_handle

        self.asr_worker = None
        self.asr_session_id = None
        self.asr_released = False
        self.asr_finalized = False

        # VAD's overlapping input window and the ASR rolling audio window are separate.
        self.vad_input_buffer = bytearray()
        self.audio_ring = bytearray()
        self.audio_base_sample = 0

        self.samples_received = 0
        self.vad_samples_processed = 0
        self.asr_audio_cursor = 0
        self.last_asr_submit_sample = 0

        self.speech_active = False
        self.speech_start_frame = None
        self.last_partial_text = ""

        self.closed = False
        self.started_at = time.perf_counter()
        self.vad_calls = 0
        self.vad_processing_time = 0.0

        self.vad.reset()

    # ---------------------------------------------------------------------
    # ASR lifecycle
    # ---------------------------------------------------------------------

    async def init_asr(self):
        started = time.perf_counter()
        result = await await_with_timeout(
            self.asr_handle.start_session.remote(self.session_id),
            CFG.asr_call_timeout_seconds,
            "ASR start_session",
        )
        ASR_REMOTE_LATENCY.labels(operation="start").observe(time.perf_counter() - started)
        self.asr_session_id = result["session_id"]
        self.asr_worker = result["worker_handle"]
        logger.info("[%s] ASR assigned | worker=%s", self.session_id, result.get("worker_id"))

    async def _release_router_session(self):
        if self.asr_released:
            return
        try:
            await await_with_timeout(
                self.asr_handle.release_session.remote(self.session_id),
                CFG.cleanup_timeout_seconds,
                "ASR release_session",
            )
            self.asr_released = True
        except Exception:
            # Leave the flag false so final cleanup can make another attempt.
            logger.exception("[%s] Failed to release router session", self.session_id)

    async def feed_asr_until(self, end_sample: int, websocket: WebSocket, force: bool = False):
        if self.asr_worker is None or self.asr_session_id is None or self.asr_finalized:
            return

        end_sample = min(max(0, end_sample), self.samples_received)
        start_sample = self.asr_audio_cursor
        if end_sample <= start_sample:
            return

        # Avoid making remote calls for every VAD chunk. Speech-end forces a final update.
        min_interval = int(CFG.asr_update_interval_seconds * SAMPLE_RATE)
        if not force and end_sample - self.last_asr_submit_sample < min_interval:
            return

        if start_sample < self.audio_base_sample:
            logger.error(
                "[%s] ASR audio fell behind rolling buffer | cursor=%s base=%s",
                self.session_id, start_sample, self.audio_base_sample,
            )
            raise RuntimeError("ASR fell behind the audio buffer")

        offset_samples = start_sample - self.audio_base_sample
        count_samples = end_sample - start_sample
        offset_bytes = offset_samples * 2
        end_bytes = offset_bytes + count_samples * 2
        if end_bytes > len(self.audio_ring):
            raise RuntimeError("Requested ASR audio is outside the rolling buffer")

        audio = np.frombuffer(
            self.audio_ring[offset_bytes:end_bytes], dtype=np.int16
        )
        if audio.size == 0:
            return

        pcm_float = pcm_to_float32(audio)
        started = time.perf_counter()
        result = await await_with_timeout(
            self.asr_worker.process_audio.remote(self.asr_session_id, pcm_float),
            CFG.asr_call_timeout_seconds,
            "ASR process_audio",
        )
        elapsed = time.perf_counter() - started
        ASR_REMOTE_LATENCY.labels(operation="process").observe(elapsed)
        ASR_AUDIO_SECONDS.inc(len(pcm_float) / SAMPLE_RATE)

        # Advance only after the worker confirms successful processing.
        self.asr_audio_cursor = end_sample
        self.last_asr_submit_sample = end_sample
        timing = result.get("timing", {})
        logger.info(
            "[%s] ASR | audio=%.3fs remote=%.3fs inference=%.3fs wait=%.3fs decoded=%s worker=%s",
            self.session_id,
            len(pcm_float) / SAMPLE_RATE,
            elapsed,
            timing.get("inference", 0.0),
            timing.get("semaphore_wait", 0.0),
            result.get("decoded"),
            result.get("worker_id"),
        )

        if result.get("decoded"):
            payload = result["transcript"]
            self.last_partial_text = payload.get("text", "")
            await websocket.send_json({
                "type": "transcript",
                **payload,
                "audio_end_sec": end_sample / SAMPLE_RATE,
                "decode_wall_sec": elapsed,
                "is_final": False,
            })

        self.trim_audio_ring()

    async def finish_asr(self, websocket: WebSocket):
        if self.asr_worker is None or self.asr_session_id is None or self.asr_finalized:
            return

        worker = self.asr_worker
        session_id = self.asr_session_id
        started = time.perf_counter()
        finished = False
        try:
            result = await await_with_timeout(
                worker.finish_session.remote(session_id),
                CFG.asr_call_timeout_seconds,
                "ASR finish_session",
            )
            elapsed = time.perf_counter() - started
            ASR_REMOTE_LATENCY.labels(operation="finish").observe(elapsed)
            self.asr_finalized = True
            finished = True
            await websocket.send_json({
                "type": "transcript",
                **result["transcript"],
                "audio_end_sec": self.samples_received / SAMPLE_RATE,
                "decode_wall_sec": elapsed,
                "is_final": True,
            })
            logger.info("[%s] ASR finalized | elapsed=%.3fs worker=%s", self.session_id, elapsed, result.get("worker_id"))
        finally:
            if not finished:
                with suppress(Exception):
                    await await_with_timeout(
                        worker.cancel_session.remote(session_id),
                        CFG.cleanup_timeout_seconds,
                        "ASR cancel after finish failure",
                    )
            await self._release_router_session()
            self.asr_worker = None
            self.asr_session_id = None

    async def close_asr(self):
        worker = self.asr_worker
        session_id = self.asr_session_id
        if session_id is None:
            return
        try:
            if worker is not None and not self.asr_finalized:
                await await_with_timeout(
                    worker.cancel_session.remote(session_id),
                    CFG.cleanup_timeout_seconds,
                    "ASR cancel_session",
                )
        except Exception:
            logger.exception("[%s] Failed to cancel ASR worker session", self.session_id)
        finally:
            await self._release_router_session()
            self.asr_worker = None
            self.asr_session_id = None

    # ---------------------------------------------------------------------
    # Audio rolling buffer
    # ---------------------------------------------------------------------

    def append_audio(self, audio: np.ndarray):
        raw = audio.tobytes()
        self.audio_ring.extend(raw)
        self.samples_received += len(audio)
        AUDIO_BUFFER_BYTES.labels(session_id=self.session_id).set(len(self.audio_ring))

        max_samples = int(CFG.max_buffer_seconds * SAMPLE_RATE)
        if len(self.audio_ring) // 2 > max_samples:
            # Silence can be discarded because only a short pre-roll is needed
            # before the next VAD speech-start event. During speech, never drop
            # unsubmitted audio: fail explicitly if ASR falls too far behind.
            if not self.speech_active:
                self._drop_audio_before(max(self.audio_base_sample, self.samples_received - int(CFG.audio_preroll_seconds * SAMPLE_RATE)))
            else:
                new_base = self.samples_received - max_samples
                if new_base > self.asr_audio_cursor:
                    raise RuntimeError(
                        "Audio buffer limit reached while ASR is behind; refusing to drop audio"
                    )
                self._drop_audio_before(max(self.audio_base_sample, self.asr_audio_cursor - int(CFG.audio_preroll_seconds * SAMPLE_RATE)))

    def _drop_audio_before(self, sample: int):
        sample = min(max(sample, self.audio_base_sample), self.samples_received)
        drop_samples = sample - self.audio_base_sample
        if drop_samples <= 0:
            return
        del self.audio_ring[: drop_samples * 2]
        self.audio_base_sample = sample
        AUDIO_BUFFER_BYTES.labels(session_id=self.session_id).set(len(self.audio_ring))

    def trim_audio_ring(self):
        preroll = int(CFG.audio_preroll_seconds * SAMPLE_RATE)
        if self.speech_active:
            keep_from = max(self.audio_base_sample, self.asr_audio_cursor - preroll)
        else:
            keep_from = max(self.audio_base_sample, self.samples_received - preroll)
        self._drop_audio_before(keep_from)

    # ---------------------------------------------------------------------
    # VAD processing
    # ---------------------------------------------------------------------

    async def detect_vad(self, audio_chunk: np.ndarray):
        started = time.perf_counter()
        async with self.vad_semaphore:
            VAD_ACTIVE.inc()
            try:
                results = await asyncio.to_thread(self.vad.detect_chunk, audio_chunk)
            finally:
                VAD_ACTIVE.dec()
        elapsed = time.perf_counter() - started
        self.vad_calls += 1
        self.vad_processing_time += elapsed
        VAD_INFERENCE.observe(elapsed)
        return results

    def _chunk_sizes(self):
        chunk_length_samples = FRAME_LENGTH_SAMPLE + (CFG.stream_chunk_frames - 1) * FRAME_SHIFT_SAMPLE
        chunk_shift_samples = CFG.stream_chunk_frames * FRAME_SHIFT_SAMPLE
        return chunk_length_samples, chunk_shift_samples

    async def process_audio(self, audio: np.ndarray):
        if self.closed:
            raise RuntimeError("Session already closed")
        self.append_audio(audio)
        self.vad_input_buffer.extend(audio.tobytes())

        chunk_length_samples, chunk_shift_samples = self._chunk_sizes()
        chunk_length_bytes = chunk_length_samples * 2
        chunk_shift_bytes = chunk_shift_samples * 2
        batches = []

        while len(self.vad_input_buffer) >= chunk_length_bytes:
            chunk = np.frombuffer(bytes(self.vad_input_buffer[:chunk_length_bytes]), dtype=np.int16)
            results = await self.detect_vad(chunk)
            self.vad_samples_processed += chunk_shift_samples
            processed_end = min(self.vad_samples_processed + (chunk_length_samples - chunk_shift_samples), self.samples_received)
            batches.append({"results": results, "processed_end_sample": processed_end})
            del self.vad_input_buffer[:chunk_shift_bytes]

        return batches

    async def flush(self):
        if self.closed:
            return []
        # Synthetic silence is fed to VAD only; it is never appended to real audio/ASR buffers.
        silence_samples = FRAME_LENGTH_SAMPLE + 50 * FRAME_SHIFT_SAMPLE
        self.vad_input_buffer.extend(np.zeros(silence_samples, dtype=np.int16).tobytes())

        chunk_length_samples, chunk_shift_samples = self._chunk_sizes()
        chunk_length_bytes = chunk_length_samples * 2
        chunk_shift_bytes = chunk_shift_samples * 2
        batches = []
        while len(self.vad_input_buffer) >= chunk_length_bytes:
            chunk = np.frombuffer(bytes(self.vad_input_buffer[:chunk_length_bytes]), dtype=np.int16)
            results = await self.detect_vad(chunk)
            self.vad_samples_processed += chunk_shift_samples
            processed_end = min(self.vad_samples_processed + (chunk_length_samples - chunk_shift_samples), self.samples_received)
            batches.append({"results": results, "processed_end_sample": processed_end})
            del self.vad_input_buffer[:chunk_shift_bytes]
        self.vad_input_buffer.clear()
        return batches

    def cleanup_local(self):
        if self.closed:
            return
        with suppress(Exception):
            self.vad.reset()
        self.vad_input_buffer.clear()
        self.audio_ring.clear()
        self.closed = True
        AUDIO_BUFFER_BYTES.labels(session_id=self.session_id).set(0)
        with suppress(Exception):
            AUDIO_BUFFER_BYTES.remove(self.session_id)
        elapsed = time.perf_counter() - self.started_at
        WS_SESSION_DURATION.observe(elapsed)
        logger.info(
            "[%s] Cleanup | duration=%.2fs vad_calls=%s vad_total=%.3fs vad_avg=%.2fms",
            self.session_id,
            elapsed,
            self.vad_calls,
            self.vad_processing_time,
            self.vad_processing_time / self.vad_calls * 1000 if self.vad_calls else 0,
        )


# ============================================================================
# VAD event handling
# ============================================================================

async def handle_vad_results(session: AudioSession, batches: list[dict[str, Any]], websocket: WebSocket):
    for batch in batches:
        for result in batch["results"]:
            if result.is_speech_start:
                start_frame = int(result.speech_start_frame)
                start_sample = min(frame_to_samples(start_frame), session.samples_received)
                session.speech_active = True
                session.speech_start_frame = start_frame
                # Retain any pre-roll still present in the ring. Do not move backwards
                # beyond the rolling buffer's current base.
                session.asr_audio_cursor = max(session.asr_audio_cursor, session.audio_base_sample, start_sample)
                session.last_asr_submit_sample = session.asr_audio_cursor
                await websocket.send_json({
                    "type": "speech_start",
                    "frame": start_frame,
                    "timestamp": frame_to_seconds(start_frame),
                })

            if result.is_speech_end:
                end_frame = int(result.speech_end_frame)
                end_sample = min(
                    frame_to_samples(end_frame) + FRAME_LENGTH_SAMPLE,
                    session.samples_received,
                )
                if session.speech_active:
                    await session.feed_asr_until(end_sample, websocket, force=True)
                session.speech_active = False
                await websocket.send_json({
                    "type": "speech_end",
                    "frame": end_frame,
                    "timestamp": frame_to_seconds(end_frame),
                })
                session.trim_audio_ring()

        if session.speech_active:
            await session.feed_asr_until(batch["processed_end_sample"], websocket)

    # Keep only pre-roll while idle. This also bounds long-silence sessions.
    if not session.speech_active:
        session.trim_audio_ring()


# ============================================================================
# WebSocket endpoint
# ============================================================================

@app.websocket("/transcribe")
async def transcribe(websocket: WebSocket):
    await websocket.accept()
    session = None
    acquired = False
    WS_SESSIONS_TOTAL.inc()

    try:
        try:
            await asyncio.wait_for(
                app.state.session_semaphore.acquire(),
                timeout=CFG.session_acquire_timeout,
            )
            acquired = True
        except asyncio.TimeoutError:
            WS_SESSION_ERRORS.labels(kind="capacity").inc()
            await websocket.send_json({"type": "error", "message": "Server is currently at capacity"})
            await websocket.close(code=1013)
            return

        vad_started = time.perf_counter()
        vad_model = await asyncio.to_thread(create_vad)
        logger.info("VAD initialized | device=%s elapsed=%.3fs", "GPU" if CFG.vad_use_gpu else "CPU", time.perf_counter() - vad_started)

        session = AudioSession(vad_model, app.state.vad_semaphore, app.state.asr_handle)
        app.state.active_sessions.add(session.session_id)
        WS_ACTIVE_SESSIONS.inc()

        await session.init_asr()
        await websocket.send_json({
            "type": "session_started",
            "session_id": session.session_id,
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "channels": 1,
        })

        while True:
            if time.perf_counter() - session.started_at > CFG.max_session_duration_seconds:
                await websocket.send_json({"type": "error", "message": "Maximum session duration exceeded"})
                await websocket.close(code=1008)
                break

            message = await websocket.receive()
            raw_bytes = message.get("bytes")
            text_message = message.get("text")

            if raw_bytes is not None:
                if len(raw_bytes) > CFG.max_audio_message_bytes:
                    WS_SESSION_ERRORS.labels(kind="oversized_message").inc()
                    await websocket.send_json({"type": "error", "message": "Audio message too large"})
                    await websocket.close(code=1009)
                    break
                if len(raw_bytes) % 2:
                    WS_SESSION_ERRORS.labels(kind="invalid_pcm").inc()
                    await websocket.send_json({"type": "error", "message": "PCM16 payload must contain an even number of bytes"})
                    continue
                if not raw_bytes:
                    continue

                audio = np.frombuffer(raw_bytes, dtype=np.int16)
                WS_AUDIO_BYTES.inc(len(raw_bytes))
                batches = await session.process_audio(audio)
                await handle_vad_results(session, batches, websocket)

            elif text_message is not None:
                try:
                    command = json.loads(text_message)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                    continue

                command_type = command.get("type")
                if command_type == "end_of_audio":
                    logger.info("[%s] End of audio | duration=%.2fs", session.session_id, session.samples_received / SAMPLE_RATE)
                    batches = await session.flush()
                    await handle_vad_results(session, batches, websocket)
                    if session.speech_active:
                        await session.feed_asr_until(session.samples_received, websocket, force=True)
                        session.speech_active = False
                    await session.finish_asr(websocket)
                    await websocket.send_json({"type": "completed", "duration": session.samples_received / SAMPLE_RATE})
                    break

                elif command_type == "cancel":
                    await websocket.send_json({"type": "cancelled", "session_id": session.session_id})
                    break

                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown command: {command_type}"})

    except Exception as exc:
        if is_disconnect(exc):
            logger.info("[%s] WebSocket disconnected", session.session_id if session else "unknown")
        else:
            WS_SESSION_ERRORS.labels(kind=type(exc).__name__).inc()
            logger.exception("[%s] WebSocket session failed", session.session_id if session else "unknown")
            with suppress(Exception):
                await websocket.send_json({"type": "error", "message": "Streaming ASR processing failed"})
    finally:
        if session is not None:
            try:
                await session.close_asr()
            finally:
                session.cleanup_local()
                app.state.active_sessions.discard(session.session_id)
                WS_ACTIVE_SESSIONS.dec()
                logger.info("[%s] Session closed | active=%s", session.session_id, len(app.state.active_sessions))
        if acquired:
            app.state.session_semaphore.release()


# ============================================================================
# Health and metrics
# ============================================================================

@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    if not getattr(app.state, "ready", False):
        return PlainTextResponse("not ready", status_code=503)
    try:
        await await_with_timeout(
            app.state.asr_handle.health.remote(),
            CFG.asr_call_timeout_seconds,
            "ASR health",
        )
        return {"status": "ready", "active_sessions": len(app.state.active_sessions)}
    except Exception:
        logger.exception("ASR readiness check failed")
        return PlainTextResponse("ASR unavailable", status_code=503)


@app.get("/health")
async def health():
    ready = await health_ready()
    if isinstance(ready, PlainTextResponse):
        return {
            "status": "error",
            "active_sessions": len(getattr(app.state, "active_sessions", [])),
            "asr": "unavailable",
        }
    return {
        "status": "ok",
        "vad": {
            "device": "GPU" if CFG.vad_use_gpu else "CPU",
            "mode": "per-session-state/model instance",
            "active_sessions": len(app.state.active_sessions),
            "max_active_sessions": CFG.max_active_sessions,
            "max_concurrent_inference": CFG.vad_max_concurrent_inference,
            "chunk_frames": CFG.stream_chunk_frames,
            "asr_update_interval_seconds": CFG.asr_update_interval_seconds,
            "max_buffer_seconds": CFG.max_buffer_seconds,
        },
        "asr": "available",
    }


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host=CFG.gateway_host, port=CFG.gateway_port, log_level=os.getenv("LOG_LEVEL", "info").lower())
