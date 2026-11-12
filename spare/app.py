
import asyncio
import json
import logging
import uuid

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

ASR_WINDOW_SECONDS = 30.0
ASR_UPDATE_SECONDS = 2.0
ASR_OVERLAP_SECONDS = 2.0
ASR_MAX_NEW_TOKENS = 256


# ============================================================
# Model Loading
# ============================================================

def create_vad():
    """
    Create an independent FireRedVAD instance.

    Each WebSocket session gets its own stateful VAD instance.
    """

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


logger.info("Loading Qwen3-ASR Transformers model...")

GLOBAL_ASR_MODEL = Qwen3ASRModel.from_pretrained(
    ASR_MODEL_NAME,
    torch_dtype="auto",
    device_map="cuda:0",
)

logger.info("Qwen3-ASR loaded.")


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


def update_to_dict(update) -> dict:
    """
    Convert an incremental update into a JSON-safe dictionary.

    Optional timing attributes are included when available.
    """

    result = {
        "text": getattr(update, "text", "") or "",
        "language": getattr(update, "language", "") or "",
    }

    optional_fields = (
        "audio_start_sec",
        "audio_end_sec",
        "decode_wall_sec",
    )

    for field in optional_fields:
        value = getattr(update, field, None)

        if value is not None:
            result[field] = value

    return result


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

        # Shared global model reference.
        # This does not load another Qwen model.
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

        # Independent streaming state for this session.
        self.incremental_state = None

        self.asr_audio_cursor = 0
        self.last_partial_text = ""

        self.finalized = False

        # Prevent overlapping feed()/finish() calls
        # on the same incremental state.
        self.asr_lock = asyncio.Lock()

        self.vad.reset()

        self.init_asr_state()

    # --------------------------------------------------------
    # Qwen initialization
    # --------------------------------------------------------

    def init_asr_state(self):

        logger.info(
            "[%s] Initializing incremental ASR state",
            self.session_id,
        )

        self.incremental_state = (
            self.asr.init_incremental_streaming(
                context="",
                language=None,
                window_seconds=ASR_WINDOW_SECONDS,
                update_seconds=ASR_UPDATE_SECONDS,
                overlap_seconds=ASR_OVERLAP_SECONDS,
                max_new_tokens=ASR_MAX_NEW_TOKENS,
            )
        )

        self.asr_audio_cursor = 0
        self.last_partial_text = ""
        self.finalized = False

    # --------------------------------------------------------
    # Incremental ASR
    # --------------------------------------------------------

    async def feed_asr_until(
        self,
        end_sample: int,
        websocket: WebSocket,
    ):

        if self.incremental_state is None or self.finalized:
            return

        # Serialize operations on this session's state.
        async with self.asr_lock:

            if self.incremental_state is None or self.finalized:
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

            logger.debug(
                "[%s] Feeding ASR %.3fs",
                self.session_id,
                len(audio) / SAMPLE_RATE,
            )

            # Native Transformers incremental API.
            updates = await asyncio.to_thread(
                self.incremental_state.feed,
                pcm16k,
            )

            # Advance cursor only after successful inference.
            self.asr_audio_cursor = end_sample

        # Send results outside the lock.
        for update in updates:

            payload = update_to_dict(update)

            text = payload["text"]
            language = payload["language"]

            if text and text != self.last_partial_text:

                self.last_partial_text = text

                await websocket.send_json({
                    "type": "transcript",
                    **payload,
                    "is_final": False,
                })

    # --------------------------------------------------------
    # Final ASR
    # --------------------------------------------------------

    async def finish_asr(
        self,
        websocket: WebSocket,
    ):

        if self.incremental_state is None or self.finalized:
            return

        logger.info(
            "[%s] Finalizing incremental ASR",
            self.session_id,
        )

        async with self.asr_lock:

            if self.incremental_state is None or self.finalized:
                return

            final_update = await asyncio.to_thread(
                self.incremental_state.finish,
            )

            self.finalized = True

        if final_update is None:
            return

        payload = update_to_dict(final_update)

        logger.info(
            "[%s] Final transcript: %s",
            self.session_id,
            payload["text"],
        )

        await websocket.send_json({
            "type": "transcript",
            **payload,
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

            chunk_results = await asyncio.to_thread(
                self.vad.detect_chunk,
                chunk,
            )

            self.vad_samples_processed += chunk_shift

            chunk_end = min(
                chunk_start + chunk_length,
                self.samples_received,
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

        # Synthetic silence goes to VAD only.
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

            chunk_results = await asyncio.to_thread(
                self.vad.detect_chunk,
                chunk,
            )

            self.vad_samples_processed += chunk_shift

            batches.append({
                "results": chunk_results,
                "processed_end_sample": min(
                    chunk_start + chunk_length,
                    self.samples_received,
                ),
            })

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

        self.incremental_state = None

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

                # Skip audio before the detected speech start.
                # Never move the cursor backwards.
                session.asr_audio_cursor = max(
                    session.asr_audio_cursor,
                    start_sample,
                )

                logger.info(
                    "[%s] Speech start: %.2fs",
                    session.session_id,
                    frame_to_seconds(start_frame),
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

                logger.info(
                    "[%s] Speech end: %.2fs",
                    session.session_id,
                    frame_to_seconds(end_frame),
                )

                await websocket.send_json({
                    "type": "speech_end",
                    "frame": end_frame,
                    "timestamp": frame_to_seconds(end_frame),
                })

                # Keep the same Qwen incremental state
                # across speech segments.

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

        logger.info(
            "New WebSocket connection. Creating independent VAD..."
        )

        # Each connection gets its own VAD instance.
        vad_model = await asyncio.to_thread(
            create_vad
        )

        # Each connection gets its own streaming state,
        # but shares the same global Qwen model.
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

                    logger.info(
                        "[%s] WebSocket disconnected by client",
                        session.session_id,
                    )

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

                batches = await session.process_audio(
                    audio
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
                        "[%s] End of audio received",
                        session.session_id,
                    )

                    # Flush remaining VAD audio.
                    flush_batches = await session.flush()

                    await handle_vad_results(
                        session,
                        flush_batches,
                        websocket,
                    )

                    # Feed remaining speech audio.
                    if session.speech_active:

                        await session.feed_asr_until(
                            session.samples_received,
                            websocket,
                        )

                        session.speech_active = False

                    # Finalize the same incremental state.
                    await session.finish_asr(
                        websocket
                    )

                    await websocket.send_json({
                        "type": "completed",
                        "duration": (
                            session.samples_received / SAMPLE_RATE
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

    except Exception as exc:

        if is_websocket_disconnect(exc):

            logger.info(
                "[%s] WebSocket disconnected",
                session.session_id if session else "unknown",
            )

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
                pass

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
        "asr": "transformers",
        "asr_model": ASR_MODEL_NAME,
    }
