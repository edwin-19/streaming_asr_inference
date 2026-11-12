
import argparse
import asyncio
import json
import logging
import time
import wave

import websockets


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

# Diagnostic setting:
# Disable automatic ping/pong temporarily to isolate heartbeat
# timeouts during long-running ASR inference.
PING_INTERVAL = None
PING_TIMEOUT = None


# ============================================================
# Audio Sender
# ============================================================

async def send_audio(
    websocket,
    wav,
    frames_per_chunk,
    frame_ms,
):
    frame_count = 0
    total_bytes = 0
    start_time = time.perf_counter()

    logger.info(
        "Starting audio transmission: chunk=%dms",
        frame_ms,
    )

    try:
        while True:

            pcm = wav.readframes(frames_per_chunk)

            if not pcm:
                break

            await websocket.send(pcm)

            frame_count += 1
            total_bytes += len(pcm)

            await asyncio.sleep(frame_ms / 1000)

    except websockets.ConnectionClosed as exc:

        logger.error(
            "Audio sender: connection closed "
            "code=%s reason=%s",
            exc.code,
            exc.reason,
        )

        raise

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Finished sending %d chunks | %.2f seconds | "
        "%.2f MB transmitted",
        frame_count,
        elapsed,
        total_bytes / (1024 * 1024),
    )

    # Tell the server that the audio stream has ended.
    await websocket.send(json.dumps({
        "type": "end_of_audio",
    }))

    logger.info("Sent end_of_audio")


# ============================================================
# Message Receiver
# ============================================================

async def receive_messages(websocket):

    final_transcript = None
    latest_partial = ""
    speech_segments = []

    try:

        async for response in websocket:

            try:
                data = json.loads(response)

            except json.JSONDecodeError:

                logger.info(
                    "Non-JSON response: %s",
                    response,
                )

                continue

            message_type = data.get("type")

            # ------------------------------------------------
            # Session started
            # ------------------------------------------------

            if message_type == "session_started":

                logger.info(
                    "Session started: %s",
                    data.get("session_id"),
                )

                logger.info(
                    "Sample rate: %s",
                    data.get("sample_rate"),
                )

            # ------------------------------------------------
            # Speech start
            # ------------------------------------------------

            elif message_type == "speech_start":

                timestamp = data.get("timestamp", 0)

                logger.info(
                    "[VAD] Speech started at %.2fs (frame %s)",
                    timestamp,
                    data.get("frame"),
                )

                speech_segments.append({
                    "start": timestamp,
                    "end": None,
                })

            # ------------------------------------------------
            # Speech end
            # ------------------------------------------------

            elif message_type == "speech_end":

                timestamp = data.get("timestamp", 0)

                logger.info(
                    "[VAD] Speech ended at %.2fs (frame %s)",
                    timestamp,
                    data.get("frame"),
                )

                if speech_segments:
                    speech_segments[-1]["end"] = timestamp

            # ------------------------------------------------
            # Transcript
            # ------------------------------------------------

            elif message_type == "transcript":

                text = data.get("text", "")
                language = data.get("language", "")
                is_final = data.get("is_final", False)

                if is_final:

                    final_transcript = text

                    logger.info("")
                    logger.info("=" * 70)
                    logger.info("[FINAL TRANSCRIPT]")
                    logger.info("=" * 70)
                    logger.info("%s", text)
                    logger.info("Language: %s", language)
                    logger.info("=" * 70)

                else:

                    latest_partial = text

                    logger.info(
                        "[PARTIAL] %s",
                        latest_partial,
                    )

            # ------------------------------------------------
            # Completed
            # ------------------------------------------------

            elif message_type == "completed":

                logger.info(
                    "Processing completed: %s",
                    data,
                )

                logger.info("")
                logger.info("=" * 70)
                logger.info("STREAM SUMMARY")
                logger.info("=" * 70)

                logger.info(
                    "Duration: %.2fs",
                    data.get("duration", 0),
                )

                logger.info(
                    "Speech segments: %d",
                    len(speech_segments),
                )

                logger.info("Speech boundaries:")

                for index, segment in enumerate(
                    speech_segments,
                    1,
                ):

                    logger.info(
                        "[%d] %.2fs - %.2fs",
                        index,
                        segment["start"] or 0,
                        segment["end"] or 0,
                    )

                logger.info("")
                logger.info("Final transcript:")

                logger.info(
                    "%s",
                    final_transcript or latest_partial,
                )

                logger.info("=" * 70)

                return {
                    "transcript": (
                        final_transcript or latest_partial
                    ),
                    "speech_segments": speech_segments,
                }

            # ------------------------------------------------
            # Cancelled
            # ------------------------------------------------

            elif message_type == "cancelled":

                logger.info(
                    "Stream cancelled: %s",
                    data,
                )

                return {
                    "transcript": None,
                    "speech_segments": speech_segments,
                }

            # ------------------------------------------------
            # Server error
            # ------------------------------------------------

            elif message_type == "error":

                logger.error(
                    "Server error: %s",
                    data,
                )

            else:

                logger.info(
                    "Server: %s",
                    data,
                )

    except websockets.ConnectionClosed as exc:

        logger.error(
            "WebSocket receiver disconnected: "
            "code=%s reason=%s",
            exc.code,
            exc.reason,
        )

        raise

    logger.warning(
        "Receiver ended without receiving completed message"
    )

    return {
        "transcript": final_transcript or latest_partial,
        "speech_segments": speech_segments,
    }


# ============================================================
# Test Connection
# ============================================================

async def test_connection(
    url,
    audio_path,
    frame_ms=100,
):

    with wave.open(audio_path, "rb") as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()

        # ----------------------------------------------------
        # Validate audio
        # ----------------------------------------------------

        if (
            channels != 1
            or sample_width != 2
            or sample_rate != 16000
        ):

            raise ValueError(
                f"Expected 16kHz mono PCM16, got "
                f"{sample_rate}Hz, {channels} channels, "
                f"{sample_width * 8}-bit"
            )

        if frame_ms <= 0:

            raise ValueError(
                "frame_ms must be greater than zero"
            )

        frames_per_chunk = int(
            sample_rate * frame_ms / 1000
        )

        if frames_per_chunk <= 0:
            raise ValueError("Invalid audio chunk size")

        # ----------------------------------------------------
        # WebSocket connection
        # ----------------------------------------------------

        async with websockets.connect(
            url,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT,
            close_timeout=10,
            open_timeout=30,
            max_size=None,
        ) as websocket:

            logger.info(
                "Connected to %s",
                url,
            )

            logger.info(
                "WebSocket ping interval: %s",
                PING_INTERVAL,
            )

            # ------------------------------------------------
            # Initial handshake
            # ------------------------------------------------

            response = await websocket.recv()

            session = json.loads(response)

            logger.info(
                "Server: %s",
                session,
            )

            if session.get("type") != "session_started":

                raise RuntimeError(
                    f"Unexpected handshake: {session}"
                )

            # ------------------------------------------------
            # Start sender and receiver
            # ------------------------------------------------

            sender = asyncio.create_task(
                send_audio(
                    websocket,
                    wav,
                    frames_per_chunk,
                    frame_ms,
                ),
                name="audio-sender",
            )

            receiver = asyncio.create_task(
                receive_messages(websocket),
                name="message-receiver",
            )

            tasks = [sender, receiver]

            try:

                results = await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

                # ------------------------------------------------
                # Inspect task results
                # ------------------------------------------------

                errors = []

                for name, result in zip(
                    ("sender", "receiver"),
                    results,
                ):

                    if isinstance(result, BaseException):

                        logger.error(
                            "%s failed: %r",
                            name,
                            result,
                        )

                        errors.append(result)

                if errors:
                    raise errors[0]

                logger.info(
                    "Both sender and receiver completed"
                )

                return results[1]

            finally:

                # ------------------------------------------------
                # Cleanup
                # ------------------------------------------------

                for task in tasks:

                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

                logger.info(
                    "Client tasks cleaned up"
                )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="ws://localhost:8000/transcribe",
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to 16kHz mono PCM16 WAV",
    )

    parser.add_argument(
        "--frame-ms",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    result = asyncio.run(
        test_connection(
            args.url,
            args.audio,
            args.frame_ms,
        )
    )

    logger.info(
        "Returned result: %s",
        result,
    )
