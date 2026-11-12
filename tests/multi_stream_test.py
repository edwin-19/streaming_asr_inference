
import argparse
import asyncio
import json
import random
import statistics
import time
import wave

from pathlib import Path

import websockets

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel


# ==================================================
# Configuration
# ==================================================

DEFAULT_URL = "ws://127.0.0.1:8090/transcribe"

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

DEFAULT_FRAME_MS = 100

console = Console()


# ==================================================
# Audio
# ==================================================

def load_wav(path: Path):

    with wave.open(str(path), "rb") as wf:

        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()

        audio = wf.readframes(frames)

    if channels != CHANNELS:
        raise ValueError(
            f"{path}: expected mono audio, got {channels} channels"
        )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE}Hz, got {sample_rate}Hz"
        )

    if sample_width != SAMPLE_WIDTH:
        raise ValueError(
            f"{path}: expected PCM16, got {sample_width} bytes/sample"
        )

    return audio, frames / sample_rate


# ==================================================
# Statistics helpers
# ==================================================

def average(values):

    values = [
        float(v)
        for v in values
        if v is not None
    ]

    return statistics.fmean(values) if values else None


def minimum(values):

    values = [
        float(v)
        for v in values
        if v is not None
    ]

    return min(values) if values else None


def maximum(values):

    values = [
        float(v)
        for v in values
        if v is not None
    ]

    return max(values) if values else None


def percentile(values, pct):

    values = sorted(
        float(v)
        for v in values
        if v is not None
    )

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * pct / 100

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def fmt(value, suffix="s"):

    if value is None:
        return "-"

    return f"{value:.2f}{suffix}"


# ==================================================
# Arrival scheduling
# ==================================================

def make_arrival_offsets(
    count,
    mode,
    arrival_rate,
    ramp_duration,
    seed,
):

    rng = random.Random(seed)

    if mode == "sync":

        return [0.0] * count

    if mode == "ramp":

        if count <= 1:
            return [0.0]

        return [
            ramp_duration * i / (count - 1)
            for i in range(count)
        ]

    if mode == "poisson":

        if arrival_rate <= 0:
            raise ValueError(
                "Arrival rate must be greater than zero"
            )

        offsets = [0.0]

        for _ in range(1, count):

            interval = rng.expovariate(arrival_rate)

            offsets.append(
                offsets[-1] + interval
            )

        return offsets

    raise ValueError(f"Unknown arrival mode: {mode}")


async def sleep_until(deadline):

    remaining = deadline - time.perf_counter()

    if remaining > 0:
        await asyncio.sleep(remaining)


# ==================================================
# Progress
# ==================================================

class ProgressState:

    def __init__(self, streams):

        self.streams = streams
        self.start_time = None
        self.finished = False

        self.max_active = 0

        self.data = {}

        for i in range(1, streams + 1):

            self.data[i] = {
                "status": "Scheduled",
                "sent_bytes": 0,
                "total_bytes": 0,
                "elapsed": 0.0,
                "first_partial": None,
                "transcript": "",
                "last_event": "-",
                "error": None,
            }

    def update(self, stream_id, **kwargs):

        self.data[stream_id].update(kwargs)

        active = sum(
            item["status"] not in (
                "Scheduled",
                "Completed",
                "Failed",
            )
            for item in self.data.values()
        )

        self.max_active = max(
            self.max_active,
            active,
        )

    def render(self):

        table = Table(
            expand=True,
            padding=(0, 1),
            pad_edge=False,
        )

        table.add_column("ID", width=4, justify="right")
        table.add_column("Status", width=12)
        table.add_column("Upload", width=7, justify="right")
        table.add_column("Elapsed", width=8, justify="right")
        table.add_column("First", width=7, justify="right")
        table.add_column("Transcript", ratio=1, no_wrap=True)

        completed = 0
        failed = 0
        sent = 0
        total = 0
        active = 0
        scheduled = 0

        styles = {
            "Completed": "[green]DONE[/green]",
            "Failed": "[red]FAILED[/red]",
            "Connecting": "[yellow]CONNECT[/yellow]",
            "Ready": "[cyan]READY[/cyan]",
            "Sending": "[cyan]SENDING[/cyan]",
            "Waiting ASR": "[yellow]ASR WAIT[/yellow]",
            "Speech detected": "[green]SPEECH[/green]",
            "Scheduled": "[dim]SCHEDULED[/dim]",
        }

        for stream_id, item in self.data.items():

            status = item["status"]

            if status == "Completed":
                completed += 1

            elif status == "Failed":
                failed += 1

            elif status == "Scheduled":
                scheduled += 1

            else:
                active += 1

            sent += item["sent_bytes"]
            total += item["total_bytes"]

            upload = (
                f"{item['sent_bytes'] / item['total_bytes'] * 100:.0f}%"
                if item["total_bytes"]
                else "-"
            )

            first = (
                fmt(item["first_partial"])
                if item["first_partial"] is not None
                else "-"
            )

            transcript = item["transcript"].replace("\n", " ")

            if len(transcript) > 65:
                transcript = transcript[:62] + "..."

            table.add_row(
                str(stream_id),
                styles.get(status, status),
                upload,
                fmt(item["elapsed"]),
                first,
                transcript or "-",
            )

        elapsed = (
            time.perf_counter() - self.start_time
            if self.start_time
            else 0
        )

        upload_pct = (
            sent / total * 100
            if total
            else 0
        )

        summary = (
            f"Started {self.streams - scheduled}/{self.streams}  |  "
            f"Active {active}  |  "
            f"[green]Done {completed}[/green]  |  "
            f"[red]Failed {failed}[/red]  |  "
            f"Peak active {self.max_active}  |  "
            f"Wall {elapsed:.1f}s  |  "
            f"Upload {upload_pct:.0f}%"
        )

        return Panel(
            table,
            title="[bold cyan]ASR REALTIME LOAD MONITOR[/bold cyan]",
            subtitle=summary,
            border_style="cyan",
            padding=(0, 1),
        )


# ==================================================
# Stream
# ==================================================

async def run_stream(
    stream_id,
    url,
    audio_pool,
    frame_ms,
    speed,
    jitter_ms,
    arrival_offset,
    benchmark_start,
    progress,
    verbose,
    seed,
    session_timeout,
):

    rng = random.Random(
        seed + stream_id * 7919
    )

    audio_path, audio, audio_duration = rng.choice(
        audio_pool
    )

    result = {
        "stream_id": stream_id,
        "audio_file": audio_path.name,
        "audio_duration": audio_duration,
        "arrival_offset": arrival_offset,

        "stream_elapsed": 0.0,

        "first_partial_latency": None,
        "final_latency": None,

        "decode_wall_sec": None,
        "inference_wall_sec": None,
        "semaphore_wait_sec": None,
        "queue_wait_sec": None,

        "transcript_updates": 0,
        "max_audio_endpoint": 0.0,

        "e2e_rtf": None,
        "asr_rtf": None,

        "transcript": "",
        "error": None,

        "last_event": "-",
        "last_message": None,
        "upload_complete": False,
        "completed_event": False,
    }

    sender_task = None
    stream_start = None

    total_bytes = len(audio)

    bytes_per_frame = (
        SAMPLE_RATE * SAMPLE_WIDTH * frame_ms // 1000
    )

    if bytes_per_frame <= 0:
        raise ValueError("Invalid frame duration")

    try:

        progress.update(
            stream_id,
            status="Scheduled",
            total_bytes=total_bytes,
        )

        # ------------------------------------------
        # Wait for arrival time
        # ------------------------------------------

        await sleep_until(
            benchmark_start + arrival_offset
        )

        progress.update(
            stream_id,
            status="Connecting",
        )

        async with websockets.connect(
            url,
            ping_interval=None,
            ping_timeout=None,
            open_timeout=15,
            close_timeout=5,
            max_size=None,
        ) as ws:

            # --------------------------------------
            # Wait for session initialization
            # --------------------------------------

            while True:

                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=session_timeout,
                )

                if isinstance(raw, bytes):
                    result["last_event"] = "binary_message"
                    continue

                message = json.loads(raw)

                msg_type = message.get("type", "unknown")

                result["last_event"] = msg_type
                result["last_message"] = message

                if verbose:

                    console.log(
                        f"[cyan]Stream {stream_id}[/cyan] "
                        f"{json.dumps(message, ensure_ascii=False)[:500]}"
                    )

                if msg_type == "error":

                    raise RuntimeError(
                        message.get("message", str(message))
                    )

                if msg_type == "session_started":
                    break

            stream_start = time.perf_counter()

            progress.update(
                stream_id,
                status="Sending",
                last_event="session_started",
            )

            # --------------------------------------
            # Sender
            # --------------------------------------

            async def send_audio():

                sent_bytes = 0

                frame_interval = (
                    frame_ms / 1000 / speed
                    if speed > 0
                    else 0.0
                )

                sender_start = time.perf_counter()
                previous_send = sender_start

                frame_index = 0

                for offset in range(
                    0,
                    total_bytes,
                    bytes_per_frame,
                ):

                    if frame_interval > 0:

                        nominal_deadline = (
                            sender_start
                            + frame_index * frame_interval
                        )

                        jitter = (
                            rng.uniform(
                                -jitter_ms,
                                jitter_ms,
                            ) / 1000
                        )

                        deadline = max(
                            previous_send,
                            nominal_deadline + jitter,
                        )

                        await sleep_until(deadline)

                    chunk = audio[
                        offset:offset + bytes_per_frame
                    ]

                    await ws.send(chunk)

                    previous_send = time.perf_counter()

                    sent_bytes += len(chunk)
                    frame_index += 1

                    progress.update(
                        stream_id,
                        sent_bytes=sent_bytes,
                    )

                    if frame_interval <= 0:
                        await asyncio.sleep(0)

                # Send end-of-audio only after all frames.
                await ws.send(
                    json.dumps({
                        "type": "end_of_audio"
                    })
                )

                result["upload_complete"] = True

                progress.update(
                    stream_id,
                    status="Waiting ASR",
                )

            sender_task = asyncio.create_task(
                send_audio()
            )

            # --------------------------------------
            # Receiver
            # --------------------------------------

            completed = False

            while not completed:

                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=session_timeout,
                )

                if isinstance(raw, bytes):
                    result["last_event"] = "binary_message"
                    continue

                message = json.loads(raw)

                msg_type = message.get("type", "unknown")

                result["last_event"] = msg_type
                result["last_message"] = message

                elapsed = (
                    time.perf_counter() - stream_start
                )

                if verbose:

                    console.log(
                        f"[cyan]Stream {stream_id}[/cyan] "
                        f"{json.dumps(message, ensure_ascii=False)[:500]}"
                    )

                progress.update(
                    stream_id,
                    elapsed=elapsed,
                    last_event=msg_type,
                )

                # ----------------------------------
                # VAD events
                # ----------------------------------

                if msg_type == "speech_start":

                    progress.update(
                        stream_id,
                        status="Speech detected",
                    )

                elif msg_type == "speech_end":

                    progress.update(
                        stream_id,
                        status="Sending",
                    )

                # ----------------------------------
                # Transcript events
                # ----------------------------------

                elif msg_type in (
                    "transcript",
                    "partial_transcript",
                    "final_transcript",
                ):

                    transcript = message.get("text", "") or ""

                    is_final = (
                        message.get("is_final", False)
                        or msg_type == "final_transcript"
                    )

                    if (
                        not is_final
                        and result["first_partial_latency"] is None
                    ):

                        result["first_partial_latency"] = elapsed

                        progress.update(
                            stream_id,
                            first_partial=elapsed,
                        )

                    for field in (
                        "decode_wall_sec",
                        "inference_wall_sec",
                        "semaphore_wait_sec",
                        "queue_wait_sec",
                    ):

                        value = message.get(field)

                        if value is None:
                            continue

                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue

                        if result[field] is None:
                            result[field] = 0.0

                        result[field] += value

                    result["transcript_updates"] += 1

                    endpoint = (
                        message.get("audio_end_sec", 0.0)
                        or 0.0
                    )

                    try:

                        result["max_audio_endpoint"] = max(
                            result["max_audio_endpoint"],
                            float(endpoint),
                        )

                    except (TypeError, ValueError):
                        pass

                    if transcript:

                        result["transcript"] = transcript

                        progress.update(
                            stream_id,
                            transcript=transcript,
                        )

                    if is_final:

                        result["final_latency"] = elapsed

                # ----------------------------------
                # Errors and completion
                # ----------------------------------

                elif msg_type == "error":

                    raise RuntimeError(
                        message.get("message", str(message))
                    )

                elif msg_type == "completed":

                    completed = True
                    result["completed_event"] = True

            # --------------------------------------
            # Ensure sender has finished
            # --------------------------------------

            await sender_task
            sender_task = None

            stream_end = time.perf_counter()

            result["stream_elapsed"] = (
                stream_end - stream_start
            )

            if audio_duration > 0:

                result["e2e_rtf"] = (
                    result["stream_elapsed"] / audio_duration
                )

                if result["decode_wall_sec"] is not None:

                    result["asr_rtf"] = (
                        result["decode_wall_sec"] / audio_duration
                    )

            progress.update(
                stream_id,
                status="Completed",
                elapsed=result["stream_elapsed"],
                first_partial=result["first_partial_latency"],
                transcript=result["transcript"],
            )

    except asyncio.TimeoutError:

        elapsed = (
            time.perf_counter() - stream_start
            if stream_start is not None
            else None
        )

        sent = progress.data[stream_id]["sent_bytes"]

        error = (
            f"TimeoutError: no WebSocket message received for "
            f"{session_timeout}s; "
            f"last_event={result['last_event']}; "
            f"upload={sent}/{total_bytes} bytes; "
            f"elapsed={elapsed:.2f}s"
            if elapsed is not None
            else (
                f"TimeoutError: session initialization did not complete "
                f"within {session_timeout}s; "
                f"last_event={result['last_event']}"
            )
        )

        result["error"] = error

        progress.update(
            stream_id,
            status="Failed",
            error=error,
        )

        console.log(
            f"[red]Stream {stream_id} failed: {error}[/red]"
        )

    except Exception as exc:

        error = f"{type(exc).__name__}: {exc}"

        result["error"] = error

        progress.update(
            stream_id,
            status="Failed",
            error=error,
        )

        console.log(
            f"[red]Stream {stream_id} failed: {error}[/red]"
        )

    finally:

        if sender_task and not sender_task.done():

            sender_task.cancel()

            await asyncio.gather(
                sender_task,
                return_exceptions=True,
            )

    return result


# ==================================================
# Run one concurrency level
# ==================================================

async def run_benchmark_level(
    url,
    audio_pool,
    streams,
    frame_ms,
    speed,
    jitter_ms,
    arrival_mode,
    arrival_rate,
    ramp_duration,
    verbose,
    seed,
    session_timeout,
):

    console.print(
        f"\n[bold cyan]Starting {streams}-session test "
        f"({arrival_mode})[/bold cyan]"
    )

    offsets = make_arrival_offsets(
        count=streams,
        mode=arrival_mode,
        arrival_rate=arrival_rate,
        ramp_duration=ramp_duration,
        seed=seed + streams,
    )

    progress = ProgressState(streams)

    benchmark_start = time.perf_counter()

    progress.start_time = benchmark_start

    tasks = []

    for i in range(streams):

        task = asyncio.create_task(
            run_stream(
                stream_id=i + 1,
                url=url,
                audio_pool=audio_pool,
                frame_ms=frame_ms,
                speed=speed,
                jitter_ms=jitter_ms,
                arrival_offset=offsets[i],
                benchmark_start=benchmark_start,
                progress=progress,
                verbose=verbose,
                seed=seed,
                session_timeout=session_timeout,
            )
        )

        tasks.append(task)

    results = []

    with Live(
        progress.render(),
        console=console,
        refresh_per_second=5,
        transient=False,
    ) as live:

        async def refresh():

            while not progress.finished:

                live.update(
                    progress.render()
                )

                await asyncio.sleep(0.2)

        refresh_task = asyncio.create_task(
            refresh()
        )

        try:

            results = await asyncio.gather(*tasks)

            benchmark_elapsed = (
                time.perf_counter() - benchmark_start
            )

        finally:

            progress.finished = True

            await refresh_task

            live.update(
                progress.render()
            )

    # ----------------------------------------------
    # Result groups
    # ----------------------------------------------

    successful = [
        r for r in results
        if r["error"] is None
    ]

    failed = [
        r for r in results
        if r["error"] is not None
    ]

    total_audio = sum(
        r["audio_duration"]
        for r in successful
    )

    total_decode = sum(
        r["decode_wall_sec"] or 0.0
        for r in successful
    )

    total_inference = sum(
        r["inference_wall_sec"] or 0.0
        for r in successful
    )

    total_semaphore_wait = sum(
        r["semaphore_wait_sec"] or 0.0
        for r in successful
    )

    total_queue_wait = sum(
        r["queue_wait_sec"] or 0.0
        for r in successful
    )

    # ----------------------------------------------
    # Aggregate metrics
    # ----------------------------------------------

    aggregate_rtfx = (
        total_audio / benchmark_elapsed
        if benchmark_elapsed > 0
        else None
    )

    aggregate_rtf = (
        benchmark_elapsed / total_audio
        if total_audio > 0
        else None
    )

    aggregate_asr_rtf = (
        total_decode / total_audio
        if total_audio > 0
        and any(
            r["decode_wall_sec"] is not None
            for r in successful
        )
        else None
    )

    # ----------------------------------------------
    # Per-session metrics
    # ----------------------------------------------

    elapsed_values = [
        r["stream_elapsed"]
        for r in successful
    ]

    e2e_rtfs = [
        r["e2e_rtf"]
        for r in successful
    ]

    asr_rtfs = [
        r["asr_rtf"]
        for r in successful
    ]

    firsts = [
        r["first_partial_latency"]
        for r in successful
    ]

    finals = [
        r["final_latency"]
        for r in successful
    ]

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    console.print(
        "\n[bold]Benchmark summary[/bold]"
    )

    console.print(f"Sessions requested: {streams}")
    console.print(f"Successful: {len(successful)}")
    console.print(f"Failed: {len(failed)}")

    console.print(
        f"Peak active sessions: {progress.max_active}"
    )

    console.print(
        f"Total audio: {total_audio:.2f}s"
    )

    console.print(
        f"Wall time: {benchmark_elapsed:.2f}s"
    )

    console.print(
        f"Aggregate throughput: "
        f"{aggregate_rtfx:.2f}x"
        if aggregate_rtfx is not None
        else "Aggregate throughput: -"
    )

    console.print(
        f"Aggregate RTF: {aggregate_rtf:.3f}"
        if aggregate_rtf is not None
        else "Aggregate RTF: -"
    )

    console.print(
        f"Aggregate ASR decode RTF: {aggregate_asr_rtf:.3f}"
        if aggregate_asr_rtf is not None
        else "Aggregate ASR decode RTF: unavailable"
    )

    console.print(
        f"Server decode time sum: {total_decode:.2f}s"
    )

    console.print(
        f"Server inference time sum: {total_inference:.2f}s"
    )

    console.print(
        f"Semaphore wait sum: {total_semaphore_wait:.2f}s"
    )

    console.print(
        f"Queue wait sum: {total_queue_wait:.2f}s"
    )

    # ----------------------------------------------
    # Per-session table
    # ----------------------------------------------

    table = Table(
        title="Per-session metrics",
        expand=True,
    )

    table.add_column("Session", justify="right")
    table.add_column("Audio", justify="right")
    table.add_column("Join at", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("First partial", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("E2E RTF", justify="right")
    table.add_column("ASR RTF", justify="right")
    table.add_column("Decode", justify="right")
    table.add_column("Updates", justify="right")
    table.add_column("Audio file")

    for r in successful:

        table.add_row(
            str(r["stream_id"]),
            fmt(r["audio_duration"]),
            fmt(r["arrival_offset"]),
            fmt(r["stream_elapsed"]),
            fmt(r["first_partial_latency"]),
            fmt(r["final_latency"]),
            (
                f"{r['e2e_rtf']:.3f}"
                if r["e2e_rtf"] is not None
                else "-"
            ),
            (
                f"{r['asr_rtf']:.3f}"
                if r["asr_rtf"] is not None
                else "-"
            ),
            fmt(r["decode_wall_sec"]),
            str(r["transcript_updates"]),
            r["audio_file"],
        )

    console.print(table)

    # ----------------------------------------------
    # Failed sessions
    # ----------------------------------------------

    if failed:

        console.print(
            "\n[bold red]Failed session diagnostics[/bold red]"
        )

        failure_table = Table(
            expand=True,
        )

        failure_table.add_column("ID", justify="right")
        failure_table.add_column("Error")
        failure_table.add_column("Last event")
        failure_table.add_column("Upload")
        failure_table.add_column("Elapsed", justify="right")
        failure_table.add_column("Transcript updates", justify="right")

        for r in failed:

            state = progress.data[r["stream_id"]]

            upload = (
                f"{state['sent_bytes'] / state['total_bytes'] * 100:.0f}%"
                if state["total_bytes"]
                else "-"
            )

            elapsed = (
                r["stream_elapsed"]
                if r["stream_elapsed"] > 0
                else state["elapsed"]
            )

            failure_table.add_row(
                str(r["stream_id"]),
                r["error"] or "-",
                r["last_event"],
                upload,
                fmt(elapsed),
                str(r["transcript_updates"]),
            )

        console.print(failure_table)

    return {
        "streams": streams,
        "successful": len(successful),
        "failed": len(failed),

        "peak_active": progress.max_active,

        "total_audio": total_audio,
        "wall_time": benchmark_elapsed,

        "aggregate_rtf": aggregate_rtf,
        "aggregate_rtfx": aggregate_rtfx,
        "aggregate_asr_rtf": aggregate_asr_rtf,

        "avg_stream_elapsed": average(elapsed_values),
        "avg_e2e_rtf": average(e2e_rtfs),
        "avg_asr_rtf": average(asr_rtfs),

        "first_p50": percentile(firsts, 50),
        "first_p95": percentile(firsts, 95),
        "first_p99": percentile(firsts, 99),

        "final_p50": percentile(finals, 50),
        "final_p95": percentile(finals, 95),
        "final_p99": percentile(finals, 99),

        "elapsed_p50": percentile(elapsed_values, 50),
        "elapsed_p95": percentile(elapsed_values, 95),

        "total_decode": total_decode,
        "total_inference": total_inference,
        "total_semaphore_wait": total_semaphore_wait,
        "total_queue_wait": total_queue_wait,
    }


# ==================================================
# Main
# ==================================================

async def main(args):

    levels = [
        int(x.strip())
        for x in args.levels.split(",")
        if x.strip()
    ]

    if not levels or any(x < 1 for x in levels):
        raise ValueError(
            "Stream levels must be positive integers"
        )

    # ----------------------------------------------
    # Load audio pool once
    # ----------------------------------------------

    audio_paths = [
        args.audio,
        *args.audio_pool,
    ]

    audio_paths = list(
        dict.fromkeys(
            path.resolve()
            for path in audio_paths
        )
    )

    audio_pool = []

    for path in audio_paths:

        if not path.exists():
            raise FileNotFoundError(path)

        audio, duration = load_wav(path)

        audio_pool.append(
            (path, audio, duration)
        )

    console.print(
        Panel(
            f"URL: {args.url}\n"
            f"Audio pool: {len(audio_pool)} file(s)\n"
            f"Files: {', '.join(p.name for p, _, _ in audio_pool)}\n"
            f"Frame size: {args.frame_ms}ms\n"
            f"Playback: "
            f"{'Unpaced' if args.speed == 0 else str(args.speed) + 'x'}\n"
            f"Arrival mode: {args.arrival}\n"
            f"Arrival rate: {args.arrival_rate} sessions/sec\n"
            f"Ramp duration: {args.ramp_duration}s\n"
            f"Chunk jitter: ±{args.jitter_ms}ms\n"
            f"Session timeout: {args.session_timeout}s\n"
            f"Levels: {levels}",
            title="Multi-session ASR benchmark",
        )
    )

    all_results = []

    for streams in levels:

        result = await run_benchmark_level(
            url=args.url,
            audio_pool=audio_pool,
            streams=streams,
            frame_ms=args.frame_ms,
            speed=args.speed,
            jitter_ms=args.jitter_ms,
            arrival_mode=args.arrival,
            arrival_rate=args.arrival_rate,
            ramp_duration=args.ramp_duration,
            verbose=args.verbose,
            seed=args.seed,
            session_timeout=args.session_timeout,
        )

        all_results.append(result)

        await asyncio.sleep(args.cooldown)

    # ----------------------------------------------
    # Cross-test comparison
    # ----------------------------------------------

    console.print(
        "\n[bold cyan]Concurrency comparison[/bold cyan]"
    )

    summary = Table(
        expand=True,
    )

    summary.add_column("Sessions", justify="right")
    summary.add_column("Success", justify="right")
    summary.add_column("Failed", justify="right")
    summary.add_column("Peak active", justify="right")
    summary.add_column("Wall", justify="right")
    summary.add_column("Avg elapsed", justify="right")
    summary.add_column("First P50", justify="right")
    summary.add_column("First P95", justify="right")
    summary.add_column("Final P95", justify="right")
    summary.add_column("E2E RTF", justify="right")
    summary.add_column("ASR RTF", justify="right")
    summary.add_column("Throughput", justify="right")

    for r in all_results:

        summary.add_row(
            str(r["streams"]),
            str(r["successful"]),
            str(r["failed"]),
            str(r["peak_active"]),
            fmt(r["wall_time"]),
            fmt(r["avg_stream_elapsed"]),
            fmt(r["first_p50"]),
            fmt(r["first_p95"]),
            fmt(r["final_p95"]),
            (
                f"{r['avg_e2e_rtf']:.3f}"
                if r["avg_e2e_rtf"] is not None
                else "-"
            ),
            (
                f"{r['avg_asr_rtf']:.3f}"
                if r["avg_asr_rtf"] is not None
                else "-"
            ),
            (
                f"{r['aggregate_rtfx']:.2f}x"
                if r["aggregate_rtfx"] is not None
                else "-"
            ),
        )

    console.print(summary)


# ==================================================
# CLI
# ==================================================

def cli():

    parser = argparse.ArgumentParser(
        description="Realistic multi-session streaming ASR benchmark"
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Primary WAV file",
    )

    parser.add_argument(
        "--audio-pool",
        type=Path,
        action="append",
        default=[],
        help="Additional WAV file. Can be specified multiple times.",
    )

    parser.add_argument(
        "--levels",
        default="1,2,4,8,10,20,30",
        help="Total sessions per test",
    )

    parser.add_argument(
        "--frame-ms",
        type=int,
        default=DEFAULT_FRAME_MS,
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="1=real-time, 2=2x, 0=unpaced",
    )

    parser.add_argument(
        "--arrival",
        choices=("sync", "poisson", "ramp"),
        default="poisson",
        help="Session arrival pattern",
    )

    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="Average new sessions per second in Poisson mode",
    )

    parser.add_argument(
        "--ramp-duration",
        type=float,
        default=30.0,
        help="Spread session starts across this duration in ramp mode",
    )

    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=15.0,
        help="Per-frame timing jitter in milliseconds",
    )

    parser.add_argument(
        "--session-timeout",
        type=float,
        default=180.0,
        help="Timeout for receiving server messages",
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Pause between concurrency levels",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    if args.frame_ms <= 0:
        parser.error("--frame-ms must be positive")

    if args.speed < 0:
        parser.error("--speed cannot be negative")

    if args.jitter_ms < 0:
        parser.error("--jitter-ms cannot be negative")

    if (
        args.speed > 0
        and args.jitter_ms >= args.frame_ms / args.speed
    ):
        parser.error(
            "--jitter-ms must be smaller than the playback frame interval"
        )

    if args.arrival_rate <= 0:
        parser.error("--arrival-rate must be positive")

    if args.ramp_duration < 0:
        parser.error("--ramp-duration cannot be negative")

    if args.cooldown < 0:
        parser.error("--cooldown cannot be negative")

    if args.session_timeout <= 0:
        parser.error("--session-timeout must be positive")

    if not args.audio.exists():
        parser.error(f"Audio not found: {args.audio}")

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
