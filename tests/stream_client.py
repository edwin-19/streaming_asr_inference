
from __future__ import annotations

import asyncio
import json
import time
import wave
from pathlib import Path
from typing import Optional

import typer
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

GATEWAY_URL = "ws://127.0.0.1:8090/transcribe"
SAMPLE_RATE = 16000
CHUNK_MS = 100


def load_audio(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Expected mono audio")

        if wav.getsampwidth() != 2:
            raise ValueError("Expected PCM16 audio")

        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio")

        if wav.getcomptype() != "NONE":
            raise ValueError("Expected uncompressed PCM WAV")

        return wav.readframes(wav.getnframes())


def make_display(
    status: str,
    sent: int,
    total: int,
    transcript: str,
    vad_status: str,
    elapsed: float,
):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Status", status)
    table.add_row("Chunks", f"{sent}/{total}")
    table.add_row("VAD", vad_status)
    table.add_row("Elapsed", f"{elapsed:.2f}s")
    table.add_row("Transcript", transcript or "[dim]Waiting for speech...[/dim]")

    return Panel(table, title="Streaming STT", border_style="cyan")


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, readable=True),
    url: str = typer.Option(GATEWAY_URL, "--url", help="Gateway WebSocket URL"),
    fast: bool = typer.Option(False, "--fast", help="Send without realtime pacing"),
    chunk_ms: int = typer.Option(CHUNK_MS, help="Audio chunk size in milliseconds"),
):
    """Stream a PCM16 WAV file to the ASR gateway."""

    if chunk_ms <= 0:
        raise typer.BadParameter("chunk_ms must be greater than zero")

    asyncio.run(
        run_client(
            audio=audio,
            url=url,
            realtime=not fast,
            chunk_ms=chunk_ms,
        )
    )


async def run_client(
    audio: Path,
    url: str,
    realtime: bool,
    chunk_ms: int,
):
    pcm = load_audio(audio)

    bytes_per_chunk = SAMPLE_RATE * 2 * chunk_ms // 1000
    chunks = [
        pcm[i:i + bytes_per_chunk]
        for i in range(0, len(pcm), bytes_per_chunk)
    ]

    duration = len(pcm) / (SAMPLE_RATE * 2)

    console.print(f"\n[bold]Audio:[/bold] {audio}")
    console.print(f"[bold]Duration:[/bold] {duration:.2f}s")
    console.print(f"[bold]Chunks:[/bold] {len(chunks)}")
    console.print(f"[bold]Mode:[/bold] {'realtime' if realtime else 'fast'}\n")

    sent = 0
    transcript = ""
    vad_status = "Waiting"
    status = "Connecting"
    started = time.perf_counter()

    try:
        async with websockets.connect(url, max_size=None) as ws:
            response = json.loads(await ws.recv())

            if response.get("type") != "session_started":
                raise RuntimeError(f"Unexpected initialization: {response}")

            status = "Streaming"

            with Live(
                make_display(status, sent, len(chunks), transcript,
                             vad_status, 0),
                console=console,
                refresh_per_second=10,
            ) as live:

                async def receive_events():
                    nonlocal transcript, vad_status, status

                    async for message in ws:
                        try:
                            event = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        if event_type == "transcript":
                            text = event.get("text", event.get("transcript", ""))
                            transcript = text

                            if event.get("is_final"):
                                console.print(f"[green]FINAL:[/green] {text}")
                            else:
                                console.print(f"[yellow]PARTIAL:[/yellow] {text}")

                        elif event_type == "speech_start":
                            vad_status = "Speech detected"

                        elif event_type == "speech_end":
                            vad_status = "Silence"

                        elif event_type == "completed":
                            status = "Completed"
                            break

                        elif event_type == "error":
                            status = "Error"
                            console.print(f"[red]Gateway error:[/red] {event}")
                            break

                        live.update(
                            make_display(
                                status,
                                sent,
                                len(chunks),
                                transcript,
                                vad_status,
                                time.perf_counter() - started,
                            )
                        )

                receiver = asyncio.create_task(receive_events())

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total} chunks"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:

                    task = progress.add_task("Sending audio", total=len(chunks))

                    for chunk in chunks:
                        await ws.send(chunk)
                        sent += 1
                        progress.advance(task)

                        live.update(
                            make_display(
                                status,
                                sent,
                                len(chunks),
                                transcript,
                                vad_status,
                                time.perf_counter() - started,
                            )
                        )

                        if realtime:
                            await asyncio.sleep(
                                len(chunk) / (SAMPLE_RATE * 2)
                            )

                await ws.send(json.dumps({"type": "end_of_audio"}))
                await receiver

                live.update(
                    make_display(
                        status,
                        sent,
                        len(chunks),
                        transcript,
                        vad_status,
                        time.perf_counter() - started,
                    )
                )

    except Exception as exc:
        console.print(f"\n[bold red]Inference failed:[/bold red] {exc}")
        raise

    elapsed = time.perf_counter() - started

    console.print()
    console.print(
        Panel(
            f"Audio duration: {duration:.2f}s\n"
            f"Elapsed time: {elapsed:.2f}s\n"
            f"RTF: {elapsed / duration:.3f}x",
            title="Inference Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
