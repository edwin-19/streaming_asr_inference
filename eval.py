
import asyncio
import json
import math
import time

import jiwer
import numpy as np
import typer

from datasets import load_dataset
from pathlib import Path
from scipy.signal import resample_poly
from typing import Optional

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.utils import SAMPLE_RATE, parse_asr_output


app = typer.Typer()

MODEL_NAME = "Qwen/Qwen3-ASR-1.7B-hf"


def prepare_audio(decoded, target_sample_rate: int = SAMPLE_RATE):
    """Convert a Hugging Face decoded audio sample to mono float32 16 kHz."""

    audio = decoded.data

    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()

    audio = np.asarray(audio)
    sr = int(decoded.sample_rate)

    # Hugging Face AudioSamples normally returns [channels, samples].
    if audio.ndim == 2:
        if audio.shape[0] <= 8:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if sr != target_sample_rate:
        divisor = math.gcd(sr, target_sample_rate)

        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            sr // divisor,
        ).astype(np.float32)

    return np.ascontiguousarray(audio)


async def infer_one(asr, prompt, audio):
    """
    Submit one independent request to vLLM AsyncLLM.

    Each invocation has its own request ID inside
    _generate_streaming_text_async().
    """

    raw_output = await asr._generate_streaming_text_async(
        prompt,
        audio,
    )

    language, text = parse_asr_output(
        raw_output,
        user_language=None,
    )

    return {
        "text": text,
        "language": language,
    }


async def run_benchmark(
    dataset_path: str,
    batch_size: int,
    max_samples: Optional[int],
    output_path: str,
    max_model_len: int,
    max_num_batched_tokens: int,
):
    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    typer.echo("Loading dataset...")

    ds = load_dataset(dataset_path)["train"]

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    total_samples = len(ds)

    if total_samples == 0:
        raise typer.BadParameter("Dataset contains no samples.")

    typer.echo(f"Dataset samples: {total_samples}")
    typer.echo(f"Concurrent requests: {batch_size}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    typer.echo("\nLoading Qwen3-ASR async vLLM backend...")

    asr = Qwen3ASRModel.LLMAsync(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        max_num_seqs=batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        speculative_config={
            "model": "mattymchen/Qwen3-ASR-1.7B-speculator.dflash2",
            "method": "dflash",
            "num_speculative_tokens": 7,
        },
        max_new_tokens=256,
    )

    predictions = []
    references = []
    results = []

    total_audio_seconds = 0.0
    total_decode_seconds = 0.0
    total_inference_seconds = 0.0

    prompt = asr._build_text_prompt(
        context="",
        force_language=None,
    )

    try:
        # --------------------------------------------------
        # Warmup
        # --------------------------------------------------

        typer.echo("\nPreparing warmup request...")

        warmup_sample = ds[0]
        warmup_decoded = warmup_sample["audio"].get_all_samples()
        warmup_audio = prepare_audio(warmup_decoded)

        typer.echo("Running warmup...")

        warmup_start = time.perf_counter()

        await infer_one(
            asr,
            prompt,
            warmup_audio,
        )

        warmup_elapsed = time.perf_counter() - warmup_start

        typer.echo(f"Warmup completed: {warmup_elapsed:.2f}s")

        # --------------------------------------------------
        # Benchmark
        # --------------------------------------------------

        benchmark_start = time.perf_counter()

        for start_idx in range(0, total_samples, batch_size):

            end_idx = min(start_idx + batch_size, total_samples)

            typer.echo(
                f"\nPreparing batch: "
                f"{start_idx + 1}-{end_idx}/{total_samples}"
            )

            batch_samples = []
            batch_references = []
            batch_durations = []

            # ----------------------------------------------
            # Decode and preprocess audio
            # ----------------------------------------------

            decode_start = time.perf_counter()

            for index in range(start_idx, end_idx):

                sample = ds[index]
                decoded = sample["audio"].get_all_samples()

                audio = prepare_audio(decoded)

                duration = len(audio) / SAMPLE_RATE

                batch_samples.append(audio)
                batch_references.append(sample["text"])
                batch_durations.append(duration)

                total_audio_seconds += duration

            decode_elapsed = time.perf_counter() - decode_start
            total_decode_seconds += decode_elapsed

            # ----------------------------------------------
            # Concurrent async inference
            # ----------------------------------------------

            inference_start = time.perf_counter()

            tasks = [
                infer_one(
                    asr,
                    prompt,
                    audio,
                )
                for audio in batch_samples
            ]

            batch_results = await asyncio.gather(*tasks)

            inference_elapsed = time.perf_counter() - inference_start
            total_inference_seconds += inference_elapsed

            batch_predictions = [
                result["text"]
                for result in batch_results
            ]

            predictions.extend(batch_predictions)
            references.extend(batch_references)

            # ----------------------------------------------
            # Save per-sample results
            # ----------------------------------------------

            for i, (reference, prediction, result) in enumerate(
                zip(
                    batch_references,
                    batch_predictions,
                    batch_results,
                )
            ):
                sample_index = start_idx + i

                results.append({
                    "index": sample_index,
                    "reference": reference,
                    "prediction": prediction,
                    "language": result["language"],
                    "duration": batch_durations[i],
                })

            batch_audio_seconds = sum(batch_durations)

            batch_rtfx = (
                batch_audio_seconds / inference_elapsed
                if inference_elapsed > 0
                else 0
            )

            typer.echo(
                f"Batch completed | "
                f"Decode: {decode_elapsed:.2f}s | "
                f"Inference: {inference_elapsed:.2f}s | "
                f"RTFx: {batch_rtfx:.2f}x"
            )

            # ----------------------------------------------
            # Show predictions
            # ----------------------------------------------

            for i, (reference, prediction) in enumerate(
                zip(batch_references, batch_predictions)
            ):
                typer.echo(f"  [{start_idx + i}] REF: {reference}")
                typer.echo(f"  [{start_idx + i}] HYP: {prediction}")

        benchmark_elapsed = time.perf_counter() - benchmark_start

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)

        rtfx = (
            total_audio_seconds / total_inference_seconds
            if total_inference_seconds > 0
            else 0
        )

        end_to_end_rtfx = (
            total_audio_seconds / benchmark_elapsed
            if benchmark_elapsed > 0
            else 0
        )

        # --------------------------------------------------
        # Save results
        # --------------------------------------------------

        output = {
            "backend": "vllm_async",
            "model": MODEL_NAME,
            "speculative_decoding": {
                "method": "dflash",
                "num_speculative_tokens": 7,
            },
            "batch_size": batch_size,
            "samples": len(predictions),
            "audio_duration": total_audio_seconds,
            "decode_time": total_decode_seconds,
            "inference_time": total_inference_seconds,
            "benchmark_wall_time": benchmark_elapsed,
            "warmup_time": warmup_elapsed,
            "wer": wer,
            "cer": cer,
            "rtfx": rtfx,
            "end_to_end_rtfx": end_to_end_rtfx,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        typer.echo("\n" + "=" * 50)
        typer.echo("ASYNC vLLM BENCHMARK RESULTS")
        typer.echo("=" * 50)

        typer.echo(f"Samples:             {len(predictions)}")
        typer.echo(f"Concurrent requests: {batch_size}")
        typer.echo(f"Audio duration:      {total_audio_seconds:.2f}s")
        typer.echo(f"Decode time:         {total_decode_seconds:.2f}s")
        typer.echo(f"Inference time:      {total_inference_seconds:.2f}s")
        typer.echo(f"Benchmark wall time: {benchmark_elapsed:.2f}s")
        typer.echo(f"WER:                 {wer:.4%}")
        typer.echo(f"CER:                 {cer:.4%}")
        typer.echo(f"Inference RTFx:      {rtfx:.2f}x")
        typer.echo(f"End-to-end RTFx:     {end_to_end_rtfx:.2f}x")

        typer.echo(f"\nResults saved to: {output_path}")

    finally:
        # Stop the engine after all benchmark requests have completed.
        asr.shutdown()


@app.command()
def main(
    dataset_path: str = typer.Option(
        "./dataset/Revolab/ASR-Benchmark-Public"
    ),
    batch_size: int = typer.Option(8, min=1),
    max_samples: Optional[int] = typer.Option(None, min=1),
    output_path: str = typer.Option("vllm_async_results.json"),
    max_model_len: int = typer.Option(8192, min=1),
    max_num_batched_tokens: int = typer.Option(8192, min=1),
):
    asyncio.run(
        run_benchmark(
            dataset_path=dataset_path,
            batch_size=batch_size,
            max_samples=max_samples,
            output_path=output_path,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
        )
    )
    
@app.command()
def benchmark_stream(
    dataset_path: str = typer.Option(
        "./dataset/Revolab/ASR-Benchmark-Public"
    ),
    max_samples: int = typer.Option(None, min=1),
    window_seconds: float = typer.Option(30.0, min=1),
    update_seconds: float = typer.Option(2.0, min=0.1),
    overlap_seconds: float = typer.Option(2.0, min=0),
    output_path: str = typer.Option("streaming_results.json"),
):  
    from jiwer import wer, cer
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
    
    TARGET_SAMPLE_RATE = 16000
    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    typer.echo("Loading dataset...")

    ds = load_dataset(dataset_path)["train"]

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    typer.echo(f"Dataset samples: {len(ds)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    typer.echo("Loading Qwen3-ASR...")

    asr = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B-hf",
        torch_dtype="auto",
        device_map="cuda:0",
    )

    # --------------------------------------------------
    # Streaming benchmark
    # --------------------------------------------------

    predictions = []
    references = []
    results = []

    total_audio_seconds = 0.0
    total_stream_seconds = 0.0
    total_update_latency = 0.0
    total_updates = 0

    for index, sample in enumerate(ds):

        typer.echo(
            f"\n[{index + 1}/{len(ds)}] Processing sample..."
        )

        # ----------------------------------------------
        # Decode audio
        # ----------------------------------------------

        decoded = sample["audio"].get_all_samples()
        audio = prepare_audio(decoded)

        duration = len(audio) / TARGET_SAMPLE_RATE
        reference = sample["text"]

        total_audio_seconds += duration

        # ----------------------------------------------
        # Initialize streaming session
        # ----------------------------------------------

        stream = asr.init_incremental_streaming(
            context="",
            language=None,
            window_seconds=window_seconds,
            update_seconds=update_seconds,
            overlap_seconds=overlap_seconds,
        )

        chunk_samples = int(
            update_seconds * TARGET_SAMPLE_RATE
        )

        sample_start = time.perf_counter()
        sample_updates = 0
        sample_update_latency = 0.0

        # ----------------------------------------------
        # Feed audio chunks
        # ----------------------------------------------

        for start in range(0, len(audio), chunk_samples):

            chunk = audio[start:start + chunk_samples]

            feed_start = time.perf_counter()
            updates = stream.feed(chunk)
            feed_elapsed = time.perf_counter() - feed_start

            for update in updates:
                sample_updates += 1
                total_updates += 1

                latency = update.decode_wall_sec

                sample_update_latency += latency
                total_update_latency += latency

                typer.echo(
                    f"  Epoch {update.epoch_index} | "
                    f"Latency: {latency:.3f}s | "
                    f"KV tokens: {update.kv_tokens} | "
                    f"Audio KV: {update.audio_kv_tokens} | "
                    f"Cache rebuilt: {update.cache_rebuilt}"
                )

        # ----------------------------------------------
        # Finish session
        # ----------------------------------------------

        finish_start = time.perf_counter()
        final = stream.finish()
        finish_elapsed = time.perf_counter() - finish_start

        sample_elapsed = time.perf_counter() - sample_start

        total_stream_seconds += sample_elapsed

        prediction = final.text

        predictions.append(prediction)
        references.append(reference)

        sample_wer = wer(reference, prediction)
        sample_cer = cer(reference, prediction)

        results.append({
            "index": index,
            "duration_seconds": duration,
            "reference": reference,
            "prediction": prediction,
            "wer": sample_wer,
            "cer": sample_cer,
            "stream_seconds": sample_elapsed,
            "finish_seconds": finish_elapsed,
            "updates": sample_updates,
            "average_update_latency": (
                sample_update_latency / sample_updates
                if sample_updates else 0.0
            ),
        })

        typer.echo(f"  Duration: {duration:.2f}s")
        typer.echo(f"  Stream time: {sample_elapsed:.2f}s")
        typer.echo(f"  Updates: {sample_updates}")
        typer.echo(f"  REF: {reference}")
        typer.echo(f"  HYP: {prediction}")
        typer.echo(f"  WER: {sample_wer:.2%}")
        typer.echo(f"  CER: {sample_cer:.2%}")

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------

    corpus_wer = wer(references, predictions)
    corpus_cer = cer(references, predictions)

    rtfx = (
        total_audio_seconds / total_stream_seconds
        if total_stream_seconds > 0
        else 0.0
    )

    avg_update_latency = (
        total_update_latency / total_updates
        if total_updates > 0
        else 0.0
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output = {
        "config": {
            "model": "Qwen/Qwen3-ASR-1.7B-hf",
            "window_seconds": window_seconds,
            "update_seconds": update_seconds,
            "overlap_seconds": overlap_seconds,
        },
        "summary": {
            "samples": len(predictions),
            "audio_seconds": total_audio_seconds,
            "stream_seconds": total_stream_seconds,
            "rtfx": rtfx,
            "wer": corpus_wer,
            "cer": corpus_cer,
            "total_updates": total_updates,
            "average_update_latency": avg_update_latency,
        },
        "results": results,
    }

    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    typer.echo("\n" + "=" * 55)
    typer.echo("REVOLAB STREAMING BENCHMARK")
    typer.echo("=" * 55)

    typer.echo(f"Samples:               {len(predictions)}")
    typer.echo(f"Audio duration:        {total_audio_seconds:.2f}s")
    typer.echo(f"Streaming time:        {total_stream_seconds:.2f}s")
    typer.echo(f"WER:                   {corpus_wer:.4%}")
    typer.echo(f"CER:                   {corpus_cer:.4%}")
    typer.echo(f"RTFx:                  {rtfx:.2f}x")
    typer.echo(f"Total updates:         {total_updates}")
    typer.echo(f"Avg update latency:    {avg_update_latency:.3f}s")
    typer.echo(f"Results saved to:      {output_path}")

if __name__ == "__main__":
    app()
