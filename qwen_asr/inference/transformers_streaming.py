
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""Bounded-context streaming ASR using native Hugging Face Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
import time
import unicodedata
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


SAMPLE_RATE = 16_000
UPDATE_SECONDS = 2.0

_END_IDS = {151645, 151643}
_PUNCTUATION = set("，。！？、；：“”‘’\"'")


@dataclass(frozen=True)
class IncrementalKVUpdate:
    """Complete transcript snapshot emitted on one update tick."""

    text: str
    window_text: str
    audio_start_sec: float
    audio_end_sec: float
    window_duration_sec: float
    decode_wall_sec: float
    final: bool = False
    epoch_index: int = 0
    kv_tokens: int = 0
    audio_kv_tokens: int = 0
    cache_rebuilt: bool = False
    feature_invalidated_blocks: Tuple[int, ...] = ()
    language: str = ""


def _normalized_with_positions(text: str):
    kept = [
        (ch, index)
        for index, ch in enumerate(text)
        if not ch.isspace() and ch not in _PUNCTUATION
    ]
    return (
        "".join(ch for ch, _ in kept),
        [index for _, index in kept],
    )


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


def _join_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()

    if not left:
        return right

    if not right:
        return left

    if unicodedata.category(right[0]).startswith("P"):
        return left + right

    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right

    if left[-1].isalnum() and right[0].isalnum():
        return left + " " + right

    return left + right


def _merge_overlap(committed: str, current: str) -> str:
    """Merge adjacent epochs without duplicating their shared audio."""

    if not committed:
        return current.strip()

    if not current:
        return committed.strip()

    left, _ = _normalized_with_positions(committed)
    right, right_positions = _normalized_with_positions(current)

    if not left or not right:
        return _join_text(committed, current)

    max_overlap = min(24, len(left), len(right))
    matched_prefix = 0

    for size in range(max_overlap, 2, -1):
        if left[-size:] == right[:size]:
            matched_prefix = size
            break

    if not matched_prefix:
        left_tail = left[-max_overlap:]
        right_head = right[:max_overlap]

        matcher = SequenceMatcher(
            None,
            left_tail,
            right_head,
            autojunk=False,
        )

        candidates = [
            block
            for block in matcher.get_matching_blocks()
            if block.size >= 3
            and block.b <= 1
            and len(left_tail) - (block.a + block.size) <= 2
        ]

        if candidates:
            block = max(
                candidates,
                key=lambda item: (item.size, -item.b),
            )
            matched_prefix = block.b + block.size

    if not matched_prefix:
        return _join_text(committed, current)

    cut = right_positions[matched_prefix - 1] + 1
    remainder = current[cut:]

    while remainder and (
        remainder[0].isspace()
        or remainder[0] in _PUNCTUATION
    ):
        remainder = remainder[1:]

    return _join_text(committed, remainder)


class TransformersIncrementalStreamingASR:
    """
    Bounded-context Qwen3-ASR streaming using native Hugging Face Transformers.

    Expected input:
        16 kHz mono PCM, float [-1, 1] or int16.

    Expected ASR wrapper:
        asr.model
        asr.processor
        asr._build_text_prompt(...)

    The model must be a native Hugging Face
    Qwen3ASRForConditionalGeneration instance.

    This implementation performs a fresh generate() per update.
    It does not claim incremental encoder or decoder KV reuse.
    """

    def __init__(
        self,
        asr: Any,
        *,
        language: Optional[str] = None,
        context: str = "",
        window_seconds: float = 30.0,
        update_seconds: float = UPDATE_SECONDS,
        overlap_seconds: float = 2.0,
        rollback_tokens: int = 8,
        max_new_tokens: int = 512,
        feature_mean_threshold: float = 0.10,
        feature_changed_fraction_threshold: float = 0.90,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Qwen3-ASR expects {SAMPLE_RATE} Hz audio"
            )

        if abs(update_seconds - UPDATE_SECONDS) > 1e-6:
            raise ValueError(
                "ASR update period is fixed at 2 seconds"
            )

        if window_seconds <= overlap_seconds:
            raise ValueError(
                "window_seconds must exceed overlap_seconds"
            )

        if overlap_seconds < update_seconds:
            raise ValueError(
                "overlap_seconds must be >= update_seconds"
            )

        stride = window_seconds - overlap_seconds

        for value, name in (
            (window_seconds, "window_seconds"),
            (overlap_seconds, "overlap_seconds"),
            (stride, "epoch stride"),
        ):
            if abs(
                value / UPDATE_SECONDS
                - round(value / UPDATE_SECONDS)
            ) > 1e-6:
                raise ValueError(
                    f"{name} must be a multiple of 2 seconds"
                )

        if max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be positive"
            )

        self.asr = asr
        self.model = asr.model.eval()
        self.processor = asr.processor

        # Native Transformers architecture.
        if type(self.model).__name__ != "Qwen3ASRForConditionalGeneration":
            raise TypeError(
                "Expected native Qwen3ASRForConditionalGeneration, "
                f"got {type(self.model).__name__}"
            )

        self.audio_tower = self.model.model.audio_tower
        self.language_model = self.model.model.language_model
        self.projector = self.model.model.multi_modal_projector
        self.lm_head = self.model.lm_head

        self.force_language = language
        self.context = context or ""

        self.window_seconds = float(window_seconds)
        self.update_seconds = UPDATE_SECONDS
        self.overlap_seconds = float(overlap_seconds)
        self.epoch_stride_seconds = stride

        self.max_new_tokens = max_new_tokens
        self.sample_rate = sample_rate
        
        self.rollback_tokens = rollback_tokens
        self.feature_mean_threshold = feature_mean_threshold
        self.feature_changed_fraction_threshold = (
            feature_changed_fraction_threshold
        )

        self.window_samples = int(
            round(window_seconds * sample_rate)
        )
        self.stride_samples = int(
            round(stride * sample_rate)
        )
        self.hop_samples = int(
            round(UPDATE_SECONDS * sample_rate)
        )

        self.device = self.model.device
        self.dtype = self.model.dtype

        self.prompt = asr._build_text_prompt(
            context=self.context,
            force_language=self.force_language,
        )

        # Audio buffers.
        self._ring = np.empty((0,), dtype=np.float32)
        self._pending = np.empty((0,), dtype=np.float32)

        self._total_samples = 0
        self._closed = False

        # Transcript state.
        self._last_update: Optional[IncrementalKVUpdate] = None
        self._text = ""
        self._language = ""

        self._committed_text = ""
        self._epoch_index: Optional[int] = None
        self._epoch_start_sample = 0
        self._epoch_text = ""

        # Metrics.
        self.update_count = 0
        self.total_encoder_input_feature_frames = 0
        self.total_naive_prefix_replay_feature_frames = 0
        self.feature_invalidation_count = 0
        self.decoder_kv_rebuild_count = 0
        self.max_audio_kv_tokens = 0
        self.max_persistent_kv_tokens = 0

    @property
    def text(self) -> str:
        return self._text

    @property
    def language(self) -> str:
        return self._language

    @property
    def buffered_audio_seconds(self) -> float:
        return len(self._ring) / self.sample_rate

    @property
    def audio_seconds(self) -> float:
        return (
            self._total_samples + len(self._pending)
        ) / self.sample_rate

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare_batch(self, audio: np.ndarray):
        """
        Prepare native Transformers model inputs.

        The native processor returns feature_attention_mask.
        The native model forward expects input_features_mask.
        """

        batch = self.processor(
            text=[self.prompt],
            audio=[audio],
            return_tensors="pt",
        )

        if "input_features" not in batch:
            raise RuntimeError(
                "Processor did not return input_features"
            )

        if "input_ids" not in batch:
            raise RuntimeError(
                "Processor did not return input_ids"
            )

        if "feature_attention_mask" in batch:
            feature_mask = batch["feature_attention_mask"]
        elif "input_features_mask" in batch:
            feature_mask = batch["input_features_mask"]
        else:
            raise RuntimeError(
                "Processor did not return an audio feature mask"
            )

        inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "input_features": batch["input_features"].to(
                device=self.device,
                dtype=self.dtype,
            ),
            "input_features_mask": feature_mask.to(
                device=self.device,
                dtype=torch.long,
            ),
        }

        return inputs

    def warmup(self, seconds: float = 2.0) -> float:
        """Warm up model kernels. The silence is discarded."""

        if self._closed:
            raise RuntimeError(
                "Cannot warm up after finish()"
            )

        if seconds <= 0:
            raise ValueError(
                "seconds must be positive"
            )

        silence = np.zeros(
            int(round(seconds * self.sample_rate)),
            dtype=np.float32,
        )

        inputs = self._prepare_batch(silence)

        self._sync()
        started = time.perf_counter()

        with torch.inference_mode():
            self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            )

        self._sync()

        return time.perf_counter() - started

    def feed(
        self,
        pcm16k: np.ndarray,
    ) -> List[IncrementalKVUpdate]:

        if self._closed:
            raise RuntimeError(
                "Cannot feed audio after finish()"
            )

        if pcm16k is None:
            raise ValueError(
                "pcm16k must not be None"
            )

        x = np.asarray(pcm16k)

        if x.ndim > 1:
            if x.shape[-1] == 1:
                x = x.reshape(-1)
            else:
                raise ValueError(
                    "Expected mono audio; downmix stereo first"
                )

        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        updates = []
        cursor = 0

        while cursor < len(x):

            needed = self.hop_samples - len(self._pending)

            take = min(
                needed,
                len(x) - cursor,
            )

            self._pending = np.concatenate(
                (
                    self._pending,
                    x[cursor:cursor + take],
                )
            )

            cursor += take

            if len(self._pending) == self.hop_samples:

                self._append_audio(self._pending)

                self._pending = np.empty(
                    (0,),
                    dtype=np.float32,
                )

                updates.append(
                    self._decode_update(final=False)
                )

        return updates

    def finish(self) -> IncrementalKVUpdate:

        if self._closed:
            raise RuntimeError(
                "Stream is already closed"
            )

        if self._last_update is not None and not len(self._pending):
            self._closed = True

            final = replace(
                self._last_update,
                final=True,
                decode_wall_sec=0.0,
            )

            self._last_update = final
            return final

        if len(self._pending):
            self._append_audio(self._pending)

            self._pending = np.empty(
                (0,),
                dtype=np.float32,
            )

        self._closed = True

        if not len(self._ring):
            final = IncrementalKVUpdate(
                text="",
                window_text="",
                audio_start_sec=0.0,
                audio_end_sec=0.0,
                window_duration_sec=0.0,
                decode_wall_sec=0.0,
                final=True,
            )

            self._last_update = final
            return final

        final = self._decode_update(final=True)
        self._last_update = final

        return final

    def _append_audio(self, samples: np.ndarray):

        self._total_samples += len(samples)

        if len(samples) >= self.window_samples:
            self._ring = np.array(
                samples[-self.window_samples:],
                dtype=np.float32,
                copy=True,
            )
        else:
            self._ring = np.concatenate(
                (self._ring, samples)
            )[-self.window_samples:].copy()

    def _epoch_for_time(self, end_sec: float) -> int:

        epochs_after_first = math.ceil(
            (
                end_sec
                - self.window_seconds
                - 1e-6
            ) / self.epoch_stride_seconds
        )

        return max(0, int(epochs_after_first))

    def _decode_update(
        self,
        *,
        final: bool,
    ) -> IncrementalKVUpdate:

        from qwen_asr.inference.utils import parse_asr_output

        total_samples = self._total_samples
        end_sec = total_samples / self.sample_rate

        window_start_sample = (
            total_samples - len(self._ring)
        )

        window_start_sec = (
            window_start_sample / self.sample_rate
        )

        epoch_index = self._epoch_for_time(end_sec)

        epoch_start_sample = (
            epoch_index * self.stride_samples
        )

        epoch_start_sec = (
            epoch_start_sample / self.sample_rate
        )

        if epoch_start_sample < window_start_sample:
            raise RuntimeError(
                "Ring buffer no longer contains active epoch"
            )

        offset = (
            epoch_start_sample - window_start_sample
        )

        epoch_audio = self._ring[offset:].copy()

        epoch_duration = (
            len(epoch_audio) / self.sample_rate
        )

        # Commit the previous epoch when the next epoch begins.
        if self._epoch_index is None:

            self._epoch_index = epoch_index
            self._epoch_start_sample = epoch_start_sample

        elif epoch_index != self._epoch_index:

            self._committed_text = _merge_overlap(
                self._committed_text,
                self._epoch_text,
            )

            self._epoch_index = epoch_index
            self._epoch_start_sample = epoch_start_sample
            self._epoch_text = ""

        self._sync()
        tick_start = time.perf_counter()

        inputs = self._prepare_batch(epoch_audio)

        feature_mask = inputs["input_features_mask"]

        feature_len = int(
            feature_mask.sum().item()
        )

        # Native Transformers generation.
        with torch.inference_mode():

            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # generate() returns the original prompt plus generated tokens.
        input_length = inputs["input_ids"].shape[1]

        output_ids = generated[:, input_length:]

        decoded = self.processor.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self._language, self._epoch_text = parse_asr_output(
            decoded,
            user_language=self.force_language,
        )

        self._text = _merge_overlap(
            self._committed_text,
            self._epoch_text,
        )

        self._sync()

        latency = time.perf_counter() - tick_start

        self.total_encoder_input_feature_frames += feature_len
        self.total_naive_prefix_replay_feature_frames += feature_len

        self.update_count += 1

        update = IncrementalKVUpdate(
            text=self._text,
            window_text=self._epoch_text,
            audio_start_sec=epoch_start_sec,
            audio_end_sec=end_sec,
            window_duration_sec=epoch_duration,
            decode_wall_sec=latency,
            final=final,
            epoch_index=epoch_index,

            # No persistent KV cache is reused between updates.
            kv_tokens=0,
            audio_kv_tokens=0,
            cache_rebuilt=False,
            feature_invalidated_blocks=(),

            language=self._language,
        )

        self._last_update = update

        return update
