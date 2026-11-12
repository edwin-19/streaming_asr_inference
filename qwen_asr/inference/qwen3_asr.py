# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from qwen_asr.startup_logging import StartupTimer, log_startup, optional_timer


def _trace_requests_enabled() -> bool:
    return os.getenv("QWEN_ASR_TRACE_REQUESTS", "0").strip().lower() in {
        "1", "true", "yes", "y"
    }


def _resolve_stop_token_ids(processor: Any) -> List[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return []

    token_ids: List[int] = []

    for token in ("<|im_end|>", "<|endoftext|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue

        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id not in token_ids
        ):
            token_ids.append(token_id)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        isinstance(eos_token_id, int)
        and eos_token_id >= 0
        and eos_token_id not in token_ids
    ):
        token_ids.append(eos_token_id)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if (
        isinstance(pad_token_id, int)
        and pad_token_id >= 0
        and pad_token_id not in token_ids
    ):
        token_ids.append(pad_token_id)

    return token_ids


from .compile_utils import compile_model_forward
from .qwen3_forced_aligner import Qwen3ForcedAligner
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    chunk_list,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)


@dataclass
class ASRTranscription:
    language: str
    text: str
    time_stamps: Optional[Any] = None
    duration: Optional[float] = None


@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int
    max_window_sec: float
    max_window_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray
    audio_samples_seen: int

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str
    _committed_token_ids: List[int]
    _window_token_ids: List[int]


class Qwen3ASRModel:
    """
    Unified inference wrapper for Qwen3-ASR.

    Backends:
      - Transformers
      - vLLM synchronous offline engine
      - vLLM asynchronous engine

    For the asynchronous vLLM backend, use:
      - streaming_transcribe_async()
      - finish_streaming_transcribe_async()

    The async engine is intended for concurrent streaming requests.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        backend: str = "transformers",
        sampling_params: Any | None = None,
        forced_aligner: Optional[Qwen3ForcedAligner] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
        torch_compile_enabled: bool = False,
        generation_cache_implementation: str | None = None,
        async_engine: bool = False,
    ):
        self.backend = backend
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.forced_aligner = forced_aligner
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = max_new_tokens
        self.torch_compile_enabled = bool(torch_compile_enabled)
        self.generation_cache_implementation = generation_cache_implementation

        self._async_engine = bool(async_engine)

        # This lock protects the synchronous generate() path.
        # AsyncLLM requests must NOT use this lock: the async engine
        # performs request scheduling and batching internally.
        self._generate_lock = threading.Lock()

        if backend == "transformers":
            self.device = getattr(model, "device", None)

            if self.device is None:
                try:
                    self.device = next(model.parameters()).device
                except StopIteration:
                    self.device = torch.device("cpu")

            self.dtype = getattr(model, "dtype", torch.float32)
        else:
            self.device = None
            self.dtype = None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        if self._async_engine:
            raise RuntimeError(
                "Synchronous generation is unavailable when using LLMAsync(). "
                "Use the asynchronous streaming methods instead."
            )

        if (
            self.backend == "transformers"
            and self.generation_cache_implementation is not None
        ):
            kwargs.setdefault(
                "cache_implementation",
                self.generation_cache_implementation,
            )

        with self._generate_lock:
            return self.model.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Transformers backend
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: Optional[int] = 512,
        torch_compile: bool = False,
        torch_compile_backend: str = "inductor",
        torch_compile_mode: str = "default",
        torch_compile_fullgraph: bool = False,
        torch_compile_dynamic: bool | None = None,
        generation_cache_implementation: str | None = None,
        **kwargs,
    ) -> "Qwen3ASRModel":

        log_startup(f"Transformers backend kwargs: {kwargs}")

        with StartupTimer("load Transformers model"):
            model = AutoModelForMultimodalLM.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )

        model.eval()

        log_startup(
            "Transformers attention implementation: "
            f"{getattr(model.config, '_attn_implementation', 'auto')}"
        )

        compile_enabled = compile_model_forward(
            model,
            enabled=torch_compile,
            backend=torch_compile_backend,
            mode=torch_compile_mode,
            fullgraph=torch_compile_fullgraph,
            dynamic=torch_compile_dynamic,
            label="ASR model",
        )

        with StartupTimer("load Transformers processor"):
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path,
                fix_mistral_regex=True,
            )

        forced_aligner_model = None

        if forced_aligner is not None:
            with StartupTimer("load forced aligner"):
                forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                    forced_aligner,
                    **(forced_aligner_kwargs or {}),
                )

        return cls(
            model=model,
            processor=processor,
            backend="transformers",
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            torch_compile_enabled=compile_enabled,
            generation_cache_implementation=generation_cache_implementation,
        )

    # ------------------------------------------------------------------
    # Synchronous vLLM backend
    # ------------------------------------------------------------------

    @classmethod
    def LLM(
        cls,
        model: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> "Qwen3ASRModel":

        log_startup(f"vLLM backend kwargs: {kwargs}")

        try:
            with StartupTimer("import vLLM runtime classes"):
                from vllm import LLM as vLLM
                from vllm import SamplingParams
        except Exception as exc:
            raise ImportError(
                "vLLM is not available. Install qwen-asr with vLLM support."
            ) from exc

        with StartupTimer("initialize vLLM LLM"):
            llm = vLLM(model=model, **kwargs)

        with StartupTimer("load Qwen3-ASR processor"):
            processor = AutoProcessor.from_pretrained(
                model,
                fix_mistral_regex=True,
            )

        sampling_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": int(max_new_tokens),
        }

        stop_token_ids = _resolve_stop_token_ids(processor)

        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids
            log_startup(f"vLLM stop_token_ids: {stop_token_ids}")

        sampling_params = SamplingParams(**sampling_kwargs)

        forced_aligner_model = None

        if forced_aligner is not None:
            with StartupTimer("load forced aligner"):
                forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                    forced_aligner,
                    **(forced_aligner_kwargs or {}),
                )

        return cls(
            backend="vllm",
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Asynchronous vLLM backend
    # ------------------------------------------------------------------

    @classmethod
    def LLMAsync(
        cls,
        model: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> "Qwen3ASRModel":
        """
        Initialize the vLLM AsyncLLM engine.

        This uses the vLLM 0.32 AsyncLLM API.

        Unlike the synchronous LLM backend, each streaming session can
        submit inference requests concurrently. vLLM's scheduler manages
        the requests and batches their decoding work.

        Note:
            This does not make the synchronous transcribe() API async.
            Use streaming_transcribe_async() with this backend.
        """

        log_startup(f"vLLM async backend kwargs: {kwargs}")

        try:
            with StartupTimer("import vLLM async runtime classes"):
                from vllm import SamplingParams
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.v1.engine.async_llm import AsyncLLM
        except Exception as exc:
            raise ImportError(
                "Could not import vLLM AsyncLLM. "
                "This implementation expects vLLM 0.32.0."
            ) from exc

        with StartupTimer("initialize vLLM AsyncLLM"):
            engine_args = AsyncEngineArgs(
                model=model,
                **kwargs,
            )

            engine = AsyncLLM.from_engine_args(
                engine_args,
                start_engine_loop=True,
            )

        with StartupTimer("load Qwen3-ASR processor"):
            processor = AutoProcessor.from_pretrained(
                model,
                fix_mistral_regex=True,
            )

        sampling_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": int(max_new_tokens),
        }

        stop_token_ids = _resolve_stop_token_ids(processor)

        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids
            log_startup(f"vLLM async stop_token_ids: {stop_token_ids}")

        sampling_params = SamplingParams(**sampling_kwargs)

        forced_aligner_model = None

        if forced_aligner is not None:
            with StartupTimer("load forced aligner"):
                forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                    forced_aligner,
                    **(forced_aligner_kwargs or {}),
                )

        return cls(
            backend="vllm",
            model=engine,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            async_engine=True,
        )

    async def _generate_streaming_text_async(
        self,
        prompt: str,
        audio: np.ndarray,
    ) -> str:
        """
        Submit one multimodal ASR request to vLLM AsyncLLM.

        The method consumes the async generator until the request finishes
        and returns the final generated text.
        """

        if not self._async_engine:
            raise RuntimeError(
                "_generate_streaming_text_async requires LLMAsync()."
            )

        request_id = str(uuid.uuid4())

        item = {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [audio],
            },
        }

        final_output = None

        async for output in self.model.generate(
            item,
            self.sampling_params,
            request_id,
        ):
            final_output = output

        if final_output is None or not final_output.outputs:
            return ""

        return str(final_output.outputs[0].text)

    # ------------------------------------------------------------------
    # General utilities
    # ------------------------------------------------------------------

    def get_supported_languages(self) -> List[str]:
        return list(SUPPORTED_LANGUAGES)

    @torch.inference_mode()
    def warm_up(
        self,
        *,
        max_new_tokens: int = 1,
        iterations: int = 1,
        audio: AudioLike | None = None,
        aligner_text: str = "Warmup",
    ) -> None:

        if self._async_engine:
            raise RuntimeError(
                "warm_up() uses synchronous inference and cannot be used "
                "with LLMAsync(). Warm up the async engine using an async "
                "streaming request instead."
            )

        warmup_audio: AudioLike = (
            audio
            if audio is not None
            else (np.zeros((SAMPLE_RATE // 2,), dtype=np.float32), SAMPLE_RATE)
        )

        original_max_new_tokens = self.max_new_tokens
        original_sampling_params = self.sampling_params

        try:
            warmup_max_tokens = max(1, int(max_new_tokens))
            self.max_new_tokens = warmup_max_tokens

            if self.backend == "vllm" and original_sampling_params is not None:
                sampling_kwargs: Dict[str, Any] = {
                    "temperature": 0.0,
                    "max_tokens": warmup_max_tokens,
                }

                stop_token_ids = getattr(
                    original_sampling_params,
                    "stop_token_ids",
                    None,
                )

                if stop_token_ids:
                    sampling_kwargs["stop_token_ids"] = list(stop_token_ids)

                self.sampling_params = type(original_sampling_params)(
                    **sampling_kwargs
                )

            for iteration in range(max(1, int(iterations))):
                language = None if iteration % 2 == 0 else "English"

                self.transcribe(
                    audio=warmup_audio,
                    language=language,
                    return_time_stamps=False,
                )

            if self.forced_aligner is not None:
                self.forced_aligner.warm_up(
                    audio=warmup_audio,
                    text=aligner_text,
                    language="English",
                )

        finally:
            self.max_new_tokens = original_max_new_tokens
            self.sampling_params = original_sampling_params

    # ------------------------------------------------------------------
    # Synchronous transcription API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:

        if self._async_engine:
            raise RuntimeError(
                "transcribe() is synchronous and cannot be used with "
                "LLMAsync(). Use an async streaming API or initialize "
                "the model with LLM()."
            )

        if return_time_stamps and self.forced_aligner is None:
            raise ValueError(
                "return_time_stamps=True requires `forced_aligner` "
                "to be provided at initialization."
            )

        trace_requests = _trace_requests_enabled()

        with optional_timer("normalize request audio", trace_requests):
            wavs = normalize_audios(audio)

        n = len(wavs)

        ctxs = context if isinstance(context, list) else [context]

        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n

        if len(ctxs) != n:
            raise ValueError(
                f"Batch size mismatch: audio={n}, context={len(ctxs)}"
            )

        if language is None:
            langs_in: List[Optional[str]] = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]

            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n

            if len(langs_in) != n:
                raise ValueError(
                    f"Batch size mismatch: audio={n}, language={len(langs_in)}"
                )

        langs_norm: List[Optional[str]] = []

        for lang in langs_in:
            if lang is None or str(lang).strip() == "":
                langs_norm.append(None)
            else:
                normalized = normalize_language_name(str(lang))
                validate_language(normalized)
                langs_norm.append(normalized)

        max_chunk_sec = (
            MAX_FORCE_ALIGN_INPUT_SECONDS
            if return_time_stamps
            else MAX_ASR_INPUT_SECONDS
        )

        with optional_timer("split request audio into chunks", trace_requests):
            chunks: List[AudioChunk] = []

            for i, wav in enumerate(wavs):
                parts = split_audio_into_chunks(
                    wav=wav,
                    sr=SAMPLE_RATE,
                    max_chunk_sec=max_chunk_sec,
                )

                for j, (chunk_wav, offset_sec) in enumerate(parts):
                    chunks.append(
                        AudioChunk(
                            orig_index=i,
                            chunk_index=j,
                            wav=chunk_wav,
                            sr=SAMPLE_RATE,
                            offset_sec=offset_sec,
                        )
                    )

        chunk_ctx = [ctxs[c.orig_index] for c in chunks]
        chunk_lang = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs = [c.wav for c in chunks]

        with optional_timer(
            f"infer ASR chunks count={len(chunks)}",
            trace_requests,
        ):
            raw_outputs = self._infer_asr(
                chunk_ctx,
                chunk_wavs,
                chunk_lang,
            )

        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []

        with optional_timer("parse ASR output", trace_requests):
            for output, forced_lang in zip(raw_outputs, chunk_lang):
                lang, text = parse_asr_output(
                    output,
                    user_language=forced_lang,
                )
                per_chunk_lang.append(lang)
                per_chunk_text.append(text)

        per_chunk_align: List[Optional[Any]] = [None] * len(chunks)

        if return_time_stamps:
            to_align_audio = []
            to_align_text = []
            to_align_lang = []
            to_align_idx = []

            for idx, (chunk, text, lang_pred) in enumerate(
                zip(chunks, per_chunk_text, per_chunk_lang)
            ):
                if text.strip() == "":
                    continue

                to_align_audio.append((chunk.wav, chunk.sr))
                to_align_text.append(text)
                to_align_lang.append(lang_pred)
                to_align_idx.append(idx)

            aligned_results: List[Any] = []

            for audio_batch, text_batch, lang_batch in zip(
                chunk_list(to_align_audio, self.max_inference_batch_size),
                chunk_list(to_align_text, self.max_inference_batch_size),
                chunk_list(to_align_lang, self.max_inference_batch_size),
            ):
                aligned_results.extend(
                    self.forced_aligner.align(
                        audio=audio_batch,
                        text=text_batch,
                        language=lang_batch,
                    )
                )

            for k, idx in enumerate(to_align_idx):
                chunk = chunks[idx]
                result = aligned_results[k]
                per_chunk_align[idx] = self._offset_align_result(
                    result,
                    chunk.offset_sec,
                )

        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        out_aligns: List[List[Any]] = [[] for _ in range(n)]

        for chunk, lang, text, align in zip(
            chunks,
            per_chunk_lang,
            per_chunk_text,
            per_chunk_align,
        ):
            out_langs[chunk.orig_index].append(lang)
            out_texts[chunk.orig_index].append(text)

            if return_time_stamps and align is not None:
                out_aligns[chunk.orig_index].append(align)

        results: List[ASRTranscription] = []

        for i in range(n):
            merged_text = "".join(
                text for text in out_texts[i] if text is not None
            )
            merged_language = merge_languages(out_langs[i])

            merged_align = None

            if return_time_stamps:
                merged_align = self._merge_align_results(out_aligns[i])

            results.append(
                ASRTranscription(
                    language=merged_language,
                    text=merged_text,
                    time_stamps=merged_align,
                    duration=round(len(wavs[i]) / SAMPLE_RATE, 3),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Prompt construction and synchronous inference
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        context: str,
        audio_payload: Any,
    ) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": audio_payload,
                    }
                ],
            },
        ]

    def _build_text_prompt(
        self,
        context: str,
        force_language: Optional[str],
    ) -> str:

        messages = self._build_messages(
            context=context,
            audio_payload="",
        )

        base = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if force_language:
            base += f"language {force_language}{'<asr_text>'}"

        return base

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:

        if self._async_engine:
            raise RuntimeError(
                "Synchronous ASR inference is unavailable with LLMAsync()."
            )

        if self.backend == "vllm":
            return self._infer_asr_vllm(contexts, wavs, languages)

        if self.backend == "transformers":
            return self._infer_asr_transformers(contexts, wavs, languages)

        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _infer_asr_transformers(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:

        outputs: List[str] = []

        texts = [
            self._build_text_prompt(context=c, force_language=lang)
            for c, lang in zip(contexts, languages)
        ]

        batch_size = self.max_inference_batch_size

        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i:i + batch_size]
            sub_wavs = wavs[i:i + batch_size]

            inputs = self.processor(
                text=sub_text,
                audio=sub_wavs,
                return_tensors="pt",
                padding=True,
            )

            inputs = inputs.to(self.model.device).to(self.model.dtype)

            generated = self._generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            sequences = getattr(generated, "sequences", generated)

            decoded = self.processor.batch_decode(
                sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            outputs.extend(list(decoded))

        return outputs

    def _infer_asr_vllm(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:

        inputs: List[Dict[str, Any]] = []
        trace_requests = _trace_requests_enabled()

        with optional_timer(
            f"build vLLM ASR prompts count={len(wavs)}",
            trace_requests,
        ):
            for context, wav, language in zip(contexts, wavs, languages):
                inputs.append(
                    {
                        "prompt": self._build_text_prompt(
                            context=context,
                            force_language=language,
                        ),
                        "multi_modal_data": {
                            "audio": [wav],
                        },
                    }
                )

        outputs: List[str] = []

        for batch in chunk_list(inputs, self.max_inference_batch_size):
            with optional_timer(
                f"vLLM generate batch size={len(batch)}",
                trace_requests,
            ):
                generated = self._generate(
                    batch,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

            outputs.extend(
                item.outputs[0].text for item in generated
            )

        return outputs

    def _generate_streaming_text(
        self,
        prompt: str,
        audio: np.ndarray,
    ) -> str:

        if self._async_engine:
            raise RuntimeError(
                "Use _generate_streaming_text_async() with LLMAsync()."
            )

        if self.backend == "vllm":
            item = {
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": [audio],
                },
            }

            generated = self._generate(
                [item],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            return str(generated[0].outputs[0].text)

        inputs = self.processor(
            text=[prompt],
            audio=[audio],
            return_tensors="pt",
            padding=True,
        )

        inputs = inputs.to(self.model.device).to(self.model.dtype)

        generated = self._generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        sequences = getattr(generated, "sequences", generated)

        decoded = self.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return str(decoded[0])

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _offset_align_result(
        self,
        result: Any,
        offset_sec: float,
    ) -> Any:

        if result is None:
            return None

        items = []

        for item in result.items:
            items.append(
                type(item)(
                    text=item.text,
                    start_time=round(item.start_time + offset_sec, 3),
                    end_time=round(item.end_time + offset_sec, 3),
                )
            )

        return type(result)(items=items)

    def _merge_align_results(
        self,
        results: List[Any],
    ) -> Optional[Any]:

        if not results:
            return None

        all_items = []

        for result in results:
            if result is None:
                continue

            all_items.extend(list(result.items))

        if not all_items:
            return None

        return type(results[0])(items=all_items)

    # ------------------------------------------------------------------
    # Experimental Transformers incremental streaming
    # ------------------------------------------------------------------

    def init_incremental_streaming(
        self,
        context: str = "",
        language: Optional[str] = None,
        window_seconds: float = 30.0,
        update_seconds: float = 2.0,
        overlap_seconds: float = 2.0,
        rollback_tokens: int = 8,
        max_new_tokens: Optional[int] = None,
        feature_mean_threshold: float = 0.10,
        feature_changed_fraction_threshold: float = 0.90,
    ):

        if self.backend != "transformers":
            raise RuntimeError(
                "Incremental KV streaming currently supports only "
                "the Transformers backend. vLLM does not expose the "
                "required audio-tower outputs and decoder KV cache."
            )

        force_language = None

        if language is not None and str(language).strip():
            force_language = normalize_language_name(str(language))
            validate_language(force_language)

        from .transformers_streaming import TransformersIncrementalStreamingASR

        return TransformersIncrementalStreamingASR(
            self,
            language=force_language,
            context=context or "",
            window_seconds=window_seconds,
            update_seconds=update_seconds,
            overlap_seconds=overlap_seconds,
            rollback_tokens=rollback_tokens,
            max_new_tokens=(
                int(max_new_tokens)
                if max_new_tokens is not None
                else int(self.max_new_tokens)
            ),
            feature_mean_threshold=feature_mean_threshold,
            feature_changed_fraction_threshold=feature_changed_fraction_threshold,
            sample_rate=SAMPLE_RATE,
        )

    # ------------------------------------------------------------------
    # Native streaming state
    # ------------------------------------------------------------------

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        max_window_sec: float = 30.0,
    ) -> ASRStreamingState:

        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(
                f"chunk_size_sec must be > 0, got: {chunk_size_sec}"
            )

        if (
            max_window_sec is None
            or float(max_window_sec) < float(chunk_size_sec)
        ):
            raise ValueError(
                "max_window_sec must be greater than or equal to "
                f"chunk_size_sec, got: {max_window_sec}"
            )

        force_language = None

        if language is not None and str(language).strip() != "":
            normalized = normalize_language_name(str(language))
            validate_language(normalized)
            force_language = normalized

        chunk_size_samples = int(
            round(float(chunk_size_sec) * SAMPLE_RATE)
        )
        chunk_size_samples = max(1, chunk_size_samples)

        max_window_samples = int(
            round(float(max_window_sec) * SAMPLE_RATE)
        )
        max_window_samples = max(
            chunk_size_samples,
            max_window_samples,
        )

        prompt_raw = self._build_text_prompt(
            context=context,
            force_language=force_language,
        )

        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            max_window_sec=float(max_window_sec),
            max_window_samples=int(max_window_samples),
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            audio_samples_seen=0,
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
            _committed_token_ids=[],
            _window_token_ids=[],
        )

    # ------------------------------------------------------------------
    # Streaming token helpers
    # ------------------------------------------------------------------

    def _stream_encode(self, text: str) -> List[int]:
        if not text:
            return []

        try:
            return list(
                self.processor.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
        except TypeError:
            return list(self.processor.tokenizer.encode(text))

    def _stream_decode(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""

        try:
            return self.processor.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.processor.tokenizer.decode(token_ids)

    def _streaming_prefix(self, state: ASRStreamingState) -> str:
        if state.chunk_id < state.unfixed_chunk_num:
            return ""

        current_ids = self._stream_encode(state._raw_decoded)
        rollback = max(0, int(state.unfixed_token_num))

        while True:
            end_idx = max(0, len(current_ids) - rollback)

            prefix = (
                self._stream_decode(current_ids[:end_idx])
                if end_idx
                else ""
            )

            if "\ufffd" not in prefix or end_idx == 0:
                return prefix

            rollback += 1

    @staticmethod
    def _append_streaming_audio(
        state: ASRStreamingState,
        audio: np.ndarray,
    ) -> int:

        state.audio_samples_seen += int(audio.shape[0])

        if state.audio_accum.shape[0] == 0:
            state.audio_accum = audio
        else:
            state.audio_accum = np.concatenate(
                [state.audio_accum, audio],
                axis=0,
            )

        overflow = max(
            0,
            int(state.audio_accum.shape[0])
            - int(state.max_window_samples),
        )

        if overflow:
            state.audio_accum = state.audio_accum[overflow:]

        return overflow

    @staticmethod
    def _rolling_token_alignment(
        previous_ids: List[int],
        current_ids: List[int],
        *,
        dropped_samples: int,
        previous_window_samples: int,
    ) -> tuple[int, int]:

        if not previous_ids or not current_ids:
            return 0, 0

        expected_commit = min(
            len(previous_ids),
            max(
                0,
                round(
                    len(previous_ids)
                    * dropped_samples
                    / max(1, previous_window_samples)
                ),
            ),
        )

        minimum_match = max(
            3,
            min(
                24,
                round(
                    min(len(previous_ids), len(current_ids)) * 0.1
                ),
            ),
        )

        matches = SequenceMatcher(
            None,
            previous_ids,
            current_ids,
            autojunk=False,
        ).get_matching_blocks()

        viable = [
            match for match in matches
            if match.size >= minimum_match
        ]

        if not viable:
            return expected_commit, 0

        best = max(
            viable,
            key=lambda match: (
                match.size,
                -abs(match.a - expected_commit),
                -match.b,
            ),
        )

        return best.a, best.b

    # ------------------------------------------------------------------
    # Synchronous native streaming decode
    # ------------------------------------------------------------------

    def _apply_streaming_decode(
        self,
        state: ASRStreamingState,
        *,
        dropped_samples: int,
    ) -> None:

        rolling = dropped_samples > 0
        prefix = "" if rolling else self._streaming_prefix(state)

        if rolling:
            previous_ids = state._window_token_ids

            expected_commit = min(
                len(previous_ids),
                max(
                    0,
                    round(
                        len(previous_ids)
                        * dropped_samples
                        / max(1, state.max_window_samples)
                    ),
                ),
            )

            rolling_ids = previous_ids[expected_commit:]

            stable_end = max(
                0,
                len(rolling_ids) - max(0, int(state.unfixed_token_num)),
            )

            text_prefix = self._stream_decode(
                rolling_ids[:stable_end]
            )

            if state.force_language:
                prefix = text_prefix
            else:
                latest_language = (
                    state.language.split(",")[-1].strip()
                )

                prefix = (
                    f"language {latest_language}<asr_text>{text_prefix}"
                    if latest_language
                    else text_prefix
                )

        generated = self._generate_streaming_text(
            state.prompt_raw + prefix,
            state.audio_accum,
        )

        state._raw_decoded = prefix + generated

        language, window_text = parse_asr_output(
            state._raw_decoded,
            user_language=state.force_language,
        )

        if not rolling:
            state.language = language
            state.text = window_text
            state._window_token_ids = self._stream_encode(window_text)
            return

        current_ids = self._stream_encode(window_text)
        previous_ids = state._window_token_ids

        if not current_ids:
            commit_count = min(
                len(previous_ids),
                max(
                    0,
                    round(
                        len(previous_ids)
                        * dropped_samples
                        / max(1, state.max_window_samples)
                    ),
                ),
            )

            state._committed_token_ids.extend(
                previous_ids[:commit_count]
            )
            state._window_token_ids = previous_ids[commit_count:]

        else:
            commit_count, current_start = self._rolling_token_alignment(
                previous_ids,
                current_ids,
                dropped_samples=dropped_samples,
                previous_window_samples=state.max_window_samples,
            )

            state._committed_token_ids.extend(
                previous_ids[:commit_count]
            )

            state._window_token_ids = current_ids[current_start:]

        state.text = self._stream_decode(
            state._committed_token_ids + state._window_token_ids
        ).strip()

        state.language = merge_languages(
            [*state.language.split(","), language]
        )

    # ------------------------------------------------------------------
    # Asynchronous native streaming decode
    # ------------------------------------------------------------------

    async def _apply_streaming_decode_async(
        self,
        state: ASRStreamingState,
        *,
        dropped_samples: int,
    ) -> None:
        """
        Async equivalent of _apply_streaming_decode().

        The transcript alignment and rolling-window logic are intentionally
        identical to the synchronous implementation. Only model generation
        is asynchronous.
        """

        if not self._async_engine:
            raise RuntimeError(
                "_apply_streaming_decode_async() requires LLMAsync()."
            )

        rolling = dropped_samples > 0
        prefix = "" if rolling else self._streaming_prefix(state)

        if rolling:
            previous_ids = state._window_token_ids

            expected_commit = min(
                len(previous_ids),
                max(
                    0,
                    round(
                        len(previous_ids)
                        * dropped_samples
                        / max(1, state.max_window_samples)
                    ),
                ),
            )

            rolling_ids = previous_ids[expected_commit:]

            stable_end = max(
                0,
                len(rolling_ids) - max(0, int(state.unfixed_token_num)),
            )

            text_prefix = self._stream_decode(
                rolling_ids[:stable_end]
            )

            if state.force_language:
                prefix = text_prefix
            else:
                latest_language = (
                    state.language.split(",")[-1].strip()
                )

                prefix = (
                    f"language {latest_language}<asr_text>{text_prefix}"
                    if latest_language
                    else text_prefix
                )

        generated = await self._generate_streaming_text_async(
            state.prompt_raw + prefix,
            state.audio_accum,
        )

        state._raw_decoded = prefix + generated

        language, window_text = parse_asr_output(
            state._raw_decoded,
            user_language=state.force_language,
        )

        if not rolling:
            state.language = language
            state.text = window_text
            state._window_token_ids = self._stream_encode(window_text)
            return

        current_ids = self._stream_encode(window_text)
        previous_ids = state._window_token_ids

        if not current_ids:
            commit_count = min(
                len(previous_ids),
                max(
                    0,
                    round(
                        len(previous_ids)
                        * dropped_samples
                        / max(1, state.max_window_samples)
                    ),
                ),
            )

            state._committed_token_ids.extend(
                previous_ids[:commit_count]
            )

            state._window_token_ids = previous_ids[commit_count:]

        else:
            commit_count, current_start = self._rolling_token_alignment(
                previous_ids,
                current_ids,
                dropped_samples=dropped_samples,
                previous_window_samples=state.max_window_samples,
            )

            state._committed_token_ids.extend(
                previous_ids[:commit_count]
            )

            state._window_token_ids = current_ids[current_start:]

        state.text = self._stream_decode(
            state._committed_token_ids + state._window_token_ids
        ).strip()

        state.language = merge_languages(
            [*state.language.split(","), language]
        )

    # ------------------------------------------------------------------
    # Synchronous streaming API
    # ------------------------------------------------------------------

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: ASRStreamingState,
    ) -> ASRStreamingState:

        if self._async_engine:
            raise RuntimeError(
                "streaming_transcribe() is synchronous. "
                "Use streaming_transcribe_async() with LLMAsync()."
            )

        if state is None:
            raise ValueError(
                "state must not be None. Call init_streaming_state() first."
            )

        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        x = np.asarray(pcm16k)

        if x.ndim != 1:
            x = x.reshape(-1)

        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate(
                [state.buffer, x],
                axis=0,
            )

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[:state.chunk_size_samples]

            state.buffer = state.buffer[state.chunk_size_samples:]

            dropped_samples = self._append_streaming_audio(
                state,
                chunk,
            )

            self._apply_streaming_decode(
                state,
                dropped_samples=dropped_samples,
            )

            state.chunk_id += 1

        return state

    # ------------------------------------------------------------------
    # Asynchronous streaming API
    # ------------------------------------------------------------------

    async def streaming_transcribe_async(
        self,
        pcm16k: np.ndarray,
        state: ASRStreamingState,
    ) -> ASRStreamingState:
        """
        Async streaming transcription.

        Each session should have its own ASRStreamingState and should
        serialize calls to this method for that session.

        Different sessions can call this method concurrently. Their
        requests are submitted independently to vLLM AsyncLLM, allowing
        the vLLM scheduler to batch model execution.
        """

        if not self._async_engine:
            raise RuntimeError(
                "streaming_transcribe_async() requires LLMAsync()."
            )

        if state is None:
            raise ValueError(
                "state must not be None. Call init_streaming_state() first."
            )

        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        x = np.asarray(pcm16k)

        if x.ndim != 1:
            x = x.reshape(-1)

        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate(
                [state.buffer, x],
                axis=0,
            )

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[:state.chunk_size_samples]

            state.buffer = state.buffer[state.chunk_size_samples:]

            dropped_samples = self._append_streaming_audio(
                state,
                chunk,
            )

            await self._apply_streaming_decode_async(
                state,
                dropped_samples=dropped_samples,
            )

            state.chunk_id += 1

        return state

    # ------------------------------------------------------------------
    # Synchronous finish
    # ------------------------------------------------------------------

    def finish_streaming_transcribe(
        self,
        state: ASRStreamingState,
    ) -> ASRStreamingState:

        if self._async_engine:
            raise RuntimeError(
                "finish_streaming_transcribe() is synchronous. "
                "Use finish_streaming_transcribe_async() with LLMAsync()."
            )

        if state is None:
            raise ValueError("state must not be None.")

        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        dropped_samples = self._append_streaming_audio(
            state,
            tail,
        )

        self._apply_streaming_decode(
            state,
            dropped_samples=dropped_samples,
        )

        state.chunk_id += 1

        return state

    # ------------------------------------------------------------------
    # Asynchronous finish
    # ------------------------------------------------------------------

    async def finish_streaming_transcribe_async(
        self,
        state: ASRStreamingState,
    ) -> ASRStreamingState:
        """
        Flush the remaining audio for an async streaming session.

        The caller should ensure that no other operation is currently
        modifying this session's state.
        """

        if not self._async_engine:
            raise RuntimeError(
                "finish_streaming_transcribe_async() requires LLMAsync()."
            )

        if state is None:
            raise ValueError("state must not be None.")

        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        dropped_samples = self._append_streaming_audio(
            state,
            tail,
        )

        await self._apply_streaming_decode_async(
            state,
            dropped_samples=dropped_samples,
        )

        state.chunk_id += 1

        return state

    # ------------------------------------------------------------------
    # Async engine lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Shut down the underlying async engine.

        Call this only after stopping new requests and allowing active
        requests to finish. This is not automatically invoked by defining
        this method; wire it into your deployment's shutdown lifecycle.
        """

        if not self._async_engine:
            return

        shutdown_fn = getattr(self.model, "shutdown", None)

        if shutdown_fn is not None:
            shutdown_fn()
