
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import ray
from ray import serve

from fireredvad.stream_vad import (
    FireRedStreamVad,
    FireRedStreamVadConfig,
)
from fireredvad.core.stream_vad_postprocessor import (
    StreamVadPostprocessor,
)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = os.getenv(
    "FIRERED_VAD_MODEL_DIR",
    "/mnt/data/workspace/freelance/voice/diar/"
    "pretrained_models/FireRedVAD/Stream-VAD",
)

USE_GPU = os.getenv("VAD_USE_GPU", "1") == "1"

VAD_MAX_CONCURRENT_INFERENCE = int(
    os.getenv("VAD_MAX_CONCURRENT_INFERENCE", "1")
)

VAD_CONFIG = {
    "use_gpu": USE_GPU,
    "smooth_window_size": 5,
    "speech_threshold": 0.3,
    "pad_start_frame": 5,
    "min_speech_frame": 8,
    "max_speech_frame": 2000,
    "min_silence_frame": 40,
    "chunk_max_frame": 30000,
}


# ============================================================
# Session state
# ============================================================

@dataclass
class SessionState:
    model_caches: Any
    postprocessor: StreamVadPostprocessor
    lock: asyncio.Lock


# ============================================================
# Ray Serve deployment
# ============================================================

@serve.deployment(
    name="FireRedVAD",
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 0.25 if USE_GPU else 0,
    },
    max_ongoing_requests=64,
    max_queued_requests=256,
    graceful_shutdown_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "target_ongoing_requests": 16,
        "upscale_delay_s": 15,
        "downscale_delay_s": 60,
    },
)
class FireRedVADDeployment:

    def __init__(
        self,
        model_dir: str,
        config_kwargs: dict,
    ):
        torch.set_num_threads(1)

        self.config = FireRedStreamVadConfig(**config_kwargs)

        print(f"Loading FireRedVAD from: {model_dir}")
        print(f"VAD GPU enabled: {self.config.use_gpu}")

        template = FireRedStreamVad.from_pretrained(
            model_dir,
            self.config,
        )

        self.audio_feat = template.audio_feat
        self.vad_model = template.vad_model

        del template

        # Conservative default: only one forward pass at a time.
        # Increase only after verifying shared-model thread safety.
        self.inference_semaphore = asyncio.Semaphore(
            VAD_MAX_CONCURRENT_INFERENCE
        )

        self.sessions: dict[str, SessionState] = {}

        print("FireRedVAD deployment ready.")

    # --------------------------------------------------------
    # Session lifecycle
    # --------------------------------------------------------

    async def create_session(self, session_id: str) -> dict:

        if session_id in self.sessions:
            raise ValueError(
                f"VAD session already exists: {session_id}"
            )

        config = self.config

        self.sessions[session_id] = SessionState(
            model_caches=None,
            postprocessor=StreamVadPostprocessor(
                config.smooth_window_size,
                config.speech_threshold,
                config.pad_start_frame,
                config.min_speech_frame,
                config.max_speech_frame,
                config.min_silence_frame,
            ),
            lock=asyncio.Lock(),
        )

        return {
            "session_id": session_id,
            "status": "created",
        }

    # --------------------------------------------------------
    # Streaming detection
    # --------------------------------------------------------

    async def detect(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
    ):

        session = self.sessions.get(session_id)

        if session is None:
            raise KeyError(
                f"Unknown VAD session: {session_id}"
            )

        audio_chunk = np.asarray(
            audio_chunk,
            dtype=np.float32,
        )

        if audio_chunk.ndim != 1:
            raise ValueError("Expected mono audio")

        if not audio_chunk.flags.c_contiguous:
            audio_chunk = np.ascontiguousarray(audio_chunk)

        # Maintain ordering within this session.
        async with session.lock:

            loop = asyncio.get_running_loop()

            semaphore_wait_start = loop.time()

            async with self.inference_semaphore:

                semaphore_wait_sec = (
                    loop.time() - semaphore_wait_start
                )

                inference_start = loop.time()

                result = await asyncio.to_thread(
                    self._detect_sync,
                    session,
                    audio_chunk,
                )

                inference_sec = (
                    loop.time() - inference_start
                )

            return {
                "frames": result,
                "semaphore_wait_sec": semaphore_wait_sec,
                "inference_sec": inference_sec,
                "total_sec": semaphore_wait_sec + inference_sec,
            }

    def _detect_sync(
        self,
        session: SessionState,
        audio_chunk: np.ndarray,
    ):

        with torch.inference_mode():

            feats, _ = self.audio_feat.extract(audio_chunk)

            if self.config.use_gpu:
                feats = feats.cuda()

            probs, session.model_caches = self.vad_model.forward(
                feats.unsqueeze(0),
                caches=session.model_caches,
            )

            raw_probs = probs.cpu().squeeze().tolist()

            if isinstance(raw_probs, float):
                raw_probs = [raw_probs]

            return [
                session.postprocessor.process_one_frame(prob)
                for prob in raw_probs
            ]

    # --------------------------------------------------------
    # Session reset
    # --------------------------------------------------------

    async def reset_session(self, session_id: str) -> dict:

        session = self.sessions.get(session_id)

        if session is None:
            raise KeyError(
                f"Unknown VAD session: {session_id}"
            )

        async with session.lock:

            session.model_caches = None
            session.postprocessor.reset()

        return {
            "session_id": session_id,
            "status": "reset",
        }

    # --------------------------------------------------------
    # Session removal
    # --------------------------------------------------------

    async def remove_session(self, session_id: str) -> dict:

        session = self.sessions.get(session_id)

        if session is None:
            return {
                "session_id": session_id,
                "status": "not_found",
            }

        async with session.lock:
            self.sessions.pop(session_id, None)

        return {
            "session_id": session_id,
            "status": "removed",
        }

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    async def health(self) -> dict:

        return {
            "status": "healthy",
            "model": "FireRedVAD",
            "sessions": len(self.sessions),
            "gpu": self.config.use_gpu,
            "max_concurrent_inference": VAD_MAX_CONCURRENT_INFERENCE,
        }


app = FireRedVADDeployment.bind(
    MODEL_DIR,
    VAD_CONFIG,
)


if __name__ == "__main__":

    ray.init(
        address="auto",
        ignore_reinit_error=True,
    )

    serve.run(
        app,
        name="firered_vad",
        route_prefix=None,
        blocking=True,
    )
