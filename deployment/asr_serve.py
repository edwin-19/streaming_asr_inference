
import asyncio
import logging
import os
import time
import uuid

import ray
from ray import serve

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("asr_serve")


# ============================================================
# Resource Configuration
# ============================================================

# Total GPUs available to ASR.
# One worker is created per configured GPU.
ASR_GPU_COUNT = int(
    os.getenv("ASR_GPU_COUNT", "1")
)

# GPU resources reserved by each worker actor.
ASR_WORKER_NUM_GPUS = float(
    os.getenv("ASR_WORKER_NUM_GPUS", "1")
)

# CPU resources reserved by each worker actor.
ASR_WORKER_NUM_CPUS = float(
    os.getenv("ASR_WORKER_NUM_CPUS", "4")
)

# Router state is stored in local memory.
# Keep exactly one replica until session routing is moved
# to shared storage or another coordinated mechanism.
ASR_ROUTER_NUM_REPLICAS = 1

ASR_ROUTER_NUM_CPUS = float(
    os.getenv("ASR_ROUTER_NUM_CPUS", "1")
)


# ============================================================
# Resource Validation
# ============================================================

if ASR_GPU_COUNT < 1:
    raise ValueError("ASR_GPU_COUNT must be >= 1")

if ASR_WORKER_NUM_GPUS <= 0:
    raise ValueError("ASR_WORKER_NUM_GPUS must be > 0")

if ASR_WORKER_NUM_CPUS <= 0:
    raise ValueError("ASR_WORKER_NUM_CPUS must be > 0")

if ASR_ROUTER_NUM_CPUS <= 0:
    raise ValueError("ASR_ROUTER_NUM_CPUS must be > 0")


# ============================================================
# Model Configuration
# ============================================================

ASR_MODEL_NAME = os.getenv(
    "ASR_MODEL_NAME",
    "Qwen/Qwen3-ASR-1.7B-hf",
)

ASR_CHUNK_SIZE_SECONDS = float(
    os.getenv("ASR_CHUNK_SIZE_SECONDS", "2.0")
)

ASR_UNFIXED_CHUNK_NUM = int(
    os.getenv("ASR_UNFIXED_CHUNK_NUM", "2")
)

ASR_UNFIXED_TOKEN_NUM = int(
    os.getenv("ASR_UNFIXED_TOKEN_NUM", "5")
)

ASR_MAX_NEW_TOKENS = int(
    os.getenv("ASR_MAX_NEW_TOKENS", "256")
)

ASR_MAX_MODEL_LEN = int(
    os.getenv("ASR_MAX_MODEL_LEN", "8192")
)

ASR_GPU_MEMORY_UTILIZATION = float(
    os.getenv("ASR_GPU_MEMORY_UTILIZATION", "0.90")
)

# Concurrent inference submissions per GPU worker.
ASR_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("ASR_MAX_CONCURRENT_REQUESTS", "8")
)

# Ray actor concurrency.
ASR_ACTOR_MAX_CONCURRENCY = int(
    os.getenv("ASR_ACTOR_MAX_CONCURRENCY", "64")
)

SPECULATIVE_CONFIG = {
    "model": "mattymchen/Qwen3-ASR-1.7B-speculator.dflash2",
    "method": "dflash",
    "num_speculative_tokens": 7,
}


# ============================================================
# GPU Worker
#
# One actor = one Qwen3-ASR model.
#
# Responsibilities:
# - Model lifecycle
# - Streaming session state
# - Per-session serialization
# - GPU inference concurrency
#
# The gateway calls this actor directly after assignment.
# ============================================================

@ray.remote(
    num_gpus=ASR_WORKER_NUM_GPUS,
    num_cpus=ASR_WORKER_NUM_CPUS,
    max_concurrency=ASR_ACTOR_MAX_CONCURRENCY,
)
class ASRWorker:

    def __init__(self, worker_id: int):

        self.worker_id = worker_id

        self.sessions = {}

        # Controls simultaneous inference submissions.
        self.inference_semaphore = asyncio.Semaphore(
            ASR_MAX_CONCURRENT_REQUESTS
        )

        logger.info(
            "[Worker %s] Loading ASR model | model=%s",
            self.worker_id,
            ASR_MODEL_NAME,
        )

        model_started = time.perf_counter()

        self.asr = Qwen3ASRModel.LLMAsync(
            model=ASR_MODEL_NAME,
            trust_remote_code=True,
            max_model_len=ASR_MAX_MODEL_LEN,
            gpu_memory_utilization=ASR_GPU_MEMORY_UTILIZATION,
            max_inference_batch_size=-1,
            max_new_tokens=ASR_MAX_NEW_TOKENS,
            speculative_config=SPECULATIVE_CONFIG,
        )

        model_elapsed = time.perf_counter() - model_started

        logger.info(
            "[Worker %s] Qwen3-ASR ready | initialization=%.2fs",
            self.worker_id,
            model_elapsed,
        )

    # --------------------------------------------------------
    # Internal session validation
    # --------------------------------------------------------

    def _get_session(self, session_id: str):

        session = self.sessions.get(session_id)

        if session is None:
            raise KeyError(
                f"Unknown ASR session: {session_id}"
            )

        return session

    def _validate_session(
        self,
        session_id: str,
        session: dict,
    ):

        # A session might have been cancelled while this
        # request was waiting for its session lock.
        if self.sessions.get(session_id) is not session:
            raise RuntimeError(
                f"ASR session is no longer active: {session_id}"
            )

        if session["finalized"]:
            raise RuntimeError(
                f"Session already finalized: {session_id}"
            )

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    async def ready(self):

        return {
            "worker_id": self.worker_id,
            "status": "ready",
            "model": ASR_MODEL_NAME,
        }

    async def health(self):

        return {
            "worker_id": self.worker_id,
            "status": "ready",
            "active_sessions": len(self.sessions),
            "max_concurrent_requests": (
                ASR_MAX_CONCURRENT_REQUESTS
            ),
            "max_actor_concurrency": (
                ASR_ACTOR_MAX_CONCURRENCY
            ),
        }

    # --------------------------------------------------------
    # Session initialization
    # --------------------------------------------------------

    async def start_session(self, session_id: str):

        if session_id in self.sessions:
            raise ValueError(
                f"Session already exists: {session_id}"
            )

        state = self.asr.init_streaming_state(
            context="",
            language=None,
            unfixed_chunk_num=ASR_UNFIXED_CHUNK_NUM,
            unfixed_token_num=ASR_UNFIXED_TOKEN_NUM,
            chunk_size_sec=ASR_CHUNK_SIZE_SECONDS,
        )

        self.sessions[session_id] = {
            "state": state,
            "lock": asyncio.Lock(),
            "finalized": False,
            "chunks_processed": 0,
            "audio_seconds_submitted": 0.0,
            "started_at": time.perf_counter(),
        }

        logger.info(
            "[Worker %s] Session started | session=%s | active=%s",
            self.worker_id,
            session_id,
            len(self.sessions),
        )

        return {
            "session_id": session_id,
            "worker_id": self.worker_id,
        }

    # --------------------------------------------------------
    # Incremental inference
    #
    # One session cannot process two chunks simultaneously.
    # Different sessions can submit inference concurrently.
    # --------------------------------------------------------

    async def process_audio(
        self,
        session_id: str,
        audio,
    ):

        total_started = time.perf_counter()

        session = self._get_session(session_id)

        lock_started = time.perf_counter()

        async with session["lock"]:

            lock_wait = (
                time.perf_counter() - lock_started
            )

            self._validate_session(
                session_id,
                session,
            )

            old_state = session["state"]

            old_chunk_id = getattr(
                old_state,
                "chunk_id",
                0,
            )

            old_text = (
                getattr(old_state, "text", "")
                or ""
            )

            semaphore_started = time.perf_counter()

            async with self.inference_semaphore:

                semaphore_wait = (
                    time.perf_counter() - semaphore_started
                )

                inference_started = time.perf_counter()

                new_state = (
                    await self.asr.streaming_transcribe_async(
                        audio,
                        old_state,
                    )
                )

                inference_time = (
                    time.perf_counter() - inference_started
                )

            # Commit state only after successful inference.
            session["state"] = new_state

            session["chunks_processed"] += 1

            # Input is float32 PCM at SAMPLE_RATE.
            session["audio_seconds_submitted"] += (
                len(audio) / 16000
            )

            new_chunk_id = getattr(
                new_state,
                "chunk_id",
                old_chunk_id,
            )

            new_text = (
                getattr(new_state, "text", "")
                or ""
            )

            decoded = (
                new_chunk_id != old_chunk_id
                and bool(new_text)
                and new_text != old_text
            )

            total_time = (
                time.perf_counter() - total_started
            )

            logger.info(
                "[Worker %s] ASR timing | "
                "session=%s | "
                "audio=%.3fs | "
                "lock_wait=%.3f | "
                "semaphore_wait=%.3f | "
                "inference=%.3f | "
                "total=%.3f | "
                "decoded=%s",
                self.worker_id,
                session_id,
                len(audio) / 16000,
                lock_wait,
                semaphore_wait,
                inference_time,
                total_time,
                decoded,
            )

            return {
                "decoded": decoded,
                "transcript": {
                    "text": new_text,
                    "language": (
                        getattr(new_state, "language", "")
                        or ""
                    ),
                },
                "chunk_id": new_chunk_id,
                "worker_id": self.worker_id,
                "timing": {
                    "lock_wait": lock_wait,
                    "semaphore_wait": semaphore_wait,
                    "inference": inference_time,
                    "total": total_time,
                },
            }

    # --------------------------------------------------------
    # Final inference
    # --------------------------------------------------------

    async def finish_session(self, session_id: str):

        session = self._get_session(session_id)

        async with session["lock"]:

            self._validate_session(
                session_id,
                session,
            )

            semaphore_started = time.perf_counter()

            async with self.inference_semaphore:

                semaphore_wait = (
                    time.perf_counter() - semaphore_started
                )

                inference_started = time.perf_counter()

                final_state = (
                    await self.asr.finish_streaming_transcribe_async(
                        session["state"]
                    )
                )

                inference_time = (
                    time.perf_counter() - inference_started
                )

            session["state"] = final_state
            session["finalized"] = True

            transcript = {
                "text": (
                    getattr(final_state, "text", "")
                    or ""
                ),
                "language": (
                    getattr(final_state, "language", "")
                    or ""
                ),
            }

            session_duration = (
                time.perf_counter()
                - session["started_at"]
            )

            self.sessions.pop(session_id, None)

            logger.info(
                "[Worker %s] Session finalized | "
                "session=%s | "
                "chunks=%s | "
                "audio=%.2fs | "
                "duration=%.2fs | "
                "inference=%.3fs | "
                "semaphore_wait=%.3fs | "
                "active=%s",
                self.worker_id,
                session_id,
                session["chunks_processed"],
                session["audio_seconds_submitted"],
                session_duration,
                inference_time,
                semaphore_wait,
                len(self.sessions),
            )

            return {
                "transcript": transcript,
                "worker_id": self.worker_id,
                "timing": {
                    "inference": inference_time,
                    "semaphore_wait": semaphore_wait,
                    "session_duration": session_duration,
                },
            }

    # --------------------------------------------------------
    # Cancellation
    # --------------------------------------------------------

    async def cancel_session(self, session_id: str):

        session = self.sessions.get(session_id)

        if session is None:
            return {
                "cancelled": False,
                "reason": "not_found",
            }

        async with session["lock"]:

            # Another operation might have removed it while
            # this cancellation was waiting.
            if self.sessions.get(session_id) is not session:
                return {
                    "cancelled": False,
                    "reason": "not_found",
                }

            self.sessions.pop(session_id, None)

        logger.info(
            "[Worker %s] Session cancelled | session=%s | active=%s",
            self.worker_id,
            session_id,
            len(self.sessions),
        )

        return {
            "cancelled": True,
            "session_id": session_id,
            "worker_id": self.worker_id,
        }


# ============================================================
# Ray Serve Router
#
# Responsibilities:
# - Create GPU workers
# - Assign sessions
# - Maintain session-to-worker mapping
# - Return direct worker handles
#
# Audio inference never passes through this router.
# ============================================================

@serve.deployment(
    num_replicas=ASR_ROUTER_NUM_REPLICAS,
    ray_actor_options={
        "num_cpus": ASR_ROUTER_NUM_CPUS,
    },
    max_ongoing_requests=1000,
)
class ASRRouter:

    def __init__(self, num_workers: int = 1):

        if num_workers < 1:
            raise ValueError(
                "num_workers must be >= 1"
            )

        self.num_workers = num_workers

        self.workers = [
            ASRWorker.remote(worker_id=i)
            for i in range(num_workers)
        ]

        self.worker_sessions = [
            set() for _ in range(num_workers)
        ]

        self.session_to_worker = {}

        # Protects session assignment and release.
        self.assignment_lock = asyncio.Lock()

        logger.info(
            "Initializing ASR workers | count=%s",
            num_workers,
        )

        # Worker constructors load their models before
        # ready.remote() can execute.
        ready_results = ray.get([
            worker.ready.remote()
            for worker in self.workers
        ])

        logger.info(
            "All ASR workers ready: %s",
            ready_results,
        )

    # --------------------------------------------------------
    # Worker assignment
    # --------------------------------------------------------

    def select_worker(self):

        return min(
            range(self.num_workers),
            key=lambda i: len(self.worker_sessions[i]),
        )

    # --------------------------------------------------------
    # Start session
    # --------------------------------------------------------

    async def start_session(
        self,
        session_id: str = None,
    ):

        if session_id is None:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"

        # Reserve the worker assignment atomically.
        async with self.assignment_lock:

            if session_id in self.session_to_worker:
                raise ValueError(
                    f"Session already exists: {session_id}"
                )

            worker_id = self.select_worker()

            worker = self.workers[worker_id]

            self.session_to_worker[session_id] = worker_id

            self.worker_sessions[worker_id].add(
                session_id
            )

        try:

            result = await worker.start_session.remote(
                session_id
            )

            logger.info(
                "Session assigned | "
                "session=%s | "
                "worker=%s | "
                "worker_sessions=%s",
                session_id,
                worker_id,
                len(self.worker_sessions[worker_id]),
            )

            return {
                **result,
                "worker_handle": worker,
            }

        except Exception:

            # Roll back reservation if worker initialization
            # or actor communication fails.
            async with self.assignment_lock:

                self.session_to_worker.pop(
                    session_id,
                    None,
                )

                self.worker_sessions[worker_id].discard(
                    session_id
                )

            logger.exception(
                "Failed to start session | session=%s | worker=%s",
                session_id,
                worker_id,
            )

            raise

    # --------------------------------------------------------
    # Release session bookkeeping
    # --------------------------------------------------------

    async def release_session(self, session_id: str):

        async with self.assignment_lock:

            worker_id = self.session_to_worker.pop(
                session_id,
                None,
            )

            if worker_id is None:
                return {
                    "released": False,
                    "reason": "not_found",
                }

            self.worker_sessions[worker_id].discard(
                session_id
            )

        logger.info(
            "Session released | session=%s | worker=%s",
            session_id,
            worker_id,
        )

        return {
            "released": True,
            "session_id": session_id,
            "worker_id": worker_id,
        }

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    async def health(self):

        async with self.assignment_lock:

            active_sessions = len(
                self.session_to_worker
            )

            worker_session_counts = [
                len(sessions)
                for sessions in self.worker_sessions
            ]

        worker_results = await asyncio.gather(
            *[
                worker.health.remote()
                for worker in self.workers
            ],
            return_exceptions=True,
        )

        workers_health = []

        for i, result in enumerate(worker_results):

            if isinstance(result, Exception):

                workers_health.append({
                    "worker_id": i,
                    "status": "error",
                    "error": str(result),
                })

            else:

                workers_health.append(result)

        healthy_workers = all(
            worker.get("status") == "ready"
            for worker in workers_health
        )

        return {
            "status": (
                "ok" if healthy_workers else "degraded"
            ),
            "num_workers": self.num_workers,
            "active_sessions": active_sessions,
            "worker_session_counts": worker_session_counts,
            "workers": workers_health,
        }


# ============================================================
# Serve application builder
#
# Deployment:
#
# ASR_GPU_COUNT=2 \
# ASR_WORKER_NUM_GPUS=1 \
# ASR_WORKER_NUM_CPUS=4 \
# ASR_ROUTER_NUM_CPUS=1 \
# serve run asr_serve:build_app
#
# Or override GPU count through the Serve builder argument:
# serve run asr_serve:build_app gpu_count=2
# ============================================================

def build_app(args: dict = None):

    args = args or {}

    # Explicit Serve builder argument overrides the environment variable.
    gpu_count = int(
        args.get("gpu_count", ASR_GPU_COUNT)
    )

    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")

    # One worker is created per configured GPU.
    num_workers = gpu_count

    logger.info(
        "Building ASR application | "
        "gpu_count=%s | "
        "worker_count=%s | "
        "worker_gpus_per_actor=%s | "
        "worker_cpus=%s | "
        "router_replicas=%s | "
        "router_cpus=%s",
        gpu_count,
        num_workers,
        ASR_WORKER_NUM_GPUS,
        ASR_WORKER_NUM_CPUS,
        ASR_ROUTER_NUM_REPLICAS,
        ASR_ROUTER_NUM_CPUS,
    )

    return ASRRouter.bind(
        num_workers=num_workers
    )
