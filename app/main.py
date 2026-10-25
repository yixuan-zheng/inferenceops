import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from app.config import Settings
from app.batcher import Batcher
from app.model import Model
from app.metrics import Metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from app.preprocessing import ImageInputError, preprocess
from app.schemas import PredictResponse

logger = logging.getLogger(__name__)


class ReferenceEngine:
    """One CPU session, one inference thread, bounded preprocessing and scheduling."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics = Metrics()
        self.model = Model(settings)
        self.preprocessing = ThreadPoolExecutor(max_workers=settings.preprocessing_workers,
                                                thread_name_prefix="preprocess")
        self.inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.batcher = Batcher(self.model, self.inference, settings.mode, settings.batch_size, settings.queue_delay_ms / 1000, settings.queue_capacity, self.metrics)
        self.active = 0
        self.tasks: set[asyncio.Task] = set()
        self.ready = True

    def admit(self):
        # Called only on the event loop, with no await between check and increment.
        if not self.ready:
            raise HTTPException(503, "Model is not ready")
        if self.active >= self.settings.max_requests:
            self.metrics.rejections.labels("admission").inc()
            raise HTTPException(503, "Request capacity exhausted", headers={"Retry-After": "1"})
        self.active += 1
        self.metrics.admitted.inc()

    async def process(self, payload: bytes, request_id: str, started: float, cancelled=None):
        cancelled = cancelled or asyncio.Event()
        deadline = started + self.settings.request_timeout_ms / 1000
        try:
            loop = asyncio.get_running_loop()
            if cancelled.is_set() or perf_counter() >= deadline:
                raise HTTPException(504, "Request expired or cancelled")
            submitted = perf_counter()

            def prepare():
                entered = perf_counter()
                tensor = preprocess(payload, self.settings)
                return tensor, (perf_counter() - entered) * 1000, (entered - submitted) * 1000

            tensor, preprocessing_ms, preprocessing_wait_ms = await loop.run_in_executor(
                self.preprocessing, prepare)

            if cancelled.is_set() or perf_counter() >= deadline:
                raise HTTPException(504, "Request expired or cancelled")
            predictions, inference_ms, inference_wait_ms, accumulation_ms, batch_size = await self.batcher.submit(tensor, deadline, cancelled)
            return {"request_id": request_id, "model_version": self.model.manifest["version"],
                    "predictions": predictions,
                    "timings": {"preprocessing_ms": preprocessing_ms,
                                "preprocessing_wait_ms": preprocessing_wait_ms,
                                "inference_wait_ms": inference_wait_ms,
                                "inference_ms": inference_ms,
                                "accumulation_ms": accumulation_ms, "batch_size": batch_size,
                                "server_total_ms": (perf_counter() - started) * 1000}}
        finally:
            self.active -= 1

    def track(self, task):
        self.tasks.add(task)

        def completed(done):
            self.tasks.discard(done)
            # Retrieve errors even if the HTTP handler was cancelled.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)

    async def close(self):
        self.ready = False
        if self.tasks:
            await asyncio.wait(list(self.tasks), timeout=self.settings.shutdown_timeout_ms / 1000)
        await self.batcher.close()
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        self.preprocessing.shutdown(wait=True)
        self.inference.shutdown(wait=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = ReferenceEngine(settings)
        try:
            yield
        finally:
            await app.state.engine.close()

    app = FastAPI(title="InferenceOps", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def response_time(request, call_next):
        started = perf_counter()
        response = await call_next(request)
        if request.url.path == "/predict":
            app.state.engine.metrics.http.labels(str(response.status_code)).inc()
            app.state.engine.metrics.latency.observe(perf_counter()-started)
        response.headers["X-Response-Time-Ms"] = f"{(perf_counter() - started) * 1000:.3f}"
        return response

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(app.state.engine.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        engine = getattr(app.state, "engine", None)
        if engine is None or not engine.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {"status": "ready"}

    @app.get("/model/info")
    async def model_info():
        engine = app.state.engine
        return {"name": engine.model.manifest["name"], "version": engine.model.manifest["version"],
                "weights": engine.model.manifest["weights"],
                "execution_provider": engine.model.session.get_providers()[0],
                "onnx_sha256": engine.model.manifest["sha256"]["resnet18.onnx"],
                "input_shape": ["batch", 3, 224, 224], "mode": settings.mode, "batch_size": settings.batch_size,
                "queue_capacity": settings.queue_capacity, "queue_delay_ms": settings.queue_delay_ms,
                "request_timeout_ms": settings.request_timeout_ms,
                "shutdown_timeout_ms": settings.shutdown_timeout_ms,
                "inference_workers": 1, "intra_op_threads": settings.intra_op_threads,
                "inter_op_threads": settings.inter_op_threads,
                "active_requests": engine.active, "queue_depth": len(engine.batcher.queue),
                "max_requests": settings.max_requests,
                "preprocessing_workers": settings.preprocessing_workers,
                "max_upload_bytes": settings.max_upload_bytes,
                "max_image_pixels": settings.max_image_pixels,
                "max_image_side": settings.max_image_side}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request):
        started = perf_counter()
        engine = app.state.engine
        engine.admit()
        transferred = False
        task = None
        cancelled = asyncio.Event()
        try:
            content_type = request.headers.get("content-type", "").split(";")[0].lower()
            if content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
                raise HTTPException(415, "Send raw JPEG/PNG bytes, not multipart or JSON")
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as exc:
                    raise HTTPException(400, "Invalid Content-Length") from exc
                if declared < 0:
                    raise HTTPException(400, "Invalid Content-Length")
                if declared > settings.max_upload_bytes:
                    raise HTTPException(413, "Image exceeds upload byte limit")
            payload = bytearray()
            async with asyncio.timeout(settings.request_timeout_ms / 1000):
                async for chunk in request.stream():
                    if len(payload) + len(chunk) > settings.max_upload_bytes:
                        raise HTTPException(413, "Image exceeds upload byte limit")
                    payload.extend(chunk)
            task = asyncio.create_task(engine.process(bytes(payload), str(uuid4()), started, cancelled))
            engine.track(task)
            transferred = True
            # Retain capacity until native work ends, even if the handler is cancelled.
            return await asyncio.wait_for(asyncio.shield(task), max(0, settings.request_timeout_ms / 1000 - (perf_counter()-started)))
        except asyncio.CancelledError:
            cancelled.set()
            engine.batcher.changed.set()
            raise
        except TimeoutError as exc:
            cancelled.set()
            engine.batcher.changed.set()
            raise HTTPException(504, "Request deadline exceeded") from exc
        except ImageInputError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except ClientDisconnect as exc:
            raise HTTPException(400, "Client disconnected during upload") from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(500, "Inference failed") from exc
        finally:
            if not transferred:
                engine.active -= 1

    return app


app = create_app()
