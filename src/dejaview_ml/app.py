import asyncio
import logging
import time
from contextlib import asynccontextmanager
from importlib.resources import files

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from dejaview_ml.config import Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.errors import MlError
from dejaview_ml.media import LIMITS, MEDIA_TYPES, parse_text_query
from dejaview_ml.storage import SchemaMismatch, VectorStore

logger = logging.getLogger(__name__)


async def read_body(request: Request, limit: int, *, text_query: bool = False) -> bytes:
    invalid = "INVALID_QUERY" if text_query else "FILE_CORRUPTED"
    oversized = (
        MlError(422, invalid, "JSON body exceeds 64 KiB")
        if text_query
        else MlError(413, "FILE_TOO_LARGE", f"body limit is {limit} bytes")
    )
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
            if length < 0:
                raise ValueError
        except ValueError as exc:
            raise MlError(422, invalid, "invalid Content-Length") from exc
        if length > limit:
            raise oversized
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise oversized
        data.extend(chunk)
    if not data:
        raise MlError(422, invalid, "empty request body")
    return bytes(data)


def create_app(
    settings: Settings | None = None,
    encoders: Encoders | None = None,
    store: VectorStore | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.encoders = encoders or Encoders(settings)
        app.state.store = store or VectorStore(settings)
        app.state.inference_lock = asyncio.Lock()
        task = None
        try:
            # Retry cold Qdrant startup, but never recreate an incompatible collection.
            for attempt in range(10):
                try:
                    await app.state.store.ensure_collections()
                    break
                except SchemaMismatch:
                    logger.exception("incompatible Qdrant schema; startup aborted")
                    raise
                except Exception:
                    if attempt == 9:
                        raise
                    await asyncio.sleep(1)

            async def load_models():
                for modality in ("image", "video", "text", "audio"):
                    state = app.state.encoders.states[modality]
                    if state == "loading":
                        try:
                            await asyncio.to_thread(app.state.encoders.load, modality)
                        except Exception:
                            app.state.encoders.states[modality] = "failed"
                            logger.exception("model loading failed modality=%s", modality)

            task = asyncio.create_task(load_models())
            yield
        finally:
            if task is not None:
                # Do not leave a model-loading thread using resources during shutdown.
                await task
            await app.state.store.close()

    app = FastAPI(title="DejaView ML", version="0.1.0", lifespan=lifespan)
    app.openapi = lambda: yaml.safe_load(files("dejaview_ml").joinpath("openapi.yaml").read_text())

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        started = time.monotonic()
        response = await call_next(request)
        logger.info(
            "request_id=%r method=%s path=%s status=%s duration_ms=%.1f",
            request.headers.get("x-request-id"),
            request.method,
            request.url.path,
            response.status_code,
            (time.monotonic() - started) * 1000,
        )
        return response

    @app.exception_handler(MlError)
    async def ml_error(request: Request, exc: MlError):
        logger.warning(
            "request_id=%r code=%s message=%s",
            request.headers.get("x-request-id"),
            exc.code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status, content={"code": exc.code, "message": exc.message}
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        logger.error(
            "request_id=%r internal error",
            request.headers.get("x-request-id"),
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "unexpected error",
            },
        )

    @app.get("/health")
    async def health():
        qdrant_ok = await app.state.store.healthy()
        states = dict(app.state.encoders.states)
        ready = qdrant_ok and all(state in {"ready", "disabled"} for state in states.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "unavailable",
                "models": states,
                "qdrant": "ok" if qdrant_ok else "unavailable",
            },
        )

    async def search(request: Request, modality: str):
        state = app.state.encoders.states[modality]
        if state != "ready":
            raise MlError(503, "NOT_READY", f"model {modality} is {state}")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in MEDIA_TYPES[modality]:
            raise MlError(415, "UNSUPPORTED_MEDIA_TYPE", f"{content_type} is not supported")
        if app.state.inference_lock.locked():
            raise MlError(503, "NOT_READY", "inference capacity is occupied; retry later")
        async with app.state.inference_lock:
            data = await read_body(request, LIMITS[modality], text_query=modality == "text")
            if modality == "text":
                query = parse_text_query(data)
                vectors = await asyncio.to_thread(app.state.encoders.encode_text, query)
            else:
                vectors = await asyncio.to_thread(
                    app.state.encoders.encode, modality, data, content_type
                )
            return {"movie_ids": await app.state.store.search(modality, vectors)}

    @app.post("/v1/search/image")
    async def image(request: Request):
        return await search(request, "image")

    @app.post("/v1/search/video")
    async def video(request: Request):
        return await search(request, "video")

    @app.post("/v1/search/text")
    async def text(request: Request):
        return await search(request, "text")

    @app.post("/v1/search/audio")
    async def audio(request: Request):
        return await search(request, "audio")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
