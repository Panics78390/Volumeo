"""Point d'entrée FastAPI : ``uvicorn app.main:app``."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings, settings as default_settings
from .jobs.queue import JobQueue
from .jobs.store import JobStore
from .pipeline.device import system_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("volumeo")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or default_settings
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = JobStore(settings.db_path)
        queue = JobQueue(settings, store)
        app.state.settings = settings
        app.state.queue = queue
        await queue.start()
        info = system_info(settings.device)
        if info["warning"]:
            log.warning(info["warning"])
        else:
            log.info("GPU détecté : %s (%s Go)", info["gpu_name"], info["gpu_memory_gb"])
        yield
        await queue.stop()
        store.close()

    app = FastAPI(title="Volumeo", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # En production (image Docker), le frontend compilé est servi par FastAPI.
    index = settings.static_dir / "index.html"
    if index.exists():
        app.mount("/assets", StaticFiles(directory=settings.static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Route inconnue")
            candidate = (settings.static_dir / path).resolve()
            if path and candidate.is_file() and settings.static_dir in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app


app = create_app()
