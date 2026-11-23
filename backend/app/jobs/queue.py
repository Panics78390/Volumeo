"""File d'attente asynchrone des générations.

Un unique worker asyncio dépile les tâches une par une (le GPU ne peut pas
servir deux modèles en parallèle de façon fiable) et exécute le pipeline,
bloquant et gourmand en CPU, dans un thread via ``asyncio.to_thread`` : la
boucle d'événements de FastAPI reste disponible pour l'interface.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from PIL import Image

from ..config import Settings
from ..pipeline import text2image
from ..pipeline.background import ImageError
from ..pipeline.runner import PipelineParams, _stage, reprocess, run_pipeline
from ..uploads import save_image
from .store import JobStore

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


class JobQueue:
    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task | None = None
        self._cancel: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._pending: list[str] = []  # ordre d'attente, pour afficher la position
        self._delete_after: set[str] = set()  # supprimées pendant leur exécution
        self.current: str | None = None
        # Un seul thread dédié au pipeline : les bibliothèques natives (PyMeshLab/Qt,
        # PyTorch) sont toujours appelées depuis le même thread.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")

    # -- cycle de vie --------------------------------------------------------
    async def start(self) -> None:
        self._queue = asyncio.Queue()
        # Une tâche « running » au démarrage a été interrompue par un arrêt du serveur.
        for job_id in self.store.ids_with_status("running"):
            self.store.update(job_id, status="failed", error="Interrompue par un redémarrage du serveur",
                              message="Interrompue", finished_at=time.time())
        for job_id in self.store.ids_with_status("queued"):
            self._enqueue(job_id)
        self._worker = asyncio.create_task(self._run(), name="job-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    # -- API publique ----------------------------------------------------------
    def submit(self, name: str, params: PipelineParams, *, upload_id: str | None = None,
               parent_id: str | None = None, kind: str = "generate") -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = self.store.create(job_id, name, {**params.to_dict()}, kind=kind, upload_id=upload_id,
                                parent_id=parent_id)
        self._enqueue(job_id)
        return job

    def _enqueue(self, job_id: str) -> None:
        self._pending.append(job_id)
        assert self._queue is not None, "JobQueue.start() n'a pas été appelé"
        self._queue.put_nowait(job_id)

    def position(self, job_id: str) -> int | None:
        """Position dans la file (0 = prochaine à passer), None si non en attente."""
        try:
            return self._pending.index(job_id)
        except ValueError:
            return None

    def cancel(self, job_id: str) -> None:
        with self._cancel_lock:
            self._cancel.add(job_id)
        job = self.store.get(job_id)
        if job and job["status"] == "queued":
            self.store.update(job_id, status="cancelled", message="Annulée", finished_at=time.time())

    def delete(self, job_id: str) -> None:
        self.cancel(job_id)
        if self.current == job_id:
            # La tâche tourne : le worker la supprimera dès qu'elle s'arrête.
            self._delete_after.add(job_id)
            return
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        self.store.delete(job_id)

    # -- worker ----------------------------------------------------------------
    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            if job_id in self._pending:
                self._pending.remove(job_id)
            try:
                await self._process(job_id)
            except Exception:  # pragma: no cover - filet de sécurité
                log.exception("Erreur inattendue du worker")
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job["status"] != "queued":
            return
        with self._cancel_lock:
            if job_id in self._cancel:
                return
        self.current = job_id
        self.store.update(job_id, status="running", started_at=time.time(), progress=0.0, message="Démarrage")
        last = [0.0]

        def progress(frac: float, message: str) -> None:
            with self._cancel_lock:
                if job_id in self._cancel:
                    raise JobCancelled()
            now = time.monotonic()
            if now - last[0] > 0.2 or frac >= 1.0:  # limite les écritures SQLite
                last[0] = now
                self.store.update(job_id, progress=round(float(frac), 4), message=message)

        try:
            result = await asyncio.get_running_loop().run_in_executor(self._executor, self._execute, job, progress)
            self.store.update(job_id, status="done", progress=1.0, message="Terminé", result=result,
                              finished_at=time.time())
            log.info("Génération %s terminée en %.1f s", job_id, result.get("timings", {}).get("total", 0))
        except JobCancelled:
            self.store.update(job_id, status="cancelled", message="Annulée", finished_at=time.time())
        except Exception as exc:  # noqa: BLE001 - toute erreur est remontée à l'utilisateur
            log.error("Génération %s en échec : %s\n%s", job_id, exc, traceback.format_exc())
            self.store.update(job_id, status="failed", error=str(exc) or exc.__class__.__name__,
                              message="Échec", finished_at=time.time())
        finally:
            self.current = None
            with self._cancel_lock:
                self._cancel.discard(job_id)
            if job_id in self._delete_after:
                self._delete_after.discard(job_id)
                shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
                self.store.delete(job_id)
            self._prune_history()

    def _execute(self, job: dict, progress) -> dict:
        params = PipelineParams(**job["params"])
        out_dir = self.job_dir(job["id"])
        if job["kind"] == "reprocess":
            return reprocess(self.job_dir(job["parent_id"]), out_dir, params, progress)
        if params.prompt:
            # Texte vers 3D : le prompt devient d'abord une image, enregistrée comme un import.
            init = None
            if params.source_upload_id:
                # Image + texte : le prompt transforme l'image importée (détourée).
                init = Image.open(self.settings.uploads_dir / params.source_upload_id / "cutout.png")
                init.load()
            image = text2image.generate_image(params.prompt, params.seed, _stage(progress, 0.0, 0.3),
                                              init_image=init, strength=params.strength)
            progress(0.3, "Suppression de l'arrière-plan")
            try:
                save_image(self.settings, image, job["name"], remove_bg=True, upload_id=job["upload_id"],
                           extra={"prompt": params.prompt})
            except ImageError as exc:
                raise ImageError("L'image créée ne montre pas de sujet net : reformulez la description ou relancez.") from exc
            progress = _stage(progress, 0.32, 1.0)
        cutout_path = self.settings.uploads_dir / job["upload_id"] / "cutout.png"
        img = Image.open(cutout_path)
        img.load()
        return run_pipeline(img, out_dir, params, progress)

    def _prune_history(self) -> None:
        limit = self.settings.history_limit
        if limit <= 0:
            return
        for old in self.store.oldest_finished_beyond(limit):
            shutil.rmtree(self.job_dir(old), ignore_errors=True)
            self.store.delete(old)
