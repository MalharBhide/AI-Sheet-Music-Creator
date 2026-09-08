import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.middleware import UploadLimitMiddleware
from app.routes import downloads, jobs, uploads
from app.routes.dependencies import dependencies
from app.services.job_store import JobStore
from app.workers.job_runner import JobRunner


def create_app(settings: Settings | None = None, *, start_worker: bool = True) -> FastAPI:
    settings = settings or Settings()
    store = JobStore(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        store.initialize()
        app.state.store = store
        app.state.settings = settings
        runner = JobRunner(settings, store)
        if start_worker:
            runner.start()
        try:
            yield
        finally:
            if start_worker:
                runner.stop()

    app = FastAPI(title='AI Sheet Music Creator API', version='0.2.0', lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, limit=settings.max_upload_mb * 1024 * 1024 + 65536)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'],
                       expose_headers=['Location', 'Content-Disposition', 'Retry-After'])
    app.include_router(uploads.router)
    app.include_router(jobs.router)
    app.include_router(downloads.router)

    @app.get('/health', include_in_schema=False)
    @app.get('/api/health')
    def health():
        checks = dependencies(settings)
        return {'status': 'ok', 'ready': all(checks.values()), 'dependencies': checks,
                'limits': {'max_upload_mb': settings.max_upload_mb,
                           'max_audio_seconds': settings.max_audio_seconds,
                           'retention_hours': settings.retention_hours}}

    return app


logging.basicConfig(level=logging.INFO)
app = create_app()
