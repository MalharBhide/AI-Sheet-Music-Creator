from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import downloads, jobs, uploads


settings = get_settings()

app = FastAPI(
    title="AI Sheet Music Creator API",
    version="0.1.0",
    description="Upload audio and generate piano sheet music as MIDI, MusicXML, PDF, and SVG.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(uploads.router)
app.include_router(jobs.router)
app.include_router(downloads.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

