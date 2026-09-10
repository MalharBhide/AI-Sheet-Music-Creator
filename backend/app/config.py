import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / '.env', REPO_ROOT / 'backend' / '.env'),
        env_file_encoding='utf-8', extra='ignore',
    )
    storage_root: Path = REPO_ROOT / 'storage'
    cors_origins: Annotated[list[str], NoDecode] = ['http://localhost:5173', 'http://127.0.0.1:5173']
    # Zero disables optional operator limits. Audio is streamed and transcribed in chunks.
    max_upload_mb: int = Field(default=0, ge=0)
    max_audio_seconds: int = Field(default=0, ge=0)
    max_queued_jobs: int = Field(default=10, ge=1, le=100)
    job_timeout_seconds: int = Field(default=0, ge=0)
    render_timeout_seconds: int = Field(default=0, ge=0)
    retention_hours: int = Field(default=24, ge=1, le=720)
    default_tempo_bpm: int = Field(default=120, ge=30, le=240)
    ffmpeg_bin: str = 'ffmpeg'
    musescore_bin: str | None = None

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str):
            if value.strip().startswith('['):
                return json.loads(value)
            return [item.strip() for item in value.split(',') if item.strip()]
        return value

    @field_validator('storage_root', mode='after')
    @classmethod
    def resolve_storage(cls, value: Path) -> Path:
        return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()

    @property
    def data_dir(self) -> Path:
        return self.storage_root

    @property
    def database_path(self) -> Path:
        return self.storage_root / 'jobs.sqlite3'

    @property
    def jobs_dir(self) -> Path:
        return self.storage_root / 'jobs'

    def renderer(self) -> str | None:
        if self.musescore_bin:
            return shutil.which(self.musescore_bin)
        for candidate in ('musescore3', 'mscore', 'musescore', 'musescore4', 'MuseScore4',
                          '/Applications/MuseScore 3.app/Contents/MacOS/mscore',
                          '/Applications/MuseScore 4.app/Contents/MacOS/mscore'):
            if executable := shutil.which(candidate):
                return executable
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
