from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["queued", "processing", "completed", "failed"]


class ScoreOptions(BaseModel):
    tempo_bpm: float = Field(default=120, ge=30, le=240, allow_inf_nan=False)
    time_signature: Literal["4/4", "3/4", "6/8"] = "4/4"
    grid: Literal["eighth", "sixteenth"] = "sixteenth"


class Artifact(BaseModel):
    name: str
    label: str
    media_type: str
    url: str


class PipelineError(Exception):
    """A failure with a message safe to show to the person who uploaded the file."""
