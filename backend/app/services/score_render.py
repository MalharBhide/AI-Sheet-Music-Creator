import logging
import os
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader

from app.config import Settings
from app.models import PipelineError

logger = logging.getLogger(__name__)


def render_score(xml_path: Path, directory: Path, settings: Settings) -> list[dict]:
    executable = settings.renderer()
    if executable is None:
        raise PipelineError("Sheet rendering is unavailable. Install MuseScore and configure MUSESCORE_BIN.")
    runtime_dir = directory / "qt-runtime"
    runtime_dir.mkdir(mode=0o700, exist_ok=True)
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "XDG_RUNTIME_DIR": str(runtime_dir)}
    for extension in ("pdf", "svg"):
        destination = directory / f"score.{extension}"
        try:
            result = subprocess.run([executable, "-o", str(destination), str(xml_path)],
                                    env=environment, capture_output=True,
                                    timeout=settings.render_timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PipelineError("Sheet rendering timed out. Try a shorter or simpler recording.") from exc
        if result.returncode != 0:
            logger.error("MuseScore failed: %s", result.stderr.decode(errors="replace")[-4000:])
            raise PipelineError("MuseScore could not render this score. Check the server log or try a simpler clip.")

    pdf = directory / "score.pdf"
    # Different MuseScore releases produce score.svg or score-1.svg, score-2.svg, etc.
    pages = sorted((p for p in directory.glob("score*.svg")
                    if re.fullmatch(r"score(?:-\d+)?\.svg", p.name)),
                   key=lambda p: int(re.search(r"(\d+)\.svg$", p.name).group(1))
                   if re.search(r"(\d+)\.svg$", p.name) else 0)
    if not pdf.is_file() or pdf.stat().st_size < 5 or not pages or any(p.stat().st_size == 0 for p in pages):
        raise PipelineError("MuseScore finished without producing all expected score files.")
    with pdf.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise PipelineError("MuseScore produced an invalid PDF.")
    try:
        page_count = len(PdfReader(pdf).pages)
        for page in pages:
            if ElementTree.parse(page).getroot().tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError("Invalid SVG root")
    except Exception as exc:
        raise PipelineError("MuseScore produced unreadable score files.") from exc
    if len(pages) != page_count:
        raise PipelineError("MuseScore did not export every SVG page. Use the MuseScore 3 Docker setup and try again.")

    artifacts = [
        {"name": "score.pdf", "label": "PDF", "media_type": "application/pdf"},
        {"name": "score.musicxml", "label": "MusicXML", "media_type": "application/vnd.recordare.musicxml+xml"},
        {"name": "transcription.mid", "label": "MIDI", "media_type": "audio/midi"},
    ]
    for index, page in enumerate(pages, 1):
        target = directory / f"page-{index}.svg"
        page.rename(target)
        artifacts.append({"name": target.name, "label": f"SVG · page {index}", "media_type": "image/svg+xml"})
    with zipfile.ZipFile(directory / "score-svgs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            if artifact["media_type"] == "image/svg+xml":
                archive.write(directory / artifact["name"], artifact["name"])
    artifacts.append({"name": "score-svgs.zip", "label": "All SVG pages", "media_type": "application/zip"})
    return artifacts
