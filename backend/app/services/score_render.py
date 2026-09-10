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
        raise PipelineError("Sheet rendering is unavailable. Install MuseScore and configure MUSESCORE_BIN. MIDI and MusicXML are available to download.")
    runtime_dir = directory / "qt-runtime"
    runtime_dir.mkdir(mode=0o700, exist_ok=True)
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "XDG_RUNTIME_DIR": str(runtime_dir)}

    def export(extension: str) -> None:
        destination = directory / f"score.{extension}"
        try:
            result = subprocess.run([executable, "-o", str(destination), str(xml_path)],
                                    env=environment, capture_output=True,
                                    timeout=settings.render_timeout_seconds or None, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PipelineError("Sheet rendering exceeded RENDER_TIMEOUT_SECONDS. Increase it or set it to 0. MIDI and MusicXML are available to download.") from exc
        except OSError as exc:
            raise PipelineError("MuseScore could not start. Check MUSESCORE_BIN. MIDI and MusicXML are available to download.") from exc
        if result.returncode != 0:
            logger.error("MuseScore failed: %s", result.stderr.decode(errors="replace")[-4000:])
            raise PipelineError("MuseScore could not render this score. MIDI and MusicXML are available to download; check the server log for rendering details.")

    export("pdf")
    pdf = directory / "score.pdf"
    try:
        with pdf.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("Invalid PDF signature")
        page_count = len(PdfReader(pdf).pages)
        if not page_count:
            raise ValueError("PDF has no pages")
    except Exception as exc:
        raise PipelineError("MuseScore did not produce a readable PDF. MIDI and MusicXML are available to download.") from exc

    artifacts = [
        {"name": "score.pdf", "label": "PDF", "media_type": "application/pdf"},
        {"name": "score.musicxml", "label": "MusicXML", "media_type": "application/vnd.recordare.musicxml+xml"},
        {"name": "transcription.mid", "label": "MIDI", "media_type": "audio/midi"},
    ]
    # Preview export is optional: MuseScore 4 can return just its first SVG page.
    # Never discard the full, validated PDF because a preview export is incomplete.
    try:
        export("svg")
        pages = sorted((p for p in directory.glob("score*.svg")
                        if re.fullmatch(r"score(?:-\d+)?\.svg", p.name)),
                       key=lambda p: int(re.search(r"(\d+)\.svg$", p.name).group(1))
                       if re.search(r"(\d+)\.svg$", p.name) else 0)
        if len(pages) != page_count:
            raise ValueError("MuseScore did not export every SVG page")
        for page in pages:
            if ElementTree.parse(page).getroot().tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError("Invalid SVG root")
    except (PipelineError, ValueError, OSError, ElementTree.ParseError):
        logger.warning("SVG preview unavailable; retaining complete PDF, MIDI, and MusicXML", exc_info=True)
        return artifacts

    for index, page in enumerate(pages, 1):
        target = directory / f"page-{index}.svg"
        page.replace(target)
        artifacts.append({"name": target.name, "label": f"SVG · page {index}", "media_type": "image/svg+xml"})
    with zipfile.ZipFile(directory / "score-svgs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            if artifact["media_type"] == "image/svg+xml":
                archive.write(directory / artifact["name"], artifact["name"])
    artifacts.append({"name": "score-svgs.zip", "label": "All SVG pages", "media_type": "application/zip"})
    return artifacts
