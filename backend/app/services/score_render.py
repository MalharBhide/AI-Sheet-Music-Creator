from pathlib import Path
import shutil
import subprocess

from app.config import get_settings


class ScoreRenderError(RuntimeError):
    pass


def render_musicxml_to_pdf_and_svg(musicxml_path: Path, pdf_path: Path, svg_path: Path) -> tuple[Path, Path]:
    settings = get_settings()
    musescore = _resolve_musescore(settings.musescore_bin)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    _run_musescore(musescore, musicxml_path, pdf_path)
    _run_musescore(musescore, musicxml_path, svg_path)

    actual_svg = _resolve_svg_output(svg_path)
    return pdf_path, actual_svg


def _resolve_musescore(configured_binary: str) -> str:
    candidates = [
        configured_binary,
        "mscore",
        "musescore",
        "musescore3",
        "musescore4",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise ScoreRenderError(
        "MuseScore CLI was not found. Install MuseScore or set MUSESCORE_BIN."
    )


def _run_musescore(musescore: str, input_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        [musescore, "-o", str(output_path), str(input_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ScoreRenderError(f"MuseScore could not render {output_path.name}. {detail}")


def _resolve_svg_output(expected_svg: Path) -> Path:
    if expected_svg.exists():
        return expected_svg

    generated_svgs = sorted(expected_svg.parent.glob(f"{expected_svg.stem}*.svg"))
    if generated_svgs:
        return generated_svgs[0]

    raise ScoreRenderError("MuseScore did not create an SVG preview.")

