from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.routes.dependencies import lookup
from app.services.storage import output_path

router = APIRouter(tags=['downloads'])
FILENAMES = {'midi': 'transcription.mid', 'musicxml': 'score.musicxml', 'pdf': 'score.pdf',
             'svg': 'page-1.svg', 'svg_zip': 'score-svgs.zip'}


def serve_artifact(request: Request, job_id: UUID, name: str, preview: bool):
    job = lookup(request, job_id)
    artifact = next((a for a in job['artifacts'] if a['name'] == name), None)
    if artifact is None:
        if job['status'] not in ('completed', 'failed'):
            raise HTTPException(409, 'This file will be available after its processing stage completes.')
        raise HTTPException(404, 'This file is not available.')
    try:
        path = output_path(request.app.state.settings, job_id, name)
    except ValueError as exc:
        raise HTTPException(404, 'This file is not available.') from exc
    if not path.is_file():
        raise HTTPException(404, 'This file was not found or has expired.')
    return FileResponse(path, media_type=artifact['media_type'], filename=name,
                        content_disposition_type='inline' if preview else 'attachment',
                        headers={'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store',
                                 'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; sandbox"})


@router.get('/api/downloads/{job_id}/{kind}')
def download_file(job_id: UUID, kind: str, request: Request, preview: bool = False):
    if kind not in FILENAMES:
        raise HTTPException(404, 'Unknown download type.')
    return serve_artifact(request, job_id, FILENAMES[kind], preview)


@router.get('/api/jobs/{job_id}/files/{name}')
def download_page(job_id: UUID, name: str, request: Request, preview: bool = False):
    return serve_artifact(request, job_id, name, preview)
