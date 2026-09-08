"""Submit a recording to a running backend and verify every returned download."""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio_file', type=Path)
    parser.add_argument('--api', default='http://localhost:8000')
    args = parser.parse_args()
    path = args.audio_file.expanduser().resolve()
    if not path.is_file():
        print(f'Audio file not found: {path}', file=sys.stderr)
        return 1
    boundary = 'score-' + uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="recording{path.suffix}"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode()
    body += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    origin = args.api.rstrip('/')
    try:
        with urlopen(Request(origin + '/api/upload', data=body,
                             headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}), timeout=120) as response:
            job = json.load(response)
        status_url = origin + '/api/jobs/' + job['job_id']
        deadline = time.monotonic() + 960
        while job['status'] not in ('done', 'failed') and time.monotonic() < deadline:
            time.sleep(2)
            with urlopen(status_url, timeout=30) as response:
                job = json.load(response)
        print('Job:', job['job_id'], 'Status:', job['status'])
        if job['status'] != 'done':
            print(job.get('error') or 'Timed out waiting for the job.', file=sys.stderr)
            return 1
        for artifact in job['artifacts']:
            with urlopen(origin + artifact['url'], timeout=30) as response:
                assert response.read(1), 'Empty artifact: ' + artifact['name']
            print(artifact['label'] + ':', origin + artifact['url'])
        return 0
    except HTTPError as exc:
        print(exc.read().decode(errors='replace'), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
