"""Continue the verified acquisition into a frozen candidate study; never deploy."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

source = Path('/training/musicnet-piano-subset-v1')
proc = Path('/proc/48/cmdline')
expected = [b'python', b'/src/training/acquire_musicnet_subset.py', b'/training/musicnet/musicnet_metadata.csv', str(source).encode()]
def identity():
    try:
        return proc.read_bytes().rstrip(b'\0').split(b'\0')
    except FileNotFoundError:
        return None

if identity() != expected:
    raise ValueError('Original acquisition process is not live with its expected identity')
plan = {
    'version': 'musicnet-v33-local-sequential-driver-v1',
    'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'original_acquisition_pid': 48,
    'original_acquisition_command': [s.decode() for s in expected],
    'no_restart_or_deployment_or_user_audio': True,
    'phases': ['verified-source', 'training-validation-observations', 'frozen-fit', 'consumed-regression-if-selected', 'first-pass-if-regression-passes'],
}
witness = Path('/training/musicnet-v33-local-driver.json')
if witness.exists():
    raise ValueError('Preserve original queued candidate study')
witness.write_text(json.dumps(plan, indent=2)+'\n')
print(json.dumps({'phase':'waiting-for-original-acquisition','pid':48}),flush=True)
while (current := identity()) is not None:
    if current != expected:
        raise ValueError('Original acquisition PID changed identity; no replacement download')
    time.sleep(10)

# The observer checks full publisher checksum, source contract and work splits.
from prepare_musicnet_v32 import sources
rows = sources(source)
print(json.dumps({'phase':'verified-source','recordings':len(rows)}),flush=True)
environment = dict(os.environ)
environment.update(PYTHONPATH='/src/backend:/src/training',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
run = Path('/training/musicnet-anchor-v33-v1')
def child(script, *args):
    command = [sys.executable, '/src/training/'+script, *map(str,args)]
    print(json.dumps({'phase':script,'command':command}),flush=True)
    subprocess.run(command, env=environment, cwd='/src/backend',check=True)
child('prepare_musicnet_v32.py',source,'/training/musicnet-v32-observations-v1')
child('train_musicnet_anchor_v33.py','/training/bass-v32-baseline-v1','/training/musicnet-v32-observations-v1','/training/bass-v32-regression-v1','/training/teacher-embedding-v32-v1/embedding-conservative/epoch-060.pt',run)
selection=json.loads((run/'batch-selection.json').read_text())
if selection['selected']:
    child('evaluate_musicnet_anchor_v33.py',run)
    regression=json.loads((run/'consumed-regression.json').read_text())
    if regression['passes']:
        child('fresh_musicnet_anchor_v33.py',run,'/training/musicnet-anchor-v33-first-pass-v1')
print(json.dumps({'phase':'candidate-study-ended','selected':selection['selected'],'new_weights_deployed':False}),flush=True)
