"""Run the frozen W6 stages in isolated Conda interpreters."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGES = [('dense', 'DENSE'), ('support', 'SUPPORT'), ('score', 'SCORE'),
          ('gctrl_plain_dense', 'DENSE'), ('gctrl_dense', 'DENSE'),
          ('gctrl_score', 'SCORE'), ('refiner', 'REFINER'), ('postcal_w2', 'SCORE')]


def stage_environment(interpreter, kind):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('JPY_', 'JUPYTER_', 'IPYTHON')) or key in (
            'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE',
            'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER'):
            env.pop(key)
    # CPU evaluation imports can leave CUDA hidden in their parent process.
    if kind in ('DENSE', 'SUPPORT', 'REFINER') and env.get('CUDA_VISIBLE_DEVICES') == '':
        env.pop('CUDA_VISIBLE_DEVICES')
    env.update(PYTHONNOUSERSITE='1', MPLBACKEND='Agg', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env['PATH'] = str(Path(interpreter).resolve().parent) + os.pathsep + env.get('PATH', '')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'models', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--final', choices=['w6'], default='w6')
    parser.add_argument('--dense-long2x-tta', action='store_true')
    parser.add_argument('--postcal-w2', action='store_true', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.cache.mkdir(parents=True, exist_ok=True)
    for stage, kind in STAGES:
        interpreter = os.environ.get('SOLAR_' + kind + '_PYTHON', sys.executable)
        env = stage_environment(interpreter, kind)
        audit = args.cache / (stage + '_ACCESS.json')
        env['RF_COLD_AUDIT_OUTPUT'] = str(audit.resolve())
        command = [interpreter, '-u']
        if os.environ.get('SOLAR_AUDIT_ENTRY'):
            command.append(os.environ['SOLAR_AUDIT_ENTRY'])
        command += [str(root/'scripts/w6_infer.py')]
        for name in ('images', 'models', 'cache', 'output'):
            command += ['--' + name, str(getattr(args, name).resolve())]
        command += ['--stage', stage, '--final', 'w6', '--dense-long2x-tta',
                    '--postcal-w2', '--device', args.device]
        started = time.monotonic()
        print('STAGE_START', stage, flush=True)
        with (args.cache/(stage+'.log')).open('w') as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        if audit.exists():
            assert json.loads(audit.read_text())['denied'] == 0
        receipt = dict(stage=stage, status='PASS', elapsed_seconds=time.monotonic()-started)
        (args.cache/(stage+'_DEMO_DONE.json')).write_text(json.dumps(receipt)+'\n')
        print('STAGE_DONE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
