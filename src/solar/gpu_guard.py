"""Conservative compute admission and periodic resource checks for Solar workers."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

_CACHE = (0., [])


def classify_processes(rows, own_pid):
    """Exclude only this worker and its descendants, never peer Solar jobs."""
    own = {own_pid}
    while True:
        children = {r['pid'] for r in rows if r['ppid'] in own}
        if children <= own:
            break
        own |= children
    return [r for r in rows if r['pid'] not in own and (
        r.get('gpu_fd') or (
            any(t in (r['cwd'] + ' ' + r['command']).lower() for t in ('umud', 'k12', 'rsna'))
            and r['comm'].lower().startswith(('python', 'torchrun', 'vllm', 'sglang', 'llama'))
            and not any(t in r['command'].lower() for t in ('--device cpu', '--device=cpu', '--cpu-only'))
            and any(t in r['command'].lower() for t in ('cuda', 'train', 'vllm', 'sglang', 'serve'))
        )
    )], own


def resource_conflicts(force=False):
    """Inspect NVML, ps, device descriptors and memory at most five minutes apart."""
    global _CACHE
    if not force and time.monotonic() - _CACHE[0] < 300:
        return _CACHE[1]
    rows, conflicts = [], []
    try:
        output = subprocess.run(['ps', '-eo', 'pid=,ppid=,comm=,args='],
                                capture_output=True, text=True, check=True, timeout=20).stdout
        for line in output.splitlines():
            parts = line.strip().split(maxsplit=3)
            if len(parts) != 4:
                continue
            pid, ppid, comm, command = parts
            proc = Path('/proc') / pid
            try:
                cwd = str((proc / 'cwd').resolve())
                gpu_fd = any('/dev/nvidia' in str(fd.resolve()) for fd in (proc / 'fd').iterdir())
            except OSError:
                cwd, gpu_fd = '', False
            rows.append(dict(pid=int(pid), ppid=int(ppid), comm=comm, command=command, cwd=cwd, gpu_fd=gpu_fd))
        conflicts, own = classify_processes(rows, os.getpid())
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                               capture_output=True, text=True, check=True, timeout=20)
        for row in csv.DictReader(query.stdout.splitlines()):
            pid = int(row['pid'].strip())
            if pid not in own and not any(r.get('pid') == pid for r in conflicts):
                conflicts.append(dict(pid=pid, source='nvidia-smi'))
        memory = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
        if memory < 30 * 1024 * 1024:
            conflicts.append(dict(reason='MemAvailable below 30 GiB', mem_available_kib=memory))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        conflicts.append(dict(reason='Resource discovery failed', error=str(error)))
    root = Path(os.environ.get('SOLAR_ARTIFACTS', 'artifacts'))
    path = root / 'resource_checks' / f'{os.getpid()}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                                    inspected_processes=len(rows), conflicts=[{k: v for k, v in r.items() if k != 'command'} for r in conflicts]), indent=2) + '\n')
    _CACHE = time.monotonic(), conflicts
    return conflicts
