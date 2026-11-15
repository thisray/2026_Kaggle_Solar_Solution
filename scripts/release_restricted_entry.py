"""Record rejected cold reads without opening source while collecting stacks."""
import json
import os
from pathlib import Path
import runpy
import sys

entry = Path(sys.argv[1]).resolve()
art = Path(os.environ['RF_COLD_ARTIFACT_ROOT']).resolve()
run = Path(os.environ['RF_COLD_RUN_ROOT']).resolve()
images = Path(os.environ['RF_COLD_IMAGES_ROOT']).resolve()
vendor = art / 'w52/vendor/rf-detr'
counts = dict(test_jpeg=0, clean_runtime=0, dependency=0, denied=0)
denials = []


def deny(path, reason):
    counts['denied'] += 1
    frames = []
    frame = sys._getframe(2)
    while frame:
        frames.append(dict(file=frame.f_code.co_filename, line=frame.f_lineno, function=frame.f_code.co_name))
        frame = frame.f_back
    denials.append(dict(path=str(path), reason=reason, stack=frames))
    raise PermissionError(reason + ': ' + str(path))


def hook(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(images):
        if path.suffix.lower() not in ('.jpeg', '.jpg'):
            deny(path, 'Only official test JPEG reads are allowed')
        counts['test_jpeg'] += 1
    elif path.is_relative_to(run):
        counts['clean_runtime'] += 1
    elif path.is_relative_to(vendor) and path.suffix in ('.py', '.pyc', '.json', '.yaml', '.txt'):
        counts['dependency'] += 1
    elif any(path.is_relative_to(root) for root in (art, art.parent/'261008_Kaggle_Solar', art.parent/'261008_Kaggle_Solar_workers')):
        deny(path, 'Cold inference attempted a historical private input')


sys.argv = sys.argv[1:]
sys.path.insert(0, str(entry.parent))
sys.addaudithook(hook)
try:
    runpy.run_path(str(entry), run_name='__main__')
finally:
    Path(os.environ['RF_COLD_AUDIT_OUTPUT']).write_text(json.dumps(dict(**counts, denials=denials), indent=2) + '\n')
