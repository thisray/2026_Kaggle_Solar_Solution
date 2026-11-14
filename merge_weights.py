"""Verify shards, optionally download them, and reconstruct trusted model files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def merge(root, destination, base_url=None):
    root, destination = Path(root), Path(destination)
    manifest = json.loads((root/'weights_manifest.json').read_text())
    for row in manifest['files']:
        target = destination/row['path']
        if target.exists() and sha(target) == row['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix+'.tmp')
        with temporary.open('wb') as out:
            for part in row['parts']:
                source = root/part['path']
                if not source.exists():
                    if not base_url:
                        raise FileNotFoundError(source)
                    source.parent.mkdir(parents=True, exist_ok=True)
                    download = source.with_suffix('.download')
                    urllib.request.urlretrieve(base_url.rstrip('/')+'/'+part['path'], download)
                    if sha(download) != part['sha256']:
                        download.unlink()
                        raise ValueError('Downloaded shard hash mismatch')
                    download.replace(source)
                if source.stat().st_size != part['bytes'] or sha(source) != part['sha256']:
                    raise ValueError('Shard hash mismatch: '+str(source))
                with source.open('rb') as handle:
                    for block in iter(lambda:handle.read(8<<20), b''):
                        out.write(block)
        if temporary.stat().st_size != row['bytes'] or sha(temporary) != row['sha256']:
            raise ValueError('Reconstructed model hash mismatch: '+row['path'])
        temporary.replace(target)
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--destination',type=Path,default=Path('models'))
    p.add_argument('--base-url')
    a=p.parse_args();merge(a.root,a.destination,a.base_url)
