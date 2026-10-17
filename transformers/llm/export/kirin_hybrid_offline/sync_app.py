"""Sync only the independent Offline NPU-2 bridge; never changes libMNN/old NPU.

Invoke explicitly with --write; default checks copies and their SHA256 manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sync(app, write=False):
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    sources = {name: here / 'runtime' / name for name in ('OfflineNpu2.hpp', 'OfflineNpu2.cpp', 'Vision.cpp', 'Protocol.cpp')}
    for name in ('llmconfig.hpp', 'ujson.hpp'):
        sources[name] = root / 'transformers/llm/engine/src' / name
    dest = app / 'entry/src/main/cpp/offline_npu2'
    if write:
        previous = dest / 'source_manifest.json'
        if previous.exists():
            for name, digest in json.loads(previous.read_text()).items():
                local = dest / name
                if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('App bridge was edited locally; merge before syncing: ' + name)
        dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, src in sorted(sources.items()):
        content = src.read_bytes()
        manifest[name] = hashlib.sha256(content).hexdigest()
        if write:
            (dest / name).write_bytes(content)
        elif not (dest / name).is_file() or (dest / name).read_bytes() != content:
            raise RuntimeError('App bridge out of sync: ' + name)
    target = dest / 'source_manifest.json'
    if write:
        target.write_text(json.dumps(manifest, indent=2) + '\n')
    elif json.loads(target.read_text()) != manifest:
        raise RuntimeError('App bridge manifest out of sync')
    print(json.dumps({'app': str(app), 'files': len(sources), 'write': write}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', type=Path, required=True)
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    sync(a.app, a.write)
