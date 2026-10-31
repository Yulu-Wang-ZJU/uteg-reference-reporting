#!/usr/bin/env python3
"""Verify current code and provenance without importing or running analyses."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


RELEASE = Path(__file__).resolve().parents[1]


def relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError(f'Expected a relative manifest path: {value}')
    return Path(*path.parts)


def check(data: bytes, entry: dict) -> None:
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'Hash/size mismatch: {entry}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optional original workspace for local hash checks')
    args = parser.parse_args()
    current = json.loads((RELEASE / 'provenance/CURRENT_MANUSCRIPT.json').read_text(encoding='utf-8'))
    manifest = json.loads((RELEASE / relative(current['supplementary_code'])).read_text(encoding='utf-8'))
    count = python_count = source_count = 0
    expected = set()
    for bundle in manifest['bundles']:
        for entry in bundle['members']:
            path = RELEASE / relative(entry['repository_path'])
            expected.add(path.relative_to(RELEASE).as_posix())
            if path.suffix not in {'.py', '.md', '.txt'}:
                raise ValueError(f'Unexpected public file type: {path}')
            data = path.read_bytes()
            check(data, entry)
            if path.suffix == '.py':
                ast.parse(data.decode('utf-8-sig'), filename=entry['repository_path'])
                python_count += 1
            count += 1
        if args.source_root:
            archive = args.source_root / relative(bundle['source_archive']['source_path'])
            check(archive.read_bytes(), bundle['source_archive'])
            with zipfile.ZipFile(archive) as z:
                for entry in bundle['members']:
                    check(z.read(entry['archive_member']), entry)
                    source_count += 1
                accounted = {x['archive_member'] for x in bundle['members']} | {x['member'] for x in bundle['excluded_members']}
                if accounted != set(z.namelist()):
                    raise ValueError(f'Unaccounted archive members: {archive}')
    actual = {p.relative_to(RELEASE).as_posix() for p in (RELEASE / 'supplementary').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != expected:
        raise ValueError(f'Unmanifested or missing supplementary files: {actual ^ expected}')
    anchors = current['manuscript_anchors'] + current['companion_anchors'] + current['figure_anchors']
    if args.source_root:
        for entry in anchors:
            check((args.source_root / relative(entry['source_path'])).read_bytes(), entry)
    print(json.dumps({'status': 'PASS', 'bundles': len(manifest['bundles']), 'files': count,
                      'python_files_syntax_checked': python_count, 'source_archive_members_checked': source_count,
                      'local_anchors_checked': len(anchors) if args.source_root else 0,
                      'analysis_entrypoints_executed': False}, indent=2))


if __name__ == '__main__':
    main()
