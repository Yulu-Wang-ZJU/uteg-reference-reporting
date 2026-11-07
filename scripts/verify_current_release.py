#!/usr/bin/env python3
"""Verify published sources and aggregate files without running analyses."""
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
        raise ValueError(f'Hash/size mismatch: {entry.get("repository_path", entry.get("source_path"))}')


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inventory(directory: str) -> set[str]:
    return {p.relative_to(RELEASE).as_posix()
            for p in (RELEASE/directory).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optional original workspace for exact source checks')
    args = parser.parse_args()
    current = load(RELEASE/'provenance/CURRENT_MANUSCRIPT.json')
    code = load(RELEASE/relative(current['current_code']))
    expected = set()
    python_count = json_count = 0
    for entry in code['members']:
        path = RELEASE/relative(entry['repository_path'])
        if path.suffix not in {'.py', '.json', '.md', '.mjs', '.txt'}:
            raise ValueError(f'Unexpected current source type: {path}')
        data = path.read_bytes()
        check(data, entry)
        expected.add(entry['repository_path'])
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=entry['repository_path'])
            python_count += 1
        elif path.suffix == '.json':
            json.loads(data)
            json_count += 1
    if inventory('current') != expected:
        raise ValueError(f'Current inventory mismatch: {inventory("current") ^ expected}')

    legacy = load(RELEASE/relative(current['supplementary_code']))
    legacy_expected = set()
    for bundle in legacy['bundles']:
        for entry in bundle['members']:
            path = RELEASE/relative(entry['repository_path'])
            check(path.read_bytes(), entry)
            legacy_expected.add(entry['repository_path'])
    if inventory('supplementary') != legacy_expected:
        raise ValueError('Historical supplementary inventory mismatch')

    tables = load(RELEASE/relative(current['public_data']))
    for entry in tables['files']:
        check((RELEASE/relative(entry['repository_path'])).read_bytes(), entry)
        if args.source_root:
            check((args.source_root/relative(entry['source_path'])).read_bytes(), entry)
    if inventory('data') != {e['repository_path'] for e in tables['files']} | {'data/README.md'}:
        raise ValueError('Public-data inventory mismatch')

    source_members = anchors_checked = 0
    if args.source_root:
        archive = args.source_root/relative(code['source_archive']['source_path'])
        check(archive.read_bytes(), code['source_archive'])
        with zipfile.ZipFile(archive) as z:
            for entry in code['members']:
                check(z.read(entry['archive_member']), entry)
                source_members += 1
            accounted = {e['archive_member'] for e in code['members']} | {e['member'] for e in code['excluded_members']}
            if accounted != set(z.namelist()):
                raise ValueError('Unaccounted submission archive member')
        for entry in current['manuscript_anchors'] + current['companion_anchors'] + current['figure_anchors']:
            check((args.source_root/relative(entry['source_path'])).read_bytes(), entry)
            anchors_checked += 1

    print(json.dumps(dict(status='PASS', current_files=len(expected),
                         python_files_syntax_checked=python_count, json_files_checked=json_count,
                         historical_supplementary_files=len(legacy_expected),
                         aggregate_workbooks_checked=len(tables['files']),
                         source_archive_members_checked=source_members,
                         local_anchors_checked=anchors_checked,
                         analysis_entrypoints_executed=False), indent=2))


if __name__ == '__main__':
    main()
