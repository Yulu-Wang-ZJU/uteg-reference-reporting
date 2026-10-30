#!/usr/bin/env python3
"""Restore the release's exact code/protocol paths into a fresh private root.

The release stores curated snapshots under ``snapshots/`` so it remains a
portable code package.  The historical runners instead resolve dependencies
from their original workspace-relative paths.  This command restores those
paths beneath a caller-selected root without copying any private inputs,
trained objects, predictions or result tables.

Dry-run is the default.  ``--write`` is required before any filesystem write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath


RELEASE = Path(__file__).resolve().parents[1]
PROJECT = RELEASE.parents[1]
MANIFEST = RELEASE / "provenance" / "SOURCE_PROVENANCE.json"
WINDOWS_SAFE_PATH_LIMIT = 248


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_source_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid source_path in provenance: {value!r}")
    # Provenance paths are deliberately POSIX-style workspace-relative names,
    # even on Windows.  Reject absolute and parent-traversing paths before
    # joining them to a user-selected target root.
    portable = PurePosixPath(value)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise ValueError(f"source_path must be a normalized relative path: {value!r}")
    path = Path(*portable.parts)
    if path.is_absolute() or path.drive:
        raise ValueError(f"source_path must be relative: {value!r}")
    return path


def load_entries() -> list[dict[str, object]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = []
    for section in ("snapshots", "protocols"):
        values = manifest.get(section)
        if not isinstance(values, list):
            raise ValueError(f"provenance section {section!r} is missing or not a list")
        for item in values:
            if not isinstance(item, dict):
                raise ValueError(f"invalid {section} entry: {item!r}")
            source_rel = _relative_source_path(item.get("source_path"))
            snapshot_value = item.get("snapshot_path")
            if not isinstance(snapshot_value, str) or not snapshot_value:
                raise ValueError(f"invalid snapshot_path in {section}: {snapshot_value!r}")
            snapshot_rel = _relative_source_path(snapshot_value)
            snapshot = RELEASE / snapshot_rel
            expected = item.get("snapshot_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"missing snapshot hash for {snapshot_value}")
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            actual = sha256(snapshot)
            if actual != expected:
                raise ValueError(
                    f"snapshot hash mismatch for {snapshot_value}: {actual} != {expected}"
                )
            entries.append(
                {
                    "section": section,
                    "source_path": source_rel.as_posix(),
                    "snapshot_path": snapshot_value,
                    "snapshot_sha256": expected,
                    "bytes": snapshot.stat().st_size,
                    "snapshot": snapshot,
                }
            )
    return entries


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _destination_paths(target: Path, entries: list[dict[str, object]]) -> list[Path]:
    return [
        target / Path(*PurePosixPath(str(item["source_path"])).parts)
        for item in entries
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="dry-run or restore exact snapshot paths into a private code root"
    )
    parser.add_argument(
        "--target-root",
        required=True,
        help="fresh private workspace root beneath which source_path values are restored",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="perform the materialization; without this flag only a dry-run plan is printed",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="permit a pre-existing target root (existing files remain protected by default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing target file only when its bytes differ",
    )
    args = parser.parse_args()

    target = Path(args.target_root).expanduser().resolve()
    release = RELEASE.resolve()
    project = PROJECT.resolve()
    if target == release or _inside(target, release):
        raise SystemExit("refusing to materialize inside the release directory")
    if target == project:
        raise SystemExit("refusing to materialize into the project root")
    if target.exists() and not target.is_dir():
        raise SystemExit(f"target root is not a directory: {target}")
    if target.exists() and not args.allow_existing:
        raise SystemExit(
            f"target root already exists; use a fresh path or pass --allow-existing: {target}"
        )

    entries = load_entries()
    destinations = _destination_paths(target, entries)
    max_destination_length = max((len(str(path)) for path in destinations), default=len(str(target)))
    if args.write and os.name == "nt":
        long_paths = [path for path in destinations if len(str(path)) >= WINDOWS_SAFE_PATH_LIMIT]
        if long_paths:
            longest = max(long_paths, key=lambda path: len(str(path)))
            raise SystemExit(
                "refusing to materialize because a destination path is at/near "
                f"the Windows MAX_PATH limit ({len(str(longest))} characters): {longest}\n"
                "Choose a short fresh target such as E:/uteg_replay (or enable "
                "Windows long-path support) and retry; this invocation performed "
                "no writes."
            )
    plan = [
        {
            "source_path": item["source_path"],
            "snapshot_path": item["snapshot_path"],
            "snapshot_sha256": item["snapshot_sha256"],
            "bytes": item["bytes"],
        }
        for item in entries
    ]

    if not args.write:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "target_root": str(target),
                    "count": len(plan),
                    "max_destination_path_length": max_destination_length,
                    "files": plan,
                    "writes_performed": False,
                },
                indent=2,
            )
        )
        return 0

    target.mkdir(parents=True, exist_ok=True)
    written = 0
    unchanged = 0
    for item in entries:
        destination = target / Path(*PurePosixPath(str(item["source_path"])).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot = Path(item["snapshot"])
        if destination.exists():
            if not destination.is_file():
                raise SystemExit(f"refusing to replace non-file target: {destination}")
            if sha256(destination) == item["snapshot_sha256"]:
                unchanged += 1
                continue
            if not args.overwrite:
                raise SystemExit(
                    f"target differs from exact snapshot; pass --overwrite to replace: {destination}"
                )
        shutil.copyfile(snapshot, destination)
        written += 1

    print(
        json.dumps(
            {
                "status": "MATERIALIZED",
                "target_root": str(target),
                "count": len(plan),
                "max_destination_path_length": max_destination_length,
                "written": written,
                "unchanged": unchanged,
                "writes_performed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
