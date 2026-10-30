#!/usr/bin/env python3
"""Verify snapshot hashes and optionally verify the original workspace files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    release = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=release.parents[1],
        help="workspace root containing the original outputs tree (default: this workspace)",
    )
    args = parser.parse_args()
    manifest_path = release / "provenance" / "SOURCE_PROVENANCE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    source_checked = 0
    for item in manifest["snapshots"]:
        snapshot = release / item["snapshot_path"]
        if not snapshot.exists():
            failures.append(f"missing snapshot: {snapshot}")
            continue
        observed = sha256(snapshot)
        if observed != item["snapshot_sha256"]:
            failures.append(f"snapshot hash mismatch: {snapshot}")
        original = args.source_root / item["source_path"]
        if original.exists():
            source_checked += 1
            if sha256(original) != item["source_sha256"]:
                failures.append(f"original hash mismatch: {original}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps({"status": "PASS", "snapshots": len(manifest["snapshots"]), "originals_checked": source_checked}, indent=2))


if __name__ == "__main__":
    main()
