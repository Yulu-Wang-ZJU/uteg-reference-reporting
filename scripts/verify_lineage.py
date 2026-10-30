#!/usr/bin/env python3
"""Verify the current manuscript artifact-to-source lineage map."""

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


def main() -> int:
    release = Path(__file__).resolve().parents[1]
    project = release.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project,
        help="workspace root containing the current manuscript outputs",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="record missing local artifacts instead of failing (useful off-workspace)",
    )
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    lineage_path = release / "provenance" / "MANUSCRIPT_LINEAGE.json"
    provenance_path = release / "provenance" / "SOURCE_PROVENANCE.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    by_snapshot = {
        item["snapshot_path"]: item for item in provenance["snapshots"]
    }
    failures: list[str] = []
    missing: list[str] = []
    checked_artifacts = 0
    checked_sources = 0
    mappings: list[dict[str, object]] = []

    for entry in lineage["entries"]:
        code_snapshots = entry.get("code_snapshots", [])
        for snapshot_path in code_snapshots:
            item = by_snapshot.get(snapshot_path)
            snapshot = release / snapshot_path
            if item is None:
                failures.append(f"code snapshot is not in SOURCE_PROVENANCE: {snapshot_path}")
                continue
            if not snapshot.is_file():
                failures.append(f"missing code snapshot: {snapshot}")
                continue
            if sha256(snapshot) != item["snapshot_sha256"]:
                failures.append(f"code snapshot hash mismatch: {snapshot}")
                continue
            source = source_root / Path(*Path(item["source_path"]).parts)
            if source.is_file():
                checked_sources += 1
                expected = item.get("source_sha256")
                if expected and sha256(source) != expected:
                    failures.append(f"original source hash mismatch: {source}")
            mappings.append(
                {
                    "artifact_id": entry["id"],
                    "artifact_path": entry["artifact_path"],
                    "snapshot_path": snapshot_path,
                    "source_path": item["source_path"],
                    "source_present": source.is_file(),
                }
            )

        artifact_specs = [
            {"path": entry["artifact_path"], "sha256": entry["artifact_sha256"]},
            *entry.get("related_artifacts", []),
        ]
        for artifact in artifact_specs:
            artifact_path = artifact["path"]
            path = source_root / Path(*Path(artifact_path).parts)
            if not path.is_file():
                missing.append(artifact_path)
                if not args.allow_missing_artifacts:
                    failures.append(f"missing manuscript artifact: {path}")
                continue
            checked_artifacts += 1
            observed = sha256(path)
            if observed != artifact["sha256"]:
                failures.append(
                    f"manuscript artifact hash mismatch: {path} ({observed} != {artifact['sha256']})"
                )

        for private_path in entry.get("private_prerequisites", []):
            normalized = str(private_path).replace("\\", "/")
            if normalized.startswith("releases/") or "/releases/" in normalized:
                failures.append(f"private prerequisite points inside release: {private_path}")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "lineage_entries": len(lineage["entries"]),
        "artifact_files_checked": checked_artifacts,
        "original_sources_checked": checked_sources,
        "missing_artifacts": missing,
        "artifact_source_mappings": len(mappings),
    }
    print(json.dumps(payload, indent=2))
    if failures:
        print("FAILURES:")
        print("\n".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
