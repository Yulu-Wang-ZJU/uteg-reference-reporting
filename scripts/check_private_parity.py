#!/usr/bin/env python3
"""Read-only schema parity check for caller-supplied private archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "src"))

from uteg_reference_reporting.private import inspect_data15, inspect_prefix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data15", required=True, type=Path, help="private DATA15.npz")
    parser.add_argument("--prefix-x", required=True, type=Path, help="private PREFIX_X.npz")
    args = parser.parse_args()
    result = {"status": "PASS", "data15": inspect_data15(args.data15), "prefix_x": inspect_prefix(args.prefix_x)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
