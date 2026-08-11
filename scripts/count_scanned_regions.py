"""
Counts how many cases scan each body region, based on each case's
scanned-region.json (a single JSON string, e.g. "abdomen"), and lists which
case labels (e.g. A_001) fall under each region.

Usage:
    uv run python scripts/count_scanned_regions.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def main():
    labels_by_region = defaultdict(list)

    for case_dir in sorted(config.DATA_DIR.iterdir()):
        path = case_dir / "scanned-region.json"
        if not path.exists():
            continue
        region = json.loads(path.read_text())
        labels_by_region[region].append(case_dir.name)

    total = sum(len(labels) for labels in labels_by_region.values())
    if total == 0:
        print(f"No scanned-region.json files found under {config.DATA_DIR}")
        return

    for region, labels in sorted(labels_by_region.items(), key=lambda kv: -len(kv[1])):
        print(f"{region}: {len(labels)} ({100 * len(labels) / total:.1f}%)")
        print(f"  {', '.join(labels)}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
