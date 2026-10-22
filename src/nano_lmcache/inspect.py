"""Read-only inspector for a DiskKVCache directory.

Unlike opening DiskKVCache, this module reads entry files directly and never
performs recovery or cleanup. Use it to inspect a cache without changing it.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ENTRY_SUFFIX = ".entry"


def inspect_directory(directory: Path) -> dict[str, Any]:
    """Return cache metadata without creating, deleting, or touching files."""
    entries: dict[str, dict[str, Any]] = {}
    invalid_files: list[str] = []
    total_bytes = 0

    for path in sorted(directory.glob(f"*{ENTRY_SUFFIX}")):
        total_bytes += path.stat().st_size
        try:
            data = path.read_bytes()
            header_line, separator, _ = data.partition(b"\n")
            if not separator:
                raise ValueError("missing header")
            header = json.loads(header_line)
            key = header.get("key") if isinstance(header, dict) else None
            parent = header.get("parent_key") if isinstance(header, dict) else None
            if not isinstance(key, str) or (
                parent is not None and not isinstance(parent, str)
            ):
                raise ValueError("malformed header")
        except (OSError, ValueError, UnicodeDecodeError):
            invalid_files.append(path.name)
            continue

        entries[key] = {
            "key": key,
            "parent": parent,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }

    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for key, entry in entries.items():
        parent = entry["parent"]
        if parent is None:
            roots.append(key)
        elif parent in entries:
            children[parent].append(key)

    for values in children.values():
        values.sort()
    roots.sort()

    reachable: set[str] = set()

    def make_tree(key: str, depth: int = 0) -> dict[str, Any]:
        reachable.add(key)
        child_keys = children.get(key, [])
        return {
            "key": key,
            "depth": depth,
            "children": len(child_keys),
            "size_bytes": entries[key]["size_bytes"],
            "children_entries": [make_tree(child, depth + 1) for child in child_keys],
        }

    tree = [make_tree(key) for key in roots]
    # Validly formatted entries with missing parents (or cycles) are shown
    # separately instead of silently disappearing from the inspection.
    unreachable = sorted(set(entries) - reachable)
    leaf_keys = [
        key for key in entries if not children.get(key)
    ]
    leaf_keys.sort(key=lambda key: (entries[key]["mtime_ns"], key))
    leaves_lru = [
        {"key": key, "mtime_ns": entries[key]["mtime_ns"]} for key in leaf_keys
    ]

    return {
        "directory": str(directory),
        "entries": len(entries),
        "total_bytes": total_bytes,
        "tree": tree,
        "unreachable": unreachable,
        "leaves_lru": leaves_lru,
        "invalid_files": sorted(invalid_files),
    }


def _print_tree(nodes: list[dict[str, Any]], prefix: str = "") -> None:
    for index, node in enumerate(nodes):
        last = index == len(nodes) - 1
        branch = "└── " if last else "├── "
        print(
            f"{prefix}{branch}{node['key']} "
            f"(depth={node['depth']}, children={node['children']}, "
            f"{node['size_bytes']} bytes)"
        )
        _print_tree(
            node["children_entries"], prefix + ("    " if last else "│   ")
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a DiskKVCache directory without modifying it."
    )
    parser.add_argument(
        "directory", type=Path, help="directory containing.entry files"
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="print JSON output"
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    report = inspect_directory(args.directory)
    if args.as_json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Directory: {report['directory']}")
    print(f"Entries: {report['entries']}")
    print(f"Total entry-file bytes: {report['total_bytes']}")
    print("Prefix chains:")
    if report["tree"]:
        _print_tree(report["tree"])
    else:
        print("  (no roots)")
    if report["unreachable"]:
        print("Unreachable entries: " + ", ".join(report["unreachable"]))
    print("Leaves in LRU order (oldest first):")
    for item in report["leaves_lru"]:
        print(f"  {item['key']} (mtime_ns={item['mtime_ns']})")
    if report["invalid_files"]:
        print("Invalid entry files: " + ", ".join(report["invalid_files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
