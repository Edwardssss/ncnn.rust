#!/usr/bin/env python3
"""Report the C API functions the pinned adapter needs but an SDK does not declare.

The adapter compiles against the headers of one ncnn release, so a different release may lack
functions it calls. Comparing declaration lists turns that into a report instead of a build error.
"""
import argparse
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1]


def declared(path: pathlib.Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {name for name in re.findall(r"\b(ncnn_\w+)\s*\(", text) if not name.endswith("_t")}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sdk", required=True, type=pathlib.Path, help="Installed SDK prefix to inspect")
parser.add_argument(
    "--baseline",
    type=pathlib.Path,
    default=root / "vendor" / "ncnn-20261020" / "src" / "c_api.h",
    help="Header the adapter was written against",
)
parser.add_argument(
    "--require-compatible",
    action="store_true",
    help="Exit non-zero when the SDK lacks a needed function",
)
args = parser.parse_args()

target = args.sdk / "include/ncnn/c_api.h"
if not target.is_file():
    raise SystemExit(f"Missing {target}")
if not args.baseline.is_file():
    raise SystemExit(f"Missing baseline {args.baseline}")

needed = declared(args.baseline)
available = declared(target)
missing = sorted(needed - available)
extra = sorted(available - needed)

print(f"baseline declarations: {len(needed)}")
print(f"target declarations: {len(available)}")
print(f"missing in target: {len(missing)}")
for name in missing:
    print(f"  - {name}")
print(f"added in target: {len(extra)}")
for name in extra:
    print(f"  + {name}")

if missing and args.require_compatible:
    sys.exit(1)
