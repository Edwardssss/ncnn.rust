#!/usr/bin/env python3
"""Resolve the ncnn version matrix from ci/ncnn-versions.json.

Prints a GitHub Actions output line with the version array. Dynamic entries name
tag positions: `latest` is the newest tag and `previous` the next distinct tag
below the newest that is not already listed.
"""
import json
import pathlib
import sys
import urllib.request

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "ci" / "ncnn-versions.json"
TAGS_URL = "https://api.github.com/repos/Tencent/ncnn/tags?per_page=100"


def fetch_tags() -> list[str]:
    request = urllib.request.Request(TAGS_URL, headers={"User-Agent": "ncnn-rust-ci"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return [entry["name"] for entry in json.load(response)]


def resolve() -> list[str]:
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    versions = [config["pinned"]]
    dynamic = config.get("dynamic", [])
    if not dynamic:
        return versions
    # Tag names are dates, so descending name order is descending release order.
    tags = sorted((tag for tag in fetch_tags() if tag not in versions), reverse=True)
    for entry in dynamic:
        if entry == "latest" and tags:
            versions.append(tags[0])
        elif entry == "previous":
            for tag in tags:
                if tag not in versions:
                    versions.append(tag)
                    break
        else:
            print(f"Unknown dynamic entry: {entry}", file=sys.stderr)
            sys.exit(1)
    return versions


if __name__ == "__main__":
    print(f"versions={json.dumps(resolve())}")
