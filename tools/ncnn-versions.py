#!/usr/bin/env python3
"""Resolve the ncnn version matrix from ci/ncnn-versions.json.

Prints a GitHub Actions output line with the version array. Dynamic entries name
tag positions: `latest` is the newest tag and `previous` the next distinct tag
below the newest that is not already listed.
"""
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "ci" / "ncnn-versions.json"
TAGS_URL = "https://api.github.com/repos/Tencent/ncnn/tags?per_page=100"


def fetch_tags() -> list[str]:
    headers = {"User-Agent": "ncnn-rust-ci", "Accept": "application/vnd.github+json"}
    # The unauthenticated limit is per IP and hosted runners share addresses, so a token keeps
    # the monthly run from failing on a rate limit it did not cause.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(TAGS_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return [entry["name"] for entry in json.load(response)]
    except urllib.error.HTTPError as error:
        print(f"Cannot list ncnn tags: {error.code} {error.reason}", file=sys.stderr)
        if error.code in (403, 429):
            print("Set GITHUB_TOKEN to raise the API rate limit", file=sys.stderr)
        raise SystemExit(1)


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
    resolved = resolve()
    print(f"versions={json.dumps(resolved)}")
    print(f"pinned={resolved[0]}")
