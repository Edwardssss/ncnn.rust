#!/usr/bin/env python3
"""Verify that the published packages share one version.

Release notes come from GitHub Releases, which groups the Conventional Commit
subjects between two tags, so no changelog file is maintained here.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT
        )
    )
    versions = {package["name"]: package["version"] for package in metadata["packages"]}
    expected = versions.get("ncnn-rust")
    problems = []
    if expected is None:
        problems.append("workspace has no ncnn-rust package")
    for name, version in versions.items():
        if name.startswith("ncnn-rust") and version != expected:
            problems.append(f"{name} is {version}, expected {expected}")
    print("versions:", versions)
    for problem in problems:
        print("error:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
