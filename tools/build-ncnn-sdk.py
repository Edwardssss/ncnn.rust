#!/usr/bin/env python3
"""Build an ncnn SDK from a source tree with the project patches applied.

Usage: build-ncnn-sdk.py <source-dir> <install-prefix> [--vulkan] [--no-patches]

The workflow builds the same feature set that ncnn-rust requires, so a missing
feature fails the compatibility job instead of silently changing the ABI surface.
The patches in tools/patches fix two upstream defects with observable results:
Vulkan Cast advertises conversions it has no shader for, and the Linux sibling
parser truncates CPU masks to 32 bits. Applying them keeps every SDK built here
identical to the one the tests were validated against.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

FEATURES = [
    "NCNN_C_API=ON",
    "NCNN_STDIO=ON",
    "NCNN_STRING=ON",
    "NCNN_PIXEL=ON",
    "NCNN_PIXEL_ROTATE=ON",
    "NCNN_PIXEL_AFFINE=ON",
    "NCNN_PIXEL_DRAWING=ON",
    "NCNN_BATCH=ON",
    "NCNN_INT8=ON",
    "NCNN_BF16=ON",
    "NCNN_VALIDATION=ON",
    "NCNN_BUILD_TOOLS=OFF",
    "NCNN_BUILD_EXAMPLES=OFF",
    "NCNN_BUILD_BENCHMARK=OFF",
    "NCNN_BUILD_TESTS=OFF",
    "NCNN_SHARED_LIB=ON",
    "NCNN_OPENMP=OFF",
]

PATCH_DIR = pathlib.Path(__file__).resolve().parent / "patches"


def source_version(source):
    """Derives the release tag from a source directory such as ncnn-20261020."""
    name = source.name
    return name.split("-", 1)[1] if "-" in name else name


def patch_targets(patch):
    """Maps each patched file to the lines the patch adds."""
    targets = {}
    current = None
    for line in patch.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ b/"):
            current = line[6:].strip()
            targets.setdefault(current, [])
        elif line.startswith("+") and not line.startswith("+++") and current:
            targets[current].append(line[1:])
    return targets


def is_applied(source, patch):
    """Reports whether every added line is already present in its file."""
    for relative, added in patch_targets(patch).items():
        path = source / relative
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(line and line not in text for line in added):
            return False
    return True


def apply_patches(source):
    """Applies every project patch written for this release and not yet present.

    Patches target the pinned release. A compatibility run against another tag
    builds the unmodified upstream tree, which is what that job measures.
    """
    version = source_version(source)
    applied = []
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        if not patch.name.startswith(f"ncnn-{version}-"):
            print(f"skipped patch for another release: {patch.name}")
            continue
        if is_applied(source, patch):
            print(f"patch already applied: {patch.name}")
            continue
        apply_env = dict(os.environ)
        # Keeping git inside the source directory stops it from resolving patch
        # paths against an enclosing repository root.
        apply_env["GIT_CEILING_DIRECTORIES"] = str(source.parent)
        subprocess.run(["git", "apply", str(patch)], cwd=source, env=apply_env, check=True)
        if not is_applied(source, patch):
            raise SystemExit(f"{patch.name} did not apply cleanly")
        applied.append(patch.name)
        print(f"applied patch: {patch.name}")
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("prefix", type=pathlib.Path)
    parser.add_argument("--vulkan", action="store_true")
    parser.add_argument("--no-patches", action="store_true", help="Build the unmodified tree")
    args = parser.parse_args()
    source = args.source.resolve()
    prefix = args.prefix.resolve()
    build = prefix.parent / (prefix.name + "-build")
    if not (source / "CMakeLists.txt").is_file():
        raise SystemExit(f"missing CMakeLists.txt in {source}")
    if not args.no_patches:
        apply_patches(source)
    for directory in (build, prefix):
        shutil.rmtree(directory, ignore_errors=True)
    config = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        # ncnn stamps the configure date unless a version is given, which would make
        # platform.h disagree with the pinned release.
        f"-DNCNN_VERSION={source_version(source)}",
        f"-DNCNN_VULKAN={'ON' if args.vulkan else 'OFF'}",
    ]
    if args.vulkan:
        # System glslang keeps the build reproducible; the vendored tree has no glslang.
        config.append("-DNCNN_SYSTEM_GLSLANG=ON")
    config += [f"-D{feature}" for feature in FEATURES]
    subprocess.run(config, check=True)
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "-j", str(os.cpu_count() or 2)],
        check=True,
    )
    subprocess.run(["cmake", "--install", str(build), "--config", "Release"], check=True)
    print(f"installed ncnn to {prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
