#!/usr/bin/env python3
"""Install a prebuilt ncnn SDK published by this repository.

Usage:
  install-sdk.py --platform linux-x64-cpu --dest /opt/ncnn
  install-sdk.py --list

The archives come from the `sdk-<ncnn version>` GitHub release and carry the feature
set and project patches that tools/build-ncnn-sdk.py applies, so a machine without
CMake, glslang and a C++ build of ncnn can still build the crates. Each download is
checked against the release's SHA256SUMS and against the release stamped into the
archive's platform.h.

Neither check authenticates the archive: the checksum ships next to it, so it detects a
truncated or corrupted transfer rather than a substituted release. A build script never
runs this step; installing an SDK is explicit and NCNN_INSTALL_DIR points the build at
the result.
"""
import argparse
import hashlib
import pathlib
import platform
import re
import shutil
import sys
import tarfile
import urllib.request

VERSION = "20261020"
REPO = "Edwardssss/ncnn.rust"

PLATFORMS = {
    "linux-x64-cpu": {
        "system": "linux",
        "machine": "x86_64",
        "vulkan": False,
        "library": "lib/libncnn.so",
    },
    "linux-x64-vulkan": {
        "system": "linux",
        "machine": "x86_64",
        "vulkan": True,
        "library": "lib/libncnn.so",
    },
    "windows-x64-cpu": {
        "system": "win32",
        "machine": "x86_64",
        "vulkan": False,
        "library": "bin/ncnn.dll",
    },
    "macos-arm64-cpu": {
        "system": "darwin",
        "machine": "arm64",
        "vulkan": False,
        "library": "lib/libncnn.dylib",
    },
}


def asset_name(version, name):
    return f"ncnn-rust-sdk-{version}-{name}.tar.gz"


def detect_platform():
    """Picks the archive that matches this interpreter, or explains why none does."""
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    for name, entry in PLATFORMS.items():
        if entry["system"] == sys.platform and entry["machine"] == machine:
            return name
    published = ", ".join(sorted(PLATFORMS))
    raise SystemExit(
        f"no archive is published for {sys.platform}/{machine}; "
        f"pass --platform with one of: {published}"
    )


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ncnn.rust-install-sdk"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def read_checksums(payload, wanted):
    """Returns the recorded digest for one asset from a SHA256SUMS file."""
    for line in payload.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == wanted:
            return parts[0].lower()
    raise SystemExit(f"SHA256SUMS has no entry for {wanted}")


def extract(archive, destination):
    """Unpacks an archive after rejecting any member that leaves the destination."""
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise SystemExit(f"archive member escapes the destination: {member.name}")
        # Python 3.12 applies the data filter by default; older interpreters only get
        # the check above, which is why it runs on every member first.
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(root, **extra)


def check(prefix, entry, version):
    """Rejects an archive that holds another release, backend or layout."""
    header = prefix / "include/ncnn/platform.h"
    if not header.is_file():
        raise SystemExit(f"{prefix} carries no include/ncnn/platform.h")
    text = header.read_text(encoding="utf-8", errors="replace")
    stamped = re.search(r"^#define NCNN_VERSION_NUMBER (\S+)$", text, re.MULTILINE)
    if not stamped or stamped.group(1) != version:
        found = stamped.group(1) if stamped else "nothing"
        raise SystemExit(f"archive stamps ncnn {found}, expected {version}")
    expected = "1" if entry["vulkan"] else "0"
    if f"#define NCNN_VULKAN {expected}" not in text:
        backend = "Vulkan" if entry["vulkan"] else "CPU"
        raise SystemExit(f"archive was not built for the {backend} backend")
    library = prefix / entry["library"]
    if not library.is_file():
        raise SystemExit(f"{prefix} carries no {entry['library']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--platform", help="Archive to install; detected when omitted")
    parser.add_argument("--dest", type=pathlib.Path, help="Directory to unpack the SDK into")
    parser.add_argument("--version", default=VERSION, help=f"ncnn release (default {VERSION})")
    parser.add_argument(
        "--base-url",
        help="Directory or URL holding the archive and SHA256SUMS "
        "(default: the sdk-<version> GitHub release)",
    )
    parser.add_argument("--force", action="store_true", help="Replace a non-empty --dest")
    parser.add_argument("--list", action="store_true", help="Print the published archives")
    args = parser.parse_args()

    if args.list:
        print(f"ncnn {args.version} archives; pass one of these to --platform:")
        for name, entry in sorted(PLATFORMS.items()):
            backend = "vulkan" if entry["vulkan"] else "cpu"
            print(f"  {name:<20} {entry['library']:<18} ({backend})")
        return 0
    if not args.dest:
        parser.error("--dest is required unless --list is given")

    name = args.platform or detect_platform()
    if name not in PLATFORMS:
        parser.error(
            f"unknown platform {name}; known: {', '.join(sorted(PLATFORMS))}"
        )
    entry = PLATFORMS[name]
    asset = asset_name(args.version, name)
    base = args.base_url or (
        f"https://github.com/{REPO}/releases/download/sdk-{args.version}"
    )

    destination = args.dest.resolve()
    if destination.exists() and any(destination.iterdir()):
        if not args.force:
            raise SystemExit(f"{destination} is not empty; pass --force to replace it")
        shutil.rmtree(destination)

    print(f"downloading {asset}")
    archive = download(f"{base.rstrip('/')}/{asset}")
    recorded = read_checksums(download(f"{base.rstrip('/')}/SHA256SUMS"), asset)
    actual = hashlib.sha256(archive).hexdigest()
    if actual != recorded:
        raise SystemExit(f"{asset} does not match SHA256SUMS: {actual} != {recorded}")

    staging = destination.with_name(destination.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        pending = staging / asset
        pending.write_bytes(archive)
        extract(pending, staging)
        pending.unlink()
        check(staging, entry, args.version)
    except BaseException:
        # A rejected archive must not leave a half-installed prefix behind.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    shutil.rmtree(destination, ignore_errors=True)
    staging.rename(destination)

    print(f"installed ncnn {args.version} ({name}) to {destination}")
    print(f"  export NCNN_INSTALL_DIR={destination}")
    if entry["vulkan"]:
        print("  cargo build --features vulkan")
    if sys.platform == "win32":
        print(f"  add {destination / 'bin'} to PATH so ncnn.dll resolves at run time")
    else:
        variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        print(f"  export {variable}={destination / 'lib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
