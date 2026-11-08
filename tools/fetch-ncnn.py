#!/usr/bin/env python3
"""Fetch the pinned ncnn source archive with SHA-256 verification."""
import hashlib
import pathlib
import tarfile
import urllib.request

root = pathlib.Path(__file__).resolve().parents[1]
version = "20261020"
checksum = "968d8d8e65ccadc803322f8238736176e0f9c929fcf484895f58a63e6b227127"
vendor = root / "vendor"
vendor.mkdir(exist_ok=True)
archive = vendor / f"ncnn-{version}.tar.gz"
if not archive.exists():
    urllib.request.urlretrieve(
        f"https://github.com/Tencent/ncnn/archive/refs/tags/{version}.tar.gz", archive
    )
if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
    raise SystemExit("ncnn archive checksum mismatch")
with tarfile.open(archive) as source:
    source.extractall(vendor, filter="data")
print(f"Verified ncnn {version}: {checksum}")
