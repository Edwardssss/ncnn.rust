#!/usr/bin/env python3
"""Check enabled CPU declarations against generated bindings and native symbols."""
import argparse
import os
import pathlib
import re
import subprocess
import shlex
import sys

root = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bindings', type=pathlib.Path, help='Generated bindings for the build being audited')
parser.add_argument('--target', default=os.environ.get('CARGO_BUILD_TARGET'), help='Cargo target triple for cross-build auditing')
args = parser.parse_args()
if args.bindings:
    bindings = args.bindings.resolve()
else:
    target = pathlib.Path(os.environ.get('CARGO_TARGET_DIR', str(root / 'target')))
    if not target.is_absolute():
        target = root / target
    if args.target:
        target /= args.target
    outputs = sorted((target / 'debug/build').glob('ncnn-rust-sys-*/out/bindings.rs'), key=lambda p: p.stat().st_mtime)
    if not outputs:
        raise SystemExit("Build ncnn-rust-sys before checking coverage or pass --bindings")
    bindings = outputs[-1]
prefix = os.environ.get('NCNN_INSTALL_DIR')
if not prefix:
    raise SystemExit('Set NCNN_INSTALL_DIR to the installation used by the selected bindings')
include = pathlib.Path(prefix) / 'include/ncnn'
header = root / "crates/ncnn-sys/native/adapter.h"
defines = ['-DNCNN_STATIC_DEFINE'] if os.environ.get('NCNN_LINK_KIND', 'static') == 'static' else []
clang_args = shlex.split(os.environ.get('BINDGEN_EXTRA_CLANG_ARGS', ''))
if args.target:
    clang_args += ['--target=' + args.target]
text = subprocess.check_output([os.environ.get('CLANG', 'clang'), '-E', '-P', *clang_args, *defines, '-I' + str(include), str(header)], text=True)
declared = {name for name in re.findall(r"\b(ncnn_\w+)\s*\(", text) if not name.endswith("_t")}
bound = set(re.findall(r"pub fn (ncnn_\w+)\s*\(", bindings.read_text()))
def symbols(path, shared=False):
    if os.name == 'nt':
        command = ['dumpbin', '/nologo', '/linkermember:1', str(path)]
    elif sys.platform == 'darwin':
        command = [os.environ.get('NM', 'nm'), '-g', '-U', str(path)]
    else:
        command = [os.environ.get('NM', 'nm'), '-g', '--defined-only', *(['-D'] if shared else []), str(path)]
    output = subprocess.check_output(command, text=True, errors='replace')
    return set(re.findall(r'(?:^|\s)_?(ncnn_\w+)\b', output, re.M))

link_kind = os.environ.get('NCNN_LINK_KIND', 'static')
if link_kind not in ('static', 'dylib'):
    raise SystemExit('NCNN_LINK_KIND must be static or dylib')
library_name = 'ncnn.lib' if os.name == 'nt' else ('libncnn.a' if link_kind == 'static' else ('libncnn.dylib' if sys.platform == 'darwin' else 'libncnn.so'))
libraries = [pathlib.Path(prefix) / directory / library_name for directory in ('lib', 'lib64')]
library = next((path for path in libraries if path.is_file()), None)
if library is None:
    raise SystemExit(f'Installed library missing: {library_name}')
adapter_name = 'ncnn_rust_adapter.lib' if os.name == 'nt' else 'libncnn_rust_adapter.a'
adapter = bindings.parent / adapter_name
if not adapter.is_file():
    raise SystemExit(f'Build the adapter for the selected bindings: {adapter}')
linked = symbols(library, link_kind == 'dylib') | symbols(adapter)
missing = sorted(declared - bound)
unresolved = sorted(declared - linked)
print(f"Enabled C functions: {len(declared)}; generated: {len(declared & bound)}; library symbols: {len(declared & linked)}")
for name in sorted(declared):
    status = 'missing binding' if name in missing else ('missing symbol' if name in unresolved else 'bound and defined')
    print(f'{name}: {status}')
if missing or unresolved:
    raise SystemExit(1)
