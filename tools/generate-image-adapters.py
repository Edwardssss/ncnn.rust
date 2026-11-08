#!/usr/bin/env python3
"""Generate thin C wrappers for fixed-version public image helper overloads."""
import pathlib
import re
root = pathlib.Path(__file__).resolve().parents[1]
text = (root / 'vendor/ncnn-20261020/src/mat.h').read_text()
selected = []
for ret, name, args in re.findall(r'NCNN_EXPORT (void) (\w+)\(([^;]*)\);', text):
    if not name.startswith(('yuv420sp2rgb', 'resize_bilinear_c', 'resize_bilinear_yuv', 'kanna_rotate_', 'get_rotation_matrix', 'get_affine_transform', 'invert_affine_transform', 'warpaffine_', 'draw_', 'get_text_drawing_size')):
        continue
    args = re.sub(r' = [^,]+', '', args)
    suffix = '_strided' if 'int stride' in args else ''
    symbol = 'ncnn_rs_image_' + name + suffix
    names = [re.search(r'(\w+)$', arg.strip()).group(1) for arg in args.split(',')]
    selected.append((ret, symbol, args, name, ', '.join(names)))
native = root / 'crates/ncnn-sys/native'
header = '#ifndef NCNN_RUST_IMAGE_H\n#define NCNN_RUST_IMAGE_H\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
source = '#include "image_adapter.h"\n#include "mat.h"\n'
for ret, symbol, args, name, call in selected:
    header += f'{ret} {symbol}({args});\n'
    source += f'extern "C" {ret} {symbol}({args})\n{{\n    ncnn::{name}({call});\n}}\n'
header += '#ifdef __cplusplus\n}\n#endif\n#endif\n'
(native / 'image_adapter.h').write_text(header)
(native / 'image_adapter.cpp').write_text(source)
print(f'Generated {len(selected)} image overload adapters')
