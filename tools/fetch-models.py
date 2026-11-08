#!/usr/bin/env python3
"""Fetch optional pinned model fixtures into ignored build storage."""
import hashlib
import pathlib
import urllib.request

root = pathlib.Path(__file__).resolve().parents[1] / 'target/model-fixtures'
root.mkdir(parents=True, exist_ok=True)
fixtures = [
    ('nanodet.param', 'nihui/ncnn-android-nanodet/321276bf9fd465896dd685d64ab0bb08dd68179b/app/src/main/assets/nanodet-m.param', 'c1e7f51fd49b90b275cb6d585ea93c7d915cbfbe17874054abb57403afb9043b'),
    ('nanodet.bin', 'nihui/ncnn-android-nanodet/321276bf9fd465896dd685d64ab0bb08dd68179b/app/src/main/assets/nanodet-m.bin', '19bccebca9339bd559f05046a8a7a94a6eed0117f470b1de5ce5c89a68f0f596'),
    ('ocr.param', 'nihui/ncnn-android-ppocrv5/458269b48a434a4301070f83f7ce90e8fd11180f/app/src/main/assets/PP_OCRv5_mobile_rec.ncnn.param', 'f52a6586ac3338d8c350db0c9f3c55ff2ecd3763327f9bc7f0efc091f8a63e74'),
    ('ocr.bin', 'nihui/ncnn-android-ppocrv5/458269b48a434a4301070f83f7ce90e8fd11180f/app/src/main/assets/PP_OCRv5_mobile_rec.ncnn.bin', '49d9907a55ba20fa6637f9f788f66ab00793bc8ce57a733a09dbe86b9a2e3db0'),
]
for name, source, checksum in fixtures:
    path = root / name
    if not path.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/' + source, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise SystemExit(f'Checksum mismatch: {name}')
    print(f'Verified {name}')
