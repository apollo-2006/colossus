"""how close a cut is to full detail, measured in images: python3 docs/quality.py

each view renders at an error threshold of 0.05 pixels (full detail, as near as makes no
difference) and at 1, 2, 4, 8 and 16 pixels, with antialiasing, ambient occlusion and soft
shadows off (their per-frame noise would differ between any two renders; with them off, two
renders of one cut are identical). two comparisons:

  coverage   view 10 draws white where a model is and black elsewhere. the cut's coverage
             against the reference's: every pixel that differs, and its distance from the
             reference's silhouette. what the error bound is about: where the surface is.
  shaded     nvidia's flip (andersson et al. 2020) between the shaded images, and the share
             of pixels off by more than 8/255: what a viewer sees, shading included, which
             the bound does not cover (a coarse triangle interpolates its corners' normals
             across a larger span).

needs the models (models/fetch.sh) and `pip install flip-evaluator scipy`. writes a markdown
table to stdout and the shaded flip error map of the first view at 1 pixel to
docs/quality_flip.png.
"""
import os
import struct
import subprocess
import sys
import tempfile
import zlib

import flip_evaluator as flip
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS = [
    ("lucy, close", ["--model", "models/lucy.cgeo", "--camera", "0.32,0.78,-0.42,-2.46,-0.2"]),
    ("dragon, whole", ["--model", "models/xyzrgb_dragon.cgeo", "--camera", "0.0,0.35,1.3,0.0,-0.2"]),
    ("washington, face", ["--model", "models/washington.cgeo", "--camera", "0.32,0.78,0.0,-1.5708,-0.05"]),
    ("fox, walking", ["--model", "models/fox.cgeo", "--animation", "Walk", "--camera", "0.0,0.45,1.6,0.0,-0.1"]),
    ("crowd of 9", ["--model", "models/lucy.cgeo", "--model", "models/xyzrgb_dragon.cgeo", "--grid", "3",
                    "--spacing", "1.25", "--camera", "-0.4,0.55,2.6,0.25,-0.15"]),
]
THRESHOLDS = [1, 2, 4, 8, 16]
COMMON = ["--headless", "--warmup", "120", "--frames", "4", "--width", "1600", "--height", "900",
          "--no-taa", "--no-ao", "--hard-shadows"]


def render(view, threshold, mode, out):
    r = subprocess.run([os.path.join(ROOT, "colossus"), *view, *COMMON, "--threshold", str(threshold),
                        "--mode", str(mode), "--screenshot", out], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    line = [l for l in r.stdout.splitlines() if " tris" in l and l[:1].isdigit()][0]
    return line.split("|")[1].split("tris")[0].strip()  # triangles drawn


def coverage(ref_path, test_path):
    ref = flip.load(ref_path).mean(axis=2) > 0.5
    test = flip.load(test_path).mean(axis=2) > 0.5
    edge = np.zeros_like(ref)  # the reference's silhouette, both sides
    for axis in (0, 1):
        d = np.diff(ref.astype(np.int8), axis=axis) != 0
        if axis == 0:
            edge[:-1] |= d
            edge[1:] |= d
        else:
            edge[:, :-1] |= d
            edge[:, 1:] |= d
    dist = ndimage.distance_transform_edt(~edge)
    off = dist[ref ^ test]
    return (ref ^ test).mean(), off.max() if off.size else 0.0


def shaded(ref_path, test_path):
    error_map, mean, _ = flip.evaluate(ref_path, test_path, "LDR")
    r, t = flip.load(ref_path), flip.load(test_path)
    off = (np.abs(r.astype(np.float32) - t.astype(np.float32)).max(axis=2) > 8 / 255).mean()
    return mean, off, error_map


def write_png(path, rgb):  # 8-bit rgb, rows unfiltered
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(h))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    tmp = tempfile.mkdtemp(prefix="colossus_quality_")
    print("| view | threshold | triangles | coverage differs | farthest from silhouette | flip | pixels off by over 8/255 |")
    print("|---|---|---|---|---|---|---|")
    for k, (name, view) in enumerate(VIEWS):
        ref_tris = render(view, 0.05, 10, f"{tmp}/ref_cov.png")
        render(view, 0.05, 0, f"{tmp}/ref.png")
        print(f"| {name} | full detail | {ref_tris} | | | | |")
        for t in THRESHOLDS:
            tris = render(view, t, 10, f"{tmp}/cov.png")
            render(view, t, 0, f"{tmp}/img.png")
            cov, far = coverage(f"{tmp}/ref_cov.png", f"{tmp}/cov.png")
            mean, off, error_map = shaded(f"{tmp}/ref.png", f"{tmp}/img.png")
            print(f"| | {t} px | {tris} | {100 * cov:.3f}% | {far:.2f} px | {mean:.4f} | {100 * off:.2f}% |", flush=True)
            if k == 0 and t == 1:
                # flip's error, magma-like: black for none through orange to white.
                e = np.clip(error_map if error_map.ndim == 2 else error_map.mean(axis=2), 0, 1)
                rgb = np.stack([np.clip(3 * e, 0, 1), np.clip(3 * e - 1, 0, 1), np.clip(3 * e - 2, 0, 1)], axis=2)
                write_png(os.path.join(ROOT, "docs/quality_flip.png"), (rgb * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
