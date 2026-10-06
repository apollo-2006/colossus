"""colossus's builder against meshoptimizer's clusterlod, through the same renderer: python3
docs/compare/compare.py (after make compare and models/fetch.sh). each view is drawn from both
builds at several thresholds and measured, as docs/quality.py measures, against one full-detail
reference (the leaves are the same triangles in both)."""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "docs"))
import quality as q  # noqa: E402

BUILDS = {
    "colossus": {"lucy": "models/lucy.cgeo", "dragon": "models/xyzrgb_dragon.cgeo"},
    "clusterlod": {"lucy": "models/lucy_clod.cgeo", "dragon": "models/dragon_clod.cgeo"},
}
SOURCES = {"lucy": ("models/lucy.ply", ["--up-z"]), "dragon": ("models/xyzrgb_dragon.ply", [])}
VIEWS = [
    ("lucy, close", ["lucy"], ["--camera", "0.32,0.78,-0.42,-2.46,-0.2"]),
    ("dragon, whole", ["dragon"], ["--camera", "0.0,0.35,1.3,0.0,-0.2"]),
    ("crowd of 9", ["lucy", "dragon"], ["--grid", "3", "--spacing", "1.25", "--camera", "-0.4,0.55,2.6,0.25,-0.15"]),
]
THRESHOLDS = [0.5, 1, 2, 4]


def view_args(build, models, rest):
    args = []
    for m in models:
        args += ["--model", BUILDS[build][m]]
    return args + rest


def main():
    for m, out in BUILDS["clusterlod"].items():
        if not os.path.exists(os.path.join(ROOT, out)):
            src, flags = SOURCES[m]
            subprocess.run([os.path.join(HERE, "clod_build"), src, out, *flags], cwd=ROOT, check=True)
    tmp = tempfile.mkdtemp(prefix="colossus_compare_")
    print("| view | builder | threshold | triangles | coverage differs | farthest from silhouette | flip | pixels off by over 8/255 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, models, rest in VIEWS:
        ref = view_args("colossus", models, rest)
        q.render(ref, 0.05, 10, f"{tmp}/ref_cov.png")
        q.render(ref, 0.05, 0, f"{tmp}/ref.png")
        for build in BUILDS:
            for t in THRESHOLDS:
                v = view_args(build, models, rest)
                tris = q.render(v, t, 10, f"{tmp}/cov.png")
                q.render(v, t, 0, f"{tmp}/img.png")
                cov, far = q.coverage(f"{tmp}/ref_cov.png", f"{tmp}/cov.png")
                mean, off, _ = q.shaded(f"{tmp}/ref.png", f"{tmp}/img.png")
                print(f"| {name} | {build} | {t} px | {tris} | {100 * cov:.3f}% | {far:.2f} px | {mean:.4f} | {100 * off:.2f}% |", flush=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
