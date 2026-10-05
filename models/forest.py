"""builds a pine forest from poly haven's pine forest collection (cc0; models/polyhaven.py
fetches it) and scatters it into models/forest.scene for `colossus --scene models/forest.scene`:
python3 models/forest.py [--scene-only]

each variant in the collection's files (three pines, three firs, saplings, grass clumps, moss,
ferns, mossy rocks, deadwood) is built as its own model in metres (--node, --keep-scale) into
models/forest/built; models already built are kept. the scatter is seeded, so every run makes
the same forest: about 140 trees on 100 x 100 metres with a clearing in the middle, saplings
between them, and undergrowth everywhere but on the trunks. it streams about 3 gb of pages
from 2 gb on disk, so the scene asks for a 4 gb pool.
"""
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(os.path.dirname(HERE), "colossus_build")
ASSETS = {  # asset: nodes
    "pine_tree_01": ["pine_tree_01_a_LOD0", "pine_tree_01_b_LOD0", "pine_tree_01_c_LOD0"],
    "fir_tree_01": ["fir_tree_01_a_LOD0", "fir_tree_01_b_LOD0", "fir_tree_01_c_LOD0"],
    "pine_sapling_medium": ["pine_sapling_medium_a_LOD0", "pine_sapling_medium_b_LOD0", "pine_sapling_medium_c_LOD0"],
    "fir_sapling_medium": ["fir_sapling_medium_a_LOD0", "fir_sapling_medium_b_LOD0", "fir_sapling_medium_c_LOD0"],
    "pine_sapling_small": ["pine_sapling_small_a", "pine_sapling_small_b", "pine_sapling_small_c"],
    "fir_sapling": ["fir_sapling_a", "fir_sapling_b", "fir_sapling_c"],
    "grass_medium_01": ["grass_medium_01_large_a_LOD0", "grass_medium_01_large_b_LOD0", "grass_medium_01_mid_a_LOD0",
                        "grass_medium_01_mid_b_LOD0", "grass_medium_01_tall_a_LOD0", "grass_medium_01_tall_b_LOD0",
                        "grass_medium_01_small_a_LOD0", "grass_medium_01_small_b_LOD0"],
    "moss_01": ["moss_01_a_LOD0", "moss_01_b_LOD0", "moss_01_c_LOD0", "moss_01_tall_a_LOD0"],
    "fern_02": ["fern_02_a", "fern_02_b", "fern_02_c", "fern_02_d"],
    "rock_moss_set_01": [f"rock_moss_set_01_rock0{k}" for k in range(1, 7)],
    "rock_moss_set_02": [f"rock_moss_set_02_rock{k:02d}" for k in range(7, 14)],
    "dead_tree_trunk": ["dead_tree_trunk"],
    "tree_stump_01": ["tree_stump_01"],
    "tree_stump_02": ["tree_stump_02"],
    "pine_roots": ["pine_roots_a", "pine_roots_b"],
    "dry_branches_medium_01": ["dry_branches_medium_01_a", "dry_branches_medium_01_b", "dry_branches_medium_01_c"],
}
SIZE = 100.0


def build():
    os.makedirs(os.path.join(HERE, "forest", "built"), exist_ok=True)
    for asset, nodes in ASSETS.items():
        gltf = os.path.join(HERE, "forest", asset, f"{asset}_2k.gltf")
        if not os.path.exists(gltf):
            subprocess.run([sys.executable, os.path.join(HERE, "polyhaven.py"), asset, "2k", os.path.join(HERE, "forest", asset)], check=True)
        for node in nodes:
            out = os.path.join(HERE, "forest", "built", node.replace("_LOD0", "") + ".cgeo")
            if os.path.exists(out):
                continue
            print(f"building {node}", flush=True)
            subprocess.run([BUILD, gltf, out, "--node", node, "--keep-scale"], check=True, stdout=subprocess.DEVNULL)


def scatter():
    rng = random.Random(1840)
    names = {asset: [n.replace("_LOD0", "") for n in nodes] for asset, nodes in ASSETS.items()}
    lines = ["# a pine forest from poly haven's pine forest collection (cc0), made by models/forest.py.",
             "ground 0.16 0.13 0.09", "fog 0.006", "grid off", "pool 4095"]
    for nodes in names.values():
        for n in nodes:
            lines.append(f"model {n} forest/built/{n}.cgeo")
    trunks = []  # (x, z, radius kept clear)

    def free(x, z, gap):
        return all((x - tx) ** 2 + (z - tz) ** 2 >= (gap + tr) ** 2 for tx, tz, tr in trunks)

    def place(assets, count, gap, scale, clear, keep=0.0, tries=40):
        for _ in range(count):
            for _ in range(tries):
                x, z = rng.uniform(-SIZE / 2, SIZE / 2), rng.uniform(-SIZE / 2, SIZE / 2)
                if math.hypot(x, z) < clear or not free(x, z, gap):
                    continue
                name = rng.choice([n for a in assets for n in names[a]])
                s = rng.uniform(*scale)
                lines.append(f"place {name} {x:.3f} 0 {z:.3f} {rng.uniform(0, 2 * math.pi):.4f} {s:.3f}")
                if keep:
                    trunks.append((x, z, keep * s))
                break

    # kinds of placement: assets, how many, smallest gap to a trunk, scale range, clearing
    # radius, and the gap each keeps around itself (per unit of scale).
    place(["pine_tree_01", "fir_tree_01"], 140, 3.5, (0.85, 1.15), 9.0, keep=1.5)
    place(["pine_sapling_medium", "fir_sapling_medium"], 120, 1.5, (0.8, 1.2), 6.0, keep=1.0)
    place(["pine_sapling_small", "fir_sapling"], 300, 0.8, (0.8, 1.3), 4.0, keep=0.4)
    place(["rock_moss_set_01", "rock_moss_set_02"], 60, 0.6, (0.6, 1.4), 3.0, keep=1.0)
    place(["dead_tree_trunk", "tree_stump_01", "tree_stump_02", "pine_roots", "dry_branches_medium_01"], 160, 0.5, (0.8, 1.2), 2.0)
    place(["fern_02"], 3000, 0.3, (0.7, 1.3), 1.5)
    place(["moss_01"], 2000, 0.2, (0.7, 1.4), 0.0)
    place(["grass_medium_01"], 9000, 0.2, (0.7, 1.4), 0.0)
    open(os.path.join(HERE, "forest.scene"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(HERE, 'forest.scene')}: {sum(l.startswith('place') for l in lines)} placed")


if __name__ == "__main__":
    if "--scene-only" not in sys.argv:
        build()
    scatter()
