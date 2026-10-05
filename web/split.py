"""splits a .cgeo for the web demo: name.meta.gz (all but page data,
gzipped) and name.pages (raw page data, read a page at a time by http
range request). a .ctex beside it (include/texture_file.hpp) becomes
name.texture.json (its levels) and name.tiles (its tiles, read the same
way); a .cskn (include/skeleton.hpp) is kept as it is."""
import gzip
import json
import os
import struct
import sys

TILE_BYTES = 128 * 128 // 2


def split_texture(stem):
    data = open(stem + ".ctex", "rb").read()
    if data[:8] != b"CTEXv001":
        raise SystemExit(stem + ".ctex: not a texture file")
    width, height, count = struct.unpack_from("<3I", data, 8)
    levels = []
    for l in range(count):
        tx, ty, first = struct.unpack_from("<3I", data, 20 + 12 * l)
        levels.append({"tilesX": tx, "tilesY": ty, "first": first})
    start = (20 + 12 * count + 15) // 16 * 16
    tiles = levels[-1]["first"] + levels[-1]["tilesX"] * levels[-1]["tilesY"]
    json.dump({"width": width, "height": height, "levels": levels, "tileBytes": TILE_BYTES}, open(stem + ".texture.json", "w"))
    open(stem + ".tiles", "wb").write(data[start : start + tiles * TILE_BYTES])
    print(f"{stem}: texture {width} x {height}, {tiles} tiles, {tiles * TILE_BYTES / 1e6:.1f} MB")

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    at = 8 + 48  # magic, bounds, lod bounds, grid

    def skip(element):
        global at
        n = struct.unpack_from("<Q", data, at)[0]
        at += 8 + n * element

    for element in (48, 20, 24, 4, 40):  # clusters, page bounds, pages, dependencies, levels
        skip(element)
    size = struct.unpack_from("<Q", data, at)[0]
    at += 8
    start = (at + 15) // 16 * 16
    stem = path[: -len(".cgeo")]
    with gzip.open(stem + ".meta.gz", "wb", 9) as f:
        f.write(data[:start])
    open(stem + ".pages", "wb").write(data[start : start + size])
    print(f"{stem}: metadata {start / 1e6:.1f} MB, pages {size / 1e6:.1f} MB")
    if os.path.exists(stem + ".ctex"):
        split_texture(stem)
    if os.path.exists(stem + ".cskn"):
        print(f"{stem}: skeleton {os.path.getsize(stem + '.cskn') / 1e3:.0f} kb")
