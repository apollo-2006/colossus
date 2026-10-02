"""splits a .cgeo for the web demo: name.meta.gz (all but page data,
gzipped) and name.pages (raw page data, read a page at a time by http
range request)."""
import gzip
import struct
import sys

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
