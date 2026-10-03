"""crops the same rectangle from rgb pngs written by the viewer (unfiltered
rows) and joins them side by side: python3 docs/strip.py out.png x y w h in.png..."""
import struct
import sys
import zlib


def read(path):
    data = open(path, "rb").read()
    i, idat, width, height = 8, b"", 0, 0
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        kind, body = data[i + 4:i + 8], data[i + 8:i + 8 + n]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            if depth != 8 or colour != 2:
                raise SystemExit(f"{path}: not 8-bit rgb")
        elif kind == b"IDAT":
            idat += body
        i += 12 + n
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    rows = []
    for y in range(height):
        if raw[y * stride] != 0:
            raise SystemExit(f"{path}: filtered rows; written by the viewer?")
        rows.append(raw[y * stride + 1:(y + 1) * stride])
    return rows


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


out, x, y, w, h = sys.argv[1], *map(int, sys.argv[2:6])
images = [read(p) for p in sys.argv[6:]]
rows = [b"".join(img[y + r][3 * x:3 * (x + w)] for img in images) for r in range(h)]
raw = b"".join(b"\0" + row for row in rows)
header = struct.pack(">IIBBBBB", w * len(images), h, 8, 2, 0, 0, 0)
open(out, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
