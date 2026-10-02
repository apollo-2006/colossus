"""re-encodes png files in place at zlib's best compression. the viewer
writes them uncompressed to need no library."""
import struct
import sys
import zlib


def chunks(data):
    i = 8
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        yield data[i + 4:i + 8], data[i + 8:i + 8 + n]
        i += 12 + n


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


for path in sys.argv[1:]:
    data = open(path, "rb").read()
    header = b""
    idat = b""
    for kind, body in chunks(data):
        if kind == b"IHDR":
            header = body
        elif kind == b"IDAT":
            idat += body
    raw = zlib.decompress(idat)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(out)
    print(f"{path}: {len(data) // 1024} KB -> {len(out) // 1024} KB")
