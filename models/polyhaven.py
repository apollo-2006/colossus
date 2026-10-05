"""downloads a poly haven model (cc0, https://polyhaven.com) as gltf with its files, and turns
its base colour textures to ppm for colossus_build: python3 models/polyhaven.py ASSET [RES] [DIR]

RES is the texture resolution (1k, 2k, 4k, 8k; 2k by default), DIR where it goes (models/ASSET
by default). files already there are kept. needs ffmpeg for the textures.
"""
import json
import os
import subprocess
import sys
import urllib.request


def fetch(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "colossus model fetcher"})
    with urllib.request.urlopen(request) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".part", path)


def main():
    asset = sys.argv[1]
    res = sys.argv[2] if len(sys.argv) > 2 else "2k"
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, asset)
    request = urllib.request.Request(f"https://api.polyhaven.com/files/{asset}", headers={"User-Agent": "colossus model fetcher"})
    files = json.load(urllib.request.urlopen(request))
    gltf = files["gltf"][res]["gltf"]
    fetch(gltf["url"], os.path.join(out, os.path.basename(gltf["url"])))
    for name, f in gltf.get("include", {}).items():
        fetch(f["url"], os.path.join(out, name))
        # base colour textures (poly haven names them _diff_) as ppm for the builder.
        if "_diff_" in name and not name.endswith(".ppm"):
            ppm = os.path.join(out, os.path.splitext(name)[0] + ".ppm")
            if not os.path.exists(ppm):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(out, name), ppm], check=True)
    print(os.path.join(out, os.path.basename(gltf["url"])))


if __name__ == "__main__":
    main()
