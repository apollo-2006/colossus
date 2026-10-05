#!/usr/bin/env bash
# builds the web models: lucy, the dragon and washington trimmed to 4M triangles
# at their finest, into web/models, each split into gzipped metadata (loaded up
# front) and raw pages (streamed by range request) by web/split.py, washington's
# texture into tiles the same way, and the skinned fox, subdivided, whole.
# needs colossus_build (make) and the models (models/fetch.sh dragon lucy washington fox).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/models
rm -f web/models/*.cgeo.gz web/models/*.ctex web/models/*.cskn
./colossus_build models/lucy.ply web/models/lucy.cgeo --up-z --max-triangles 4000000
./colossus_build models/xyzrgb_dragon.ply web/models/dragon.cgeo --max-triangles 4000000
./colossus_build "models/washington/george-washington-greenough-statue-(1840)-master-geometry.obj" \
  web/models/washington.cgeo --max-triangles 4000000
./colossus_build models/fox/Fox.gltf web/models/fox.cgeo --subdivide 5
python3 web/split.py web/models/lucy.cgeo web/models/dragon.cgeo web/models/washington.cgeo web/models/fox.cgeo
rm web/models/lucy.cgeo web/models/dragon.cgeo web/models/washington.cgeo web/models/fox.cgeo
rm web/models/washington.ctex web/models/fox.ctex
ls -l web/models
