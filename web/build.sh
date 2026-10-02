#!/usr/bin/env bash
# Builds the web demo's models: Lucy and the dragon, trimmed to 4M
# triangles at their finest, into web/models, each split into gzipped
# metadata, loaded up front, and raw pages, streamed in with range
# requests as the view needs them (web/split.py).
# Needs colossus_build (make) and the scans (models/fetch.sh downloads them).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/models
rm -f web/models/*.cgeo.gz
./colossus_build models/lucy.ply web/models/lucy.cgeo --up-z --max-triangles 4000000
./colossus_build models/xyzrgb_dragon.ply web/models/dragon.cgeo --max-triangles 4000000
python3 web/split.py web/models/lucy.cgeo web/models/dragon.cgeo
rm web/models/lucy.cgeo web/models/dragon.cgeo
ls -l web/models
