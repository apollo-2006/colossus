#!/usr/bin/env bash
# Builds the web demo's models: Lucy and the dragon, trimmed to 400k
# triangles at their finest (about 8 MB each, gzipped), into web/models.
# Needs colossus_build (make) and the scans (models/fetch.sh downloads them).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/models
./colossus_build models/lucy.ply web/models/lucy.cgeo --up-z --max-triangles 400000
./colossus_build models/xyzrgb_dragon.ply web/models/dragon.cgeo --max-triangles 400000
gzip -9 -f web/models/lucy.cgeo web/models/dragon.cgeo
ls -l web/models
