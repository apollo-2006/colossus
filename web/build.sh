#!/usr/bin/env bash
# Builds the web demo's models: Lucy and the dragon, trimmed to 400k
# triangles at their finest (about 8 MB each, gzipped), into web/models.
# Needs ngeo_build (make) and the scans (models/fetch.sh downloads them).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/models
./ngeo_build models/lucy.ply web/models/lucy.ngeo --up-z --max-triangles 400000
./ngeo_build models/xyzrgb_dragon.ply web/models/dragon.ngeo --max-triangles 400000
gzip -9 -f web/models/lucy.ngeo web/models/dragon.ngeo
ls -l web/models
