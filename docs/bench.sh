#!/usr/bin/env bash
# the readme's performance numbers, measured: docs/bench.sh [--quick]
#
# each scene at 1920x1080, the median of 200 frames after the warmup (100, or 300 for the forest,
# while pages stream), as a markdown table: the frame and its parts, and triangles drawn. needs
# the models (models/fetch.sh lucy dragon washington fox; models/forest.py for the forest, which
# is skipped without it). run it on an idle machine: a busy cpu slows streaming, not the gpu
# timings, but warmups can end before pages settle. --quick takes 50 frames.
set -euo pipefail
cd "$(dirname "$0")/.."
frames=200
[[ ${1:-} == --quick ]] && frames=50
crowd=(--model models/lucy.cgeo --model models/xyzrgb_dragon.cgeo --spacing 1.25)
row() {
  local name=$1 warmup=$2
  shift 2
  local line
  line=$(./colossus "$@" --headless --warmup "$warmup" --frames "$frames" --width 1920 --height 1080 2>&1 | grep -E '^[0-9.]+ ms \(')
  # "1.48 ms (cull 0.17, raster 0.26, pass 2 0.10, shadow pages 0.08, shade 0.88) | 4.77M tris, ..."
  echo "$line" | sed -E 's/^([0-9.]+) ms \(cull ([0-9.]+), raster ([0-9.]+), pass 2 ([0-9.]+), shadow pages ([0-9.]+), shade ([0-9.]+)\) \| ([0-9.]+[kMB]?) tris.*/| '"$name"' | \1 ms | \2 | \3 | \4 | \5 | \6 | \7 |/'
}
echo "| scene | frame | culling | raster | pass 2 | shadow pages | shading | triangles |"
echo "|---|---|---|---|---|---|---|---|"
row "900 statues, beside lucy" 100 "${crowd[@]}" --grid 30 --camera -1.2,0.55,3.2,0.45,-0.12
row "900 statues, raised" 100 "${crowd[@]}" --grid 30 --camera 3,4,16,-0.3,-0.35
row "900 statues, ground level" 100 "${crowd[@]}" --grid 30 --camera 0,0.3,8,0,0
row "a million statues" 100 "${crowd[@]}" --grid 1000 --camera -1.2,0.55,3.2,0.45,-0.12
row "a million, 1% moving" 100 "${crowd[@]}" --grid 1000 --moving 0.01 --camera -1.2,0.55,3.2,0.45,-0.12
row "washington, close" 100 --model models/washington.cgeo --camera 0.32,0.78,0.0,-1.5708,-0.05
row "900 skinned foxes" 100 --model models/fox.cgeo --grid 30 --spacing 0.9 --camera -1.2,0.45,3.2,0.45,-0.12
if [[ -f models/forest.scene ]]; then
  row "the forest" 300 --scene models/forest.scene --camera 0,1.7,6,0,0.08
fi
