#!/usr/bin/env bash
# the fallback page's renders (web/gallery/): docs/*.png from the native viewer, as jpegs a
# phone can load. needs imagemagick; run after docs/shots.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/gallery
for n in crowd clusters lod_levels rasterizers materials; do
  convert docs/$n.png -resize '1280x>' -strip -quality 82 -sampling-factor 4:2:0 -interlace JPEG web/gallery/$n.jpg
done
ls -l web/gallery
