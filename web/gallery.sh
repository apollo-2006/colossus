#!/usr/bin/env bash
# the fallback page's renders (web/gallery/): docs/*.png from the native viewer, as jpegs a
# phone can load. needs imagemagick or ffmpeg; run after docs/shots.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p web/gallery
for n in crowd clusters lod_levels rasterizers materials washington foxes; do
  if command -v convert >/dev/null; then
    convert docs/$n.png -resize '1280x>' -strip -quality 82 -sampling-factor 4:2:0 -interlace JPEG web/gallery/$n.jpg
  else
    ffmpeg -loglevel error -y -i docs/$n.png -vf "scale='min(1280,iw)':-2" -q:v 4 web/gallery/$n.jpg
  fi
done
ls -l web/gallery
