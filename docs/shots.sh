#!/usr/bin/env bash
# renders the readme images into docs/ from models/fetch.sh's models, 60 frames
# of streaming each. the viewer writes uncompressed png; docs/recompress.py
# squeezes them.
set -euo pipefail
cd "$(dirname "$0")/.."
crowd=(--model models/lucy.cgeo --model models/xyzrgb_dragon.cgeo --grid 30 --spacing 1.25)
shot() { ./colossus "$@" --headless --warmup 60 --frames 4 --width 1600 --height 900; }
shot "${crowd[@]}" --camera -1.2,0.55,3.2,0.45,-0.12 --screenshot docs/crowd.png
shot "${crowd[@]}" --camera 3,4,16,-0.3,-0.35 --mode 3 --screenshot docs/lod_levels.png
shot --model models/lucy.cgeo --camera 0.32,0.78,-0.42,-2.46,-0.2 --mode 1 --screenshot docs/clusters.png
shot "${crowd[@]}" --camera 0.6,0.7,1.6,-0.5,-0.3 --mode 7 --screenshot docs/rasterizers.png
# one lucy in each material, cropped to her upper half and joined.
for m in marble sandstone bronze gold granite; do
  shot --model models/lucy.cgeo --materials $m --camera 0.32,0.78,-0.42,-2.46,-0.2 --screenshot /tmp/colossus_$m.png
done
python3 docs/strip.py docs/materials.png 610 150 380 620 /tmp/colossus_{marble,sandstone,bronze,gold,granite}.png
rm -f /tmp/colossus_{marble,sandstone,bronze,gold,granite}.png
# the textured scan, if fetched (models/fetch.sh washington).
if [[ -f models/washington.cgeo ]]; then
  shot --model models/washington.cgeo --camera 0.32,0.78,0.0,-1.5708,-0.05 --screenshot docs/washington.png
fi
# the skinned crowd, if fetched (models/fetch.sh fox).
if [[ -f models/fox.cgeo ]]; then
  shot --model models/fox.cgeo --grid 30 --spacing 1.2 --camera -1.2,0.45,3.2,0.45,-0.12 --screenshot docs/foxes.png
fi
python3 docs/recompress.py docs/*.png
# the web demo's fallback frames, when imagemagick or ffmpeg is around.
if command -v convert >/dev/null || command -v ffmpeg >/dev/null; then web/gallery.sh; fi
