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
python3 docs/recompress.py docs/*.png
