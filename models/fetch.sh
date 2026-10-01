#!/usr/bin/env bash
# Downloads the Stanford scans into models/ and builds their LOD hierarchies.
# They are kept out of the repository for their size (the PLY files are 130
# and 530 MB).
#
#   models/fetch.sh               # both
#   models/fetch.sh dragon        # the XYZ RGB Asian Dragon, 7.2M triangles
#   models/fetch.sh lucy          # Lucy, 28M triangles
#
# The models are from the Stanford 3D Scanning Repository,
# http://graphics.stanford.edu/data/3Dscanrep/, which asks for credit to the
# Stanford Computer Graphics Laboratory (and, for the dragon, XYZ RGB Inc.)
# and allows no commercial use without permission.
set -euo pipefail
cd "$(dirname "$0")"
names=("$@")
[[ ${#names[@]} -eq 0 ]] && names=(dragon lucy)
build=../colossus_build
[[ -x $build ]] || { echo "build colossus_build first: make"; exit 1; }
# --download-only fetches the scans without building (the web demo's
# workflow builds trimmed copies itself).
download_only=false
if [[ ${names[0]:-} == --download-only ]]; then download_only=true; names=("${names[@]:1}"); fi
[[ ${#names[@]} -eq 0 ]] && names=(dragon lucy)
for name in "${names[@]}"; do
  case $name in
    dragon)
      if [[ ! -f xyzrgb_dragon.ply ]]; then
        curl -fL -o xyzrgb_dragon.ply.gz http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz
        gunzip xyzrgb_dragon.ply.gz
      fi
      $download_only || $build xyzrgb_dragon.ply xyzrgb_dragon.cgeo ;;
    lucy)
      if [[ ! -f lucy.ply ]]; then
        curl -fL -o lucy.tar.gz http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz
        tar xzf lucy.tar.gz lucy.ply && rm lucy.tar.gz
      fi
      $download_only || $build lucy.ply lucy.cgeo --up-z ;;
    *) echo "unknown model $name (dragon, lucy)"; exit 2 ;;
  esac
done
