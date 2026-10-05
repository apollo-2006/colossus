#!/usr/bin/env bash
# downloads the stanford scans into models/ and builds their hierarchies. kept
# out of the repository for size (ply files of 130 and 530 mb).
#
#     models/fetch.sh               # both
#     models/fetch.sh dragon        # the xyz rgb asian dragon, 7.2M triangles
#     models/fetch.sh lucy          # lucy, 28M triangles
#     models/fetch.sh washington    # textured: horatio greenough's george washington
#                                   # (1840), 17M triangles, a 720 mb download
#
# the dragon and lucy are from the stanford 3d scanning repository,
# http://graphics.stanford.edu/data/3Dscanrep/, which asks credit to the
# stanford computer graphics laboratory (and, for the dragon, xyz rgb inc.) and
# allows no commercial use without permission. washington is the smithsonian
# american art museum's scan, released cc0 through smithsonian open access
# (https://3d.si.edu). its texture is converted to ppm with ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
names=("$@")
[[ ${#names[@]} -eq 0 ]] && names=(dragon lucy)
build=../colossus_build
[[ -x $build ]] || { echo "build colossus_build first: make"; exit 1; }
# --download-only fetches without building (the pages workflow builds trimmed
# copies).
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
    washington)
      stem="washington/george-washington-greenough-statue-(1840)-master"
      if [[ ! -f $stem-geometry.obj ]]; then
        mkdir -p washington
        curl -fL -o washington/scan.zip "https://3d-api.si.edu/content/document/3d_package:789cf90a-4387-4ac1-9e96-c7d6a7b9d26f/resources/george-washington-greenough-statue-(1840)-full_resolution-obj.zip"
        (cd washington && unzip -o -q scan.zip && rm scan.zip)
      fi
      [[ -f $stem-texture.ppm ]] || ffmpeg -v error -y -i "$stem-texture.jpg" "$stem-texture.ppm"
      $download_only || $build "$stem-geometry.obj" washington.cgeo ;;
    *) echo "unknown model $name (dragon, lucy, washington)"; exit 2 ;;
  esac
done
