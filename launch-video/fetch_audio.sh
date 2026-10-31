#!/bin/bash
# Download the film's music and sound effects into assets/audio/.  (OUR script.)
# They are not stored in the repo: their licences cover use inside the video, not republishing the files on their own.
#   Music: "Infected Mushroom Vibes" by Alejandro Magaña (A. M.), Mixkit Stock Music Free License (https://mixkit.co/license/)
#   Effects: HyperFrames' bundled set, from Pixabay under the Pixabay Content License
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets/audio
curl -sfL -o assets/audio/music-infected-mushroom-vibes.mp3 https://assets.mixkit.co/music/136/136.mp3
SFX=https://raw.githubusercontent.com/heygen-com/hyperframes/main/skills/media-use/audio/assets/sfx
for f in whoosh whoosh-cinematic pop impact-bass-1 sparkle; do
  curl -sfL -o "assets/audio/sfx-$f.mp3" "$SFX/$f.mp3"
done
ls -la assets/audio
