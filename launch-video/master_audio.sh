#!/bin/bash
# Master a rendered MP4's audio to -14 LUFS / -1.5 dBTP for social feeds (two-pass loudnorm); video is copied as is.
# HyperFrames lowers the whole mix when clip gains go above 1, so renders come out around -21 LUFS.  (OUR script.)
# Usage: ./master_audio.sh renders/bhaav-e-launch-16x9.mp4
set -euo pipefail
IN="$1"
OUT="${IN%.mp4}.mastered.mp4"
J=$(ffmpeg -hide_banner -i "$IN" -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
ARGS=$(echo "$J" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true\")")
ffmpeg -v error -y -i "$IN" -c:v copy -af "$ARGS" -c:a aac -b:a 192k -ar 48000 "$OUT"
mv "$OUT" "$IN"
echo "mastered $IN"
