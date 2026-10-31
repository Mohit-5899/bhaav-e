# Bhaav-E launch film: spec

30.0 s · 30 fps · 1920×1080 and 1080×1080 · silent (feeds autoplay muted; type and pictures carry it)

The one thing: **a 212M model, trained from scratch for about $12, turns a sentence into an animated sticker.**
Everything shown as output is real Bhaav-E output (the 8 released example stickers). Numbers come from the evaluation
reports.

## Palette and type

- Ink #10243f · brand orange #e8590c · white #ffffff
- Worlds use the stickers' own backgrounds: light blue #cfe7ff, navy #1d2843, peach #fde0c6
- Inter 900 for titles, Inter 400/700 for lines; JetBrains Mono only for the one equation
- Titles 6–9% of frame height; quiet lines 2.6–3.4%

## Beats

| Time | Scene | World | What happens |
|---|---|---|---|
| 0.0–4.0 | hook | light blue | the sentence turns ink word by word; its last word becomes the sticker, which plays |
| 4.0–8.5 | reveal | flood to orange | the sticker carries into a grid of all 8 stickers; title "Bhaav-E" per letter |
| 8.5–15.5 | how (3D) | navy | particles = the sticker; scatter to noise, then 50 sampler steps back: x_t = (1−t)·x + t·ε |
| 15.5–21.0 | range (3D) | peach | camera flies along a tilted wall of the 8 stickers; range and generalization lines |
| 21.0–26.5 | numbers | white | 211.92M · 85,816 clips · ≈ $12 · 3.3% unseen error, each its own entrance |
| 26.5–30.0 | end | flood to orange | sticker mark springs in, wordmark wipes in, Colab call to action holds 2+ s |

## Motion

- Arrivals `expo.out` / `power4.out`; floods and camera `power2.inOut`; exits `power3.in`, 40–50% shorter
- Transitions: match cut (hook→reveal), colour flood (reveal, end), zoom through a tile (reveal→how),
  blur swap (numbers), dip (range→numbers)
- Stickers play at their real 8 fps from timeline time (sprite sheets, frame = floor(t·8) mod 16)

BANNED: dark void with small floating things, glow, everything fading and sliding up, three cards in a row,
docked logo, breathing/floating loops on titles, invented UI or numbers.
