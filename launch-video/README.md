# Bhaav-E launch video

A 30-second launch film for Bhaav-E, built with [HyperFrames](https://hyperframes.heygen.com) (HTML + GSAP rendered
to MP4) and three.js for the 3D scenes.

- [`renders/bhaav-e-launch-16x9.mp4`](renders/bhaav-e-launch-16x9.mp4): 1920×1080, 30 s, for LinkedIn and X
- [`renders/bhaav-e-launch-1x1.mp4`](renders/bhaav-e-launch-1x1.mp4): 1080×1080, 30 s, for square feeds

Every sticker on screen is real Bhaav-E output (the released example samples), and every number comes from the
evaluation reports. The 3D scene plots the model's own rectified-flow path, `x_t = (1 − t)·sticker + t·noise`, with
the real 50-step sampler schedule (time shift 5).

## Scenes

| Time | Scene | |
|---|---|---|
| 0–4 s | `hook` | a sentence turns ink word by word; its last word becomes the sticker |
| 4–8.5 s | `reveal` | the sticker carries into a grid of all 8 stickers; title |
| 8.5–15.5 s | `how` (3D) | the sticker scatters into noise, then 50 sampler steps bring it back |
| 15.5–21 s | `range` (3D) | the camera flies along a wall of stickers |
| 21–26.5 s | `numbers` | 211.92M parameters · 85,816 clips · ≈ $12 · 3.3% unseen error |
| 26.5–30 s | `end` | mark, wordmark and call to action |

Full spec: [SPEC.md](SPEC.md).

## Build

Requires Node.js 22+ and FFmpeg.

```bash
./fetch_audio.sh                                   # music and effects (not stored in the repo)
npx hyperframes@0.8.223 preview                    # edit in the browser
npx hyperframes@0.8.223 render --quality delivery -o renders/bhaav-e-launch-16x9.mp4
./master_audio.sh renders/bhaav-e-launch-16x9.mp4  # -14 LUFS for social feeds
python3 make_square.py                             # writes ../launch-video-square for the 1:1 cut
```

## Credits

- Music: "Infected Mushroom Vibes" by Alejandro Magaña (A. M.), [Mixkit Stock Music Free License](https://mixkit.co/license/)
- Sound effects: Pixabay, under the Pixabay Content License (bundled with HyperFrames)
- Emoji art in the stickers: Microsoft Fluent Emoji (MIT)
