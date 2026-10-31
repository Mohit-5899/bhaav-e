# Bhaav-E: project log

**Bhaav** (भाव) means expression or feeling. That is what a reaction sticker carries.

This is **our own model** (Judgement and our code throughout). It follows a standard small video-diffusion recipe,
but trains only on stickers that our own code makes.

## What it does

| Mode | Input | Output |
|---|---|---|
| Text → sticker | "a crying emoji bouncing, tears falling, blue background" | 2 s clip |
| Image → sticker | a still picture (becomes frame 1) + text | 2 s clip that animates it |

Output format (v1): **256×256, 17 frames, 8 fps, 2 s, loops** (frame 17 = frame 1).

## Why our data is made by code

Web video comes with noisy captions, which makes prompt-following hard to learn and hard to measure.
Our clips come from code, so every caption is **exactly true** and we can score prompt-following automatically.

## Folders

```
bhaav-e/
  docs/project-log.md  this file (the step-by-step build log)
  generator/       our code that renders clips (Python / HyperFrames)
  data/            all clips, captions, labels, reviews, splits — see data/README.md
```

## Status

- [x] Name and folder layout
- [x] Check emoji art licences: Fluent Emoji is MIT (licence saved in `generator/assets/fluent-emoji/`)
- [x] Generate **1 clip**: `reactions-crying-000001` (bounce + tears), approved
- [x] Generator v0.3: 10 emotions (14 art files), 6 motions, 8 effects, 12 backgrounds
- [x] Batches b0001–b0011: **1,101 reaction clips** (b0002–b0011 run 8 at a time, ~50 s for 1,000)
- [x] Batches b0012–b0100: **10,001 reaction clips** in total (8,900 more in 5 min), 519 MB, all checks pass
- [x] Generator v0.4: **animated-emoji** category (10 objects), new motions spin / heartbeat / flicker,
      new effects confetti / embers / exhaust. Reactions output checked unchanged (pixel diff 0)
- [x] **50,001 clips**: 25,001 reactions (b0000–b0250) + 25,000 animated emoji (e0001–e0250), 2.4 GB,
      22 min on 8 processes. All checks pass. 3 byte-identical clips listed in `data/splits/exclude.txt`
- [x] Generator v0.5: **ui** category drawn entirely in code (no art files): spinner, dots-loader, success,
      error, like, bell, download, progress, toggle, typing; 7 accent colours. Older categories checked unchanged
- [x] Caption fix: "a angry" → "an angry" (2,563 `.txt` files rewritten from the generator, videos untouched)
- [x] **75,001 clips**: 25,001 reactions + 25,000 animated emoji + 25,000 ui (u0001–u0250), 3.1 GB.
      All checks pass. 13 byte-identical copies listed in `data/splits/exclude.txt`
- [x] Splits: train 61,569 / val 3,750 / test_seen 3,750 / test_unseen 5,918 (held-out combinations),
      14 excluded. No overlaps, no held-out combination in train (see `data/README.md`)
- [x] GPU data step on RunPod (RTX 5090): VAE test passed (38.5 dB), latent cache (74,987 clips, 9.2 GB) and caption
      cache (26,066 sentences) built and verified; pod stopped. About $2.40. Details in `docs/gpu-plan.md`
- [x] Training r1: our own VideoDiT code (211.92M), 40,000 steps, val 0.0854 → 0.0551, ≈ $2.60. Results in `runs/r1/`, details in `docs/gpu-plan.md`
- [x] Evaluation (v2 checker): subject, background and accent carry over to unseen combinations; the model's own error rate is 19% unseen vs 10.5% seen, mostly a leaked 'usual' effect. Loss gap +8%. See `runs/r1/eval/RESULTS.md`
- [x] Free improvements tested: CFG 3 beats 5 (unseen errors 10.9% → 7.3%), CFG-Zero* and autoguidance no gain; motion checker v3 shows motion error ≈ 0%. Next: r2 with rebalanced data
- [x] r2 (rebalanced effects, same size and recipe, ≈ $3.85): unseen errors at CFG 5 halved (10.9% → 5.7%, p = 0.01); at CFG 3 unseen 7.3% → 5.3% but seen 6.1% → 9.0% (both not significant). Eye check shows the effect checker misreads many generated stickers. Keep r2's data; fix the effect checker next. See `runs/r2/RESULTS.md`
- [x] Effect checker v4 (particles, not CLIP; ≈ $1.25): most "missing effect" errors were the old checker. With v4: seen 2.0% for both runs; unseen r1 5.3% vs r2 3.3%. The remaining real failure is adding a subject's usual effect, which r2 cuts from 9 clips to 3. **r2 is the release model.** See `runs/r2/RESULTS.md`, v4 section
- [ ] Still empty: greetings, text, brand (need a text plan), mascot (needs a character)
- [x] VAE latent cache, then training (done 2026-10-29)
