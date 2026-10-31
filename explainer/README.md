# Bhaav-E Explainer

An interactive page that shows how Bhaav-E turns a sentence into a 2-second sticker: words → numbers (Flan-T5),
50 refining steps (VideoDiT), summary → pixels (Wan2.2 VAE). The design follows
[Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/) (Georgia Tech and IBM Research, MIT): no model
runs in the browser. Every picture is real Bhaav-E output, computed in advance and played back.

```bash
python -m http.server 8000 --directory explainer   # then open http://localhost:8000
```

## What is precomputed

8 prompts (4 pairs that differ in one phrase) × seeds 0, 1 × guidance 0, 1, 3, 6 = 64 runs, 50 steps each.

| File | Content |
|---|---|
| `assets/runs/<run>.png` | 16 px tiles, one row per step: the latent the sampler holds, the model's guess `x − t·v`, and where the caption changed the prediction (`|v_caption − v_empty|`) |
| `assets/runs/<run>.jpg` | the guess fully decoded by the Wan2.2 VAE at steps 5, 10, 20, 30 and 50, 16 frames per row at 160 px |
| `assets/train.png` | one clip mixed with noise at t = 0, 0.1, …, 1 (training section) |
| `data.json` | prompts, T5 word pieces and a 5 × 5 corner of their vectors, step schedule, per-run caption impact and 2-D paths |

The **quick look** (every latent shown as colour) is a linear map from the 48 latent channels to RGB, fitted by least
squares against decoded frames shrunk to 16 × 16. Its R² is printed by `build.py` and shown on the page.
The **path chart** is a PCA of one pair's 102 latents (2 runs × 51 steps).

## Rebuild

```bash
cd explainer/build
HF_HUB_OFFLINE=1 ../../.venv/bin/python precompute.py --device mps   # ~1 min per run on an M1 Pro; resumable
../../.venv/bin/python build.py                                     # writes ../assets and ../data.json
```

`precompute.py` keeps the raw steps in `build/raw/` (about 1.4 GB, not committed; the built `assets/` is about 39 MB).
