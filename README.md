<div align="center">

# Bhaav-E

**A compact text-to-video model for animated stickers**

[![Project Page](https://img.shields.io/badge/Project-Page-e8590c)](https://huggingface.co/spaces/Mohit5899/bhaav-e)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Bhaav--E-ffcc4d)](https://huggingface.co/Mohit5899/bhaav-e)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mohit-5899/bhaav-e/blob/main/bhaav_e_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

| | | | |
|:---:|:---:|:---:|:---:|
| ![](docs/examples/00_seed0.gif) | ![](docs/examples/01_seed0.gif) | ![](docs/examples/02_seed0.gif) | ![](docs/examples/03_seed0.gif) |
| *a crying emoji bouncing, tears falling, light blue background* | *a red furious emoji pulsing, steam puffing from its head, dark navy background* | *a blue success checkmark drawing itself inside a circle, confetti bursting out, white background* | *a party popper emoji pulsing, sparkles twinkling around it, light grey background* |
| ![](docs/examples/04_seed0.gif) | ![](docs/examples/05_seed0.gif) | ![](docs/examples/06_seed0.gif) | ![](docs/examples/07_seed0.gif) |
| *a cool emoji with sunglasses floating gently, sparkles twinkling around it, peach background* | *a glowing star emoji wobbling side to side, sparkles twinkling around it, dark navy background* | *a purple loading spinner spinning round and round, pulsing, black background* | *a screaming in shock emoji wobbling side to side, peach background* |

<sub>Uncurated samples from Bhaav-E (seed 0, 50 steps, guidance 3.0): the first and only sample generated for each prompt.</sub>

## News

- **2026-10-31** · Interactive web demo with a prompt builder, runnable on a free GPU through the
  [Colab notebook](https://colab.research.google.com/github/Mohit-5899/bhaav-e/blob/main/bhaav_e_demo.ipynb).
- **2026-10-30** · Initial release of the Bhaav-E weights, training code, evaluation suite and data generator.

## Overview

Bhaav-E generates 2-second looping stickers (256 × 256, 17 frames, 8 fps) from a text prompt. *Bhaav* (भाव) means
feeling; the *E* stands for expressions.

The model is a 211.92M-parameter video diffusion transformer, trained from scratch with rectified flow on the
latent space of the frozen Wan2.2 video VAE and conditioned on a frozen Flan-T5-Base text encoder. The whole training
set (85,816 clips) is rendered procedurally, so every caption describes its clip exactly. This makes prompt
adherence directly measurable, including on combinations of attributes that never appear together in training.

**Highlights**

- **Small and inexpensive:** one training run takes about 3.6 hours on a single 32 GB GPU (≈ $2.60). All experiments
  in this repository, including two training runs and five evaluations, cost ≈ $12.30 in total.
- **Compositional generalization:** 3.3% attribute error on prompts whose combinations were held out of training,
  against 2.0% on combinations seen during training.
- **Fully reproducible:** the data generator, training, evaluation and export code are all included, along with
  training logs and evaluation reports.

## Model Zoo

| Model | File | Parameters | Training data | Notes |
|---|---|---|---|---|
| **Bhaav-E** | [`bhaav_e.safetensors`](https://huggingface.co/Mohit5899/bhaav-e) | 211.92M | effects balanced per subject (r2) | Recommended |
| Bhaav-E r1 | [`r1/bhaav_e.safetensors`](https://huggingface.co/Mohit5899/bhaav-e/tree/main/r1) | 211.92M | natural attribute distribution | Baseline |

Both checkpoints are EMA weights at step 40,000, stored as fp32 safetensors.

## Quick Start

### Option 1: Colab (no installation)

Open the [Colab notebook](https://colab.research.google.com/github/Mohit-5899/bhaav-e/blob/main/bhaav_e_demo.ipynb),
select **Runtime → Run all**, and open the `gradio.live` link printed by the last cell. The notebook launches the full
web demo on a free T4 GPU.

### Option 2: Run locally

```bash
git clone https://github.com/Mohit-5899/bhaav-e && cd bhaav-e
pip install torch diffusers transformers sentencepiece safetensors pillow numpy huggingface_hub gradio
```

**Web demo** (CPU or GPU). Add `--share` for a temporary public link.

```bash
cd space && python app.py
```

**Command line** (CUDA GPU):

```bash
hf download Mohit5899/bhaav-e --local-dir weights
cd model && python sample.py --ckpt ../weights/bhaav_e.safetensors --cache ../weights \
    --prompt "a crying emoji bouncing, tears falling, light blue background" --out ../samples
```

**Python** (CPU or GPU). Weights are downloaded automatically on first use.

```python
import sys; sys.path.insert(0, "space")
from bhaav import Bhaav, save_gif

pipe = Bhaav()
frames = pipe.generate("a crying emoji bouncing, tears falling, light blue background",
                       steps=50, cfg=3.0, seed=0)          # uint8 array [16, 256, 256, 3]
save_gif(frames, "sticker.gif")
```

### Prompt format

The model follows the structure of its training captions: **subject, motion, effect, background**.

| Attribute | Supported values |
|---|---|
| Subject (30) | 10 emotion faces (laugh, crying, heart eyes, facepalm, angry, shocked, wink, thinking, sleepy, cool) · 10 animated emoji (fire, heart, party popper, thumbs up, star, 100, clapping hands, rocket, trophy, light bulb) · 10 UI animations (spinner, dots loader, success, error, like, bell, download, progress, toggle, typing) |
| Motion (9) | bounce, wobble, pulse, float, shake, still, spin, heartbeat, flicker |
| Effect (12) | tears, tears of joy, hearts, sparkles, sweat, steam, Zzz, thought bubbles, confetti, sparks, smoke trail, none |
| Background (12) | light blue, pink, mint green, lavender, peach, pale yellow, white, light grey, sky blue, teal, dark navy, black |

The web demo's prompt builder lists only the combinations available for each subject.

## Model Architecture

```
 prompt ──► Flan-T5-Base encoder (frozen) ──► 35 × 768 token embeddings ──┐
                                                                          │ cross-attention
 noise [48 × 5 × 16 × 16] ──► VideoDiT, 211.92M parameters (trained) ◄────┘
                                      │ 50 Euler steps · CFG 3.0 · time shift 5
                                      ▼
 latent [48 × 5 × 16 × 16] ──► Wan2.2 VAE decoder (frozen) ──► 17 frames × 256 × 256 RGB
```

| Component | Specification |
|---|---|
| Denoiser | Diffusion Transformer: 16 blocks, width 768, 12 attention heads, 2 × 2 patches (320 tokens) |
| Conditioning | timestep via adaptive LayerNorm; text via cross-attention; optional clean first frame (image-to-video) |
| Objective | rectified flow: `x_t = (1 − t)·x₀ + t·ε`, predict `v = ε − x₀`, MSE loss |
| Text encoder | [Flan-T5-Base](https://huggingface.co/google/flan-t5-base), frozen, max 35 tokens |
| Autoencoder | [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), frozen, 48 latent channels, 4× temporal and 16× spatial compression |
| Sampler | 50 Euler steps, classifier-free guidance 3.0, time shift 5.0 |

## Training

### Data

All clips are rendered by [`generator/bhaav_gen.py`](generator/bhaav_gen.py). Each caption is written from the same
specification that produced the clip.

| | |
|---|---|
| Clips | 85,816 rendered (75,001 initial + 10,815 added to balance effects) |
| Splits (released model) | train 61,571 · validation 3,750 · test (seen) 3,750 · test (unseen) 5,918 |
| Held-out design | the unseen test set contains subject–effect–motion combinations that never occur in training |
| Artwork | [Microsoft Fluent Emoji 3D](https://github.com/microsoft/fluentui-emoji) (MIT); UI animations are drawn in code |

The released model's training split gives every effect the same number of clips within each subject
([`generator/make_r2_split.py`](generator/make_r2_split.py)). The baseline r1 keeps the natural, skewed distribution.

### Recipe

| | |
|---|---|
| Optimizer | Muon (lr 0.02) for 2-D weight matrices in the transformer blocks; AdamW (lr 2e-4) for all other parameters |
| Schedule | 1,000 warmup steps, constant, linear decay over the final 25% |
| Batch size × steps | 64 × 40,000 (≈ 41.6 epochs) |
| Regularization | 10% caption dropout (for classifier-free guidance), EMA 0.9999, gradient clipping 1.0 |
| Precision | bf16 autocast, `torch.compile` |
| Compute | 1 × NVIDIA RTX PRO 4500 (32 GB), ≈ 3.6 hours |

VAE latents and text embeddings are computed once and cached, so training never decodes pixels.

## Evaluation

Each model generates one sticker for 256 prompts with seen combinations and 256 prompts with unseen combinations.
Automatic checkers read five attributes from every generated clip: subject, motion, effect, background and UI accent.
A clip is counted as a model error only if the checker reads the corresponding real clip correctly, so checker
mistakes are not attributed to the model.

**Error rate by attribute (guidance 3.0)**

| Model | Split | Subject | Motion | Effect | Background | Accent | **Any** |
|---|---|---|---|---|---|---|---|
| Bhaav-E | seen | 0.4% | 0.0% | 1.2% | 0.0% | 0.9% | **2.0%** |
| Bhaav-E | unseen | 0.8% | 0.0% | 2.5% | 0.0% | 0.0% | **3.3%** |
| Bhaav-E r1 | seen | 0.0% | 1.4% | 1.2% | 0.0% | 0.0% | **2.0%** |
| Bhaav-E r1 | unseen | 0.4% | 0.0% | 4.9% | 0.0% | 0.0% | **5.3%** |

**Findings**

- Subject, motion, background and accent transfer to unseen combinations with at most 1% error.
- The dominant failure mode is *effect leakage*: generating the effect most associated with a subject in training
  when no effect is requested (for example, tears of joy on a laughing face). Balancing effects per subject reduced
  these cases from 9 to 3 of 243 unseen prompts.
- The unseen difference between the two models is not statistically significant (exact sign test, p = 0.23; one
  training seed per model).

Full reports: [r2 vs r1 comparison](runs/r2/RESULTS.md) · [r1 evaluation history](runs/r1/eval/RESULTS.md).

## Inference Performance

| Hardware | Steps | Time per sticker |
|---|---|---|
| Apple M1 Pro, CPU (10 threads) | 50 | ≈ 43 s |
| Apple M1 Pro, CPU (2 threads) | 20 | ≈ 48 s |
| NVIDIA L4 (24 GB) | 50 | tested; not benchmarked |

On CPU, most of the time is spent in the VAE decoder (≈ 41 s with 2 threads); each denoising step takes ≈ 0.3 s.
Decoding eight clips in one batch exceeded the 24 GB of an L4, so the provided scripts decode one clip at a time.

## Limitations

- **Closed vocabulary:** the model only renders the 30 subjects, 12 effects and 12 backgrounds it was trained on.
- **Low resolution and short duration:** 256 × 256, 2 seconds.
- **Effect leakage:** an unrequested effect is occasionally added (see Evaluation).
- **Evaluation scale:** one seed per model and about 250 prompts per split; differences of 1–2 points are within noise.
  The automatic checkers are themselves 95–97% accurate on real clips.
- **Image-to-video:** the model was trained with an optional clean first frame, but the provided samplers do not
  expose this mode yet.

## Repository Structure

```
generator/            procedural sticker renderer and split builders
data/                 dataset splits and the rebalancing plan (clips are regenerated from seeds)
model/                VideoDiT, training, sampling, evaluation and attribute checkers
gpu/                  GPU scripts: latent and text caching, training, evaluation, release export
runs/                 training logs, sample snapshots and evaluation reports for r1 and r2
space/                Gradio web demo
showcase/             source of the project page
bhaav_e_demo.ipynb    Colab notebook
docs/                 technical notes: training explained, GPU plan and costs, improvement plan, project log
```

The full pipeline (generate data → cache latents → train → evaluate) is documented step by step in
[docs/project-log.md](docs/project-log.md), with pod commands and costs in [docs/gpu-plan.md](docs/gpu-plan.md).
[docs/how-training-works.md](docs/how-training-works.md) explains the training process from first principles.

## Citation

```bibtex
@misc{bhaave2026,
  title        = {Bhaav-E: A Compact Text-to-Video Model for Animated Stickers},
  author       = {Mohit},
  year         = {2026},
  howpublished = {\url{https://github.com/Mohit-5899/bhaav-e}}
}
```

## License

The code and model weights are released under the [MIT License](LICENSE). The frozen components keep their own
licenses: the Wan2.2 VAE and Flan-T5-Base are Apache-2.0.

## Acknowledgements

Bhaav-E builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) (video VAE), [Flan-T5](https://huggingface.co/google/flan-t5-base)
(text encoder), [Microsoft Fluent Emoji](https://github.com/microsoft/fluentui-emoji) (training artwork), Diffusion
Transformers (Peebles & Xie, 2023), Rectified Flow (Liu et al., 2022) and the Muon optimizer (Jordan et al., 2024).

Special thanks to [Udgam Labs](https://udgamlabs.com) for openly publishing the code, training logs and cost reports of their small models; their published recipe and model size served as the reference for this work.
