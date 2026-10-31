# Bhaav-E: how we would add sound

Written 2026-10-31. A design note, not a finished run: nothing here has been trained yet.

We study two models that already make video and sound together, then decide how to add sound to Bhaav-E.

> References: **MiniMax H3** (open weights, Aug 2026): [announcement](https://www.minimax.io/news/minimax-h3-open-source),
> [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), [GitHub](https://github.com/MiniMax-AI/MiniMax-H3).
> **Kandinsky 6.0 Video** (Oct 2026): [paper](https://arxiv.org/abs/2610.05608), plus
> [Kandinsky S2V 1.0](https://arxiv.org/html/2610.19290v1) for one extra idea.
> **Kandinsky 5.0 has no audio** ([paper](https://arxiv.org/pdf/2511.14993)); 6.0 adds it on top of 5.0.

**How the facts were gathered.** Numbers about H3 and Kandinsky were read by subagents from the pages above and are
**Reported** (not re-checked by us). On 2026-10-31 a second pass checked the Kandinsky 6.0 and S2V numbers
against the papers' verbatim text (arXiv HTML, §2–§4 and Tables 2–3); the section or table is given with each.
The Kandinsky 5.0 PDF could not be fetched, so its VAE name comes from the 6.0 paper. Everything about Bhaav-E is
**Verified** (read in our code or README). Option designs and estimates are **Judgement**.

## 1. The analogy (one minute)

Today Bhaav-E is a silent animator: it draws 17 frames from a caption. To add sound we need two new things. First,
a way to squeeze sound into a few numbers, the same way the Wan2.2 VAE squeezes pictures. Second, a generator that
writes the picture numbers and the sound numbers *at the same time*, so the "pop" lands on the frame where the
heart pops. MiniMax H3 does this like **one band reading one shared score**: video and sound are written into a
single sequence. Kandinsky 6.0 does it like **a cameraman and a sound engineer passing notes** after every step:
two separate models that talk to each other.

## 2. The problem, from first principles

**Why not just generate the waveform?** One second of sound at 32 kHz is 32,000 numbers. Our whole video latent is
48 × 5 × 16 × 16 = 61,440 numbers for 2 s. Raw audio would be the same size as the video and full of detail that
doesn't matter (exact wave shape). So, same as video: **compress first**. The compressor is an **audio VAE**:
an encoder turns sound into a short list of "audio tokens", a decoder turns them back into sound.

**Why generate sound and video together?** If you make the video first and add sound after, the sound can follow the
video but never shape it. If you make them together, each can see the other while being made, so timing can match
in both directions. The cost is a bigger model and data that has both.

**Tiny worked example with our numbers.** Our clip loops every 2.0 s (17 frames at 8 fps, frame 17 = frame 1).
An audio VAE at 40 tokens per second (H3's rate) gives 2.0 × 40 = **80 audio tokens**. Our video is
**320 tokens** (5 × 8 × 8, Verified: `model/videodit.py`). So sound adds 80 / 320 = **25%** more tokens. Each
video latent frame covers about 0.5 s, which is about 20 audio tokens.

## 3. Reference A: MiniMax H3, one shared stream

```
 text ──► Qwen3-VL-32B, layer-50 hidden states (borrowed) ───────────────┐ conditioning
 video ─► H3-VisualVAE ──► video tokens ─┐                                │
                                         ├─► ONE sequence ──► Omni-Transformer, 33B, 50 layers (trained)
 audio ─► H3-AudioVAE  ──► audio tokens ─┘     shared attention + FFN;
                                               per-modality: input/output layers and AdaLN only
                                                     │
                       video latents ──► video decoder ──► 24 fps video, 768p
                       audio latents ──► BigVGAN-style decoder ──► 32 kHz stereo
```

| Part | Value | Source |
|---|---|---|
| Transformer | 33B, "dense, single-stream", about 13B of it in AdaLN branches | Reported, model card |
| Shape | 50 layers, hidden 5376, 56 heads × 128, FFN 14336 | Reported, `transformer/config.json` |
| Modality-specific parts | input/output layers and AdaLN; "neither the attention layers nor the FFN layers contain modality-specific structures" | Reported, model card |
| Text encoder | Qwen3-VL-32B, hidden states of layer 50 | Reported, model card |
| Audio VAE | 32 kHz stereo, each channel encoded separately, 32 latent channels, 40 Hz | Reported, `audio_vae/config.json` |
| Audio VAE design | DAC-family encoder, BigVGAN decoder; strides 2·4·4·5·5 = 800, so 32,000 / 800 = 40 Hz | Reported; product **Verified** (computed) |
| Position code | 3D multimodal RoPE over (t, h, w); how audio gets its t is not stated | Reported |
| Noise-schedule shift | video 12.0, audio 3.0 (`sigma_shift_scales`); why is not stated | Reported, `model_index.json` |
| Objective, data, steps | **not stated** | — |
| Inference | CFG-distilled weights; 4–15 s at 24 fps; step count not stated | Reported |
| Licence | custom; excludes EU, UK, Korea, USA; forbids using outputs to improve other models | Reported, `LICENSE` |

**What H3 shows.** A single shared transformer can carry both modalities if only the edges (input/output) and the
noise-level conditioning (AdaLN) are separate. **What it does not show:** no training recipe, no ablation against
two streams, and the published code is inference only.

## 4. Reference B: Kandinsky 6.0 Video, two streams

```
 text ──► Qwen2.5-VL-7B + CLIP ViT-L/14 (borrowed) ──┬────────────────────────────┐
                                                     ▼                            ▼
 video stream (initialised from Kandinsky 5.0) ◄── two-way cross-attention ──► audio stream (from scratch)
                                                                                  │
 video latents ──► HunyuanVideo-family VAE decoder             audio latents ──► MMAudio autoencoder, 44 kHz
```

| Part | Lite | Pro | Source |
|---|---|---|---|
| Total | 3B | 29B | Reported, abstract |
| Video stream | 2B | 19B | Reported, §3.2 |
| Audio stream | 0.6B | 5B | Reported, §3.2 |
| Cross-attention between streams | 0.4B | 5B | Reported, §3.2 |
| Depth | 32 blocks in **both** streams | 60 in both | Reported, Table 2 ("same number of transformer blocks") |
| Width audio / video | 896 / 1792 | 2048 / 4096 | Reported, Table 2 |
| FFN audio / video | 3584 / 7168 | 7168 / 16384 | Reported, Table 2 |
| Cross-attention placement | every block, both directions (video→audio and audio→video) | same | Reported, Fig. 1 captions (no explicit "every block" sentence) |
| Audio autoencoder | MMAudio's, 44 kHz; frozen or retrained, latent rate, channels and mono/stereo **not stated** | same | Reported, §3.1 |
| Position codes | video: 3D RoPE, frame index scaled to 24 fps; audio: 1D RoPE over its own time position; how the two map onto each other **not stated** | same | Reported, §3.2, Fig. 2 |
| Text to audio stream | its own text cross-attention and timestep embedding; text encoders Qwen2.5-VL-7B (1024 tokens) + CLIP ViT-L/14 | same | Reported, Fig. 1–2, Table 2 |
| Prompt format | speech inside `<S>…<E>`, sound description inside `<AUDCAP>…<ENDAUDCAP>` | same | Reported, figure caption |

Note that "asymmetric" means the audio stream is **narrower**, not shallower: same depth, half the width.

**Training order** (Reported, §2.4–2.5 and §4.2–4.5):

1. Audio stream alone, from scratch, on **40M audio tracks** (5–20 s): 40k steps (Lite), 30k (Pro), LR 1e-4.
2. Video stream starts from the Kandinsky 5.0 checkpoint (SD resolution, 45k steps).
3. Joint training on **7M audio-video clips**, LR 5e-5. Clips with faces need a lip-sync score ≥ 3 of 5; clips
   with sound events need desync < 0.1. Lite: 57.5k steps at batch 256. Pro: 46k steps, then 3k at 2.5e-5.
   Whether the video stream is frozen here is **not stated**.
4. Fine-tuning in two stages. Stage 2 weights the loss **video 0.85, audio 0.15** (face loss 1.0).
5. RL with GRPO (10 rollouts per prompt for Pro, 8 for Lite; LoRA r 16, α 32).
6. Distillation to 10 steps (π-Flow 44k iterations, then 300 adversarial steps).

Objective: flow matching with MSE (§4.2). Optimiser is not named; betas are (0.9, 0.95) for video and joint stages and
(0.9, 0.999) for audio-only, weight decay 1e-3, EMA 0.999 (Table 3). Main-model hardware: **not stated**.

Two extra details matter for us:
- **Independent noise levels:** there is an `independent_time` mode where the noise level *t* is drawn separately for
  video and audio (§4.2.2).
- **Damping the first blocks:** gradients through the video→audio cross-attention are scaled down early: 0.0 at
  block 0, 0.1 at blocks 1–10, 0.3 at blocks 50–59 for Pro (§4.4). This keeps the new audio path from disturbing
  the trained video stream.

**Results** (VABench, Reported verbatim). Columns are LTX 2.5 / K6 Lite / K6 Pro:

| Metric | LTX 2.5 | K6 Lite | K6 Pro |
|---|---|---|---|
| lip-sync (higher is better) | 1.567 | 1.761 | 2.072 |
| desync (lower is better) | 0.521 | 0.474 | 0.396 |
| word error rate (lower is better) | 0.099 | 0.103 | 0.117 |

Note that Pro is better on sync but *worse* on word error rate than both others.

**A third pattern, from Kandinsky S2V** (audio → video, Reported, S2V §3.1, §3.5, Table 2): speech at 16 kHz goes
through the frozen Qwen2.5-Omni audio tower. Its features are added into the video blocks through cross-attention
in 13 of 32 blocks (2B model) and 18 of 60 (21B model), so about every third block. The video tokens of each
latent frame attend **only to the audio slice of their own time window** (a block-diagonal mask). This is a hard
rule for alignment instead of hoping attention learns it.

## 5. Bhaav-E today (Verified, `model/videodit.py` and README)

- 16 blocks, width 768, 12 heads, 211.92M parameters, 320 tokens.
- Positions use **added sine/cosine codes**, not RoPE: a quarter of the 768 numbers code the latent time index
  (0..4), the rest code height and width.
- One AdaLN layer per block: `Linear(768, 6 × 768)` = **3,543,552 parameters per block, 56.7M over 16 blocks**
  (Verified, computed from layer sizes). That is 27% of the model.
- Rectified flow, 50 Euler steps, CFG 3.0, time shift 5.0.
- Training: 64 × 40,000 steps, about 3.6 h on one 32 GB GPU, about $2.60.

**The real gap is data, not architecture.** Our 85,816 clips are silent.

## 6. Our options

| | A. Single stream (H3 style) | B. Two streams (K6 style) | C. Sound afterwards |
|---|---|---|---|
| What changes | Join 320 video and 80 audio tokens in one sequence; add audio in/out layers; give the audio tokens a modality tag | Keep VideoDiT; add a small audio DiT with two-way cross-attention | A separate model or rule adds sound to the finished clip |
| Reuses trained Bhaav-E | Partly (can start from its weights; the sequence changes) | Yes, fully | Yes, untouched |
| Extra parameters | about 0.05M for in/out layers; +56.7M if each block gets a second AdaLN for audio | about 47M (audio stream width 384, 8 blocks ≈ 26M, plus cross-attention every 2nd video block ≈ 21M; Judgement, rough) | depends |
| Extra compute | +25% tokens; attention +56%; overall about +30% (Judgement) | about +25% (Judgement) | small |
| Sound can shape video | yes | yes | no |

**Time alignment, concretely.** Today the time code uses the latent index 0..4. With sound, both modalities need
one clock in **seconds**:
- audio token *j* sits at *j* / 40 s;
- video latent 0 is frame 0 (0 s), and latent *k* ≥ 1 covers frames 4*k* − 3 … 4*k* (the Wan2.2 VAE keeps frame 0
  alone, then packs 4 frames per latent), so its centre is (8*k* − 3) / 16 s.

Feed both times into the same sine/cosine code (it accepts non-integer positions). Audio tokens get zeros for
height and width, plus a learned "I am audio" vector.

## 7. Our critique of the references

| Claim or choice | What the evidence shows | Gap |
|---|---|---|
| H3: one shared stream is enough | It works at 33B (the model exists) | No ablation against two streams. No training recipe. We can't tell what made it work. |
| K6: two streams with cross-attention | It beats LTX 2.5 on sync | No single-stream baseline. "Better than LTX" mixes many changes (data, size, RL). |
| K6: audio-only pretraining first | They did it, with 40M tracks | No run without it, so the value of this step is not measured. |
| H3: audio shift 3 vs video 12 | It's in the config | No reason given. It probably reflects that audio has fewer, less redundant tokens, but that's our guess. |
| K6 Pro is the best model | Best sync | Worst word error rate. One metric hides another. |
| K6: loss weights video 0.85 / audio 0.15 in fine-tuning | Stated in §4.3 | No reason or sweep given. It's the opposite of pushing audio up, so the right balance is an open question. |
| K6: gradient damping on the first cross-attention blocks | Stated in §4.4 | No run without it. |
| Both: sync is learned by attention | Benchmarks look good | S2V switched to a hard per-frame window, which hints that learned alignment wasn't fully trusted. |

**Gap in our own plan.** In our data the sound would be **fully decided by the video attributes**: hearts →
pop, heartbeat motion → thump. So a joint model has nothing to add over a simple rule. Read the effect and motion
from the generated clip, then render the matching sound with our generator code (option C as a rule). That rule is
a **strong baseline** we must beat or match, and it's free. To make the joint model meaningful, the caption must
also control something **only in the sound**, e.g. a sound style ("soft", "8-bit", "silent") chosen independently
of the picture.

## 8. Our decisions

| Decision | Our choice | Reason | How we'd check it |
|---|---|---|---|
| Speech or sound effects | Sound effects first; no speech | Emoji stickers have no mouths to sync. Speech needs real voice data and a new text field. | — |
| Audio data | Render it with code from the same spec as the video. Sounds are synthesised (no samples, no licence risk) and loop with period 2.0 s. | Exact event times make sync measurable, the same way exact captions make prompt-following measurable. | Listen to 20 clips; check that the loop seam has no click |
| Independent sound control | Add a "sound style" attribute (3–4 values incl. silent) to captions, chosen independently of the picture | Otherwise sound is redundant with video and the rule baseline wins | Hold out some style × effect combinations, like our unseen split |
| Audio VAE | **Chosen: Stable Audio Open 1.0 VAE**, frozen (44.1 kHz, 64 channels at 21.5 Hz, so 43 tokens per 2 s loop). **Not** H3's (licence fails: territory, no training on outputs) | Ties DAC on timing, closest spectral match, and 43 tokens instead of DAC 16k's 100 (see §12) | Done: gate passed on 200 clips (§12). By-ear check still open |
| Audio VAE licence | Accept the Stability AI Community License for the audio part | Free for research and non-commercial use, and for commercial use under US $1M yearly revenue after registering. A model trained on its latents likely counts as a "Derivative Work", so an audio release can't be plain MIT: it must ship their licence, a NOTICE file and "Powered by Stability AI" | Re-read §IV(a)–(b) before any public audio release; DAC 16k (MIT) is the fallback, also passed the gate |
| Architecture | **A (single stream) first** | Retraining from scratch costs about $3.40; smallest code change; 80 extra tokens is cheap | Compare against B in §11 |
| AdaLN per modality | Start with the shared AdaLN plus a modality vector added to `cond`; test a separate audio AdaLN (+56.7M) as an ablation | H3 says per-modality AdaLN helps, but it costs 27% more parameters for us | One run each, same seed |
| Noise level | Same *t* for video and audio in training | Simplest. Separate *t* (K6's `independent_time` mode, which allows sound-from-video) later. | — |
| Audio time shift | Start at our video value 5.0, sweep {1, 3, 5} | H3 uses a lower shift for audio; we don't know why | 3 short runs, compare sync error |
| Loss | Mean MSE per modality, then video + 1.0 × audio | Averaging over all tokens would let 320 video tokens drown 80 audio tokens. K6 used 0.85 / 0.15 in fine-tuning without saying why, so the weight is open. | Watch both losses separately; sweep audio weight {0.15, 1.0} |
| Option B damping | If we build B, copy K6's damping: no gradient through the first cross-attention block, 0.1× for the next few | It protects the trained video stream from the new, random audio path | Video guard eval |
| Guard on video | Video attribute error must stay within r2's (2.0% seen, 3.3% unseen) | Adding sound must not break the picture | Re-run our existing eval |

## 9. Transferable lessons

- To add a modality, first ask what it adds that the existing modalities don't already decide. If nothing, a rule beats a model.
- Compress every modality with a frozen autoencoder first, and gate it with a round-trip test before training.
- Put all modalities on one clock in seconds, not on per-modality index counters.
- When one modality has far fewer tokens, average the loss per modality, or it gets drowned.
- "Single stream vs two streams" is still untested in public. Neither design is proven better.
- Check every borrowed component's licence for clauses about training on its outputs.

## 10. Interview questions

1. **Why not generate raw audio directly?** 32,000 numbers per second, mostly irrelevant detail. Compress with an audio VAE first (H3: 800× down to 40 Hz).
2. **Single stream vs dual stream?** Single (H3): one transformer, shared attention, only edges and AdaLN per modality. Dual (K6): separate models joined by cross-attention, so a trained video model can be reused. No public ablation decides between them.
3. **How do you keep sound in sync with frames?** Give both modalities a time code in the same units (seconds). Optionally force each frame to attend only to its own audio window (S2V).
4. **Why did Kandinsky pretrain audio alone first?** The audio stream started from scratch, while video started from 5.0. Pretraining audio first stops a random audio stream from disturbing the trained video stream. They didn't test skipping it.
5. **How would you measure sync on Bhaav-E?** Our generator knows every event time. Detect sound onsets in the generated audio and report the error in ms.
6. **What's the strongest baseline?** A rule: read effect and motion from the clip, then render the sound with the generator code. A joint model must match it on sync and beat it on anything only the caption controls.
7. **Would you borrow H3's audio VAE?** No. Its licence excludes the US, EU, UK and Korea, and forbids using outputs to improve other models.
8. **What would you do differently from both labs?** Run the single vs dual comparison at small scale, publish the training recipe, and report the sync metric next to the speech metric, since K6 Pro got better at one and worse at the other.

## 11. Our experiment

**Question.** At our size, is one shared stream (A) as good as two streams (B) at keeping sound in sync, without
hurting the video?

**Steps:**

1. **Sound generator** (CPU, about 1 h of coding; rendering is minutes). Add `render_audio(spec)` to
   `generator/bhaav_gen.py`. Each effect gets a synthesised sound (pop, chime, hiss, snore), and each motion gets a
   rhythm at its known speed in Hz (thump for heartbeat, boing at each bounce landing). Save the event times next
   to the caption.
2. **Audio VAE gate and cache** (same pod as before): round-trip 200 clips, then encode all 85,816 clips.
3. **Two runs, same data and seed**, 40,000 steps each:
   - A: single stream.
   - B: frozen r2 VideoDiT plus a 384-wide, 8-block audio stream.
4. **Evaluation:**
   - **Sync:** median onset error (ms) between detected sounds and scheduled events. **Pass if under 50 ms**
     (our threshold, written before the run).
   - **Sound class:** a checker trained on real audio, calibrated on real clips first.
   - **Video guard:** our existing video attribute eval.
   - **Rule baseline:** render sound from the checker's reading of each generated clip, and score it the same way.

**Hardware and cost (Judgement).** One 32 GB GPU at r2's rate (about $0.72/h). Each run is about 4.7 h, about $3.40.
Two runs plus caching and eval come to about **$10**.

**Done so far:** step 1 (`generator/bhaav_audio.py`) and the gate half of step 2 (§12). **Not tested yet:** the
cache, both runs and the evaluation.

## 12. Audio VAE gate results (2026-10-31)

Script: `gpu/audio_vae_gate.py` (ours). 200 clips picked round-robin over every effect, motion and UI element;
each 2 s loop tiled 3×, round-tripped, middle loop compared. Pass marks fixed before the first run: ≥97% of
onsets kept and 95th-percentile onset shift ≤10 ms. All numbers **Verified** (our runs, Apple M-series GPU via MPS).

| Codec | Onsets kept | p95 shift | Spectral distance | Median SNR | Latent per 2 s | Speed (Mac) | Pass |
|---|---|---|---|---|---|---|---|
| identity (control) | 100% | 0 ms | 0 | — | — | — | yes |
| 25 ms smear (control) | 84.0% | 26 ms | — | — | — | — | no |
| **Stable Audio Open** | 98.6% (1303/1322) | 4 ms | **4.14** | 13.2 dB | **43 × 64** | 0.2 clips/s | yes |
| DAC 16 kHz | 99.0% (1309/1322) | 4 ms | 5.08 | −0.6 dB | 100 × 1024 | 0.9 clips/s | yes |
| DAC 44 kHz | 98.7% (1305/1322) | 4 ms | 5.20 | 14.0 dB | 172 × 1024 | 0.2 clips/s | yes |

- **Weak spots of Stable Audio Open:** lost onsets in hearts (6), sweat (5), sparkles (5), exhaust (3); p95 shift
  17.6 ms for sweat and 14 ms for exhaust. Both are well inside the 50 ms sync pass mark for the model.
- **Hardest sound for every codec:** sparkles (spectral distance about 11), then steam, bell and success.
- **What this does not prove:** that a *generator* can produce latents the decoder turns into clean sound; it only
  shows the decoder can rebuild our sounds from the encoder's latents. The by-ear check is not done yet.
- **Cost note:** 0.2 clips/s means about 5 days for 85,816 clips on the Mac, so the cache must be built on a GPU pod.
