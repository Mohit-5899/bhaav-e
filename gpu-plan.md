# First GPU step: VAE test and latent cache (steps 3a–3c)

Status: **plan only. Nothing is created, nothing is paid for.** Written 2026-10-29.
Labels: **Verified** (we checked it) · **Reported** (from a page or tool, not re-checked) · **Judgement** (our choice).

## 0. In one minute

We rent one GPU for about 2–3 hours, for about **$2.50–3.50 all in**. On it we:

1. **3a:** squeeze 200 stickers through the borrowed VAE and back out, then measure how much was lost. If
   thin lines or colours break, we stop here, before spending anything more.
2. **3b:** squeeze all 74,987 clips once and save the result (the "latent cache", 9.2 GB) to a disk that
   outlives the GPU.
3. **3c:** turn every caption into T5 vectors once, the same way.

**Analogy:** before printing a whole book at a new print shop, print one test page (3a). If it looks right,
print the whole book once (3b) and store it on a shelf. Every later training run reads from that shelf
and never goes back to the shop.

## 1. Which GPU, and why

**Problem first.** The VAE has 555M parameters (Reported). Squeezing one clip is a fixed amount of maths.
75K clips is not a huge job: Drishya squeezed 475K clips in 6.2 h on an RTX 5090 (Reported), which is about
**21 clips per second**. At that speed, ours would take about **1 hour** (Judgement: an estimate; our clips may
be a different size from theirs, and 3a will measure the real speed).

So almost any modern GPU is fast enough. The choice is about **memory, matching later steps, and where the data lives**.

| Option | VRAM | Memory speed | $/hr (Secure / Community) | For us |
|---|---|---|---|---|
| **RTX 5090** | 32 GB | **1,792 GB/s** | $1.19 / $0.69 | **Pick (changed 2026-10-29, see below).** Fastest per dollar for both encoding and training, and the training run fits |
| L40S | 48 GB | 864 GB/s | $1.09 / $0.79 | **Fallback.** More memory headroom, but about half the speed |
| RTX 4090 | 24 GB | about 1,000 GB/s | $0.89 / $0.34 | Training peaks at 20.6–25.5 GB (below), which is too close to 24 GB |
| A100 80GB | 80 GB | | $1.79 / $1.19 | More than we need |

**Why the 5090 replaced the L40S: measured, not guessed.** Drishya ran the **same 211.9M generator at batch 64** on both
cards (Verified: read in their `results/stage1/RESULTS.md`, `stage3/RESULTS.md` and `log.jsonl` files):

| Drishya run | GPU | Peak memory | Speed |
|---|---|---|---|
| Stage 1 (unconditional, no captions) | RTX 5090 | **25.5 GB** | **386 clips/s** |
| Stage 3 (with captions, the release run) | L40S | 20.6 GB | 213 clips/s |

- **Memory:** training peaked at 25.5 GB, so it **fits in the 5090's 32 GB** with about 6 GB spare. My earlier worry that
  batch 64 wouldn't fit was wrong.
- **Speed:** the 5090 trained about **1.8× faster**. Per clip that is about **1.6× cheaper** ($1.19 ÷ 386 vs $1.09 ÷ 213).
  Caveat: the two runs differ (captions, image-to-video), so the exact ratio is Medium confidence; the direction is clear.
  It matches first principles: this model is small, so moving data through memory is the bottleneck, and the 5090's
  memory is about 2× faster (1,792 vs 864 GB/s, Reported: several spec sites agree).
- **Why Drishya trained on the L40S anyway:** not speed. Three RunPod 5090 starts never came up, so they moved the run
  to an L40S at another provider (Verified: their `stage3/RESULTS.md`). Their cost log also shows 5090 hosts that never
  became reachable or had a broken CUDA, mostly on Community Cloud. **Lesson: budget for a failed start or two.**
- **Software:** the 5090 is a newer chip (Blackwell) and needs CUDA 12.8+ and a recent PyTorch. RunPod's
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image (CUDA 12.8.1, PyTorch 2.8; already used on your ATAS pods)
  meets that (Medium-high: Blackwell support in these versions is from memory, not re-checked).

Prices: **Verified 2026-10-29 from RunPod's live GPU catalog** (`get-gpu-type`); all four match the table.
Prices can change, so the create step re-reads them.

**Live stock for a 1-GPU Secure pod (2026-10-29):**

| GPU | Overall | Data centers with stock (all LOW) |
|---|---|---|
| **L40S** | LOW | EU-NL-1, OC-AU-1, US-TX-3, US-TX-4 |
| RTX 4090 | MEDIUM | EU-CZ-1, EU-RO-1, EUR-IS-1, EUR-IS-2, EUR-NO-1, US-IL-1 |
| RTX 5090 | MEDIUM | EU-RO-1, EUR-IS-1, EUR-NO-1 |
| A100 80GB | LOW | CA-MTL-3, US-MO-1 |

**Plan: RTX 5090 Secure in EU-RO-1** (5090 stock MEDIUM overall; EU-RO-1, EUR-IS-1 and EUR-NO-1 each LOW). The network
volume goes in the same data center, and training later runs there on a 5090 too (it fits, see above). If no 5090 can
be placed, fall back to an L40S in whichever data center has it at that moment (the user approved either card,
2026-10-29). L40S stock moves fast: at 16:00 it had left EU-NL-1 and showed in EUR-IS-2, OC-AU-1, US-MO-1, US-NC-1, US-TX-4. Creating anything still needs a
separate yes.

**Secure vs Community Cloud.** Community is cheaper (hosts are third parties). We pick **Secure** because
**network volumes can be attached only to Secure Cloud pods** (Verified: RunPod docs, storage/network-volumes).

## 2. Pod or serverless?

- **Serverless** = RunPod starts a container only when a request arrives, runs one job, and bills per second.
  It is built for an app that answers users (e.g. "make a sticker now"). You must build a Docker image first,
  each job's input is small, cold starts take time, and the hourly rate is higher (L40S: **$1.75/hr** serverless
  vs $1.09/hr pod, Reported).
- **Pod** = a rented machine with a terminal. You SSH in, copy files, run scripts and watch them, and you pay per
  hour until you stop it.

Our job is a **one-off batch job**: one hour of work on 1.3 GB of input that we want to watch.
**→ Pod.** (Judgement, High confidence.) Serverless becomes the right choice at step 8, if we serve the model.

## 3. Storage: a network volume that outlives the pod

**Problem first.** A pod's own disk is deleted when the pod is deleted. The cache is 9.2 GB, and we will need it
on a *different* pod later for training. Without shared storage we would download 9.2 GB to the Mac and upload it again.

**Fix:** a **network volume** is a disk in one RunPod datacenter that any pod in that datacenter can attach to.

- Size **30 GB**: clips 1.3 GB + latents 9.2 GB + T5 cache (size depends on our choice of caption length, see 3c) + VAE/T5 weights + room to spare.
- Cost: $0.07/GB/month under 1 TB (Verified: RunPod pricing docs) → **about $2.10 per month** while we keep it.
  Billed hourly from creation until deletion, attached or not.
- **Pick the datacenter by where the GPU is in stock**, because training will happen there too. A network volume can only
  be mounted by compute in its own data center, and its data center can't be changed later (Verified: RunPod docs).
  **Proposed: EU-RO-1** (RTX 5090 in stock; offers STANDARD volumes; Verified 2026-10-29). Fallback with an L40S:
  whichever data center has it when we create.
- **Don't reuse `atas-bench-workspace`** (150 GB, also EU-RO-1): it belongs to the ATAS project. Ours is a new 30 GB volume.
- Container disk: 50 GB (for pip installs and the model download cache). This is deleted with the pod, which is fine.

## 4. Getting the clips to the pod

Facts (Verified, counted on the Mac):

| File type | Files | Real bytes |
|---|---|---|
| `.mp4` | 75,001 | 1.29 GB |
| `.png` | 75,001 | 1.08 GB |
| `.json` | 75,001 | 0.04 GB |
| `.txt` | 75,001 | 0.01 GB |

`du` says 3.1 GB, but the real bytes are about 2.4 GB. The gap is disk-block overhead from 300K tiny files.

**Problem first.** Copying 300,000 small files one by one over the network is slow, because each file costs a
round trip. One big file is fast. **Fix:** pack the files into a single tar file on the Mac.

- **Skip the `.png` files.** They are frame 1 of each clip, which is already inside the `.mp4`.
  This cuts the upload by 1.08 GB. **Keep** `.txt` (needed for 3c) and `.json` (needed for scoring later).
- **Don't compress.** mp4 is already compressed, and gzip would only burn CPU time.

```bash
cd bhaav-e/data && COPYFILE_DISABLE=1 tar -cf ../upload/bhaav-clips.tar --exclude='*.png' --exclude='.DS_Store' clips splits && shasum -a 256 ../upload/bhaav-clips.tar
```

**Done 2026-10-29 (Verified):** `upload/bhaav-clips.tar`, **1.77 GB** (more than the 1.35 GB of file bytes,
because tar adds a 512-byte header to each of the ~225K files and pads each file to a 512-byte boundary).
SHA-256 `33d55515886a389029a271220423fe2faedb0e545b788cb8f64ad81f04b906cc` (also in `upload/bhaav-clips.tar.sha256`).
Contents: 75,001 mp4 · 75,001 captions · 75,001 labels · all split lists incl. `all_ids.txt` (74,987 ids) · no png ·
no macOS `._` files (`COPYFILE_DISABLE=1` stops tar adding them).

Upload about **1.77 GB** with `rsync -P` over the pod's SSH (it can resume if the connection drops). Then run
`shasum` on the pod and compare the hash. At 50 Mbit/s upload this takes about 4 minutes (Judgement:
depends on your connection).
Alternative: `runpodctl send` / `receive` (needs no SSH setup, but it is not installed on the Mac yet; Verified).

Untar onto the **network volume** at `/workspace/clips`.

## 5. Step 3a: VAE round-trip test (script: `gpu/vae_roundtrip.py`)

**What:** encode then decode 200 clips, and compare with the originals.
**Why:** if the VAE can't rebuild a sticker, the generator can't either, because the VAE decoder is always the last step.
**What if we skip it:** we would spend about $14 or more and 15 hours training (Drishya's numbers, Reported) and
only then find out the outputs are blurry, with no way to fix it.

What the script does:
- **Picks clips by subcategory** from `val.txt` (seed 0), the same number from each. Rare kinds such as the
  ui toggle get tested, not just the common ones.
- **PSNR** (peak signal-to-noise ratio): a score in dB of how close two images are, based on the average squared
  pixel difference. Higher is better. Each +10 dB means 10× less squared error.
  Tiny example: if every pixel is off by 5 out of 255, then PSNR = 10·log10(255²/25) = **34.2 dB**.
- **SSIM** (structural similarity): a score from 0 to 1 for whether edges and structure match. It catches blur that PSNR averages away.
- **Error-map GIFs:** original | rebuilt | difference × 4. This is the check that matters most for stickers.
  **PSNR is an average**, so a broken 1-pixel tick mark barely moves it, but it lights up in the error map.
- **bf16 check:** it decodes the *same* latent twice, once at full precision and once rounded to bf16 (the
  format the cache will use). If the PSNR loss is small (well under 0.1 dB, Judgement), storing half-size latents is safe.
- **Speed:** it times clips per second, so step 3b's time and cost are **measured**, not guessed.

**Decision rule. Set it *before* looking, so we can't move the goalposts** (Judgement):

| Result | Action |
|---|---|
| Mean PSNR ≥ 32 dB, p10 ≥ 28 dB, and no broken thin lines in the GIFs | Go to 3b |
| Numbers are fine but GIFs show broken ticks, Zzz or outlines | Stop. Options: render thicker lines, or try 512 px |
| Mean PSNR < 30 dB | Stop. Rethink which VAE to use |

The 32 / 28 thresholds are **our guess** at "looks right for flat-colour stickers". We have not checked them
against any source. Our ground truth is itself h264 video with halved colour resolution (yuv420p, Verified
by ffprobe), so a perfect VAE would still not score infinity.

## 6. Step 3b: encode everything (script: `gpu/encode_latents.py`)

- **Which clips:** the 74,987 clips in the four splits (Verified: 75,001 − 14 excluded). Make
  `all_ids.txt` by joining the four split files.
- **Size:** 74,987 × 48 × 5 × 16 × 16 × 2 bytes = **9.21 GB** (Verified by arithmetic).
- **What we store:** the posterior **mean** (the centre of the VAE's "cloud", see how-training-works 3b), in bf16,
  **not normalised**. We also save the VAE config (its `latents_mean` / `latents_std`, 48 values each, Reported
  from the Hugging Face config) next to the cache, so training can normalise the latents itself.
  - Why the mean and not a random sample: a sample would freeze one random draw forever. Storing mean + spread
    would double the size to 18.4 GB. Drishya's choice is not stated. (Judgement)
- **Speed tricks:** mp4 decoding runs on 8 CPU worker processes, so the GPU never waits. The batch size comes from 3a.
- **Crash-safe:** it writes shards of 4,096 clips, each saved under a temp name and renamed when complete. A
  re-run skips finished shards, so a crash loses at most about 3 minutes of work.
- **After it finishes:** decode 20 random clips *from the cache* and check that their PSNR matches 3a. This
  proves the saving and loading code didn't scramble anything.

## 7. Step 3c: captions → T5 cache (same pod, about 10 min)

Flan-T5-Base runs on every caption line of every clip (`.txt` has several phrasings). Script not written yet.
One choice is still open: **max token length**. It sets the cache size: 768 numbers per token × tokens ×
number of caption lines. Count the caption lengths first, then pick the length.

## 8. Cost and time

| Item | Time | Cost |
|---|---|---|
| Pod start, `pip install diffusers av safetensors scikit-image`, model download | ~20 min | |
| Upload tar + untar | ~10 min | |
| 3a | ~15 min | |
| 3b | ~1–1.5 h (3a will tell us) | |
| 3c + cache check | ~20 min | |
| **RTX 5090 Secure pod total** | **~2–2.5 h × $1.19** (likely less: the 5090 is faster) | **≈ $2.40–3.00** |
| Network volume, 30 GB | per month | ≈ $2.10 |
| **Budget cap (stop and ask if we would go over)** | | **$5** |

Compare Drishya's encoding: $7.40 for 475K clips (Reported). Ours is about 6× fewer clips.

**Stop the pod the moment 3c finishes.** A forgotten RTX 5090 costs about $28.50 a day ($1.19 × 24).

## Live resources (created 2026-10-29, user approved)

| Resource | Id | Details | Cost |
|---|---|---|---|
| Network volume `bhaav-tiny-cache` | `qdyupny3w7` | 30 GB STANDARD, EU-RO-1, mounted at `/workspace` | $2.10/month until deleted |
| Pod `bhaav-encode` | `z8olzxqapr1if1` | 1× RTX 5090 Secure, EU-RO-1, 32 vCPU, 123 GB RAM, started 10:38:58 UTC | **$1.19/h while running** |

SSH (direct, needed for rsync): `ssh -i ~/.ssh/id_ed25519 -p 21323 root@213.173.103.166`. **The port changes after every
stop/start**: re-read it with `get-pod`. Checked on the pod (Verified): RTX 5090 32 GB, driver 595.91, PyTorch 2.8.0+cu128
sees it (compute capability 12.0), a bf16 matrix multiply ran, `/workspace` is the network volume.

## Step 3a result (2026-10-29, Verified: run on the pod)

**Gate passed: the Wan2.2 VAE rebuilds our stickers well.** 180 val clips (6 per subcategory), RTX 5090:

| Category | PSNR mean | 10th percentile | Worst | SSIM |
|---|---|---|---|---|
| animated-emoji | 37.72 dB | 33.61 | 31.79 (party-002082) | 0.9815 |
| reactions | 36.46 dB | 34.24 | 33.58 (laugh-000789) | 0.9796 |
| ui | 41.36 dB | 38.40 | 34.01 (success-002127) | 0.9905 |
| **All** | **38.51 dB** | 34.22 | 31.79 | **0.9839** |

- **PSNR** = how close the rebuilt pixels are to the original, on a log scale; above about 35 dB is hard to tell apart by eye,
  30–35 dB is good. **SSIM** = structural similarity, 1.0 = identical.
- **Looked at by eye** (`data/review/vae-roundtrip/worst-cases.png`): even the worst clips look the same. The error sits only
  on sharp outlines and tiny confetti pieces, the "thin lines" risk from how-training-works.md 3e, and it is small.
- Hardest subjects: party popper (33.1 dB avg), 100, crying, sleepy (about 34.5). Easiest: UI (flat shapes), up to 46 dB.
- **Encode speed: 21.5 clips/s** (batch 8, bf16). So step 3b ≈ 75K ÷ 21.5 ≈ **58 min** (≈ $1.15).
- **Script bug found:** the "bf16-latent loss" column is always 0.000 because, inside autocast, the latent is already bf16, so
  rounding it again changes nothing. The number is meaningless; storing latents in bf16 is still fine (that is what was decoded).
- Practical lessons: pip into the network volume took >10 min, into the pod's local disk 7 s; unpacking 225K files locally
  took 7 s. Keep many-small-file work on local disk and only big outputs on the volume.

## Steps 3b + 3c result (2026-10-29, Verified: run on the pod)

**Done. The training data is ready on volume `qdyupny3w7` at `/workspace/cache/`.**

| Cache | Contents | Size |
|---|---|---|
| `cache/latents/` | 19 shards, **74,987 clips**, bf16 `[n, 48, 5, 16, 16]`, posterior mean, not normalised; `vae_config.json` has `latents_mean`/`latents_std` | 8.6 GiB (9.2 GB) |
| `cache/text/` | **26,066 unique sentences** (row 0 = empty caption) × **35 tokens** × 768, bf16 + mask; `clip_captions.json` maps each clip to its 3 rows | 1.4 GB |

- 3b: 21.1 clips/s, 11:39 → about 12:37 UTC (58 min), batch 16, 16 decode workers. 3c: 23 s (`gpu/encode_captions.py`).
- Captions: max 35 tokens, median 20, **none truncated** (so length 35, not Drishya's 77).
- `gpu/verify_cache.py`: duplicates 0, missing 0; **20 random clips decoded from the cache: mean 38.88 dB, min 33.51 dB**
  (matches 3a's 38.51 dB, so saving/loading scrambled nothing); every clip has captions. **ALL CHECKS PASSED.**
- **Pod stopped 12:39 UTC.** GPU time 10:39 → 12:39 ≈ 2 h × $1.19 ≈ **$2.38** (estimate from the clock; check the bill).
  The stopped pod keeps no disk of its own (the network volume replaced it), so it costs nothing while stopped; only the
  volume's $2.10/month continues.

## Training run r1 (started 2026-10-29 13:38 UTC; user authorised running it end to end)

- **Our own code** (`model/`: videodit.py, muon.py, data.py, train.py, sample.py, render_snapshots.py). Same blueprint as
  Drishya's s210: **211.92M parameters, exact match** (Verified by `model/tests/run_tests.py`).
- The 5090 pod (`z8olzxqapr1if1`) could not restart (its host's GPU was taken), so it was **terminated** and a new pod
  created: `skr4jccmwj4lz6`, **RTX PRO 4500 Blackwell 32 GB, $0.72/h**, EU-RO-1, same volume. EU-RO-1 had no 5090/4090 then.
- Tests before the run (Verified): Mac CPU: size, shapes, patchify, Muon; overfit 2 clips → loss 0.0 (tiny model must be
  wider than 192, the per-piece output size). GPU: overfit 8 real clips with the full model → loss 0.66 → 0.00002 in 300
  steps; 400-step rehearsal: **214 clips/s, 21.3 GB**, val 0.346 → 0.209, checkpoint + snapshot + render all work.
- Settings: 40,000 steps × batch 64 (2.56M clips, ~42 passes), warmup 1,000, decay last 25%, Muon 0.02 + AdamW 2e-4,
  caption drop 10%, image→sticker 25%, EMA 0.9999, bf16 + torch.compile. Val every 1,000; checkpoint every 2,500;
  **snapshot every 5,000** (EMA weights + 8 sample stickers: 3 seen, 3 unseen combinations, 2 image→sticker).
- Output on the volume: `/workspace/runs/r1/` (train.out, log.jsonl, last.pt, snapshots/). Expected end ≈ 17:05 UTC, ≈ $2.50.
- The pod has no RunPod key, so it can't stop itself; a watcher on the Mac wakes Claude when training ends, then
  render → download → stop the pod.

### r1 results (finished 2026-10-29 ~17:00 UTC)

- **Finished all 40,000 steps, no errors** (Verified: train.out). Speed steady at 213–216 clips/s, 21.8 GB peak.
- **Val loss** (EMA weights, fixed clips/noise/t): 0.0854 (5k) → 0.0629 (20k) → 0.0586 (30k, decay starts) → **0.0551 (40k)**.
  Still falling at the end, about 0.0004 per 1,000 steps, so a longer run would likely help a little (Judgement).
- Train loss at 40k: 0.032 (single batch, noisy). Gap to val is steady; val never rose, so no sign of memorising (Verified from log).
- **Snapshots:** 8 × (EMA bf16 weights + sample latents) on the volume at `/workspace/runs/r1/snapshots/`; rendered GIFs,
  `progress.png`, log.jsonl, train.out downloaded to `runs/r1/` on the Mac (`.pt` files left on the volume).
- **What the stickers look like** (Judgement, from looking at frames): right subject, colour, background and effect for all 8,
  including the 3 **never-seen combinations** (clapping + sparkles on mint, red furious face on black, orange bell + badge).
  Motion is visible (confetti bursting, steam puffing, badge popping, wobble). By eye, most of the quality was there by
  step 5k; later steps mainly cleaned up small effects (confetti, steam edges).
- Weakness of the sample set: 3 of 8 prompts are clapping hands, so it covers few subjects. Fix next time: pick by subject.
- **Pod `skr4jccmwj4lz6` stopped** at ~17:03 UTC. Run cost ≈ **$2.60** (3h36m × $0.72). Volume stays ($2.10/month).
- **Evaluation (2026-10-30, ≈ $0.59, 49 min):** seen vs unseen. Full write-up: `runs/r1/eval/RESULTS.md`. Loss
  unseen 0.0504 vs seen 0.0467 (+8%, all from animated-emoji). Generated stickers all-correct 87.5% seen / 64.5% unseen,
  against checker ceilings of 92.2% / 70.3%. Pod stopped again after.
- **Evaluation v2 (2026-10-30, ≈ $0.41, 34 min):** fairer checker plus paired scoring. Model error rate 10.5% seen vs
  19.1% unseen; corrects the v1 reading. Pod stopped after. Total GPU spend on Bhaav so far ≈ $6.00.
- **Evaluation v3 (2026-10-30, ≈ $1.22):** old pod's GPU was taken, so it was terminated. New pod `8nmhsj7klhrskk`
  (RTX PRO 4500, $0.72/h, same volume), now **stopped**. One run died when the volume filled (sample-file bug, fixed).
  Total GPU spend ≈ $7.20.
- **r2 = rebalanced data (2026-10-30, ≈ $3.85):** new pod `8rhpcikeuzhr1v` "bhaav-r2" (RTX PRO 4500, $0.72/h, same
  volume, grown to 50 GB). Encoded 10,815 new clips, re-encoded captions, trained 40k steps (val 0.0562 vs r1 0.0551),
  ran the v3 evaluation; 4.85 h uptime ≈ $3.49, plus ≈ $0.36 for a first start that died on a script bug. Pod
  **stopped**. Results in `runs/r2/RESULTS.md`. Total GPU spend ≈ $11.05.
- **Effect checker v4 re-score + release export (2026-10-30, ≈ $1.25):** new pod `zowou6c7pwgi4l` "bhaav-eval4" (L4,
  $0.59/h, same volume `qdyupny3w7`; the old pod had lost its GPU). Re-scored r1 and r2 at CFG 3 and 5, then exported
  both EMA weights to safetensors and rendered 8 example GIFs. 7,634 s uptime. **Stopped.** Total GPU spend ≈ $12.30.

## 9. Order of operations (each paid step needs your yes)

1. ✅ Free: build the tar and hash it; create `all_ids.txt` (done 2026-10-29).
2. **Paid, ask first:** re-check RTX 5090 stock, then create a 30 GB network volume in that data center (proposed EU-RO-1).
   Create the volume only once the stock check passes, so we don't pay for a volume next to no GPU.
3. **Paid, ask first:** create an RTX 5090 Secure pod (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 50 GB container
   disk) with the volume attached. If it can't be placed or never comes up, stop and switch to the L40S fallback (section 1).
4. Upload, run 3a, and **stop to look at the GIFs together** before 3b.
5. 3b, then 3c, then the cache check, then stop the pod.

## 10. What is not verified

- Both scripts are **untested sketches** (they only pass a Python syntax check, Verified). The diffusers calls
  (`AutoencoderKLWan`, `.latent_dist.mean`, `.decode(z).sample`) are written from memory of the API (Medium-high confidence).
  The 48-channel / 16× / 4× config is confirmed from the Hugging Face config (Reported).
- The VAE weights' download size: not checked.
- ~~RunPod prices come from the public pricing page through a summary tool~~ **Fixed 2026-10-29:** GPU prices and
  stock read live from RunPod's catalog; storage price and the Secure-only rule from RunPod's docs.
- Stock is **LOW** in every single data center for both cards; it can disappear between the check and the create.
- Drishya hit several RunPod 5090 hosts that never came up or had broken CUDA. Expect to retry once or twice.
- The speed estimate assumes Drishya's clips are a similar size to ours; their clip resolution is not stated in our note.
