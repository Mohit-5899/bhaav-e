# r1 evaluation: seen vs unseen combinations

Run: 2026-10-30 on the RTX PRO 4500 pod, ~49 min, ≈ $0.59. Script: `model/evaluate.py` (OUR code). Raw numbers:
`report.json`; full log: `eval.out`; flagged failures: `mistakes_test_seen.png`, `mistakes_test_unseen.png`.

**The question.** test_unseen holds subject × motion × effect (and UI subject × accent) combinations that never appear in
training. If the model only memorised the combinations it saw, it should be clearly worse on them.

**Short answer, corrected after the v2 checker (see the end):** subject, background and accent carry over to new
combinations almost perfectly. Once checker errors are taken out, though, finished stickers go wrong about **twice as often**
on unseen combinations (19% vs 10.5% of clips). The extra errors are mostly the wrong **effect**, and the model tends to
add the effect that subject usually comes with (sparkles on bells, tears on the laughing face). The v1 summary ("only a
little worse") was too generous, because the v1 checker hid this.

## Part A: loss on every test clip (Verified: computed)

Same training loss, measured with fixed noise, t and first caption line on all 3,750 seen, 5,918 unseen and 3,750 val
clips. (These numbers are lower than the training log's val, 0.0551, because the log picks a random caption line; here it is
always line 0. Compare within this table only.)

| step | test_seen | test_unseen | val | gap (unseen − seen) |
|---|---|---|---|---|
| 5,000 | 0.0747 | 0.0779 | 0.0765 | 0.0032 |
| 20,000 | 0.0537 | 0.0571 | 0.0553 | 0.0034 |
| 30,000 | 0.0499 | 0.0534 | 0.0513 | 0.0035 |
| 40,000 | **0.0467** | **0.0504** | 0.0480 | **0.0037** (+8%) |

At step 40,000, broken down by category:

| category | seen | unseen |
|---|---|---|
| animated-emoji | 0.0545 | **0.0679** (+25%) |
| reactions | 0.0523 | 0.0503 (slightly *better*) |
| ui | 0.0332 | 0.0334 (same) |

- Both kept falling until the end, and the gap grew only a little (0.0032 → 0.0037). There's no point where unseen stops
  improving while seen keeps going, which is what memorising would look like.
- **The whole gap is animated-emoji.** Our guess (Judgement, not tested): it has only 10 subjects, so each held-out
  combination removes a bigger share of what the model sees for that subject.
- Most of the loss sits at low noise (t < 0.35: 0.089 seen, 0.097 unseen, against about 0.02–0.03 elsewhere). That is the
  fine-detail stage.

## Part B: do finished stickers follow the caption? (Verified: computed; final EMA, 256 per split, 50 steps, CFG 5)

A checker reads each attribute back out of the pixels. It compares against 3,000 real training clips using nearest
neighbours: CLIP embeddings for subject and effect, movement numbers for motion, border colour for background, and
foreground colour for accent. The **real ceiling** row runs the same checker on the real clips of the same captions.

| | subject | motion | effect | background | accent (UI) | all correct |
|---|---|---|---|---|---|---|
| seen: generated | 99.6% | 94.5% | 92.6% | 100% | 100% | **87.5%** (83–91) |
| seen: real ceiling | 100% | 95.3% | 96.9% | 100% | 100% | 92.2% (88–95) |
| unseen: generated | 99.6% | 84.8% | 79.7% | 100% | 100% | **64.5%** (58–70) |
| unseen: real ceiling | 100% | 83.6% | 85.9% | 100% | 100% | 70.3% (64–76) |

(Brackets: 95% range for 256 samples.)

How to read it:
- **Subject, background and accent are solved**, unseen included. The 40 unseen UI subject × accent clips (for example a
  *blue* progress bar, a colour it never trained on for that icon) are 97.5% all-correct.
- **Motion on unseen is at the checker's ceiling** (84.8% generated, 83.6% real). The checker mixes up **pulse and float**,
  both small slow movements. It does that on real clips too, so this is mostly the checker's limit.
- **Effect is the one real gap:** 79.7% generated against an 85.9% ceiling, so about 6 points.
- **The checker is itself much worse on unseen real clips** (70% against 92%). It learned from training clips, where some
  subjects almost always come with a given effect, so it also "expects" the usual combination. Our checker partly
  memorised combinations, the same failure we were testing the model for.
- Looking at the 24 flagged unseen failures by eye (Judgement): most are bells, hearts or the party popper flagged
  "sparkles" when none are visible, or a sweat drop that is visible but was missed. So these are checker errors. Errors
  that look like the model's: the laughing face drawn *with* tear drops when the caption asked for none, and a star whose
  sparkles are faint.

## What this does not prove

- The checker is not perfect, and its ceiling differs between splits. We compare each split with its own ceiling, not
  seen with unseen directly. The model-only error can't be read from these totals, because we didn't save which clips the
  checker got right on real data. **Next fix: save per-clip predictions and count "right on real, wrong on generated".**
- Motion is judged from shape position, size and tilt only. Speed, rhythm and loop smoothness are not scored.
- Captions are in our own template style. Free-form prompts ("make the bell jiggle") are not tested.
- One seed, one sample per caption, CFG 5 only.
- Cost note: building the checker's features ran on the CPU and took 25 of the 49 minutes. Move it to the GPU next time.

## v2 checker: separating the model's errors from the checker's (2026-10-30, ≈ $0.41, 34 min)

Raw numbers are in `../eval_v2/report.json`, with every clip's readings under `per_clip`. Same 256 + 256 stickers (same
seeds) and the same 3,000-clip reference pool as v1. What changed in `model/evaluate.py`:

1. **Effect checker:** it now subtracts the average look of the subject and the background from the CLIP embedding
   before voting. v1 voted on the whole picture, so a bell found other bells and copied their usual effect.
2. **Motion checker:** adds the main object's box (10%–90% extents), which sparkles can't drag around, and the biggest swing.
3. **Paired scoring:** a model error only counts on clips whose *real* version the checker reads correctly.
4. Motion features moved to the GPU. Pool building still took about 20 min; the remaining time is probably CPU-side CLIP
   preprocessing and loading the train cache (not profiled).

**Did the checker improve? (Verified: computed)**

| checker ceiling on real clips | v1 | v2 |
|---|---|---|
| effect, seen | 96.9% | 96.9% |
| effect, **unseen** | 85.9% | **96.5%** |
| motion, seen | 95.3% | 95.7% |
| motion, unseen | 83.6% | 83.2% (no better) |
| all correct, unseen | 70.3% | 79.7% |

The effect fix worked, and the checker no longer "expects" the usual combination. The motion fix did not help on unseen:
pulse, float and still still get mixed up on real clips (pulse → float 18 times, pulse → still 11 times).

**The model's own error rate (paired, Verified: computed)**, counted only on clips the checker reads correctly when real:

| attribute | seen | unseen |
|---|---|---|
| subject | 0.4% (1/256) | 0.4% (1/256) |
| background, accent | 0% | 0% |
| motion | 2.9% (7/245) | 7.0% (15/213) |
| effect | 7.3% (18/248) | **10.5%** (26/247) |
| **any attribute wrong** | **10.5%** (25/237; 95% range 7–15%) | **19.1%** (39/204; 14–25%) |

By held-out kind (unseen): subject × motion × effect 36 model errors in 216; UI subject × accent 3 in 40.

**What kind of errors (from `per_clip`; Verified counts, Judgement on cause):**
- Effect, unseen: "none" read as **sparkles** 9 times (mostly bells and the success tick), "sparkles" drawn as none 8,
  **tears-joy** added to the laughing face 4 times when the caption said "none", and sweat added twice. Most of these are
  the subject's usual partner in training. That is the combination bias we were testing for, and it shows up in the
  effect, not in the subject or colour.
- Motion, unseen: float ↔ pulse and still → pulse. These are the same pairs the checker confuses, so part of this 7% may
  still be checker noise.
- By eye on `model_errors_test_unseen.png` (Judgement): some are clearly real (tears on the laughing face). Some can't be
  judged from one frame (bells flagged "sparkles" with none visible in frame 8; sparkles may appear in other frames). One or
  two still look like checker errors (the "100" has a sparkle but was read as none).

**Confidence:**

| Claim | Level | How known |
|---|---|---|
| The v2 effect checker is much fairer on unseen (86% → 96.5% on real clips) | High | Verified: same clips, same pool |
| Unseen combinations fail about 2× as often as seen (19% vs 10.5%) | Medium-high | Verified counts; the 95% ranges barely touch; one seed |
| The extra errors are mainly the subject's usual effect leaking in | Medium | Verified error counts; cause is our reading of them |
| The motion error numbers | Medium-low | The motion checker is still blind on 17% of unseen real clips |

**What we would do next (Judgement):** (1) a motion checker that reads rhythm, for example a per-frame object-size curve
compared with the generator's known formula, instead of nearest neighbours. (2) Raise CFG or train with more effect
variety per subject, then check whether "effect leaks from the usual partner" drops. (3) Score several frames per clip in
the error sheets, not just frame 8.

## v3: a formula-based motion checker, and four ways to sample (2026-10-30, ≈ $1.22 incl. a crash and re-run)

Raw numbers: `../eval_v3/report.json`. Same 256 + 256 captions and seeds, same 3,000-clip pool.

**Motion checker v3** (`model/motion_check.py`, OUR code): instead of copying the label of similar clips, it fits each
motion's *exact generator formula* (bounce, pulse ±8%, float ±10 px, wobble ±12°, spin…) to the measured movement, at
every allowed speed and starting point, and picks the best fit. Turning is measured by rotating each frame's picture to
match the next, so a round face shows its turn through its eyes. On real clips (Verified, Mac test on the original mp4s,
40 per motion): reactions 90/90, animated-emoji 78/80 on unseen. **UI icons are excluded from motion scoring:** they
animate on their own (a spinner turns, a progress bar fills), and on real UI clips no checker we built could tell their
extra still / float / pulse apart (54/70).

Checker ceiling with v3 (real clips through the VAE): motion 97.2% seen / **100% unseen** (was 83% with v2).

**Correction:** the 7% unseen motion error reported in v2 was the checker, not the model. With v3, **the model's motion
error is 0% at CFG 5 on both seen and unseen** (non-UI clips). Motion is not our problem; effects are.

**Four ways to sample, compared (model error rate, paired; Verified: computed):**

| sampling | seen: any wrong | unseen: any wrong | unseen effect | unseen accent | vs CFG 5 on unseen (clips fixed / broken, sign test) |
|---|---|---|---|---|---|
| CFG 5 (r1 default) | 7.4% | 10.9% | 10.5% | 0% | n/a |
| **CFG 3** | **6.1%** | **7.3%** | **6.9%** | 0% | **11 fixed / 2 broken, p = 0.02** |
| CFG 8 | 10.2% | 14.6% | 13.8% | 0% | 4 / 13, p = 0.05 (worse) |
| CFG-Zero\* (w 5) | 7.8% | 10.5% | 10.1% | 0% | 3 / 2, no difference |
| autoguidance (w 2, weak = step 5k) | 7.8% | 11.3% | 6.9% | **6.8%** | 16 / 17, no difference |
| autoguidance (w 2, weak = step 10k) | 9.4% | 11.7% | 7.3% | **8.8%** | 15 / 17, no difference |

(v3 totals are lower than v2's 19% because v2's motion errors were checker noise and UI motion is no longer scored.)

What it means (Judgement):
- **Lower guidance leaks less.** Guidance pushes toward everything the caption *implies*, and to this model "bell" implies
  sparkles (87% of training bells have them). Turning guidance down from 5 to 3 removes about a third of the unseen
  errors at no cost. Turning it up to 8 makes leakage worse. Same mechanism, both directions.
- **CFG-Zero\* did nothing here.** Its fix targets early-step velocity errors; ours is a learned data correlation.
- **Autoguidance traded one error for another.** It fixed effects about as well as CFG 3, but broke UI accent colours
  (a blue progress bar came out another colour). The weak model's colour mistakes get amplified too. Net: no gain.
- **What guidance can't fix:** "bell + none" still comes out with sparkles 5 times in 256 at CFG 3. That needs
  rebalanced data (improvement plan #1).

Decision taken: `model/sample.py` now defaults to **CFG 3** (was 5), which also changes r2's training snapshots.
Not tested: CFG 2 or 4, or a CFG schedule that's low early and high late. Each costs about $0.15 with the pool now cached
(`/workspace/runs/r1/eval_pool_3000.pt`).

Confidence: CFG 3 beats CFG 5 on unseen: **Medium-high** (p = 0.02, one seed, 256 captions). CFG-Zero\* and autoguidance
give no gain at these settings: **Medium** (one weight each, not swept). Model motion error ≈ 0% on emoji and reactions:
**High** for "no big motion problem" (checker at 97–100% on real clips); UI motion is **not measured**.

Incident: the first v3 run died because the 30 GB volume filled up. `torch.save(gen_vid[:32])` saved the whole
256-clip tensor (a slice shares its parent's storage), about 850 MB a file. Fixed with `.clone()`, and the files were
deleted.
