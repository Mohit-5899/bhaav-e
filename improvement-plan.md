# Bhaav-E: how to make r1 better

Written 2026-10-30 after the r1 evaluation (`runs/r1/eval/RESULTS.md`). Each idea is matched to a failure we actually
measured, ranked by expected gain per dollar. Web sources are listed at the end; claims taken from papers are
**Reported** (read in abstracts and summaries, not re-checked against their tables).

## Where r1 stands (Verified: our evaluation)

| attribute | model error, seen | model error, unseen | verdict |
|---|---|---|---|
| subject, background, UI accent | 0–0.4% | 0–0.4% | solved |
| motion | 2.9% | 7.0% | good; our checker is still weak here (blind on 17% of unseen real clips) |
| effect | 7.3% | 10.5% | **main weakness** |
| any attribute wrong | 10.5% | 19.1% | unseen fails about 2× as often |

Validation loss was still falling at step 40,000 (about 0.0004 per 1,000 steps), so the model is not finished learning.

## Root cause of the effect errors (Verified: counted in train.txt)

The model adds the effect a subject *usually* has, because our generator gave subjects lopsided effect mixes:

| subject | effects in training |
|---|---|
| bell | sparkles 87%, none 13% |
| success tick | confetti 52%, sparkles 42%, none 5% |
| laughing face | tears-joy 44%, sparkles 43%, none 14% |
| shocked face | sparkles 46%, sweat 36%, none 18% |

"Bell + none" is rare in training, and "bell + pulse/float + none" is held out entirely. So the model learned that bells
come with sparkles. This is a data problem first, not an architecture problem.

## Ranked techniques

| # | technique | targets | what it is, in one line | cost | evidence |
|---|---|---|---|---|---|
| 1 | **Rebalance the data** | effect leakage | generator picks effect and motion independently of subject, roughly uniform, with plenty of "none" | CPU time on the Mac + re-cache ≈ $1 + retrain ≈ $2.60 | Verified skew above; Okawa et al. 2023 found rare combinations need far more training steps to compose (Reported) |
| 2 | **Autoguidance** | effect leakage, overall quality | at sampling, guide with a weaker copy of the model (our step-5,000 snapshot) instead of the empty caption | **$0 training**, about $0.40 to evaluate | Karras et al. 2024: record ImageNet FID (Reported). We already own the "bad version" |
| 3 | **CFG-Zero\*** | sampling quality | per-step corrected guidance scale, plus skip guidance on the first steps; built for flow-matching models like ours | $0 training, about $0.40 to evaluate | tested on Wan-2.1 video (Reported) |
| 4 | **Fix the motion checker first** | measurement | score motion against the generator's own formulas (we know exactly how pulse and float move), not nearest neighbours | $0, Mac only | Verified need: the checker confuses pulse, float and still on real clips |
| 5 | **Motion supervision (VideoJAM idea)** | motion | the model also predicts a motion signal alongside the video, so it must learn movement; Inner-Guidance uses it at sampling | code change + retrain ≈ $3 | ICML 2025 (Reported). **Our edge:** our data is code-made, so we have *exact* motion for free (position, scale, angle per frame), with no optical-flow estimator needed |
| 6 | **Train longer** | everything a little | 80k steps instead of 40k | ≈ $5.20 | Verified: val loss still falling |
| 7 | FlowMo | motion smoothness (jitter) | training-free: at sampling, nudge toward low frame-to-frame patch variance | $0 training | NeurIPS 2025 (Reported). Low priority: our errors are the *wrong kind* of motion, not jittery motion |
| 8 | REPA | training speed | align middle DiT layers to a frozen vision encoder (DINOv2) | small code change | about 17× faster on images (Reported); video results mixed (frame-by-frame drift, needs CREPA-style fixes). Low priority at our size |
| 9 | Frame-difference loss | motion | extra loss on the difference between consecutive frames | small code change | mixed: one paper found plain denoising loss better (Reported). Skip unless 5 fails |

What we would **not** change yet (Judgement): the text encoder (captions are short templates and the subject is already
99.6% right), and the model size (the errors are data-shaped, not capacity-shaped).

## Results of the free steps (2026-10-30, details in `runs/r1/eval/RESULTS.md`, v3 section)

- #4 motion checker: done. Formula-based; reads real emoji and reaction motion 97–100% right. **Model motion error ≈ 0%**
  on those, so motion is not the problem (UI motion not measurable).
- #2 autoguidance: **no net gain**. Fixes effects, breaks UI accent colours.
- #3 CFG-Zero\*: **no change**.
- Bonus: **CFG 3 instead of 5** cuts unseen errors 10.9% → 7.3% (p = 0.02). Now the default.
- So the plan moves to **r2 = rebalanced data**. A motion head (#5) is no longer justified by our numbers; keep it for when
  we add harder motions.

## Result of r2 (2026-10-30, details in `runs/r2/RESULTS.md`)

- #1 rebalanced data, alone (no motion head): unseen errors at CFG 5 **10.9% → 5.7%** (p = 0.01). At CFG 3, unseen
  7.3% → 5.3% and seen 6.1% → 9.0%, neither significant. "Added an effect not asked for" on unseen halved (12 → 6 clips).
- Eye check of the error sheets: the **effect checker misreads many generated stickers** (visible exhaust, sweat, tears,
  zzz read as "none"). Effect differences of a few points are not trustworthy yet.
- New order: (a) effect checker v4 against the generator's particle formulas ($0); (b) re-score r1 and r2 with it;
  (c) only then autoguidance on r2 or longer training.

## Result of effect checker v4 (2026-10-30, details in `runs/r2/RESULTS.md`, v4 section)

Most of the "omission" errors were the v3 checker misreading small particles. With v4: seen error is 2.0% for both runs;
unseen r1 5.3% vs r2 3.3% (p = 0.23). The one real failure left is **adding the subject's usual effect**
(r1 9 clips, r2 3). The omission fixes listed above (loss weighting, explicit "no effect" captions) are on hold until a
bigger test set shows omissions are real. Release model: r2. (Verified counts; Judgement on what to do.)

## Suggested order (Judgement, written before the free steps)

1. **Free first:** fix the motion checker (#4), then try autoguidance and CFG-Zero\* on r1 (#2, #3). One eval each,
   ≈ $0.80 total. If effect leakage drops, guidance alone helps.
2. **r2 = rebalanced data (#1) + motion head (#5), 40k steps,** about $4. Compare with r1 on the same eval: the decision
   rule is unseen "any attribute wrong" must fall from 19% to below 12%, with seen not worse than 10.5%.
3. Only then train longer (#6) on whichever recipe wins.

One change at a time where it's cheap. #1 and #5 are bundled in r2 to save a run, so if r2 wins we can't credit either one
alone. Accept that, or spend about $2.60 more on a separate run.

## Sources

- VideoJAM (Chefer et al., ICML 2025): https://arxiv.org/abs/2502.02492
- FlowMo (Shaulov et al., NeurIPS 2025): https://arxiv.org/abs/2506.01144
- Autoguidance (Karras et al., NeurIPS 2024): https://arxiv.org/abs/2406.02507
- CFG-Zero\* (Fan et al., 2025): https://arxiv.org/abs/2503.18886
- Compositional abilities emerge multiplicatively (Okawa et al., NeurIPS 2023): https://arxiv.org/abs/2310.09336
- REPA overview and video caveats: https://www.emergentmind.com/topics/repa
- T2V-CompBench (CVPR 2025), compositional video benchmark: https://arxiv.org/abs/2407.14505
- Latent Temporal Discrepancy (motion-weighted loss): https://arxiv.org/abs/2601.20504
- Negative result on frame-difference loss: https://arxiv.org/abs/2603.17398
