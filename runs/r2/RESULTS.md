# r2 vs r1: does rebalancing the data fix effect leakage?

Run 2026-10-30. r2 = r1's exact recipe (211.92M VideoDiT, 40k steps, same schedule, same seed handling) with **one change:
the training split** (`generator/make_r2_split.py`). Within every subject, each effect now has the same number of
training clips. Train size is unchanged (61,571 vs 61,569). Val and both test sets are the same clips.

Evaluation: same harness as r1's v3 (`model/evaluate.py`), same 256 + 256 test clips (checked: identical id lists),
**same checker pool file** (copied from r1, so both models are judged by identical checkers), 50 Euler steps, shift 5.
Per-clip comparison: `compare_r1_r2.py` (ours, in the session scratchpad), exact two-sided sign test on clips where one
run is right and the other wrong.

> **Update, same day: re-scored with effect checker v4 (section at the end).** v4 reads effects from particles, not
> CLIP. It shows that **most "left out the effect" errors below were the v3 checker, not the model**: seen-set effect
> errors drop from 15 (r1) / 20 (r2) to **3 / 3**. With v4, r1 and r2 tie on seen (2.0% each); on unseen r2 has 3.3%
> vs r1's 5.3% (not significant, p = 0.23). The real remaining failure is **adding the usual effect** on unseen
> combinations, and r2 cuts it from 9 clips to 3. The "attack omission next" plan below is withdrawn.

## One-paragraph answer

Rebalancing **halved measured errors on unseen combinations at CFG 5** (10.9% → 5.7%, p = 0.011) and made the model
far less sensitive to the guidance weight. At our default CFG 3 the unseen gain is smaller (7.3% → 5.3%, not
significant) and seen got worse (6.1% → 9.0%, not significant), so the totals roughly tie. But looking at the error
sheets by eye shows **the effect checker misreads many generated stickers** (a visible rocket exhaust read as "none",
and so on), so a few points of difference is not trustworthy. Verdict: **keep r2's balanced data as the base** (the one
significant result favours it, and nothing significant goes against it), and **fix the effect checker before the next
comparison**. (Numbers Verified; verdict Judgement, Medium.)

## Part A: loss on held-out clips at step 40,000 (Verified: report.json)

| | test_seen | test_unseen | gap |
|---|---|---|---|
| r1 | 0.0467 | 0.0504 | 0.0037 (+8%) |
| r2 | 0.0475 | 0.0503 | 0.0029 (+6%) |

Unseen loss is the same; seen loss rose 1.6%; the seen–unseen gap shrank by a fifth. The validation loss logged during
training also ended slightly higher (0.0562 vs 0.0551). Expected: val and test_seen follow r1's lopsided mix, and r2 saw
fewer of those common combinations.

## Part B: model error (clips the checker reads correctly when real) — Verified

| split | guidance | r1 | r2 | r2 fixed | r2 broke | sign-test p |
|---|---|---|---|---|---|---|
| unseen | CFG 3 | 7.3% (18/247) | **5.3%** (13/247) | 12 | 7 | 0.36 |
| unseen | CFG 5 | 10.9% (27/247) | **5.7%** (14/247) | 18 | 5 | **0.011** |
| seen | CFG 3 | **6.1%** (15/244) | 9.0% (22/244) | 7 | 14 | 0.19 |
| seen | CFG 5 | **7.4%** (18/244) | 9.8% (24/244) | 8 | 14 | 0.29 |

Subject, background, UI accent and motion stay at 0–0.9% for both runs. Effect is still the only attribute that
matters; every row above is almost entirely effect errors.

Pre-registered rule (`docs/improvement-plan.md`): unseen "any wrong" below 12%, seen not worse than 10.5%. **r2 passes. So
does r1 at CFG 3**, which we only found after writing the rule. The rule no longer separates the two runs.

## What the mistakes are (CFG 3, Verified: counted per clip)

| | r1 | r2 |
|---|---|---|
| unseen: bell asked "none", got sparkles | 5 | 3 |
| unseen: laughing face asked "none", got tears-joy | 4 | **0** |
| unseen: shocked face asked "none", got sweat | 2 | 2 |
| unseen: **added** an effect not asked for (all subjects) | 12 of 17 effect errors | **6** of 12 |
| unseen: **left out** the asked effect | 4 | 5 |
| seen: **left out** the asked effect | 12 of 15 effect errors | 14 of 20 |
| seen: added an effect not asked for | 2 | 2 |

Reading (Judgement, Medium): there are two separate failures. **Addition** ("bells usually sparkle") is a data-prior
problem, and rebalancing halved it, as designed. **Omission** (asked for sweat or sparkles, got a clean sticker) was
already r1's main seen error and is slightly worse in r2, plausibly because rebalancing raised the share of "none" for
many subjects (bell: 13% → 50%). Omission looks like **the effect is too faint or too weakly bound to its caption word**,
which no amount of rebalancing fixes.

## Visual audit of the error sheets (Judgement, eyeballed; Medium-low confidence)

We looked at every CFG 3 error sheet (`eval_v3/model_errors_*.png`, one middle frame per clip, both runs). **Many
"effect errors" are the checker's mistakes, in both directions:**

- **Left out, but visible:** rocket exhaust puffs, facepalm sweat drop, crying-face tears, zzz above the sleepy face, a
  thought bubble, confetti around the 100. Seen, CFG 3: about **7 of r1's 15** effect errors and **7 of r2's 20** show the
  asked-for effect clearly.
- **Added, but not visible:** r2's three "bell asked none, got sparkles" and two "shocked face, got sweat" show no
  sparkle or sweat in the middle frame. Sparkles twinkle, so they may appear on other frames; we can't tell from one frame.

So the effect checker, which reads 96–97% of *real* clips right, is much less reliable on *generated* clips, probably
because generated effects are smaller, softer, or blurrier than the real ones it was calibrated on. Removing the obvious
misreads leaves roughly r1 ≈ 8 and r2 ≈ 13 real effect errors on seen (out of ~244). That is a small gap and still within
noise. **The headline percentages above overstate both runs' effect errors, and the r1-vs-r2 seen gap may be mostly
checker noise.**

Lesson: calibrating a checker on real data is necessary but not enough. Check a sample of its verdicts on *generated*
data by eye before trusting a small difference.

## Our critique of this comparison

- **One seed per run, 256 clips per split.** Only one of four comparisons is significant. A 2–3 point swing on seen is
  inside the noise. (High confidence this is a limitation.)
- **The seen test set favours r1.** test_seen follows r1's training mix, so it rewards knowing that bells usually
  sparkle. A balanced seen set would be fairer to r2. We did not build one.
- **CFG sensitivity changed.** r1 needed CFG 3 to look good; r2 is the same at 3 and 5. That is a real practical
  gain (less tuning), but it means "r1 at its best setting vs r2" is the honest comparison, and there the totals tie.
- **Omission may be partly the checker.** The effect checker reads 3–4% of *real* effects wrong; a generated effect that
  is present but faint could be read as "none". Look at `eval_v3/model_errors_*.png` before spending on a fix.
  (Medium confidence the omissions are mostly real.)
- **Not tested:** whether the two fixes stack (r2 + autoguidance), longer training, or a caption change that makes
  "no effect" explicit.

## Our decision

| decision | choice | reason | how we'd check |
|---|---|---|---|
| keep r2's data balance? | **yes, as the new base** | big significant unseen gain at CFG 5, robust to guidance, gap shrank; seen loss is small and not significant | balanced seen test set |
| default guidance | keep CFG 3 | ties or beats 5 on both runs | — |
| next fix | ~~attack omission~~ withdrawn after v4: omissions were mostly the checker | — | v4 section |

Next experiments, cheapest first (Judgement):
1. **Fix the effect checker first** ($0, Mac): score each effect over all 17 frames against the generator's own
   particle formulas, as we did for motion (v3). Until then, effect differences of a few points can't be trusted.
2. **Autoguidance on r2** ($0.40): on r1 it cut effect errors but broke UI accents.
3. If effects are faint: **weight the loss towards effect pixels** or train longer (r2 to 80k ≈ $5.20; val loss still
   falling). If they are missing outright: make captions name the effect more strongly, e.g. say "no effect" explicitly
   so "none" is a word the model can attend to, not just an absence.

## Cost (Verified: pod uptime 17,458 s × $0.72/h)

Pod `bhaav-r2`: ≈ $3.49 for prep + training + this eval. Plus ≈ $0.36 for the failed first start. **r2 total ≈ $3.85.**

## Re-score with effect checker v4 (2026-10-30)

**What changed.** `model/effect_check.py` (ours). The v3 checker read effects from CLIP pictures of the whole frame; a
6-pixel sweat drop barely moves those. v4 looks for the particles themselves: find the sticker's body (the biggest blob
that differs from the background), call everything else that differs a particle, then measure particle area per colour
family (blue drops, pink hearts, yellow sparkles, grey steam...), where the particles sit relative to the body, and how
many pieces there are, over all 17 frames. It then asks "which real clips of *this same subject* look most like this?"
(7 nearest neighbours). Same 256 + 256 test clips, same generated videos per run (seed fixed), 50 Euler steps, shift 5.
Run on pod `bhaav-eval4` (L4, $0.59/h, 7,634 s ≈ **$1.25**). Raw output: `runs/eval_v4.out`, `runs/r*/eval_v4/`.

**Checker ceiling (real clips after the VAE round trip, Verified: report.json).** Effect read correctly on 98.4% of
seen and 94.9% of unseen real clips (v3: 96–97%). On raw, never-encoded clips v4 scored 97.8% / 96.8%, so the VAE
round trip costs it about 2 points on unseen.

**Model error, effect v4, clips the checker reads correctly when real (Verified: `compare_v4.py`, ours).**

| split | guidance | r1 | r2 | r2 fixed | r2 broke | sign-test p |
|---|---|---|---|---|---|---|
| seen | CFG 3 | 2.0% (5/248) | 2.0% (5/248) | 2 | 2 | 1.00 |
| seen | CFG 5 | 1.2% (3/248) | 2.0% (5/248) | 1 | 3 | 0.63 |
| unseen | CFG 3 | 5.3% (13/243) | **3.3%** (8/243) | 8 | 3 | 0.23 |
| unseen | CFG 5 | 5.8% (14/243) | **3.7%** (9/243) | 9 | 4 | 0.27 |

Effect only, unseen, CFG 3: r1 12 errors, r2 6 (p = 0.11). Of those, **additions** (asked for no effect, got one):
r1 9, r2 3; **omissions**: 3 each. The biggest single r1 mistake is still "laughing face, asked none, got tears of
joy" (4 clips); r2 has none.

**v3 vs v4 on the very same generated videos (CFG 3, Verified).** Seen: v3 counted 15 (r1) and 20 (r2) effect
errors, v4 counts 3 and 3. In 12 (r1) and 16 (r2) of those clips v3 said "none" where v4 reads the asked-for effect.
This matches the eye check above. **Correction:** earlier in this note we called omission "the main seen error" and
planned to fix it. That was mostly checker error. (High confidence in the counts; Medium-high that v4 is right where
they disagree, since the earlier eye check found the effects visible.)

**Eye check of the v4 unseen error sheets (Judgement, Medium-low).** The laughing-face tears r1 adds are clearly
visible: real model errors. But v4 still misses a sweat drop that sits *on* the face (facepalm, shocked face): about 3
of r1's 13 unseen errors and 1–2 of r2's look like checker misses. A star's sparkles and a faint sparkle near the 100
are also present but small. Adjusted guess: r1 ≈ 10, r2 ≈ 6 real unseen errors out of 243.

**What this means (Judgement).**
- Both models are better than v2/v3 said. With v4: about 2% error on seen combinations and 3–6% on unseen.
- The one real, repeatable failure is **adding the effect a subject usually has** in training. Rebalancing (r2) cuts
  it by two thirds. The direction is consistent across both guidance weights, but one seed and ~240 clips cannot make
  it significant.
- r2 is the release model.

**Updated decision.**

| decision | choice | reason | how we'd check |
|---|---|---|---|
| release | **r2**, CFG 3 | ties on seen, fewer unseen errors at both CFGs, fewer additions | — |
| effect checker | v4 for future runs | 12–16 fewer false "none" per 250 clips | fix on-face drops: compare the body's blue area with the subject's own real clips frame by frame |
| next experiment | larger unseen test (1,000 clips) or a second seed | 8 vs 3 discordant clips is too few to decide | sign test at n ≈ 1,000 |

Cost so far for r2 + checker v4: $3.85 + $1.25 = **$5.10**.
