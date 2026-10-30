"""Bhaav-E train / val / test splits (our code).

Four lists of clip ids in data/splits/:
  train.txt        what the model learns from
  val.txt          ~5%, same combinations as train; for watching training
  test_seen.txt    ~5%, same combinations as train; final score on familiar combinations
  test_unseen.txt  ~8%, combinations never in train; final score on combining known parts

A combination is subject x motion x effect (e.g. angry + float + steam). For ui we also hold out
subject x accent pairs (e.g. purple spinner). A combination is only held out if every part of it
still appears in train, paired with the other parts: so test_unseen asks "can it combine things it
knows?", never "can it draw something it has never seen?".

Usage: .venv/bin/python generator/make_splits.py [--seed 0]
"""

import argparse
import collections
import glob
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
UNSEEN_SHARE, VAL_SHARE, TEST_SEEN_SHARE = 0.08, 0.05, 0.05


def load_clips():
    excluded = {line.split()[0] for line in (DATA / "splits" / "exclude.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#")}
    clips = [json.loads(Path(p).read_text()) for p in glob.glob(str(DATA / "clips" / "*" / "*" / "*.json"))]
    return [c for c in clips if c["id"] not in excluded], excluded


def pairs_of(key):
    """The 2-part pairs inside a combination: (subject, motion), (subject, effect), (motion, effect)."""
    return [("sm", key[0], key[1]), ("se", key[0], key[2]), ("me", key[1], key[2])]


def hold_out_combos(counts, budget, rng):
    """Greedily hold out whole combinations while every pair inside them keeps a combination in train."""
    pair_left = collections.Counter(p for k in counts for p in pairs_of(k))
    held, used = [], 0
    for key in rng.sample(sorted(counts), len(counts)):
        if used + counts[key] > budget:
            continue
        if all(pair_left[p] > 1 for p in pairs_of(key)):
            for p in pairs_of(key):
                pair_left[p] -= 1
            held.append(key)
            used += counts[key]
    return held, used


def hold_out_accents(counts, budget, rng):
    """ui only: hold out subject x accent pairs; each subject keeps >= 2 accents, each accent >= 3 subjects."""
    subj_left = collections.Counter(s for s, _ in counts)
    acc_left = collections.Counter(a for _, a in counts)
    held, used = [], 0
    for s, a in rng.sample(sorted(counts), len(counts)):
        if used + counts[(s, a)] > budget or subj_left[s] <= 2 or acc_left[a] <= 3:
            continue
        subj_left[s] -= 1
        acc_left[a] -= 1
        held.append((s, a))
        used += counts[(s, a)]
    return held, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    rng = random.Random(seed)

    clips, excluded = load_clips()
    by_cat = collections.defaultdict(list)
    for c in clips:
        by_cat[c["category"]].append(c)

    splits = {"train": [], "val": [], "test_seen": [], "test_unseen": []}
    held_report = {}
    for cat in sorted(by_cat):
        cs = by_cat[cat]
        combo = lambda c: (c["subcategory"], c["motion"], c["effect"])
        counts = collections.Counter(combo(c) for c in cs)
        budget = UNSEEN_SHARE * len(cs)
        if cat == "ui":                              # half the budget on combinations, half on colours
            held_sme, used = hold_out_combos(counts, budget / 2, rng)
            acc_counts = collections.Counter((c["subcategory"], c["accent"]) for c in cs
                                             if combo(c) not in set(held_sme))
            held_acc, _ = hold_out_accents(acc_counts, budget - used, rng)
        else:
            held_sme, _ = hold_out_combos(counts, budget, rng)
            held_acc = []
        held_sme, held_acc = set(held_sme), set(held_acc)
        held_report[cat] = {"subject_motion_effect": sorted(map(list, held_sme)),
                            "subject_accent": sorted(map(list, held_acc))}

        is_unseen = lambda c: combo(c) in held_sme or (c["subcategory"], c.get("accent")) in held_acc
        unseen = [c for c in cs if is_unseen(c)]
        seen = sorted((c for c in cs if not is_unseen(c)), key=lambda c: c["id"])   # sort: same input order every run
        rng.shuffle(seen)
        n_val, n_ts = round(VAL_SHARE * len(cs)), round(TEST_SEEN_SHARE * len(cs))
        splits["test_unseen"] += [c["id"] for c in unseen]
        splits["val"] += [c["id"] for c in seen[:n_val]]
        splits["test_seen"] += [c["id"] for c in seen[n_val:n_val + n_ts]]
        splits["train"] += [c["id"] for c in seen[n_val + n_ts:]]

    out = DATA / "splits"
    for name, ids in splits.items():
        (out / f"{name}.txt").write_text("\n".join(sorted(ids)) + "\n")
    (out / "heldout.json").write_text(json.dumps({"seed": seed, "excluded": len(excluded), **held_report}, indent=1) + "\n")

    total = sum(len(v) for v in splits.values())
    for name, ids in splits.items():
        print(f"{name:12s} {len(ids):6d}  {100 * len(ids) / total:5.1f}%")
    for cat, h in held_report.items():
        print(f"{cat}: held out {len(h['subject_motion_effect'])} combinations, {len(h['subject_accent'])} subject x accent pairs")


if __name__ == "__main__":
    main()
