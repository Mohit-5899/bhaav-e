"""Rebalance the training split for r2.  (OUR code.)

Problem (runs/r1/eval/RESULTS.md): each subject's effects were lopsided in training (bell: sparkles 87%, none 13%), and
the model learned "bell means sparkles". Fix: within every subject, give each of its effects the same number of training
clips.

To keep r1 vs r2 a fair comparison, only the *balance* changes:
  - train stays the same size (61,569 clips); val, test_seen and test_unseen are untouched
  - for each subject, target = its train count / number of its effects
  - cells above target: drop random clips down to target (they stay on disk, just not in train)
  - cells below target: generate new clips up to target (generator `fill` command, never a held-out combination)

Two steps:
  python generator/make_r2_split.py plan --workers 10      -> data/r2/plan_<k>.json (one per generator process)
  (run the generator fill command for each plan file)
  python generator/make_r2_split.py split                  -> data/r2/splits/ (train.txt new; others copied)
"""
import argparse, collections, json, random, shutil
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
R2 = DATA / "r2"


def cells(train, meta):
    by = collections.defaultdict(list)
    for i in train:
        m = meta[i]
        by[(m["category"], m["subcategory"], m["effect"])].append(i)
    subj = collections.defaultdict(dict)
    for (c, s, e), v in by.items():
        subj[(c, s)][e] = v
    return subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "split"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    train = (DATA / "splits" / "train.txt").read_text().split()
    meta = {}
    for f in (DATA / "clips").glob("*/*/*.json"):
        m = json.loads(f.read_text())
        meta[m["id"]] = m
    subj = cells(train, meta)
    R2.mkdir(exist_ok=True)

    if a.cmd == "plan":
        need, drop = [], {}
        for (c, s), effs in sorted(subj.items()):
            target = round(sum(len(v) for v in effs.values()) / len(effs))
            for e, v in sorted(effs.items()):
                if len(v) < target:
                    need.append([c, s, e, target - len(v)])
                elif len(v) > target:
                    drop[f"{c}/{s}/{e}"] = len(v) - target
        total = sum(n for *_, n in need)
        # split every cell across workers so each process makes a balanced mix (and finishes at the same time)
        plans = [[] for _ in range(a.workers)]
        for c, s, e, n in need:
            for k in range(a.workers):
                share = n // a.workers + (1 if k < n % a.workers else 0)
                if share:
                    plans[k].append([c, s, e, share])
        for k, p in enumerate(plans):
            (R2 / f"plan_{k}.json").write_text(json.dumps(p))
        (R2 / "rebalance.json").write_text(json.dumps(dict(generate=need, drop=drop, total_new=total), indent=1))
        print(f"generate {total} clips in {a.workers} plans; drop {sum(drop.values())}")

    else:
        rng = random.Random(a.seed)
        reb = json.loads((R2 / "rebalance.json").read_text())
        new = [m["id"] for m in meta.values() if str(m.get("batch", "")).startswith("f")]
        dropped = set()
        for key, n in reb["drop"].items():
            c, s, e = key.split("/")
            pool = sorted(subj[(c, s)][e])
            rng.shuffle(pool)
            dropped.update(pool[:n])
        r2_train = sorted(set(train) - dropped) + sorted(new)
        out = R2 / "splits"
        out.mkdir(exist_ok=True)
        (out / "train.txt").write_text("\n".join(r2_train) + "\n")
        for f in ("val.txt", "test_seen.txt", "test_unseen.txt", "heldout.json", "exclude.txt"):
            shutil.copy(DATA / "splits" / f, out / f)
        # checks: same size, no test clip, no held-out combination, every subject's effects now equal
        held = json.loads((DATA / "splits" / "heldout.json").read_text())
        tests = set()
        for f in ("val.txt", "test_seen.txt", "test_unseen.txt"):
            tests.update((DATA / "splits" / f).read_text().split())
        assert not tests & set(r2_train), "a val/test clip is in r2 train"
        for i in new:
            m = meta[i]
            assert [m["subcategory"], m["motion"], m["effect"]] not in held[m["category"]]["subject_motion_effect"], i
            assert [m["subcategory"], m.get("accent")] not in held[m["category"]]["subject_accent"], i
        after = cells(r2_train, meta)
        worst = max(max(len(v) for v in e.values()) - min(len(v) for v in e.values()) for e in after.values())
        all_ids = sorted(set((DATA / "splits" / "all_ids.txt").read_text().split()) | set(new))
        (out / "all_ids.txt").write_text("\n".join(all_ids) + "\n")
        (out / "new_ids.txt").write_text("\n".join(sorted(new)) + "\n")
        print(f"r2 train {len(r2_train)} (r1 {len(train)}): dropped {len(dropped)}, new {len(new)}; "
              f"largest effect-count gap within a subject: {worst}")


if __name__ == "__main__":
    main()
