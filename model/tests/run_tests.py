"""Mac CPU tests for our training code.  (OUR code.)  Run: .venv/bin/python model/tests/run_tests.py

1. size: the full s210 model has about the same number of parameters as Drishya's (211.92M), so it is the same blueprint
2. shapes: input and output shapes match; at the start the model predicts exactly 0 (adaLN-zero + zero output layer)
3. patchify then unpatchify gives back the exact input (cutting into pieces loses nothing)
4. Muon's orthogonalize really makes all singular values close to 1
5. overfit: a tiny model on 8 fixed clips, with fixed noise and t, drives the loss towards 0 through the real train.py
   (uses a fake cache in the exact on-disk format of steps 3b/3c)
"""
import json, subprocess, sys, tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from muon import orthogonalize                     # noqa: E402
from videodit import SIZES, VideoDiT, count_params  # noqa: E402

DRISHYA_PARAMS_M = 211.92   # from Drishya's published stage-3 log (params_M)


def test_size():
    with torch.device("meta"):
        n = count_params(VideoDiT(SIZES["s210"])) / 1e6
    gap = 100 * (n - DRISHYA_PARAMS_M) / DRISHYA_PARAMS_M
    print(f"1. size: ours {n:.2f}M vs Drishya {DRISHYA_PARAMS_M}M ({gap:+.2f}%)")
    assert abs(gap) < 0.5


def test_shapes():
    torch.manual_seed(0)
    m = VideoDiT(SIZES["tiny"])
    x = torch.randn(2, 48, 5, 16, 16)
    txt, mask = torch.randn(2, 35, 768), torch.ones(2, 35, dtype=torch.uint8)
    mask[0, 10:] = 0
    v = m(x, torch.tensor([0.3, 0.8]), txt, mask, torch.tensor([True, False]))
    assert v.shape == x.shape and v.abs().max().item() == 0.0
    print("2. shapes: ok; first prediction is exactly 0")


def test_patchify():
    m = VideoDiT(SIZES["tiny"])
    x = torch.randn(3, 48, 5, 16, 16)
    p = m.patchify(x)
    assert p.shape == (3, 320, 192) and torch.equal(m.unpatchify(p), x)
    print("3. patchify: 320 pieces of 192 numbers, round trip exact")


def test_muon():
    # Uneven directions (about 180x between strongest and weakest). Muon's 5 Newton-Schulz steps are designed to give
    # "roughly equal" (about 0.7-1.2), not exactly 1: faster, and good enough. Very extreme spreads (1,000x+) leave
    # the weakest directions short after 5 steps; that is expected, not a bug.
    torch.manual_seed(0)
    G = torch.randn(256, 128) @ torch.diag(torch.logspace(0, 2, 128))
    s = torch.linalg.svdvals(orthogonalize(G).float())
    print(f"4. muon: singular values before {torch.linalg.svdvals(G).min():.1f}..{torch.linalg.svdvals(G).max():.0f}, "
          f"after {s.min():.2f}..{s.max():.2f}")
    assert 0.6 < s.min() and s.max() < 1.25


def fake_cache(root, n=24, L=35):
    """Same files and formats as steps 3b/3c, with random numbers."""
    ids = [f"reactions-crying-{i:06d}" for i in range(n)]
    lat = root / "latents"; lat.mkdir(parents=True)
    torch.manual_seed(0)
    save_file({"latents": torch.randn(n, 48, 5, 16, 16).to(torch.bfloat16)}, str(lat / "shard_00000.safetensors"),
              metadata={"ids": json.dumps(ids)})
    (lat / "vae_config.json").write_text(json.dumps({"latents_mean": [0.1] * 48, "latents_std": [1.2] * 48}))
    txt = root / "text"; txt.mkdir()
    S = 6
    mask = torch.zeros(S, L, dtype=torch.uint8); mask[:, :12] = 1; mask[0, 1:] = 0
    save_file({"emb": torch.randn(S, L, 768).to(torch.bfloat16), "mask": mask}, str(txt / "text.safetensors"))
    (txt / "captions.json").write_text(json.dumps([""] + [f"caption {i}" for i in range(1, S)]))
    (txt / "clip_captions.json").write_text(json.dumps({c: [1 + i % 5, 1 + (i + 1) % 5, 1 + (i + 2) % 5]
                                                       for i, c in enumerate(ids)}))
    sp = root / "splits"; sp.mkdir()
    (sp / "train.txt").write_text("\n".join(ids[:16]) + "\n")
    (sp / "val.txt").write_text("\n".join(ids[16:]) + "\n")
    (sp / "test_seen.txt").write_text("\n".join(ids[16:20]) + "\n")       # train.py's snapshot samples prompts from these
    (sp / "test_unseen.txt").write_text("\n".join(ids[20:]) + "\n")
    return sp


def test_overfit():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sp = fake_cache(d / "cache")
        cmd = [sys.executable, str(HERE / "train.py"), "--cache", str(d / "cache"), "--splits", str(sp),
               "--out", str(d / "run"), "--size", "tiny", "--overfit", "8", "--steps", "400", "--log_every", "50",
               "--adam_lr", "1e-3", "--muon_lr", "0.02", "--device", "cpu"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        assert r.returncode == 0, r.stderr[-2000:]
        rows = [json.loads(l) for l in (d / "run" / "log.jsonl").read_text().splitlines()]
        first, last = rows[0]["loss"], rows[-1]["loss"]
        print(f"5. overfit: loss {first:.4f} (step {rows[0]['step']}) -> {last:.4f} (step {rows[-1]['step']}), "
              f"{100 * (1 - last / first):.1f}% lower")
        assert last < 0.1 * first
        # resume: running again with more steps must continue from the checkpoint, not restart
        r2 = subprocess.run(cmd[:-12] + ["--steps", "450"] + cmd[-10:], capture_output=True, text=True, cwd=HERE)
        assert r2.returncode == 0 and "resumed at step 400" in r2.stdout, (r2.stdout[-500:], r2.stderr[-500:])
        print("   resume from checkpoint: ok")


if __name__ == "__main__":
    test_size(); test_shapes(); test_patchify(); test_muon(); test_overfit()
    print("ALL TESTS PASSED")
