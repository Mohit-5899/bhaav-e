"""Step 3c: encode every caption once with frozen Flan-T5-Base -> text cache.  (OUR code.)

Why: T5 never changes, so running it on every training step would redo the same work. Encode once, save.
Why unique sentences: each clip has 3 caption lines, but many clips share the same sentence
(75K clips x 3 lines, only ~26K different sentences). Encoding each sentence once is ~9x less work and disk.

Usage on the pod:
    python encode_captions.py --clips /root/data/clips --ids /root/data/splits/all_ids.txt --out /workspace/cache/text
Output (in --out):
    text.safetensors    "emb":  bf16 [N, L, 768]  T5 output per token (padding positions are zeros)
                        "mask": uint8 [N, L]      1 = real token, 0 = padding
    captions.json       the N sentences, in row order; row 0 is "" (the empty caption, for caption dropout)
    clip_captions.json  {clip id: [row, row, row]}  which rows hold that clip's caption lines
    info.json           max length L, token-length stats, model name
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, T5EncoderModel

MODEL = "google/flan-t5-base"   # the frozen text encoder Drishya used (110M, 768-dim)
MAX_LEN = 77                    # Drishya's cap; we use the shortest length that fits every caption, up to this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids = Path(a.ids).read_text().split()
    paths = {p.stem: p for p in Path(a.clips).rglob("*.txt")}

    clip_lines = {c: [l.strip() for l in paths[c].read_text().splitlines() if l.strip()] for c in ids}
    sentences = [""] + sorted({l for ls in clip_lines.values() for l in ls})
    row = {s: i for i, s in enumerate(sentences)}

    tok = AutoTokenizer.from_pretrained(MODEL)
    lengths = [len(tok(s).input_ids) for s in sentences]          # includes the end-of-text token
    L = min(MAX_LEN, max(lengths))
    truncated = sum(n > L for n in lengths)

    enc = T5EncoderModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    embs, masks = [], []
    for i in range(0, len(sentences), a.batch):
        t = tok(sentences[i:i + a.batch], padding="max_length", truncation=True, max_length=L, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = enc(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state
        h = h * t.attention_mask.unsqueeze(-1)                     # zero the padding positions
        embs.append(h.to(torch.bfloat16).cpu()); masks.append(t.attention_mask.to(torch.uint8).cpu())
    emb, mask = torch.cat(embs), torch.cat(masks)
    assert emb.shape == (len(sentences), L, 768) and torch.isfinite(emb.float()).all()

    save_file({"emb": emb.contiguous(), "mask": mask.contiguous()}, str(out / "text.safetensors"))
    (out / "captions.json").write_text(json.dumps(sentences))
    (out / "clip_captions.json").write_text(json.dumps({c: [row[l] for l in ls] for c, ls in clip_lines.items()}))
    info = dict(model=MODEL, max_len=L, sentences=len(sentences), clips=len(ids), truncated=truncated,
                tokens_p50=int(np.percentile(lengths, 50)), tokens_p99=int(np.percentile(lengths, 99)),
                tokens_max=int(max(lengths)))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
