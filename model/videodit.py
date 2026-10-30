"""Bhaav Tiny generator: a video diffusion transformer (VideoDiT).  (OUR code, written from the public design.)

Design source: the DiT idea (Peebles & Xie, 2023) at the size Drishya Tiny published ("s210": width 768, 16 blocks,
12 heads, about 212M parameters). We copy the blueprint, not code; `count_params` checks we built the same size.

What it does, in one line: given a noisy sketch x_t, the noise level t and a caption, predict the velocity
v = noise - clean (rectified flow, see how-training-works.md 5a).

Shapes (B = batch):
  x      [B, 48, 5, 16, 16]   VAE latent: 48 channels, 5 time steps, 16 x 16 positions
  tokens [B, 320, 768]        after cutting into 1 x 2 x 2 blocks: 5 * 8 * 8 = 320 pieces
  text   [B, L, 768]          cached Flan-T5 vectors, L = 35, with a 0/1 mask for padding
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    in_ch: int = 48          # VAE latent channels
    frames: int = 5          # latent time steps
    size: int = 16           # latent height = width
    patch: int = 2           # 2 x 2 positions per piece (1 time step deep)
    width: int = 768         # numbers per piece inside the transformer
    depth: int = 16          # transformer blocks
    heads: int = 12          # attention heads (768 / 12 = 64 numbers each)
    mlp_ratio: float = 4.0
    txt_dim: int = 768       # Flan-T5-Base output size


# tiny = for CPU tests. Its width must exceed one piece's 192 output numbers (48 x 2 x 2); a narrower model can't
# express every possible output and can never fit random targets exactly.
SIZES = {"s210": Config(), "tiny": Config(width=256, depth=2, heads=4)}


# ---------- fixed position and time codes (no learned numbers) ----------

def sincos(n, dim):
    """Classic sine/cosine code for positions 0..n-1: nearby positions get similar codes."""
    pos = torch.arange(n, dtype=torch.float64)[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, dtype=torch.float64) / (dim // 2))
    a = pos * freq
    return torch.cat([a.sin(), a.cos()], dim=1).float()


def position_code(cfg):
    """Each piece learns *where* and *when* it is: a quarter of the numbers code time, the rest height and width."""
    n_h = cfg.size // cfg.patch
    dt = cfg.width // 4
    dh = (cfg.width - dt) // 2
    dw = cfg.width - dt - dh
    t = sincos(cfg.frames, dt)[:, None, None, :].expand(-1, n_h, n_h, -1)
    h = sincos(n_h, dh)[None, :, None, :].expand(cfg.frames, -1, n_h, -1)
    w = sincos(n_h, dw)[None, None, :, :].expand(cfg.frames, n_h, -1, -1)
    return torch.cat([t, h, w], dim=-1).reshape(1, -1, cfg.width)   # [1, 320, width]


def time_code(t, dim=256):
    """Noise level t in [0, 1] -> a 256-number code, the same trick as positions (t scaled to 0..1000)."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000)[:, None] * freq[None]
    return torch.cat([a.cos(), a.sin()], dim=-1)


# ---------- building blocks ----------

class RMSNorm(nn.Module):
    """Rescale a vector to a fixed average size. Used on queries and keys so attention scores can't explode."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Attention(nn.Module):
    """Each piece asks a question (query), every piece offers a label (key) and content (value).
    Self-attention: questions and answers come from the video pieces. Cross: answers come from the caption."""
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.h, self.cross = heads, cross
        if cross:
            self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        else:
            self.qkv = nn.Linear(d, 3 * d)
        self.q_norm, self.k_norm = RMSNorm(d // heads), RMSNorm(d // heads)
        self.proj = nn.Linear(d, d)

    def forward(self, x, ctx=None, ctx_mask=None):
        B, N, D = x.shape
        if self.cross:
            q, (k, v) = self.q(x), self.kv(ctx).chunk(2, dim=-1)
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda y: y.view(B, -1, self.h, D // self.h).transpose(1, 2)   # [B, heads, tokens, 64]
        q, k, v = self.q_norm(split(q)), self.k_norm(split(k)), split(v)
        mask = None if ctx_mask is None else ctx_mask[:, None, None, :].bool()  # padding words get no attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    """One transformer block: look at the other pieces, look at the caption, then think (MLP).
    adaLN-zero: the noise level and caption gist set a shift, scale and gate for each part. The gates start at
    zero, so a new block begins as "do nothing" and training turns it on gradually (stable start)."""
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, heads)
        self.norm_x = nn.LayerNorm(d, eps=1e-6)
        self.xattn = Attention(d, heads, cross=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hid = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, d))
        self.ada = nn.Linear(d, 6 * d)

    def forward(self, x, cond, txt, txt_mask):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(cond)).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), sh1, sc1))
        x = x + self.xattn(self.norm_x(x), txt, txt_mask)
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), sh2, sc2))
        return x


# ---------- the whole generator ----------

class VideoDiT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.width, cfg.patch
        self.n_side = cfg.size // p
        self.patch_in = nn.Linear(cfg.in_ch * p * p, d)                  # one piece (48 x 2 x 2 numbers) -> d
        self.register_buffer("pos", position_code(cfg), persistent=False)
        self.first_frame_mark = nn.Parameter(torch.zeros(d))               # tags "this frame is a given image"
        self.t_in = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.txt_in = nn.Sequential(nn.Linear(cfg.txt_dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.txt_pool = nn.Linear(d, d)                                    # caption gist, added to the noise-level code
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, cfg.in_ch * p * p)                         # d -> one piece of velocity
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.first_frame_mark, std=0.02)
        for b in self.blocks:                       # adaLN-zero: every block starts as the identity
            nn.init.zeros_(b.ada.weight); nn.init.zeros_(b.ada.bias)
        nn.init.zeros_(self.ada_out.weight); nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # first prediction is exactly 0

    def patchify(self, x):
        """[B, C, T, 16, 16] -> [B, T*8*8, C*2*2]: cut each frame into 2 x 2 blocks."""
        B, C, T, H, W = x.shape
        p, n = self.cfg.patch, self.n_side
        x = x.view(B, C, T, n, p, n, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(B, T * n * n, C * p * p)

    def unpatchify(self, x):
        B = x.shape[0]
        p, n, C, T = self.cfg.patch, self.n_side, self.cfg.in_ch, self.cfg.frames
        x = x.view(B, T, n, n, C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(B, C, T, n * p, n * p)

    def forward(self, x, t, txt, txt_mask, first_given=None):
        """x: noisy latent; t: noise level [B]; txt/txt_mask: cached caption; first_given: [B] bool, frame 0 is clean."""
        h = self.patch_in(self.patchify(x)) + self.pos
        if first_given is not None:                                         # tag frame 0's pieces when it is given
            pf = self.n_side * self.n_side
            mark = first_given[:, None, None].to(h.dtype) * self.first_frame_mark
            h = torch.cat([h[:, :pf] + mark, h[:, pf:]], dim=1)
        c = self.txt_in(txt)
        m = txt_mask.to(c.dtype)[..., None]
        gist = (c * m).sum(1) / m.sum(1).clamp(min=1)                       # average over real words only
        cond = self.t_in(time_code(t)) + self.txt_pool(gist)
        for b in self.blocks:
            h = b(h, cond, c, txt_mask)
        sh, sc = self.ada_out(F.silu(cond)).chunk(2, dim=-1)
        return self.unpatchify(self.out(modulate(self.norm_out(h), sh, sc)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
