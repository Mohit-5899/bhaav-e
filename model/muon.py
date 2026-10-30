"""Muon optimiser for weight matrices, plus AdamW for everything else.  (OUR code, written from the published method:
Keller Jordan, "Muon: An optimizer for hidden layers in neural networks", 2024.)

The problem it solves, from first principles:
  A weight matrix's update (gradient + momentum) is usually dominated by a few directions: most of the "push"
  goes into one or two directions while the rest barely move. Muon replaces the update with the nearest
  *orthogonal* matrix: same directions, but every direction moves by the same amount. In practice this lets
  big weight matrices learn faster at the same compute.

How: an orthogonal version of a matrix G is U V^T from its SVD (G = U S V^T, set all S to 1). An SVD is slow on a
GPU, so Muon uses 5 steps of a Newton-Schulz polynomial that pushes all singular values towards ~1, in bf16.

Muon only makes sense for 2D weight matrices inside the transformer blocks. Embeddings, the output layer, biases,
norm weights and small vectors use AdamW as usual (the standard recipe).
"""
import torch


def orthogonalize(G, steps=5, eps=1e-7):
    """Approximate U V^T of G with the quintic Newton-Schulz iteration (coefficients from the Muon write-up)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + eps)               # start with every singular value <= 1
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if tall:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p].setdefault("momentum", torch.zeros_like(g))
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = orthogonalize(g)
                u *= max(1.0, p.size(0) / p.size(1)) ** 0.5   # tall matrices: keep the update size comparable
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])


def split_params(model):
    """2D weights inside the transformer blocks -> Muon; everything else -> AdamW."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        (muon if (p.ndim == 2 and name.startswith("blocks.")) else adam).append(p)
    return muon, adam


def make_optimizers(model, muon_lr=0.02, adam_lr=2e-4, use_muon=True):
    """Returns a list of optimizers. use_muon=False gives plain AdamW for all weights (for comparison runs)."""
    muon, adam = split_params(model)
    if not use_muon:
        return [torch.optim.AdamW(muon + adam, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0)]
    return [Muon(muon, lr=muon_lr), torch.optim.AdamW(adam, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0)]
