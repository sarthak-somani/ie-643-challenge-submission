"""Training and prediction loop for one network (soft-target cross-entropy + auxiliary losses)."""
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model

# A GPU is used when available. Results are bit-exact on CPU (the setting used for the submission).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = dict(arch="mlp", hidden=(512, 256), dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=100, batch_size=128,
                w_deform=0.25,         # weight of the is_deformed head (multitask archs)
                w_supp=0.35,           # weight of the supplementary-reconstruction head (multitask archs)
                label_smoothing=0.0,   # mix the targets with the uniform distribution
                cos_head=0.0)          # > 0: cosine classifier head with this scale (multitask archs)


def one_hot(y, n_classes):
    return np.eye(n_classes, dtype=np.float32)[y]


def train_model(X, Y, seed=0, aux=None, **cfg):
    """Train one network.

    X   : (n, d) inputs
    Y   : (n, k) soft target distributions (one-hot for labeled rows, teacher probabilities otherwise)
    aux : (is_deformed (n,), supp targets (n, s)) for the auxiliary heads of the multitask archs
    """
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n, k = Y.shape
    if cfg["label_smoothing"] > 0:
        Y = (1 - cfg["label_smoothing"]) * Y + cfg["label_smoothing"] / k
    Xt, Yt, Wt = (torch.tensor(a).to(DEVICE) for a in (X, Y.astype(np.float32), np.ones(n, np.float32)))
    if aux is not None:
        Dt, Gt, Mt = (torch.tensor(a, dtype=torch.float32).to(DEVICE) for a in (*aux, np.ones(n)))

    supp_dim = aux[1].shape[1] if aux is not None else 384
    model = build_model(cfg["arch"], X.shape[1], k, cfg["hidden"], cfg["dropout"], supp_dim, cfg["cos_head"])
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])

    for _ in range(cfg["epochs"]):
        model.train()
        for idx in np.array_split(rng.permutation(n), max(1, n // cfg["batch_size"])):
            xb, yb, wb = Xt[idx], Yt[idx], Wt[idx]
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = -(yb * F.log_softmax(logits, dim=1)).sum(1)       # soft cross-entropy (no class weights)
            loss = (loss * wb).sum() / wb.sum()
            if aux is not None:                                       # auxiliary heads
                db, gb, mb = Dt[idx], Gt[idx], Mt[idx]
                l_def = F.binary_cross_entropy_with_logits(out[1], db, reduction="none")
                l_supp = ((out[2] - gb) ** 2).mean(1)
                m = mb.sum().clamp(min=1.0)
                loss = loss + cfg["w_deform"] * (l_def * mb).sum() / m + cfg["w_supp"] * (l_supp * mb).sum() / m
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.cpu()


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    out = model(torch.tensor(X))
    logits = out[0] if isinstance(out, tuple) else out
    return F.softmax(logits, dim=1).numpy()


def train_ensemble(X, Y, seeds, **kw):
    return [train_model(X, Y, seed=s, **kw) for s in seeds]


def ensemble_proba(models, X):
    return np.mean([predict_proba(m, X) for m in models], axis=0)
