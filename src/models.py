"""Model definitions. Only MLPs and CNNs, as required by the challenge rules."""
import torch.nn as nn
import torch.nn.functional as F


def mlp_trunk(in_dim, hidden=(512, 256), dropout=0.3):
    """Linear -> BatchNorm -> GELU -> Dropout blocks. Output size = hidden[-1]."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Plain MLP classifier (used for the teacher and the deformation predictor)."""

    def __init__(self, in_dim, n_classes, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(*mlp_trunk(in_dim, hidden, dropout), nn.Linear(hidden[-1], n_classes))

    def forward(self, x):
        return self.net(x)


class MultiTaskMLP(nn.Module):
    """Student network: a shared MLP trunk with three heads.
       - class logits (11)
       - is_deformed logit (1)
       - reconstruction of the supplementary features (or of the teacher's hidden layer)
    The predicted deformation probability modulates the trunk output (FiLM) before the class head,
    so clean and deformed samples can use different decision boundaries."""

    def __init__(self, in_dim, n_classes, hidden=(512, 256), dropout=0.3, supp_dim=384, cos_scale=0.0):
        super().__init__()
        self.cos_scale = cos_scale               # > 0: cosine classifier head with this scale
        self.trunk = mlp_trunk(in_dim, hidden, dropout)
        d = hidden[-1]
        self.deform_head = nn.Linear(d, 1)
        self.supp_head = nn.Linear(d, supp_dim)
        self.film = nn.Linear(1, 2 * d)          # deformation probability -> (gamma, beta)
        self.cls_head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.trunk(x)
        deform_logit = self.deform_head(h)
        gamma, beta = self.film(deform_logit.sigmoid()).chunk(2, dim=1)
        h_cls = h * (1 + gamma) + beta
        if self.cos_scale > 0:
            logits = self.cos_scale * F.normalize(h_cls, dim=1) @ F.normalize(self.cls_head.weight, dim=1).T
        else:
            logits = self.cls_head(h_cls)
        return logits, deform_logit.squeeze(1), self.supp_head(h)


class MultiTaskCNN(MultiTaskMLP):
    """Same heads as MultiTaskMLP, but a convolutional trunk: the 768-vector is reshaped to
    16 channels x 48, two Conv1d blocks, flatten, then a linear projection to the trunk width.
    (The parent first builds an MLP trunk that is then replaced; kept as is so that the random
    initialisation, and therefore the results, stay exactly reproducible.)"""

    def __init__(self, in_dim, n_classes, hidden=(512, 256), dropout=0.3, supp_dim=384, cos_scale=0.0,
                 shape=(16, 48), ch=64):
        super().__init__(in_dim, n_classes, hidden, dropout, supp_dim, cos_scale)
        self.trunk = nn.Sequential(
            nn.Unflatten(1, shape),
            nn.Conv1d(shape[0], ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(ch * shape[1], hidden[-1]), nn.BatchNorm1d(hidden[-1]), nn.GELU(), nn.Dropout(dropout))


def build_model(arch, in_dim, n_classes, hidden=(512, 256), dropout=0.3, supp_dim=384, cos_scale=0.0):
    if arch == "mlp":
        return MLP(in_dim, n_classes, hidden, dropout)
    if arch == "multitask":
        return MultiTaskMLP(in_dim, n_classes, hidden, dropout, supp_dim, cos_scale)
    if arch == "multitask_cnn":
        return MultiTaskCNN(in_dim, n_classes, hidden, dropout, supp_dim, cos_scale)
    raise ValueError(arch)
