"""Teacher -> multi-task student training for one ensemble member.

For a given set of labeled rows `lab` (one CV fold):
1. Teacher: an MLP on [main 768 | supplementary 384 | is_deformed 1] trained on the labeled rows.
   - its soft predictions label the 792 unlabeled training rows;
   - out-of-fold teacher predictions on the labeled rows are blended into their targets:
     y = (1 - kd_alpha) * one_hot + kd_alpha * teacher.
2. Student: a multi-task network on the main features only (the only input available at test time),
   trained on labeled + unlabeled rows with the class head plus two auxiliary heads
   (is_deformed, and a reconstruction target built from the privileged information).

Member options (passed through the JSON config):
  kd_alpha        weight of the teacher in the labeled targets (0.3 for every member)
  supp_target     "raw" (384 supplementary features), "teacher" (teacher's 256-d hidden layer)
                  or "raw+teacher" (both, 640-d)
  deform_feature  append a predicted P(deformed) as a 769th input feature
  cos_head, w_supp, label_smoothing, epochs, arch: see train.DEFAULTS / models.py
"""
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from data import N_CLASSES
from train import one_hot, train_ensemble, ensemble_proba

# Options that only change the student; everything else in the config is also passed to the teacher
# (for example "epochs"). Teacher outputs are cached per fold and teacher config.
STUDENT_KEYS = {"arch", "w_deform", "w_supp", "label_smoothing", "cos_head"}
CACHE_DIR = "cache"


def teacher_features(d):
    return np.c_[d["X"], d["G"], d["deformed"]].astype(np.float32)


def train_teacher(d, rows, seeds, cfg):
    """Teacher MLP on [main, supplementary, is_deformed] for the given labeled rows."""
    return train_ensemble(teacher_features(d)[rows], one_hot(d["y"][rows], N_CLASSES), seeds,
                          **{**cfg, "arch": "mlp"})


def teacher_probs(d, lab, seeds, cfg):
    """Teacher probabilities for the unlabeled rows, and out-of-fold ones for the labeled rows.
    Cached on disk in cache/ (keyed by the labeled rows, seeds and teacher config)."""
    tcfg = {k: v for k, v in sorted(cfg.items()) if k not in STUDENT_KEYS}
    key = hashlib.md5(lab.tobytes() + json.dumps([list(seeds), tcfg]).encode()).hexdigest()
    path = f"{CACHE_DIR}/teacher_{key}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["P_unl"], z["P_lab"]

    XT = teacher_features(d)
    unl = np.where(d["y"] == -1)[0]
    P_unl = ensemble_proba(train_teacher(d, lab, seeds, tcfg), XT[unl])
    P_lab = np.zeros((len(lab), N_CLASSES), np.float32)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=1).split(lab, d["y"][lab]):
        P_lab[va] = ensemble_proba(train_teacher(d, lab[tr], seeds, tcfg), XT[lab[va]])
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"     # atomic write (parallel runs share the cache)
    np.savez(tmp, P_unl=P_unl, P_lab=P_lab)
    os.replace(tmp, path)
    return P_unl, P_lab


def teacher_hidden_targets(d, lab, rows, cfg):
    """Feature-distillation target: the 256-d hidden layer of a teacher (seed 0) trained on this fold's
    labeled rows, evaluated on `rows`, standardised."""
    tcfg = {k: v for k, v in cfg.items() if k not in STUDENT_KEYS}
    teacher = train_teacher(d, lab, [0], tcfg)[0].eval()
    with torch.no_grad():
        H = teacher.net[:-1](torch.tensor(teacher_features(d)[rows])).numpy()
    return ((H - H.mean(0)) / (H.std(0) + 1e-6)).astype(np.float32)


def add_deform_feature(d, X_eval, seed=0):
    """Append the predicted deformation probability as an extra input feature (769-d input).
    A small MLP learns main features -> is_deformed on all training rows (no class labels) and is applied
    to the training rows and to X_eval (test rows have no deformation flag)."""
    clf = train_ensemble(d["X"], one_hot(d["deformed"].astype(int), 2), [seed], arch="mlp", epochs=30)
    p_train, p_eval = ensemble_proba(clf, d["X"])[:, 1:], ensemble_proba(clf, X_eval)[:, 1:]
    return {**d, "X": np.c_[d["X"], p_train].astype(np.float32)}, np.c_[X_eval, p_eval].astype(np.float32)


def train_member(d, lab, X_eval, seeds, kd_alpha=0.3, supp_target="raw", deform_feature=False, **cfg):
    """Train one member on the labeled rows `lab`. Returns (probabilities for X_eval, list of student models)."""
    unl = np.where(d["y"] == -1)[0]
    P_unl, P_lab = teacher_probs(d, lab, seeds, cfg)
    Y_lab = (1 - kd_alpha) * one_hot(d["y"][lab], N_CLASSES) + kd_alpha * P_lab

    if deform_feature:
        d, X_eval = add_deform_feature(d, X_eval)
    rows = np.r_[lab, unl]
    if supp_target in ("teacher", "raw+teacher"):
        H = teacher_hidden_targets(d, lab, rows, cfg)
        supp = H if supp_target == "teacher" else np.c_[d["G"][rows], H].astype(np.float32)
    else:
        supp = d["G"][rows]

    arch = cfg.get("arch") if cfg.get("arch") in ("multitask", "multitask_cnn") else "multitask"
    students = train_ensemble(np.r_[d["X"][lab], d["X"][unl]], np.r_[Y_lab, P_unl], seeds,
                              aux=(d["deformed"][rows], supp), **{**cfg, "arch": arch})
    return ensemble_proba(students, X_eval), students
