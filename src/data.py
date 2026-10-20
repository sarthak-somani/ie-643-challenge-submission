"""Load and standardise the challenge data."""
import numpy as np
import pandas as pd

DATA_DIR = "data"
N_CLASSES = 11


def load(data_dir=DATA_DIR):
    """Return a dict of numpy arrays.

    X        : (2076, 768) main train features (e0..e767)
    G        : (2076, 384) supplementary features (g0..g383), train only
    y        : (2076,)     labels, -1 = unlabeled
    deformed : (2076,)     0/1 deformation flag
    T        : (490, 768)  test features (f0..f767)
    test_ids : (490,)      test sample ids
    """
    train = pd.read_csv(f"{data_dir}/Train_Data.csv")
    test = pd.read_csv(f"{data_dir}/Test_Data.csv")
    supp = pd.read_csv(f"{data_dir}/Supplementry_Data.csv")

    # The supplementary file has 2 blank trailing rows: drop them, then align by sample_id.
    supp = supp.dropna().astype({"sample_id": int}).set_index("sample_id")
    supp = supp.loc[train["sample_id"]]

    X = train.filter(regex=r"^e\d+$").values.astype(np.float32)
    T = test.filter(regex=r"^f\d+$").values.astype(np.float32)
    G = supp.values.astype(np.float32)

    # Standardise the main features with train + test statistics (same feature space).
    mu, sd = np.r_[X, T].mean(0), np.r_[X, T].std(0) + 1e-6
    X, T = (X - mu) / sd, (T - mu) / sd
    G = (G - G.mean(0)) / (G.std(0) + 1e-6)

    return dict(
        X=X, G=G, T=T,
        y=train["Class_Label"].values,
        deformed=train["is_deformed"].values,
        test_ids=test["sample_id"].values,
    )
