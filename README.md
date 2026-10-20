# IE 643 Programming Challenge 2026: final submission

Code, model checkpoints and instructions for my best submission to the IE 643 (2026) course programming
challenge on Kaggle.

- **Participant:** Sarthak Somani (roll number 24B3006)
- **Final submission:** `submission/sub27_ens9_g0_tau1.25.csv`
- **Public leaderboard score:** 0.6912 (organiser baseline 0.6860)
- **Cross-validated balanced accuracy:** 0.721 (5-fold, 1284 labeled rows)

The detailed report is submitted separately. Every change that improved my leaderboard score is documented
in [DOCUMENTATION_KIT.md](DOCUMENTATION_KIT.md).

## The challenge

- Classify 768-dimensional feature vectors into 11 classes (labels 0 to 10).
- Training data: 2076 rows, of which 1284 are labeled and 792 unlabeled. Each row has an `is_deformed` flag,
  and about 90% of the rows are deformed (corrupted).
- A supplementary file gives 384 extra features for every training row, but not for the test rows.
- Test data: 490 rows with only the 768 features (no labels, no deformation flag, no supplementary features).
- Metric: accuracy weighted by hidden per-class weights.
- Rules: only MLPs and/or CNNs, no external data, individual work.

## How the solution works

1. **Teacher.** An MLP sees everything available for the training rows: the 768 main features, the 384
   supplementary features and the deformation flag. Trained on the labeled rows, it
   - gives soft labels to the 792 unlabeled rows, and
   - gives out-of-fold soft labels to the labeled rows, mixed 30% into their one-hot targets.
2. **Student.** A multi-task network that sees only the 768 main features, since that is all the test set has.
   It learns three things at once: the class, the deformation flag, and a reconstruction of the privileged
   information (the supplementary features or the teacher's hidden layer). The predicted deformation
   modulates the class head (FiLM). No class weighting is used during training.
3. **Ensemble.** Nine student variants (seven MLPs, one CNN, one MLP with an extra deformation input). Each is
   trained with 5-fold cross-validation and 3 seeds, so every member is an average of 15 models.
4. **Decision rule.** The hidden metric rewards rare classes, so the training class prior is removed when
   deciding: `label = argmax(log p - 1.25 * log prior)`.

| Member | Change from the base member |
|---|---|
| m01_base | multi-task MLP, 768 to 512 to 256, reconstructs the 384 supplementary features |
| m02_cosine_head | cosine-similarity class head (scale 16) |
| m03_supp_weight1 | supplementary-reconstruction loss weight 1.0 instead of 0.35 |
| m04_30_epochs | 30 epochs instead of 100 (teacher too) |
| m05_label_smoothing | label smoothing 0.1 |
| m06_cnn_trunk | CNN trunk: vector reshaped to 16 channels x 48, two Conv1d blocks |
| m07_deform_feature | predicted P(deformed) appended as a 769th input feature |
| m08_teacher_hidden | reconstructs the teacher's 256-d hidden layer instead of the supplementary features |
| m09_supp_and_teacher_hidden | reconstructs both (640-d target) |

## Repository structure

```
README.md               this file
DOCUMENTATION_KIT.md    every change that improved my leaderboard score
requirements.txt        Python dependencies
members.txt             the 9 member configurations
predict.sh              rebuild the submission from the saved checkpoints (no training)
train_members.sh        retrain all members from scratch, then build the submission
src/
  data.py               load and standardise the CSV files
  models.py             MLP, MultiTaskMLP, MultiTaskCNN
  train.py              training loop (soft-target cross-entropy + auxiliary losses)
  methods.py            teacher, soft labels, deformation feature, student training for one member
  cv.py                 train one member with 5-fold CV, save its 15 models and test probabilities
  ensemble.py           average member probabilities, apply the decision rule, write the CSV
  predict.py            same, but starting from the saved checkpoints
checkpoints/<member>/   15 model files (fold<k>_seed<i>.pt) and args.json per member
outputs/probs/          test probabilities of each member
outputs/oof/            out-of-fold probabilities of each member (labeled rows)
outputs/cv_results.jsonl  cross-validation metrics of each member
submission/             the submitted CSV
data/                   put the competition CSVs here (not included)
```

## Setup

- Python 3.11, PyTorch 2.5.1, NumPy, pandas, scikit-learn. A CPU is enough; a GPU is used if present.
  ```bash
  pip install -r requirements.txt
  ```
- Download the competition files from Kaggle and place `Train_Data.csv`, `Test_Data.csv` and
  `Supplementry_Data.csv` in `data/`.
- Run all commands from the repository root.

## Reproducing the submission

**From the saved checkpoints (about a minute, no training):**

```bash
bash predict.sh
```

This loads the 135 saved models (9 members x 15), rebuilds the submission in
`submission/submission_from_checkpoints.csv` and checks that it matches the submitted file byte for byte.

**From scratch (retrains everything):**

```bash
bash train_members.sh               # one member after another, about 1.5 hours on CPU
PARALLEL=1 bash train_members.sh    # all 9 members at once, about 10 minutes with 36 CPU cores
```

This retrains all 9 members with 5-fold cross-validation, overwrites `checkpoints/` and `outputs/`, and writes
`submission/submission_retrained.csv`. Training is deterministic on CPU: the retrained file is identical to the
submitted one.

**One member on its own:**

```bash
python src/cv.py --name m01_base --cfg '{"kd_alpha": 0.3}'
```

## Rules compliance

- Neural networks are MLPs and CNNs only (Linear, Conv1d, BatchNorm, GELU, Dropout). No attention, recurrent,
  graph or tree models, and no pretrained networks.
- Only the provided competition data is used.
- The unlabeled test features are used for feature standardisation statistics only.
