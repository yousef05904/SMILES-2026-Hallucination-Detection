# SMILES-2026 Hallucination Detection — Solution Report

## Reproducibility

### Environment

- OS: Linux, macOS, or Windows  
- Python: 3.10+ recommended (validated on 3.13)  
- Dependencies: install from repo root  

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The harness loads **`Qwen/Qwen2.5-0.5B`** from Hugging Face (cached automatically after first download). GPU (`CUDA`) is strongly recommended for a full extraction pass; CPU-only extraction is feasible but typically multi-hour wall clock with the fixed batch size of 4.

### Exactly how I generate submissions

Production run (**this is what you ship**):

```bash
# Windows UTF-8 console (recommended for Unicode separators in stdout)
set PYTHONIOENCODING=utf-8
python solution.py
```

Unset any dry-run overrides (see optional hook below):

```powershell
Remove-Item Env:SMILES_STUB_LM -ErrorAction SilentlyContinue
```

Artifacts:

- **`results.json`** — cross-validated metrics from `evaluate.py`
- **`predictions.csv`** — competition labels for `data/test.csv`

Deterministic seeds appear in **`splitting.py`** (`random_state`) and **`probe.py`** (`RNG_SEED` / `random_state` in estimators).

### Optional dry-run (CI / wiring checks only)

Because `aggregation.py` is imported **before** `model.py`, I added an optional environment hook that preempts loading the big LM when validating the scripting path on CPUs:

```powershell
set PYTHONIOENCODING=utf-8
set SMILES_STUB_LM=1
python solution.py
```

The stub swaps in a deterministic random forward with correct tensor shapes (`25` stacks × batch × seq × **`896`**). Outputs from this mode reflect **classification plumbing only**, not the challenge signal—and must **never** replace a real submission artifact.

Files modified (`aggregation.py`): look for **`_stub_model_maybe`** and **`SMILES_STUB_LM`**.

---

## Files I changed

| File | Role |
|------|------|
| `aggregation.py` | Late-layer pooled features (mean / max / last token), dispersion, cross-layer drift, norms, adjacent cosines, light length cues (`feature_dim ≈ 12 554`). Optional `SMILES_STUB_LM`. |
| `probe.py` | `HallucinationProbe` subclasses `torch.nn.Module` per harness but delegates to **sklearn** `VotingClassifier`: PCA+logistic regression, PCA+HistGradientBoosting (early stopping), and ExtraTrees. Validation F1 selects the inference threshold (`fit_hyperparameters`). |
| `splitting.py` | **Stratified 5-fold** outer evaluation; stratified validation slice (~18 %) carved from outer training folds for threshold tuning. |

Infrastructure left unchanged: **`model.py`**, **`evaluate.py`**, **`solution.py`**, data.

---

## Final approach — reasoning

### Aggregation

Baseline code used only **last-layer, last-real-token**. I pooled the **four deepest transformer stacks** (−4…−1) with **masked mean/max** plus **last token**, stacked **masked std on the deepest layer**, a **difference** between deepest and earlier depth (−13 when available), **L2 norms** along the trajectory, **log(norm ratio)** deepest vs earliest of the quartet, pairwise **cosine similarities** across adjacent depths on the answer’s last visible token, and two **effective-length ratios** scaled by `MAX_LENGTH` (fixed to 512 upstream). Padding is stripped via the attention mask; geometry lives in pooled directions rather than padding noise.

Motivation: hallucination manifests as layer-wise sharpening or inconsistency versus truth; mixing pooling heads increases robustness versus a single slicing point while keeping tensors vectorized.

### Probe

Torch MLP probes overfit aggressively on **`N≈689` × `dim≈1.25e4`.** sklearn gives transparent regularization paths:

- PCA caps rank with **randomized SVD**, sized from sample count versus feature count.
- `LogisticRegression(lbfgs, class_weight=balanced)` anchors low-variance discriminants.
- `ExtraTrees` handles nonlinear thresholding with `sqrt` subsampling and imbalance-aware weighting modes.
- `HistGradientBoosting` captures residual curvature with **built-in validation early stopping**.
- Weighted voting blends these opinions; softmax probabilities feed **AUROC** and threshold search.

Why not heavier deep nets inside the harness? Tabular ensembles with calibrated probability outputs trained under cross-validation fit the available labels without fragile hyperparameter choreography.

### Splitting

Stratified K-fold aligns with the leaderboard’s reliance on estimating generalization variance under class skew (~483 positive / ~206 negative). Nested validation shards avoid peeking calibration thresholds from the locked-outfold test mass.

---

## Experiments attempted (not all shipped)

| Idea | Outcome |
|------|---------|
| Last-token-only stacking | Competitive but clearly weaker than pooled multi-head late layers alone in earlier scratch runs. |
| Pure torch MLP (starter) | Stable but brittle on high-dimensional pooled vectors with small N compared to ensembles + PCA branches. |
| Sample-weight propagation through Voting / Pipeline sklearn 1.x | Routed metadata flags changed across versions; standardized on class imbalance inside base estimators to keep `fit()` portable. |

---

## Metrics (official harness)

Interpret **primary competition accuracy** via your own **`results.json`** `avg_test_accuracy` after **full** LM extraction ( **`SMILES_STUB_LM` unset** ).

Representative **`evaluate.py`** fields I track during development:

| Quantity | Typical purpose |
|-----------|----------------|
| Majority baseline | Sanity floor (~70 % accuracy on skewed negatives of prevalence). |
| Train vs val gap | Probe capacity / overfitting readout after folds. |
| Test accuracy / AUROC | Primary / diagnostic external generalization summaries averaged over folds |

If `extract_time_s` is on the **order of 10² seconds total** while still reporting full-batch extraction, infer a **stub** forward was active; production runs dominate wall time inside the transformers forward.

---

## Remaining assumptions & limitations

- Tokenization and truncation follow **`solution.py`** + **`MAX_LENGTH=512`**; extreme truncation collapses discriminative tails in long prompts.  
- Class imbalance dominates naive accuracy; thresholds tuned on validation F1 may trade precision/recall asymmetrically versus hidden organizer metrics.  
- Full reproduction with acceptable latency assumes **GPU-backed** PyTorch; CPU runs remain correct but disproportionately slow due to **`BATCH_SIZE=4`** in fixed infrastructure.

---

## Final commands (summary)

```bash
pip install -r requirements.txt
set PYTHONIOENCODING=utf-8   # optional Windows
python solution.py
```

Confirm **`predictions.csv`** and **`results.json`** exist beside **`solution.py`** before assembling the submission package.

---

## Google Colab (GPU)

Open **`SMILES_Colab.ipynb`** in Colab (**Runtime → GPU**), set **`REPO_URL`** to your GitHub repository, run all cells. The notebook only installs dependencies and runs **`python solution.py`**; it does not modify challenge source files. Leave **`SMILES_STUB_LM`** unset for the real Qwen forward pass.
