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
| `aggregation.py` | Minimal clean features: selected mid-late transformer layers `[-8, -4, -1]`, each with masked mean pooling only (`feature_dim ≈ 2,688` for Qwen-0.5B). Optional `SMILES_STUB_LM`. |
| `probe.py` | `HallucinationProbe` subclasses `torch.nn.Module` per harness but delegates to a deterministic **sklearn** pipeline: `StandardScaler` → balanced `LogisticRegression`. Validation accuracy, with F1 and prediction balance as tie-breakers, selects the inference threshold (`fit_hyperparameters`). |
| `splitting.py` | **Stratified 5-fold** outer evaluation; deterministic stratified validation slice from the outer training pool for threshold tuning. |

Infrastructure left unchanged: **`model.py`**, **`evaluate.py`**, **`solution.py`**, data.

---

## Final approach — reasoning

### Aggregation

The final extractor uses three selected transformer stacks: **`-8`**, **`-4`**, and **`-1`**. For each selected layer it uses:

- masked mean pooling over valid, non-padding tokens

It deliberately excludes last-token concatenation, max pooling, std pooling, cosine chains, norm ratios, length cues, and large geometric expansion. Padding is stripped with the attention mask moved to the hidden-state device, so the same code path works on CPU, CUDA, and Colab GPU.

Motivation: huge feature versions (`feature_dim = 12,554...17,943`) overfit or underperformed, while a last-three-layer mean-only variant was too weak in our run. The selected-layer version keeps the feature count low (`2,688`) but samples a wider mid-late model trajectory instead of using only adjacent final layers.

### Probe

The probe is a single regularized sklearn pipeline:

- `StandardScaler`
- `LogisticRegression(lbfgs, C=0.2, class_weight="balanced")`

Why not PCA or a larger ensemble? At `2,688` dimensions, the feature space is small enough for a directly regularized linear probe. Avoiding PCA preserves the selected layer coordinates, while the stronger L2 penalty keeps train accuracy from drifting toward memorization.

### Splitting

Stratified K-fold aligns with the leaderboard’s reliance on estimating generalization variance under class skew. The validation shard is carved only from the corresponding outer training pool, so the outer test fold is never used for fitting or threshold calibration.

---

## Experiments attempted (not all shipped)

| Idea | Outcome |
|------|---------|
| Large pooled/geometric feature set (`feature_dim = 12,554`) | Real Colab run produced high train metrics but weaker validation/test metrics, indicating overfitting. |
| Even larger feature variants (`feature_dim ≈ 17,943`) | Added complexity without reliable generalization gains. |
| Adjacent last-three-layer mean extractor (`feature_dim = 2,688`) | Real Colab run underfit in our implementation, so the final version keeps the same dimensionality but spreads layers across `-8`, `-4`, and `-1`. |
| Heavy ensemble or tree probes | Avoided in the final version because the dataset is small and the goal is stable 5-fold generalization rather than train accuracy. |

---

## Metrics (official harness)

Interpret **primary competition accuracy** via your own **`results.json`** `avg_test_accuracy` after **full** LM extraction ( **`SMILES_STUB_LM` unset** ). Final minimal-clean metrics should be filled in only after the real Colab GPU run.

Representative **`evaluate.py`** fields I track during development:

| Quantity | Typical purpose |
|-----------|----------------|
| Majority baseline | Sanity floor for accuracy under class imbalance. |
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
