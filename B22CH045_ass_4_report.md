# Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

**Roll No:** B22CH045  
**Course:** DLOps / MLOps  
**Baseline notebook:** `B22CH045_ass_4_tuned_en_to_hi_baseline.ipynb`  
**Tuning notebook:** `B22CH045_ass_4_tuned_en_to_hi.ipynb`  
**Report (PDF) filename:** `B22CH045_ass_4_report.pdf`

---

## 1. Overview

This work establishes a **baseline** English→Hindi Transformer (100 epochs, fixed hyperparameters), then **refactors training** for **Ray Tune** with **Optuna** as the search algorithm and **ASHA** for early stopping. The goal is to find stronger hyperparameters under a **≤30 epochs per trial** cap and compare against the baseline on **training time**, **final loss**, and **NLTK corpus BLEU** on the same small validation set used in the notebooks.

**BLEU reporting:** Both **percentage-style** (score × 100) and **raw** corpus BLEU (0–1) are given below for clarity.

---

## 2. Baseline (Part 1)

| Metric | Value |
|--------|--------|
| **Total training time (100 epochs)** | **7678.15 s** (~2 h 8 min) |
| **Final training loss (epoch 100)** | **0.0962** |
| **BLEU (NLTK corpus_bleu, method4)** | **68.23** (percentage-style) / **0.6823** (raw) |

- Baseline weights from the notebook run: `transformer_translation_final.pth` (local).

---

## 3. Hyperparameter search (Part 2)

### 3.1 Integration

- Training is wrapped in **`train_tune(config)`**; the model, optimizer, and training hyperparameters come from Ray Tune’s **`config`**.
- Each epoch reports **`tune.report({"loss": mean_loss, "epoch": epoch + 1}, checkpoint=checkpoint)`** so Optuna/ASHA can use **loss** as the metric.

### 3.2 Search algorithm & scheduler

- **OptunaSearch:** `metric="loss"`, `mode="min"`.
- **ASHAScheduler:** `max_t=30`, `grace_period=5`, `reduction_factor=2`, `time_attr="epoch"`.
- **Trials:** `num_samples=20`.
- **Resources:** `cpu=2`, `gpu=1` per trial (as in the notebook).

### 3.3 At least four hyperparameters + ranges

| Hyperparameter | Search space |
|----------------|----------------|
| **Learning rate** | `tune.loguniform(1e-5, 1e-3)` |
| **Batch size** | `tune.choice([32, 64])` |
| **Attention heads** | `tune.choice([4, 8])` (with **D_MODEL = 512** fixed) |
| **Feed-forward dim** | `tune.choice([1024, 2048])` |
| **Dropout** | `tune.uniform(0.1, 0.4)` |
| **Epochs per trial (cap)** | **30** (efficiency vs baseline 100) |

---

## 4. Best configuration & final metrics (Part 3)

### 4.1 Best configuration (Ray Tune / Optuna)

| Field | Value |
|-------|--------|
| **lr** | 2.412 × 10⁻⁴ |
| **batch_size** | 64 |
| **num_heads** | 8 |
| **d_ff** | 1024 |
| **dropout** | ~0.2705 |
| **num_epochs** | 30 |

### 4.2 Best trial (Tune objective)

| Metric | Value |
|--------|--------|
| **Best reported loss** | **0.4729** |
| **Best epoch** | **30** |

### 4.3 BLEU of best model (same val set as baseline)

| Form | Value |
|------|--------|
| **Percentage-style** | **66.25** |
| **Raw** | **0.6625** |

### 4.4 Time

| Description | Value |
|-------------|--------|
| **Full Ray Tune sweep** (20 samples, ASHA, logs in notebook) | **13857.11 s** (~3 h 51 min) |
| **Epoch budget vs baseline** | **30** epochs (vs **100**) for the best trial |

---

## 5. Comparison & discussion

| Item | Baseline | Best tuned |
|------|----------|------------|
| Epochs | 100 | 30 |
| Training time (single run) | 7678.15 s | (sweep total: 13857.11 s) |
| Final loss | 0.0962 | 0.4729 (Tune-reported mean loss at best epoch) |
| BLEU | 68.23 / 0.6823 | 66.25 / 0.6625 |

The tuned configuration reaches **strong BLEU in one-third of the baseline epochs** but **does not exceed** the baseline BLEU on this validation set; search was driven by **minimizing training loss**, not BLEU. A natural extension is to **report BLEU to Ray Tune** (or select the best trial by BLEU after a short eval) to align the objective with the rubric’s quality target.

---

## 6. Model weights (submission)

Best-performing weights from the tuning run are **not attached as a local file in this submission**; they are hosted on Hugging Face:

**[B22CH045_ass_4_best_model on Hugging Face](https://huggingface.co/themaverick1/B22CH045_ass_4_best_model/tree/main)**

The repository contains **`B22CH045_ass_4_best_model.pth`** (~151 MB), matching the assignment naming convention.

---

## 7. Repository contents (checklist)

- [x] Tuned notebook: `B22CH045_ass_4_tuned_en_to_hi.ipynb` (Ray Tune + Optuna + ASHA)
- [x] Baseline notebook: `B22CH045_ass_4_tuned_en_to_hi_baseline.ipynb`
- [x] Report: **`B22CH045_ass_4_report.pdf`** 
- [x] Best model: [Hugging Face — B22CH045_ass_4_best_model](https://huggingface.co/themaverick1/B22CH045_ass_4_best_model/tree/main)

---
