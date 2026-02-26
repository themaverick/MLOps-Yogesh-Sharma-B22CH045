# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

**Course:** CSL7210 ML-DL-Ops  
**Roll Number:** B22CH045  
**Name:** Yogesh Sharma  
**Notebook run:** `B22CH045_Yogesh_Sharma_Ass3.ipynb`

---

## 1. Submission Requirements

### GitHub Repository Link
(Add your repository URL after pushing, e.g. `https://github.com/<username>/MLOps-Yogesh-Sharma-B22CH045`)

### Hugging Face Model Link
**https://huggingface.co/themaverick1/goodreads-genre-classifier**

Model, tokenizer, and `training_config.json` are pushed to this repo. 

### Docker Image Build Instructions

**Training environment (train model inside container):**
```bash
docker build -t goodreads-train .
docker run --env-file .env -v $(pwd)/results:/app/results goodreads-train
```

**Production image (eval only; pulls model from Hugging Face):**
```bash
docker build -f Dockerfile.eval -t goodreads-eval .
docker run --env-file .env -v $(pwd):/app/out goodreads-eval
```
Output: `evaluation_results.json` in the current directory.

---

## 2. Short Report

### 2.1 Model Selection

**Model used:** **DistilBERT** (`distilbert-base-cased`)

**Reasons for selection:**
- **Efficiency:** DistilBERT is a distilled version of BERT with ~40% fewer parameters, so it trains faster and uses less memory—suitable for CPU-only runs.
- **Performance:** It retains most of BERT’s performance on many NLU tasks, making it a good trade-off for genre classification.
- **Cased variant:** The cased model preserves capitalization, which can help for book titles and genre-related text.
- **Compatibility:** Same as in the instructor-provided notebook, so results and workflow are directly comparable.

---

### 2.2 Training Summary

- **Dataset:** UCSD Goodreads reviews by genre (poetry, children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, romance, young_adult). 8 classes; 800 train + 200 test samples per genre (6,400 train, 1,600 test).
- **Setup:** CPU-optimized (device = `cpu`), batch size 4, max length 512, 2 epochs.
- **Results from run** (`B22CH045_Yogesh_Sharma_Ass3.ipynb`):
  - **Training:** `global_step=3200`, `training_loss≈0.998`, `train_runtime≈1219.6 s` (~20.3 min), `epoch=2.0`.
  - **Training metrics:** `train_samples_per_second≈10.5`, `train_steps_per_second≈2.62`.
- **Logging:** Evaluation every 200 steps; logging every 50 steps (via `TrainingArguments`).

---

### 2.3 Evaluation Comparison

**Baseline (TF-IDF + Logistic Regression):**
- **Accuracy:** 0.57  
- **Macro avg (precision / recall / F1):** 0.57

**Fine-tuned DistilBERT (local model):**
- **eval_loss:** 1.2699  
- **eval_accuracy:** 0.61875 (~61.9%)  
- **eval_runtime:** ~24.12 s  
- **Per-class (precision / recall / F1):**  
  - children: 0.63 / 0.70 / 0.66  
  - comics_graphic: 0.82 / 0.77 / 0.79  
  - fantasy_paranormal: 0.44 / 0.47 / 0.45  
  - history_biography: 0.59 / 0.58 / 0.59  
  - mystery_thriller_crime: 0.59 / 0.58 / 0.59  
  - poetry: 0.80 / 0.82 / 0.81  
  - romance: 0.67 / 0.60 / 0.63  
  - young_adult: 0.43 / 0.43 / 0.43  
- **Overall (classification_report):** accuracy 0.62, macro avg 0.62.

**Model loaded from Hugging Face repo** (`themaverick1/goodreads-genre-classifier`):
- **eval_loss:** 1.2699  
- **eval_accuracy:** 0.61875  
- **eval_runtime:** ~23.90 s  

**Comparison:** Local and HF-repo evaluations match (same accuracy and loss). The small runtime difference is due to environment/load; the model and metrics are consistent after push to Hugging Face.

---

### 2.4 Challenges

- **CPU-only training:** No GPU was used; training took ~20 minutes for 2 epochs. Mitigated by smaller batch size (4), 2 epochs, and using DistilBERT instead of full BERT.
- **Data download:** Goodreads data is streamed from URLs; network or server slowness can delay data loading. Sampling (e.g. 2000 per genre) keeps runtime manageable.
- **Class imbalance and harder classes:** Some genres (e.g. fantasy_paranormal, young_adult) had lower F1 (~0.43–0.45), likely due to overlap with other genres or less distinctive wording. More data or targeted augmentation could help.
- **Hugging Face auth:** Pushing requires a valid `HF_TOKEN` (e.g. in `.env`). Notebook and scripts use `python-dotenv` and `huggingface_hub.login()` so that a single token works for both.

---

## 3. Artifacts

| Item | Location |
|------|----------|
| Run notebook | `B22CH045_Yogesh_Sharma_Ass3.ipynb` |
| Report | `B22CH045_Yogesh_Sharma_Ass3.md` |
| Evaluation results (saved in notebook) | `evaluation_results.json` (when saved from notebook) |
| Scripts | `src/train.py`, `src/eval.py`, `src/data.py`, `src/utils.py` |
| Dockerfiles | `Dockerfile` (train), `Dockerfile.eval` (eval only) |
| Dependencies | `requirements.txt` |
