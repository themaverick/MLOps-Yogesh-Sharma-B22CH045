# MLOps-Yogesh-Sharma-B22CH045
Assignments for course CSL7210 ML-DL-Ops.

## Assignment 3: Hugging Face & Docker (Goodreads genre classification)

- **Notebook:** `B22CH045_Yogesh_Sharma_Ass3.ipynb`, HF login from `.env` (HF_TOKEN), push to HF, re-evaluate from HF repo.
- **Report:** `B22CH045_Yogesh_Sharma_Ass3.md` — Short report (model selection, training summary, evaluation comparison, challenges) and submission info (HF link, Docker instructions).
- **Hugging Face model:** [themaverick1/goodreads-genre-classifier](https://huggingface.co/themaverick1/goodreads-genre-classifier) (after training and push).
- **Scripts:** `src/train.py`, `src/eval.py`, `src/data.py`, `src/utils.py`.

### Docker (training environment)
```bash
docker build -t goodreads-train .
docker run --env-file .env -v $(pwd)/results:/app/results goodreads-train
```

### Docker (production – eval only, model from HF)
```bash
docker build -f Dockerfile.eval -t goodreads-eval .
docker run --env-file .env -v $(pwd):/app/out goodreads-eval
```
Output: `evaluation_results.json` in current directory.

### Local run (CPU)
```bash
pip install -r requirements.txt
# Train (optional: add --push-to-hub to upload to HF)
python -m src.train --epochs 2 --batch-size 4
# Eval (local or from HF)
python -m src.eval --model-path themaverick1/goodreads-genre-classifier --output evaluation_results.json
```
