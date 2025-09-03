Sentiment Classifier — a concise training notebook and playbook

This project was built as a training exercise: a small, reproducible pipeline that fine-tunes a transformer on IMDB sentiment so you can learn the steps, iterate fast, and reuse the pattern elsewhere.

Checklist (what this README covers)
- Short narrative of what we did and why
- Model selection rationale
- How training was run (high-level) and how to reproduce locally
- Hardware used for the exercise and tuned defaults
- Key lessons, gotchas, and how to extend the work

What we did (story form)
We picked a compact model (DistilBERT) and wired a config-driven training script so experiments are easy to run and compare. The repo shows a typical workflow: load dataset, tokenize, build a classifier head, train with the Hugging Face Trainer, evaluate, and save the best model. We added a tiny unit test to validate the collator + forward pass so CI can catch regressions quickly.

Model choice — why DistilBERT?
- Fast to download and iterate with locally.
- Small enough to train on modest GPUs (or CPU for very small experiments).
- Good tradeoff of speed vs accuracy for learning and prototyping.

If you need higher accuracy and you have better hardware, swap `model_name` in `config.yaml` for a larger model (RoBERTa, DeBERTa, etc.).

How training works (short steps)
1. Data: IMDB via `datasets.load_dataset('imdb')`.
2. Tokenization: `AutoTokenizer.from_pretrained(model_name)`, truncate to a sensible max length.
3. Model: `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)`.
4. Training loop: `Trainer` + `TrainingArguments` driven by `config.yaml` or `dev_config.yaml`.
5. Metrics: accuracy + macro-F1 via the `evaluate` package.
6. Best model saved to `output_dir/best`.

What to run locally (PowerShell examples)

Setup and install:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r dev-requirements.txt  # optional dev tools + pytest
```

Quick dev smoke test (tiny run):

```powershell
python train.py dev_config.yaml
```

Full training:

```powershell
python train.py config.yaml
```

Predict with a saved model:

```powershell
python predict.py "This movie was fantastic!"
```

Run the smoke unit test:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Hardware used for this exercise (detected)
- CPU: Intel(R) Core(TM) i7-7700HQ @ 2.80GHz (4 physical cores / 8 logical)
- RAM: 15.89 GB
- GPUs:
  - Intel(R) HD Graphics 630 (~1 GB VRAM)
  - NVIDIA GeForce GTX 1060 (~4 GB VRAM)

Tuned conservative defaults for that machine
- model_name: `distilbert-base-uncased`
- per_device_train_batch_size: 4
- per_device_eval_batch_size: 8
- num_train_epochs: 1-3 (for early experiments)
- max_length: 128
- fp16: try `True` if CUDA/driver support is available

Key lessons & gotchas (short)
- Hugging Face API changes: we detect `evaluation_strategy` vs `eval_strategy` to avoid breakage across versions.
- `load_best_model_at_end=True` requires `save_strategy` to match eval strategy — align them to avoid runtime errors.
- YAML treats `no`/`yes` as booleans — use explicit strings for strategy fields (e.g., `"no"`, `"epoch"`).
- Prefer `processing_class` + `DataCollatorWithPadding(tokenizer)` instead of the deprecated `tokenizer=` argument.
- Add quick smoke tests (we have `tests/test_smoke.py`) to verify collator + forward pass without downloading pretrained weights.

Files of interest
- `train.py` — main training script (config-driven)
- `predict.py` — single-text prediction helper
- `config.yaml` / `dev_config.yaml` — experiment configs (tuned conservatively for GTX 1060)
- `requirements.txt` / `dev-requirements.txt` — runtime and dev deps
- `tests/test_smoke.py` — tiny unit test

Next steps you can pick from
- Add `config.gtx1060.yaml` with these defaults committed (I can add it).
- Add a GitHub Actions workflow that installs dev deps and runs `pytest` on push.
- Add an `accelerate` example and a short guide to launch multi-GPU runs.
- Integrate PEFT/LoRA for parameter-efficient fine-tuning.

If you want, tell me which next step I should implement and I'll add it now (I can add the tuned config, CI, or accelerate example).
