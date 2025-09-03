# Sentiment Classifier — training exercise

This repository was created as a training exercise to learn how to fine-tune a Transformer (DistilBERT) on the IMDB dataset for sentiment classification. The code and configs are intentionally small and conservative so they can be run on modest hardware for learning and experimentation.

## What this repo contains
- `train.py` — config-driven training script (uses Hugging Face `Trainer`).
- `predict.py` — single-text prediction helper that loads the trained model in `output_dir/best`.
- `config.yaml` / `dev_config.yaml` — training configs (dev=small smoke test).
- `requirements.txt` — Python dependencies to install in a venv.

## Quick goals
- Run a fast dev smoke training to validate the pipeline.
- Run full training by editing `config.yaml`.
- Run single-text predictions using the saved model.

---

## Prerequisites
- Windows PowerShell (instructions below use PowerShell syntax).
- Python 3.8+ (use a virtual env).

## Hardware considerations (context for this training exercise)

I don't have your exact machine specs in the repo; for the purposes of the exercise I assume a typical developer laptop/desktop running Windows with modest resources. Below are two common scenarios and recommended settings:

- Low-resource / CPU-only (assumed example):
	- CPU: 4 physical cores (e.g. Intel i5), RAM: 8-16 GB, no dedicated NVIDIA GPU.
	- Recommendations: use `dev_config.yaml` for smoke tests, small `per_device_*_batch_size` (1-8), and keep `num_train_epochs` low (1-3). Use smaller models (DistilBERT or TinyBERT).

- GPU-enabled (assumed example):
	- Single NVIDIA GPU (e.g. GTX 1660 / RTX 2060 / RTX 3060), 8+ GB VRAM, 16+ GB system RAM.
	- Recommendations: increase `per_device_train_batch_size`, enable `fp16=True` in `TrainingArguments` for faster training and lower memory usage, consider `accelerate` for multi-GPU or distributed runs.

How to detect hardware quickly (PowerShell):

```powershell
# CPU summary
Get-CimInstance -ClassName Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
# GPU summary (shows display adapters; NVIDIA-specific details require NVIDIA tooling)
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
```

If you provide exact specs I can update the README to include them and recommend tuned defaults.

### Hardware used for this project (detected)

The following hardware was detected on the machine where this exercise was executed. These exact values are recorded here so experiment settings and expectations are clear:

- CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz — 4 physical cores, 8 logical processors
- RAM: 15.89 GB total physical memory
- GPUs:
	- Intel(R) HD Graphics 630 — ~1024 MB VRAM
	- NVIDIA GeForce GTX 1060 — ~4095 MB VRAM

Recommended tuned defaults for this hardware (conservative, safe):

- Model: `distilbert-base-uncased` (already used in the repo). If you need faster, smaller runs consider TinyBERT or DistilBERT variants.
- `per_device_train_batch_size`: 4 (reduce if you see OOMs)
- `per_device_eval_batch_size`: 8
- `num_train_epochs`: 1-3 for experimentation
- `max_length` / truncation: 128 tokens (smaller sequences reduce memory)
- `fp16`: try `True` when GPU has decent CUDA support (may reduce memory use)
- `gradient_accumulation_steps`: 1 (increase to emulate larger batches without increasing VRAM)

Notes:
- The GTX 1060 with ~4GB VRAM is modest for transformer training. Expect to run small experiments and use the `dev_config.yaml` for fast iterations. For larger experiments, use a GPU with >=8GB VRAM.
- If training on CPU only, keep batch sizes very small (1-4) and prefer dev runs or small datasets.

## Setup (recommended)
Open PowerShell in `d:\Sentiment Classifier\sentiment-classifier` and run:

```powershell
# Create venv (if not already present)
python -m venv .venv
# Activate the venv
& ".\..\.venv\Scripts\Activate.ps1"
# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: you can also run the scripts without activating the venv by calling the venv python directly, for example:

```powershell
& "D:/Sentiment Classifier/.venv/Scripts/python.exe" train.py dev_config.yaml
```

(Adjust the exact venv path if you created the environment elsewhere.)

## Dev smoke run (fast)
A small `dev_config.yaml` exists for quick testing. From the `sentiment-classifier` folder run:

```powershell
# using activated venv
python train.py dev_config.yaml
# or using explicit interpreter
& "D:/Sentiment Classifier/.venv/Scripts/python.exe" "D:/Sentiment Classifier/sentiment-classifier/train.py" "D:/Sentiment Classifier/sentiment-classifier/dev_config.yaml"
```

This runs a tiny training session and writes a best model to `./results/best` (or the `output_dir` you set in the config).

## Full training
Edit `config.yaml` to your desired hyperparameters then run:

```powershell
python train.py config.yaml
```

## Predict
After training, the best model is saved at `output_dir/best`. Run:

```powershell
python predict.py "This movie was fantastic!"
```

Or import `predict.predict(text)` from Python and call it programmatically.

## Notes / Gotchas
- YAML booleans: YAML parses `no`, `yes`, `on`, `off` as booleans. For fields like `save_strategy`, `evaluation_strategy`, `logging_strategy`, prefer explicit strings in the YAML, e.g. `save_strategy: "epoch"` or `evaluation_strategy: "steps"` to avoid surprises.
- Trainer API changes: `train.py` is updated to use `processing_class` + `data_collator=DataCollatorWithPadding(tokenizer)` instead of the deprecated `tokenizer=` argument. This is forward-compatible with current HF docs.
- If you see errors from `evaluate` (metrics), install `scikit-learn` in the venv (`pip install scikit-learn`).

## Reproducibility & Pinning
For stable CI / reproducible runs, pin package versions before doing heavy training. Example:

```powershell
# after building env
pip freeze > requirements-lock.txt
```

Consider pinning `transformers` to a tested minor version (example: `transformers==4.56.0`) in `requirements.txt` if you want to standardize behavior across machines.

## CI suggestion
Add a lightweight CI job (GitHub Actions) that:
- Creates Python venv
- Installs `-r requirements.txt` (or a minimal test set)
- Runs `python -c "import train, predict"` or `python -m pytest tests/` to catch import issues early

## Scaling notes (next steps)
- Add tests that run a single forward pass using a tiny synthetic batch to validate model + collator.
- Add optional `accelerate` configuration for multi-GPU runs and a sample `accelerate config` workflow.
- Add `save_safetensors=True` in TrainingArguments or in config to use safetensors (faster and safer serialization) if desired.
- Consider adding `Trainer.model_init` and `optuna`/`Ray Tune` hooks for hyperparameter searches.

---

If you want, I can now:
- Add a minimal `README` test (unit test) and CI workflow.
- Pin dependencies and produce a `requirements-lock.txt`.

Tell me which follow-up you want and I will implement it.
