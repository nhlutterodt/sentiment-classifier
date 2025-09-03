"""
train.py
- Fine-tunes a DistilBERT classifier on IMDB sentiment.
- Uses small subsets by default for a fast first run.
- Reads hyperparameters from config.yaml.
"""

import os
import random
import sys
import numpy as np
import yaml

from datasets import load_dataset
import evaluate

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding,
)

def load_config(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # ----- Load config -----
    # Allow passing a different config file path as the first CLI arg
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]

    train_samples = cfg.get("train_samples", None)
    eval_samples = cfg.get("eval_samples", None)

    num_train_epochs = cfg["num_train_epochs"]
    learning_rate = float(cfg["learning_rate"])
    per_device_train_batch_size = cfg["per_device_train_batch_size"]
    per_device_eval_batch_size = cfg["per_device_eval_batch_size"]
    weight_decay = float(cfg["weight_decay"])
    evaluation_strategy = cfg.get("evaluation_strategy", "epoch")
    seed = cfg.get("seed", 42)

    # ----- Reproducibility -----
    seed_everything(seed)

    # ----- Load dataset (IMDB) -----
    # Each example: {"text": str, "label": 0 or 1}
    dataset = load_dataset("imdb")

    # ----- Tokenizer & Model -----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ----- Tokenization function -----
    def tokenize_function(example):
        # Truncation keeps max length; padding done dynamically by Trainer.
        return tokenizer(example["text"], truncation=True)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # ----- Optional subsetting to run fast -----
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["test"]

    if train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=seed).select(range(min(train_samples, len(train_dataset))))
    if eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(eval_samples, len(eval_dataset))))

    # ----- Metrics -----
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # ----- Training args -----
    # Some transformers versions expect `eval_strategy`, others `evaluation_strategy`.
    # Detect supported parameter name and pass the value from config accordingly.
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)

    # Read save/load options from config so dev config can override them
    save_strategy = cfg.get("save_strategy", "epoch")
    load_best = cfg.get("load_best_model_at_end", True)

    # If load_best_model_at_end is requested, ensure save_strategy matches evaluation.
    if load_best and save_strategy != evaluation_strategy:
        print(
            f"Note: changing save_strategy from '{save_strategy}' to '{evaluation_strategy}' to support load_best_model_at_end"
        )
        save_strategy = evaluation_strategy

    if "evaluation_strategy" in sig.parameters:
        ta_kwargs = {"evaluation_strategy": evaluation_strategy}
    else:
        ta_kwargs = {"eval_strategy": evaluation_strategy}

    training_args = TrainingArguments(
        output_dir=output_dir,
        **ta_kwargs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_steps=cfg.get("logging_steps", 50),
        save_strategy=save_strategy,
        load_best_model_at_end=load_best,
        metric_for_best_model="accuracy",
        report_to="none",  # disable W&B etc.
        seed=seed,
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    # `processing_class` supersedes the old `tokenizer` argument. Use an explicit
    # DataCollatorWithPadding for dynamic padding (future-proof + more explicit).
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    )

    # ----- Train -----
    trainer.train()

    # ----- Final eval -----
    metrics = trainer.evaluate()
    print("\n=== FINAL EVAL METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ----- Save best model & tokenizer -----
    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nSaved best model to: {best_dir}")

if __name__ == "__main__":
    main()
