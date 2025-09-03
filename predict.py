"""
predict.py
- Loads the fine-tuned model from ./results/best
- Predicts sentiment for a single input text
Usage:
  python predict.py "I absolutely loved this movie!"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./results/best"

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

def predict(text: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(outputs.logits.argmax(dim=-1).item())
        score = float(outputs.logits.softmax(dim=-1)[0, pred_id].item())

    return LABEL_MAP[pred_id], score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"your text here\"")
        sys.exit(1)

    text = sys.argv[1]
    label, score = predict(text)
    print(f"Text: {text}")
    print(f"Prediction: {label} (confidence={score:.3f})")
