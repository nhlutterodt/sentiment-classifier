import torch
from transformers import BertConfig, AutoModelForSequenceClassification


def simple_collator(batch):
    """Pad a list of examples (each is dict with 'input_ids') into tensors."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [0] * (max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def test_single_batch_forward():
    # Build a tiny BERT config (no pretrained weights required)
    cfg = BertConfig(
        vocab_size=1024,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_labels=2,
    )

    model = AutoModelForSequenceClassification.from_config(cfg)
    model.eval()

    # Create a tiny batch with variable lengths
    batch = [
        {"input_ids": [101, 200, 300, 400]},
        {"input_ids": [101, 50, 60]},
    ]

    inputs = simple_collator(batch)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits should be (batch_size, num_labels)
    logits = outputs.logits
    assert logits.shape[0] == 2 and logits.shape[1] == 2
