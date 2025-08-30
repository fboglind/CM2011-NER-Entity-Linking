from dataclasses import dataclass

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Constants
WHITESPACE_OR_BRACKETS = set(" \n\t()[]{}")

def trim_spans(sentence, start, end):
    while start < end and sentence[start] in WHITESPACE_OR_BRACKETS:
        start += 1
    while start < end and sentence[end-1] in WHITESPACE_OR_BRACKETS:
        end -= 1
    return (start, end)

#---------------------------Load data------------------------------------------------------------

def split_sources(loaded_dataset, val_fraction=0.05, seed=42) -> dict[str, DatasetDict]:
    """Load datasets, split into train/validation, and tag each example with source."""
    out = {}
    for loaded_dataset in loaded_dataset:
        # The config only has 'train'
        split = loaded_dataset["train"].train_test_split(test_size=val_fraction, seed=seed)
        ds = DatasetDict(train=split["train"], validation=split["test"])
        # Add 'source' column for keeping track of source dataset
        def _tag(ex, src=loaded_dataset.config_name): return {"source": src}
        ds = DatasetDict(
            train=ds["train"].map(_tag),
            validation=ds["validation"].map(_tag),
        )
        out[loaded_dataset.config_name] = ds
    return out

def build_bio_label_list_from_sources(per_source: dict[str]) -> list[str]:
    """Collect entity types using the dataset feature schemas and build BIO label list."""
    # Canonical normalization for names -> tokens used in BIO
    NAME_TO_TOKEN = {
        "Disorder and Finding": "disorder_finding",
        "Pharmaceutical Drug": "pharmaceutical_drug",
        "Body Structure": "body_structure",
    }

    types = set()
    for _, ds in per_source.items():               # e.g., {'1177': DatasetDict(...), 'lt': ..., 'wiki': ...}
        for split in ds.keys():                    # 'train', 'validation'
            feat = ds[split].features["entities"].feature
            # In this dataset, 'type' is a ClassLabel; get its names
            names = feat["type"].names
            types.update(NAME_TO_TOKEN[n] for n in names)

    labels = ["O"]
    for t in sorted(types):                        # alphabetical for determinism
        labels += [f"B-{t}", f"I-{t}"]
    return labels
