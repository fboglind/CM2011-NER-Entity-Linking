import argparse
import os
import random
import re
from dataclasses import dataclass
import logging

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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#---------------------------Load data------------------------------------------------------------

def split_sources(loaded_dataset, val_fraction=0.05, seed=42) -> dict[str, DatasetDict]:
    """Load datasets, split into train/validation, and tag each example with source."""
    out = {}
    for loaded_dataset in loaded_dataset:
        # The BigBio config only has 'train'
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


def whitespace_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Return [(token, start, end_exclusive), ...] based on whitespace segmentation."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def spans_to_bio_labels(tokens_spans, entities, label2id):
    """Assign BIO labels to word tokens based on char-span overlap."""
    labels = ["O"] * len(tokens_spans)
    ents = sorted(entities, key=lambda e: (e["start"], e["end"]))
    for e in ents:
        est, eend, etype = e["start"], e["end"], e["type"]
        covered = []
        for i, (_, s, t) in enumerate(tokens_spans):
            if not (t <= est or s >= eend):  # overlap with [est, eend)
                covered.append(i)
        if not covered:
            continue
        labels[covered[0]] = f"B-{etype}"
        for j in covered[1:]:
            labels[j] = f"I-{etype}"
    return [label2id[l] for l in labels]

# def build_bio_label_list_from_sources(per_source: dict[str, DatasetDict]) -> list[str]:
#     """Collect entity types across all provided sources/splits and build BIO label list."""
#     types = set()
#     for _, ds in per_source.items():
#         for split in ds.keys():
#             for ex in ds[split]:
#                 for ent in ex.get("entities", []):
#                     types.add(ent["type"])
#     labels = ["O"]
#     for t in sorted(types):
#         labels += [f"B-{t}", f"I-{t}"]
#     return labels

def build_bio_label_list_from_sources(per_source: dict[str]) -> list[str]:
    """Collect entity typesusing the dataset feature schemas and build BIO label list."""
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


def compute_metrics_factory(id2label):
    label_list = [id2label[i] for i in range(len(id2label))]
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids
        true_predictions, true_labels = [], []
        for pred_seq, label_seq in zip(preds, labels):
            pred_labels, gold_labels = [], []
            for p_id, l_id in zip(pred_seq, label_seq):
                if l_id == -100:
                    continue
                pred_labels.append(label_list[p_id])
                gold_labels.append(label_list[l_id])
            true_predictions.append(pred_labels)
            true_labels.append(gold_labels)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        return {"precision": precision, "recall": recall, "f1": f1}
    return compute_metrics

def make_to_features(tokenizer, label2id, max_length):
    """Create a closure that converts span schema → token-classification features."""
    def to_features(example):
        text = example["sentence"]
        entities = example.get("entities", [])
        toks_spans = whitespace_tokens_with_spans(text)
        words = [t for t, _, _ in toks_spans]
        word_labels = spans_to_bio_labels(toks_spans, entities, label2id)

        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        word_ids = enc.word_ids()
        labels = []
        prev_w = None
        for w_id in word_ids:
            if w_id is None:
                labels.append(-100)
            else:
                if w_id != prev_w:
                    labels.append(word_labels[w_id])
                else:
                    labels.append(-100)
            prev_w = w_id

        enc["labels"] = labels
        return enc
    return to_features

# def normalize_entity_types(ds_split):
#     names = ds_split.features["entities"].feature["type"].names
#     name_to_token = {
#         "Disorder and Finding": "disorder_finding",
#         "Pharmaceutical Drug": "pharmaceutical_drug",
#         "Body Structure": "body_structure",
#     }
#     def _map_one(ex):
#         if ex.get("entities"):
#             for e in ex["entities"]:
#                 t = e["type"]
#                 if isinstance(t, int):
#                     e["type"] = name_to_token[names[t]]
#         return ex
#     return ds_split.map(_map_one)

NAME_TO_TOKEN = {
    "Disorder and Finding": "disorder_finding",
    "Pharmaceutical Drug": "pharmaceutical_drug",
    "Body Structure": "body_structure",
}

def normalize_types(ds_split):
    NAME_TO_TOKEN = {
    "Disorder and Finding": "disorder_finding",
    "Pharmaceutical Drug": "pharmaceutical_drug",
    "Body Structure": "body_structure",
}
    names = ds_split.features["entities"].feature["type"].names
    def _map(ex):
        ents = ex.get("entities", [])
        # entities may be dict-of-lists; convert to list-of-dicts for uniformity
        if isinstance(ents, dict):
            ents = [
                {"start": s, "end": e, "text": txt, "type": t}
                for s, e, txt, t in zip(ents["start"], ents["end"], ents["text"], ents["type"])
            ]
        for e in ents:
            if isinstance(e["type"], int):
                e["type"] = NAME_TO_TOKEN[names[e["type"]]]
        ex["entities"] = ents
        return ex
    return ds_split.map(_map)



def process_all(per_source: dict[str, DatasetDict], to_features) -> dict[str, DatasetDict]:
    """Map span→BIO+tokenize for each source and split."""
    processed = {}
    for cfg, ds in per_source.items():

        proc = DatasetDict()
        for split in ds.keys():
            proc[split] = ds[split].map(
                to_features,
                batched=False,
                remove_columns=ds[split].column_names,
                desc=f"[{cfg}] {split}",
            )
        processed[cfg] = proc
    return processed