
import argparse
import logging
import os
import random
import re
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
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------- Utilities ----------

def whitespace_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Return [(token, start, end_exclusive), ...] using whitespace segmentation."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def build_bio_label_list(ds: DatasetDict) -> list[str]:
    """Collect entity types across splits and produce BIO label list."""
    types = set()
    for split in ds.keys():
        for ex in ds[split]:
            for ent in ex.get("entities", []):
                types.add(ent["type"])
    label_list = ["O"]
    for t in sorted(types):
        label_list.append(f"B-{t}")
        label_list.append(f"I-{t}")
    return label_list


def spans_to_bio_labels(tokens_spans, entities, label2id):
    """Assign BIO labels to word tokens based on char-span overlap."""
    labels = ["O"] * len(tokens_spans)
    ents = sorted(entities, key=lambda e: (e["start"], e["end"]))
    for e in ents:
        est, eend, etype = e["start"], e["end"], e["type"]
        covered = []
        for i, (_, s, t) in enumerate(tokens_spans):
            # overlap with [est, eend)
            if not (t <= est or s >= eend):
                covered.append(i)
        if not covered:
            continue
        labels[covered[0]] = f"B-{etype}"
        for j in covered[1:]:
            labels[j] = f"I-{etype}"
    return [label2id[l] for l in labels]


def compute_metrics_factory(id2label):
    label_list = [id2label[i] for i in range(len(id2label))]

    def compute_metrics(p):
        # p.predictions: (bsz, seq_len, num_labels)
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        # Convert to label strings and strip special tokens (-100)
        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(preds, labels):
            pred_labels = []
            gold_labels = []
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


# ---------- Main training pipeline ----------

def main():
    parser = argparse.ArgumentParser(description="Train Swedish NER (span → BIO) on BigBio 1177 (or other config).")
    parser.add_argument("--model_name", type=str, default="KB/bert-base-swedish-cased")
    parser.add_argument("--dataset_name", type=str, default="bigbio/swedish_medical_ner")
    parser.add_argument("--dataset_config", type=str, default="swedish_medical_ner_1177_source")
    parser.add_argument("--output_dir", type=str, default="outputs/ner_kbbert_1177")
    parser.add_argument("--num_train_epochs", type=float, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length (subword tokens).")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) Load dataset (span schema: sentence + entities[{start,end,text,type}])
    logger.info(f"Loading dataset {args.dataset_name} / {args.dataset_config}")
    ds = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)

    # 2) Build BIO label set from all splits
    label_list = build_bio_label_list(ds)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"Labels ({len(label_list)}): {label_list}")

    # 3) Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # 4) Preprocess: span → word BIO → align to subwords
    def to_features(example):
        text = example["sentence"]
        entities = example.get("entities", [])

        # word tokens + spans based on whitespace
        toks_spans = whitespace_tokens_with_spans(text)
        words = [t for t, _, _ in toks_spans]
        word_labels = spans_to_bio_labels(toks_spans, entities, label2id)

        # tokenize words (so we can align labels via word_ids)
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
            return_overflowing_tokens=False,
            # We do not need offset_mapping since we align by word_ids.
        )

        word_ids = enc.word_ids()
        labels = []
        prev_word_id = None
        for w_id in word_ids:
            if w_id is None:
                labels.append(-100)  # special tokens
            else:
                # label only the first subword; others -> -100
                if w_id != prev_word_id:
                    labels.append(word_labels[w_id])
                else:
                    labels.append(-100)
            prev_word_id = w_id

        enc["labels"] = labels
        return enc

    processed = DatasetDict()
    for split in ds.keys():
        processed[split] = ds[split].map(
            to_features,
            batched=False,
            remove_columns=ds[split].column_names,
            desc=f"Converting {split} spans → BIO and tokenizing",
        )

    # 5) Data collator & metrics
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = compute_metrics_factory(id2label)

    # 6) Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=50,
        evaluation_strategy="epoch" if "validation" in processed else "no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if "validation" in processed else None,
    )

    # 7) Train / Evaluate / Save
    trainer.train()
    if "validation" in processed:
        eval_metrics = trainer.evaluate()
        logger.info(f"Validation metrics: {eval_metrics}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete. Model saved.")


if __name__ == "__main__":
    main()
