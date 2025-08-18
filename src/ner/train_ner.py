# NER training script using Hugging Face Transformers
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)
import evaluate
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="KB/bert-base-swedish-cased")
    p.add_argument("--dataset_name", type=str, default="bigbio/swedish_medical_ner")
    p.add_argument("--dataset_config", type=str, default="1177")
    p.add_argument("--output_dir", type=str, default="outputs/ner_baseline")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def get_label_list(features) -> list[str]:
    # Try to infer label names from the dataset features
    # Expecting a 'ner_tags' or 'labels' sequence of ClassLabel
    for candidate in ("ner_tags", "labels"):
        if candidate in features and isinstance(features[candidate].feature, ClassLabel):
            return features[candidate].feature.names
    raise ValueError("Could not infer label list. Expected a sequence(ClassLabel) feature named 'ner_tags' or 'labels'.")

def align_labels_with_tokens(labels, word_ids, ignore_index=-100):
    # Map word-level labels to token-level (first wordpiece gets the label, rest are ignored)
    aligned = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(ignore_index)
        elif word_id != prev_word_id:
            aligned.append(labels[word_id])
        else:
            aligned.append(ignore_index)
        prev_word_id = word_id
    return aligned

def main():
    args = parse_args()

    # 1) Load dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)
    # Expect train/validation/test splits
    # Some configs use 'validation' for dev; adapt if only 'train' is present.
    has_val = "validation" in dataset
    if not has_val and "dev" in dataset:
        dataset = dataset.rename_column("dev", "validation")

    # 2) Labels
    features = dataset["train"].features
    label_list = get_label_list(features)
    num_labels = len(label_list)

    # 3) Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # 4) Preprocess
    label_col = "ner_tags" if "ner_tags" in dataset["train"].features else "labels"
    text_col = "tokens" if "tokens" in dataset["train"].features else "words"

    def tokenize_and_align(batch):
        tokenized = tokenizer(batch[text_col], is_split_into_words=True, truncation=True)
        new_labels = []
        for i, labels in enumerate(batch[label_col]):
            word_ids = tokenized.word_ids(batch_index=i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))
        tokenized["labels"] = new_labels
        return tokenized

    tokenized = dataset.map(tokenize_and_align, batched=True, remove_columns=dataset["train"].column_names)

    # 5) Data collator & metrics
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []
        for pred, lab in zip(predictions, labels):
            curr_preds = []
            curr_labs = []
            for p_i, l_i in zip(pred, lab):
                if l_i != -100:
                    curr_preds.append(label_list[p_i])
                    curr_labs.append(label_list[l_i])
            true_predictions.append(curr_preds)
            true_labels.append(curr_labs)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    # 6) Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final evaluation on test set if present
    if "test" in tokenized:
        metrics = trainer.evaluate(tokenized["test"])
        print("Test metrics:", metrics)

    # Save label list for downstream use
    with open(f"{args.output_dir}/labels.json", "w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
