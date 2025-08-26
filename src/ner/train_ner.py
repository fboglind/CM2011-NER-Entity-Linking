"""train_ner.py"""

import argparse
import logging


import numpy as np
from datasets import load_dataset, DatasetDict, Features, Sequence, ClassLabel, Value, interleave_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from seqeval.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # ---------------------------- Span→BIO helpers ----------------------------

# def whitespace_tokens_with_spans(text: str) -> List[Tuple[str, int, int]]:
#     """Return [(token, start, end_exclusive), ...] based on whitespace segmentation."""
#     return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

# def spans_to_bio_labels(tokens_spans, entities, label2id):
#     """Assign BIO labels to word tokens based on char-span overlap."""
#     labels = ["O"] * len(tokens_spans)
#     ents = sorted(entities, key=lambda e: (e["start"], e["end"]))
#     for e in ents:
#         est, eend, etype = e["start"], e["end"], e["type"]
#         covered = []
#         for i, (_, s, t) in enumerate(tokens_spans):
#             if not (t <= est or s >= eend):  # overlap with [est, eend)
#                 covered.append(i)
#         if not covered:
#             continue
#         labels[covered[0]] = f"B-{etype}"
#         for j in covered[1:]:
#             labels[j] = f"I-{etype}"
#     return [label2id[l] for l in labels]

# def build_bio_label_list_from_sources(per_source: Dict[str, DatasetDict]) -> List[str]:
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

# ---------------------------- Metrics ----------------------------

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

# ---------------------------- Data loading & processing ----------------------------

def load_datasets(dataset_name, configs, val_fraction=0.05, seed=42) -> dict[str, DatasetDict]:
    """Load multiple configs, ensure train/validation, and tag each example with its source."""
    out = {}
    for cfg in configs:
        logger.info(f"Loading {dataset_name} / {cfg}")
        ds = load_dataset(dataset_name, cfg, trust_remote_code=True)
        # The BigBio config only have 'train'
        split = ds["train"].train_test_split(test_size=val_fraction, seed=seed)
        ds = DatasetDict(train=split["train"], validation=split["test"])
        # Add a 'source' column for keeping track of source dataset
        def _tag(ex, src=cfg): return {"source": src}
        ds = DatasetDict(
            train=ds["train"].map(_tag),
            validation=ds["validation"].map(_tag),
        )
        out[cfg] = ds
    return out

# def make_to_features(tokenizer, label2id, max_length):
#     """Create a closure that converts span schema → token-classification features."""
#     def to_features(example):
#         text = example["sentence"]
#         entities = example.get("entities", [])
#         toks_spans = whitespace_tokens_with_spans(text)
#         words = [t for t, _, _ in toks_spans]
#         word_labels = spans_to_bio_labels(toks_spans, entities, label2id)

#         enc = tokenizer(
#             words,
#             is_split_into_words=True,
#             truncation=True,
#             max_length=max_length,
#             return_overflowing_tokens=False,
#         )
#         word_ids = enc.word_ids()
#         labels = []
#         prev_w = None
#         for w_id in word_ids:
#             if w_id is None:
#                 labels.append(-100)
#             else:
#                 if w_id != prev_w:
#                     labels.append(word_labels[w_id])
#                 else:
#                     labels.append(-100)
#             prev_w = w_id

#         enc["labels"] = labels
#         return enc
#     return to_features

def process_all(per_source: dict[str, DatasetDict], to_features) -> dict[str, DatasetDict]:
    """Map span→BIO+tokenize for each source and split."""
    processed = {}
    for cfg, ds in per_source.items():
        logger.info(f"[{cfg}] Converting spans → BIO and tokenizing")
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

# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Swedish biomedical NER — multi-dataset trainer (span→BIO).")
    # Core I/O
    parser.add_argument("--model_name", type=str, default="KB/bert-base-swedish-cased")
    parser.add_argument("--output_dir", type=str, default="outputs/ner_kbbert_multi")
    # Data
    parser.add_argument("--dataset_name", type=str, default="bigbio/swedish_medical_ner")
    parser.add_argument("--dataset_configs", type=str, nargs="+",
                        default=["swedish_medical_ner_lt_source", "swedish_medical_ner_wiki_source", "swedish_medical_ner_1177_source"])
    parser.add_argument("--val_fraction", type=float, default=0.05)
    
    # Training schedule
    parser.add_argument("--mix_mode", choices=["staged", "interleave"], default="staged",
                        help="staged = curriculum fine-tuning; interleave = mix datasets in one run.")
    parser.add_argument("--mix_probs", type=float, nargs="+", default=[0.85, 0.14, 0.01],
                        help="Interleave probabilities (must match dataset_configs length).")
    # Usual hyperparams
    parser.add_argument("--num_train_epochs", type=float, default=2.0, help="Only used for interleave mode.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Only used for interleave mode.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    # GPU/Memory QoL (set these on GPU)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Staged curriculum defaults (epochs & LR per stage)
    parser.add_argument("--stage_lt_epochs", type=float, default=1.0)
    parser.add_argument("--stage_lt_lr", type=float, default=2e-5)
    parser.add_argument("--stage_wiki_epochs", type=float, default=1.0)
    parser.add_argument("--stage_wiki_lr", type=float, default=1e-5)
    parser.add_argument("--stage_1177_epochs", type=float, default=2.0)
    parser.add_argument("--stage_1177_lr", type=float, default=5e-6)

    args = parser.parse_args()
    set_seed(args.seed)

    # 1) Load all requested sources
    per_source_raw = load_datasets(
        dataset_name=args.dataset_name,
        configs=args.dataset_configs,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

    # 2) Build global BIO labels (union over sources)
    label_list = build_bio_label_list_from_sources(per_source_raw)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"Global labels ({len(label_list)}): {label_list}")

    # 3) Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = compute_metrics_factory(id2label=id2label)

    # 4) Convert span schema → token-classification features
    to_features = make_to_features(tokenizer, label2id, args.max_length)
    per_source = process_all(per_source_raw, to_features)

    # 5) Prepare TrainingArguments base (avoid version-dependent kwargs like evaluation_strategy)
    base_args = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=50,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        seed=args.seed,
        report_to=[],  # disable HF trackers by default
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # 6) Train
    if args.mix_mode == "interleave":
        # Interleave train according to probabilities
        if len(args.mix_probs) != len(args.dataset_configs):
            raise ValueError("mix_probs must have same length as dataset_configs.")
        train_sets = [per_source[c]["train"] for c in args.dataset_configs]
        train_dataset = interleave_datasets(train_sets, probabilities=args.mix_probs, seed=args.seed)
        # For eval, keep per-domain validation
        eval_datasets = {c: per_source[c]["validation"] for c in args.dataset_configs}

        training_args = TrainingArguments(**base_args)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # (We’ll evaluate per-source manually after training)
            tokenizer=tokenizer,               # keep for 4.x; harmless warning about deprecation
            data_collator=collator,
            compute_metrics=None,              # do per-source eval below
        )
        trainer.train()

        # Per-domain evaluation after training
        for cfg, eval_ds in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_ds)
            logger.info(f"[EVAL @ {cfg}] {metrics}")

    else:
        # Staged curriculum: lt -> wiki -> 1177 (default order if present)
        order = []
        # map short names to full config ids if present
        short2cfg = { "lt":None, "wiki":None, "1177":None }
        for cfg in args.dataset_configs:
            if "lt" in cfg: short2cfg["lt"] = cfg
            elif "wiki" in cfg: short2cfg["wiki"] = cfg
            elif "1177" in cfg: short2cfg["1177"] = cfg
        if short2cfg["lt"]:   order.append(("lt", short2cfg["lt"], args.stage_lt_epochs, args.stage_lt_lr))
        if short2cfg["wiki"]: order.append(("wiki", short2cfg["wiki"], args.stage_wiki_epochs, args.stage_wiki_lr))
        if short2cfg["1177"]: order.append(("1177", short2cfg["1177"], args.stage_1177_epochs, args.stage_1177_lr))
        if not order:
            raise ValueError("Could not infer stages from dataset_configs. Include lt/wiki/1177 source configs.")

        for stage_name, cfg, epochs, lr in order:
            logger.info(f"\n=== Stage: {stage_name} on {cfg} | epochs={epochs} lr={lr} ===")
            stage_args = TrainingArguments(
                **{**base_args,
                   "num_train_epochs": epochs,
                   "learning_rate": lr,
                   "output_dir": os.path.join(args.output_dir, f"stage_{stage_name}")},
            )
            trainer = Trainer(
                model=model,
                args=stage_args,
                train_dataset=per_source[cfg]["train"],
                eval_dataset=per_source[cfg]["validation"],
                tokenizer=tokenizer,             # (works on 4.x; future deprec warns ok)
                data_collator=collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            metrics = trainer.evaluate()
            logger.info(f"[EVAL @ {stage_name}] {metrics}")
            # carry forward the fine-tuned weights to the next stage
            model = trainer.model

    # 7) Final save (last stage / interleave result)
    logger.info("Training complete. Saving final model to output_dir.")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
