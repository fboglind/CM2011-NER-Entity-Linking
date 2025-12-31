"""train_ner.py - Swedish biomedical NER trainer

Compatible with community-datasets/swedish_medical_ner schema.
"""

import argparse
import logging
import os

import numpy as np
from datasets import load_dataset, DatasetDict, interleave_datasets
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


# ============================================================
# LABEL SCHEMA (fixed for Swedish Medical NER)
# ============================================================
# The 3 entity types are defined by the dataset, extracted from ClassLabel
# We normalize to snake_case for BIO tags

ENTITY_TYPE_MAPPING = {
    "Disorder and Finding": "disorder_finding",
    "Pharmaceutical Drug": "pharmaceutical_drug",
    "Body Structure": "body_structure",
}

def build_label_list():
    """Build BIO label list from known entity types."""
    label_list = ["O"]
    for orig_name in sorted(ENTITY_TYPE_MAPPING.keys()):
        token = ENTITY_TYPE_MAPPING[orig_name]
        label_list.append(f"B-{token}")
        label_list.append(f"I-{token}")
    return label_list


# ============================================================
# DATA LOADING & NORMALIZATION
# ============================================================

def normalize_entities(example):
    """Convert entities from dict-of-lists to list-of-dicts format.
    
    Input:  {'start': [0, 74], 'end': [13, 85], 'text': [...], 'type': [1, 2]}
    Output: [{'start': 0, 'end': 13, 'text': ..., 'type': 1}, ...]
    """
    ents = example.get("entities", {})
    if isinstance(ents, list):
        # Already normalized
        return example
    
    # Convert dict-of-lists to list-of-dicts
    normalized = []
    n = len(ents.get("start", []))
    for i in range(n):
        normalized.append({
            "start": ents["start"][i],
            "end": ents["end"][i],
            "text": ents["text"][i],
            "type": ents["type"][i],
        })
    return {**example, "entities": normalized}


def load_datasets(dataset_name, configs, val_fraction=0.05, seed=42):
    """Load multiple configs, split into train/val, normalize entities, tag source."""
    out = {}
    for cfg in configs:
        logger.info(f"Loading {dataset_name} / {cfg}")
        ds = load_dataset(dataset_name, cfg)
        
        # Split into train/validation (dataset only has 'train')
        split = ds["train"].train_test_split(test_size=val_fraction, seed=seed)
        ds = DatasetDict(train=split["train"], validation=split["test"])
        
        # Normalize entities and tag source
        def process(ex, src=cfg):
            ex = normalize_entities(ex)
            ex["source"] = src
            return ex
        
        ds = DatasetDict(
            train=ds["train"].map(process, desc=f"[{cfg}] Normalizing train"),
            validation=ds["validation"].map(process, desc=f"[{cfg}] Normalizing val"),
        )
        out[cfg] = ds
    return out


# ============================================================
# FEATURIZATION (span → BIO labels)
# ============================================================

WHITESPACE_OR_BRACKETS = set(" \n\t()[]{}") 

def trim_spans(sentence, start, end):
    """Trim brackets and whitespace from entity spans."""
    while start < end and sentence[start] in WHITESPACE_OR_BRACKETS:
        start += 1
    while start < end and sentence[end-1] in WHITESPACE_OR_BRACKETS:
        end -= 1
    return start, end


def make_to_features(tokenizer, label2id, type_names, max_length=256):
    """Create featurization function for token classification.
    
    Args:
        tokenizer: HuggingFace tokenizer
        label2id: dict mapping label strings to IDs
        type_names: list of original type names from dataset ClassLabel
        max_length: max sequence length
    """
    def type_token(t):
        """Map entity type (int or string) to BIO token."""
        name = type_names[t] if isinstance(t, int) else str(t)
        return ENTITY_TYPE_MAPPING.get(name, name.lower().replace(" ", "_"))
    
    def to_features(example):
        text = example["sentence"]
        entities = example.get("entities", []) or []
        
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]
        
        # Build trimmed spans for labeling
        ent_spans = []
        for e in entities:
            est, eend = e["start"], e["end"]
            est, eend = trim_spans(text, est, eend)
            if est >= eend:
                continue
            etok = type_token(e["type"])
            ent_spans.append((est, eend, etok))
        
        # Assign labels to tokens
        labels = []
        for (ts, te) in offsets:
            if ts == te:  # special token
                labels.append(-100)
                continue
            lab = "O"
            for (est, eend, etok) in ent_spans:
                if not (te <= est or ts >= eend):  # overlap
                    lab = f"B-{etok}" if ts <= est < te else f"I-{etok}"
                    break
            labels.append(label2id.get(lab, label2id["O"]))
        
        enc.pop("offset_mapping")
        enc["labels"] = labels
        return enc
    
    return to_features


def process_all(per_source, to_features):
    """Apply featurization to all sources and splits."""
    processed = {}
    for cfg, ds in per_source.items():
        logger.info(f"[{cfg}] Converting to token classification features")
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


# ============================================================
# METRICS
# ============================================================

def compute_metrics_factory(id2label):
    """Create compute_metrics function for seqeval."""
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
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    return compute_metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Swedish biomedical NER trainer (community-datasets schema)"
    )
    
    # Core I/O
    parser.add_argument("--model_name", type=str, default="KB/bert-base-swedish-cased")
    parser.add_argument("--output_dir", type=str, default="outputs/ner_kbbert_multi")
    
    # Data
    parser.add_argument("--dataset_name", type=str, 
                        default="community-datasets/swedish_medical_ner")
    parser.add_argument("--dataset_configs", type=str, nargs="+", 
                        default=["lt", "wiki", "1177"])
    parser.add_argument("--val_fraction", type=float, default=0.05)
    
    # Training mode
    parser.add_argument("--mix_mode", choices=["staged", "interleave"], default="staged",
                        help="staged = curriculum; interleave = mix datasets")
    parser.add_argument("--mix_probs", type=float, nargs="+", default=[0.85, 0.14, 0.01],
                        help="Interleave probabilities (must match dataset_configs length)")
    
    # Hyperparameters
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    
    # GPU/Memory
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=2)
    
    # Staged curriculum settings
    parser.add_argument("--stage_lt_epochs", type=float, default=1.0)
    parser.add_argument("--stage_lt_lr", type=float, default=2e-5)
    parser.add_argument("--stage_wiki_epochs", type=float, default=1.0)
    parser.add_argument("--stage_wiki_lr", type=float, default=1e-5)
    parser.add_argument("--stage_1177_epochs", type=float, default=2.0)
    parser.add_argument("--stage_1177_lr", type=float, default=5e-6)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 1) Build label vocabulary
    label_list = build_label_list()
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"Labels ({len(label_list)}): {label_list}")
    
    # 2) Load and normalize datasets
    per_source_raw = load_datasets(
        dataset_name=args.dataset_name,
        configs=args.dataset_configs,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    
    # 3) Get type names from first dataset's schema for type_token mapping
    first_cfg = args.dataset_configs[0]
    first_ds = load_dataset(args.dataset_name, first_cfg, split="train")
    type_names = first_ds.features["entities"]["type"].feature.names
    logger.info(f"Entity types from schema: {type_names}")
    
    # 4) Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = compute_metrics_factory(id2label)
    
    # 5) Featurize all datasets
    to_features = make_to_features(tokenizer, label2id, type_names, args.max_length)
    per_source = process_all(per_source_raw, to_features)
    
    # 6) Base training arguments
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
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # 7) Train
    if args.mix_mode == "interleave":
        # Interleave training datasets
        if len(args.mix_probs) != len(args.dataset_configs):
            raise ValueError("mix_probs must match dataset_configs length")
        
        train_sets = [per_source[c]["train"] for c in args.dataset_configs]
        train_dataset = interleave_datasets(
            train_sets, probabilities=args.mix_probs, seed=args.seed
        )
        eval_datasets = {c: per_source[c]["validation"] for c in args.dataset_configs}
        
        training_args = TrainingArguments(**base_args)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )
        trainer.train()
        
        # Per-domain evaluation
        for cfg, eval_ds in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_ds)
            logger.info(f"[EVAL @ {cfg}] {metrics}")
    
    else:
        # Staged curriculum: lt -> wiki -> 1177
        stages = []
        for cfg in args.dataset_configs:
            if "lt" in cfg:
                stages.append(("lt", cfg, args.stage_lt_epochs, args.stage_lt_lr))
            elif "wiki" in cfg:
                stages.append(("wiki", cfg, args.stage_wiki_epochs, args.stage_wiki_lr))
            elif "1177" in cfg:
                stages.append(("1177", cfg, args.stage_1177_epochs, args.stage_1177_lr))
        
        # Sort: lt -> wiki -> 1177
        stage_order = {"lt": 0, "wiki": 1, "1177": 2}
        stages.sort(key=lambda x: stage_order.get(x[0], 99))
        
        if not stages:
            raise ValueError("No valid stages found in dataset_configs")
        
        for stage_name, cfg, epochs, lr in stages:
            logger.info(f"\n=== Stage: {stage_name} | epochs={epochs} lr={lr} ===")
            
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
                tokenizer=tokenizer,
                data_collator=collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            
            metrics = trainer.evaluate()
            logger.info(f"[EVAL @ {stage_name}] {metrics}")
            
            # Carry forward fine-tuned weights
            model = trainer.model
    
    # 8) Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
