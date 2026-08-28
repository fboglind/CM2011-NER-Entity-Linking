"""
ner_utility_funcs.py

Utility functions for Swedish Biomedical NER project.
"""

import re
import numpy as np
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score

# =============================================================================
# CONSTANTS
# =============================================================================

ENTITY_TYPES = ["Disorder and Finding", "Pharmaceutical Drug", "Body Structure"]

NAME_TO_TOKEN = {
    "Disorder and Finding": "disorder_finding",
    "Pharmaceutical Drug": "pharmaceutical_drug",
    "Body Structure": "body_structure",
}

BRACKETS = set("()[]{}")


# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_sentence_and_entities(sentence, entities):
    """
    Remove annotation brackets () [] {} from sentence and adjust entity offsets.
    
    The original dataset uses brackets to mark entity boundaries in the text:
        "( Demens ) innebär att man..."
    This causes the model to rely on brackets for recognition. 
    We remove them so the model learns to recognize entities without markers.
    
    Args:
        sentence: Original sentence with brackets
        entities: List of entity dicts with 'start', 'end', 'text', 'type'
        
    Returns:
        (cleaned_sentence, adjusted_entities)
    """
    # Track cumulative removed characters at each position
    removed = [0] * (len(sentence) + 1)
    count = 0
    for i, char in enumerate(sentence):
        removed[i] = count
        if char in BRACKETS:
            count += 1
    removed[len(sentence)] = count
    
    # Clean sentence - remove brackets
    cleaned = "".join(c for c in sentence if c not in BRACKETS)
    
    # Adjust entity offsets
    adjusted = []
    for ent in entities:
        old_start, old_end = ent["start"], ent["end"]
        
        # Adjust for removed brackets
        new_start = old_start - removed[min(old_start, len(sentence))]
        new_end = old_end - removed[min(old_end, len(sentence))]
        
        # Trim whitespace from span edges
        while new_start < new_end and new_start < len(cleaned) and cleaned[new_start] in " \t\n":
            new_start += 1
        while new_start < new_end and new_end > 0 and cleaned[new_end-1] in " \t\n":
            new_end -= 1
        
        if new_start < new_end:
            adjusted.append({
                **ent,
                "start": new_start,
                "end": new_end,
                "text": cleaned[new_start:new_end]
            })
    
    return cleaned, adjusted


def normalize_entities(example):
    """
    Convert entity format from dict-of-lists to list-of-dicts.
    
    Input format (community-datasets schema):
        {'start': [0, 10], 'end': [5, 15], 'text': ['abc', 'def'], 'type': [0, 1]}
    
    Output format:
        [{'start': 0, 'end': 5, 'text': 'abc', 'type': 0}, 
         {'start': 10, 'end': 15, 'text': 'def', 'type': 1}]
    """
    ents = example.get("entities", {})
    if isinstance(ents, list):
        return example  # Already normalized
    
    starts = ents.get("start", [])
    ends = ents.get("end", [])
    texts = ents.get("text", [])
    types = ents.get("type", [])
    
    normalized = [
        {"start": s, "end": e, "text": t, "type": typ}
        for s, e, t, typ in zip(starts, ends, texts, types)
    ]
    
    return {**example, "entities": normalized}


def preprocess_example(example):
    """
    Full preprocessing pipeline for a single example:
    1. Normalize entity format
    2. Remove brackets and adjust offsets
    """
    # Step 1: Normalize entities
    normalized = normalize_entities(example)
    
    # Step 2: Clean brackets
    cleaned_sentence, cleaned_entities = clean_sentence_and_entities(
        normalized["sentence"], 
        normalized["entities"]
    )
    
    return {
        **normalized,
        "sentence": cleaned_sentence,
        "entities": cleaned_entities,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_swedish_medical_ner(configs=["1177", "lt", "wiki"], cache_dir="data"):
    """
    Load Swedish Medical NER dataset with caching.
    
    Args:
        configs: List of dataset configurations to load
        cache_dir: Directory for caching downloaded datasets
        
    Returns:
        List of DatasetDict objects
    """
    import os
    
    datasets = []
    for config in configs:
        cache_path = f"{cache_dir}/swedish_medical_ner_{config}"
        
        if os.path.isdir(f"{cache_path}/train"):
            print(f"Loading {config} from disk cache...")
            ds = load_from_disk(cache_path)
        else:
            print(f"Downloading {config} from HuggingFace...")
            ds = load_dataset("community-datasets/swedish_medical_ner", config)
            os.makedirs(cache_dir, exist_ok=True)
            ds.save_to_disk(cache_path)
        
        ds.config_name = config
        datasets.append(ds)
    
    return datasets


def split_and_preprocess(datasets, val_fraction=0.05, seed=42):
    """
    Split datasets into train/validation and apply preprocessing.
    
    Args:
        datasets: List of DatasetDict from load_swedish_medical_ner
        val_fraction: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping config name to DatasetDict with preprocessed train/validation
    """
    result = {}
    
    for ds in datasets:
        config = ds.config_name
        print(f"Processing {config}...")
        
        # Split train into train/validation
        split = ds["train"].train_test_split(test_size=val_fraction, seed=seed)
        
        # Apply preprocessing (normalize entities, remove brackets)
        train_processed = split["train"].map(preprocess_example, desc=f"{config}/train")
        val_processed = split["test"].map(preprocess_example, desc=f"{config}/val")
        
        # Add source tag
        train_processed = train_processed.map(lambda x: {**x, "source": config})
        val_processed = val_processed.map(lambda x: {**x, "source": config})
        
        result[config] = DatasetDict({
            "train": train_processed,
            "validation": val_processed
        })
    
    return result


# =============================================================================
# LABEL SCHEMA
# =============================================================================

def build_label_schema():
    """
    Build BIO label vocabulary from known entity types.
    
    Returns:
        (label_list, label2id, id2label)
    """
    label_list = ["O"]
    for etype in sorted(NAME_TO_TOKEN.values()):
        label_list.append(f"B-{etype}")
        label_list.append(f"I-{etype}")
    
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    return label_list, label2id, id2label


# =============================================================================
# FEATURIZATION
# =============================================================================

def make_to_features(tokenizer, label2id, max_length=256):
    """
    Create a function to convert examples to model input features.
    
    Uses offset mapping to align token-level labels with subword tokens.
    """
    
    def type_token(t):
        """Map entity type index to label token."""
        if isinstance(t, int):
            name = ENTITY_TYPES[t]
        else:
            name = str(t)
        return NAME_TO_TOKEN.get(name, name.lower().replace(" ", "_"))
    
    def to_features(example):
        text = example["sentence"]
        entities = example.get("entities", []) or []
        
        # Tokenize with offset mapping
        enc = tokenizer(
            text, 
            truncation=True, 
            max_length=max_length, 
            return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]
        
        # Build entity spans
        ent_spans = []
        for e in entities:
            est, eend = e["start"], e["end"]
            etok = type_token(e["type"])
            ent_spans.append((est, eend, etok))
        
        # Assign labels to tokens based on offset overlap
        labels = []
        for (ts, te) in offsets:
            if ts == te:  # Special token
                labels.append(-100)
                continue
            
            lab = "O"
            for (est, eend, etok) in ent_spans:
                if not (te <= est or ts >= eend):  # Overlap
                    lab = f"B-{etok}" if ts <= est < te else f"I-{etok}"
                    break
            
            labels.append(label2id.get(lab, label2id["O"]))
        
        # Remove offset_mapping (not needed for training)
        enc.pop("offset_mapping")
        enc["labels"] = labels
        
        return enc
    
    return to_features


# =============================================================================
# METRICS
# =============================================================================

def make_compute_metrics(id2label):
    """
    Create a compute_metrics function for the Trainer.
    
    Uses seqeval for proper entity-level evaluation.
    """
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Convert to label strings
        true_labels = []
        true_predictions = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            pred_tags = []
            gold_tags = []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    pred_tags.append(id2label[p])
                    gold_tags.append(id2label[l])
            true_predictions.append(pred_tags)
            true_labels.append(gold_tags)
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    return compute_metrics


# =============================================================================
# ENTITY LINKING
# =============================================================================

def load_icd10se(filepath):
    """
    Load and prepare ICD-10-SE for entity linking.
    
    Args:
        filepath: Path to the ICD-10-SE TSV file
        
    Returns:
        DataFrame with one row per code, aggregated text for searching
    """
    import pandas as pd
    import re
    
    def clean_html(text):
        if pd.isna(text):
            return ""
        return re.sub(r'<[^>]+>', ' ', str(text)).strip()
    
    # Load TSV
    icd_df = pd.read_csv(filepath, sep='\t', dtype=str, quotechar='"')
    
    # Aggregate multiple rows per code
    icd_docs = []
    grouped = icd_df.groupby("Kod")
    
    for code, group in grouped:
        first = group.iloc[0]
        texts = []
        
        if pd.notna(first.get("Titel")):
            texts.append(str(first["Titel"]))
        if pd.notna(first.get("Latin")):
            texts.append(str(first["Latin"]))
        if pd.notna(first.get("Beskrivning")):
            texts.append(clean_html(first["Beskrivning"]))
        
        for _, row in group.iterrows():
            if pd.notna(row.get("Innefattar")):
                texts.append(clean_html(row["Innefattar"]))
            if pd.notna(row.get("Exempel")):
                texts.append(clean_html(row["Exempel"]))
        
        doc_text = " ".join(texts)
        if doc_text.strip():
            icd_docs.append({
                "code": code,
                "title": str(first.get("Titel", "")),
                "text": doc_text,
                "parent": str(first.get("Överordnad kod", "")),
            })
    
    return pd.DataFrame(icd_docs)


def tokenize_swedish(text):
    """Simple Swedish tokenizer: lowercase, split on non-alphanumeric."""
    import re
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


class ICD10Linker:
    """
    BM25-based entity linker for ICD-10-SE.
    
    Usage:
        linker = ICD10Linker("/path/to/icd-10-se.tsv")
        results = linker.link("diabetes", top_k=5)
    """
    
    def __init__(self, icd_filepath):
        from rank_bm25 import BM25Okapi
        
        # Load ICD-10-SE
        self.icd_docs = load_icd10se(icd_filepath)
        
        # Build BM25 index
        corpus = [tokenize_swedish(doc) for doc in self.icd_docs["text"]]
        self.bm25 = BM25Okapi(corpus)
        
        print(f"ICD10Linker initialized with {len(self.icd_docs)} codes")
    
    def link(self, entity_text, top_k=5):
        """
        Link entity text to ICD-10-SE codes.
        
        Args:
            entity_text: The entity mention (e.g., "diabetes")
            top_k: Number of candidates to return
            
        Returns:
            List of dicts with 'code', 'title', 'score'
        """
        query_tokens = tokenize_swedish(entity_text)
        
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "code": self.icd_docs.iloc[idx]["code"],
                    "title": self.icd_docs.iloc[idx]["title"],
                    "score": float(scores[idx]),
                })
        
        return results
    
    def link_entities(self, entities, top_k=3):
        """
        Link multiple entities, filtering for disorder_finding only.
        
        Args:
            entities: List of entity dicts from extract_entities()
            top_k: Candidates per entity
            
        Returns:
            List of linked entity dicts
        """
        results = []
        
        for ent in entities:
            if ent.get("type") == "disorder_finding":
                candidates = self.link(ent["text"], top_k=top_k)
                results.append({
                    **ent,
                    "icd_candidates": candidates,
                    "top_icd": candidates[0] if candidates else None,
                })
        
        return results


# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def extract_entities(text, model, tokenizer, id2label, device="cpu"):
    """
    Extract entities from text using the trained model.
    
    Returns list of dicts with 'text', 'type', 'start', 'end', 'score'
    """
    import torch
    
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)[0]
        scores = probs.max(dim=-1).values[0]
    
    # Group consecutive B-/I- tokens into entities
    entities = []
    current_entity = None
    
    for i, (pred_id, (start, end)) in enumerate(zip(predictions, offsets)):
        if start == end:  # Skip special tokens
            continue
            
        label = id2label[pred_id.item()]
        score = scores[i].item()
        
        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            # Start new entity
            etype = label[2:]
            current_entity = {
                "text": text[start:end],
                "type": etype,
                "start": start,
                "end": end,
                "score": score,
            }
        elif label.startswith("I-") and current_entity:
            etype = label[2:]
            if etype == current_entity["type"]:
                # Extend current entity
                current_entity["text"] = text[current_entity["start"]:end]
                current_entity["end"] = end
                current_entity["score"] = min(current_entity["score"], score)
            else:
                # Type mismatch - save and reset
                entities.append(current_entity)
                current_entity = None
        else:
            # O label - save any current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities
