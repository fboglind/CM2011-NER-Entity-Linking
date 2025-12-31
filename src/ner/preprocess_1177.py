"""preprocess_1177.py - Convert Swedish Medical NER to token classification format

Compatible with community-datasets/swedish_medical_ner schema.
"""

import re
from datasets import load_dataset, DatasetDict, Features, Sequence, ClassLabel, Value

DATASET_NAME = "community-datasets/swedish_medical_ner"
CONFIG_NAME = "1177"
OUT_DIR = "swedish_medical_ner_1177_bio"

# Entity type mapping (from dataset ClassLabel names to BIO tokens)
ENTITY_TYPE_MAPPING = {
    "Disorder and Finding": "disorder_finding",
    "Pharmaceutical Drug": "pharmaceutical_drug", 
    "Body Structure": "body_structure",
}


def whitespace_tokens_with_spans(text):
    """Tokenize by whitespace, returning (token, start, end) tuples."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def build_label_list_from_schema(ds):
    """Extract entity types from dataset schema and build BIO label list."""
    # Get type names from ClassLabel feature
    type_names = ds.features["entities"]["type"].feature.names
    
    labels = ["O"]
    for name in sorted(type_names):
        token = ENTITY_TYPE_MAPPING.get(name, name.lower().replace(" ", "_"))
        labels.append(f"B-{token}")
        labels.append(f"I-{token}")
    return labels, type_names


def normalize_entities(entities_dict):
    """Convert dict-of-lists to list-of-dicts format.
    
    Input:  {'start': [0, 74], 'end': [13, 85], 'text': [...], 'type': [1, 2]}
    Output: [{'start': 0, 'end': 13, 'text': ..., 'type': 1}, ...]
    """
    if isinstance(entities_dict, list):
        return entities_dict
    
    n = len(entities_dict.get("start", []))
    return [
        {
            "start": entities_dict["start"][i],
            "end": entities_dict["end"][i],
            "text": entities_dict["text"][i],
            "type": entities_dict["type"][i],
        }
        for i in range(n)
    ]


def tag_sentence(tokens_spans, entities, label2id, type_names):
    """Assign BIO labels to tokens based on entity spans.
    
    Args:
        tokens_spans: list of (token, start, end)
        entities: list of entity dicts with start, end, type
        label2id: mapping from label string to ID
        type_names: list of original type names from ClassLabel
    """
    labels = ["O"] * len(tokens_spans)
    
    # Sort entities for deterministic behavior
    ents = sorted(entities, key=lambda e: (e["start"], e["end"]))
    
    for e in ents:
        est, eend = e["start"], e["end"]
        etype_idx = e["type"]
        
        # Map type index to BIO token
        type_name = type_names[etype_idx] if isinstance(etype_idx, int) else str(etype_idx)
        etype = ENTITY_TYPE_MAPPING.get(type_name, type_name.lower().replace(" ", "_"))
        
        # Find tokens that overlap with entity span
        covered = []
        for i, (_, s, t) in enumerate(tokens_spans):
            if not (t <= est or s >= eend):  # overlap
                covered.append(i)
        
        if not covered:
            continue
        
        labels[covered[0]] = f"B-{etype}"
        for j in covered[1:]:
            labels[j] = f"I-{etype}"
    
    return [label2id[l] for l in labels]


def main():
    # Load dataset
    ds = load_dataset(DATASET_NAME, CONFIG_NAME)
    
    # Build label list from schema
    label_list, type_names = build_label_list_from_schema(ds["train"])
    label2id = {l: i for i, l in enumerate(label_list)}
    print(f"Labels: {label_list}")
    print(f"Type names: {type_names}")
    
    def convert(example):
        text = example["sentence"]
        
        # Normalize entities from dict-of-lists to list-of-dicts
        entities = normalize_entities(example["entities"])
        
        # Tokenize and tag
        toks_spans = whitespace_tokens_with_spans(text)
        tokens = [t for t, _, _ in toks_spans]
        ner_tags = tag_sentence(toks_spans, entities, label2id, type_names)
        
        return {"tokens": tokens, "ner_tags": ner_tags}
    
    # Convert all splits
    converted = DatasetDict()
    for split in ds.keys():
        converted[split] = ds[split].map(
            convert,
            remove_columns=ds[split].column_names,
            desc=f"Converting {split}"
        )
    
    # Cast to explicit features with ClassLabel
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list)),
    })
    for split in converted.keys():
        converted[split] = converted[split].cast(features)
    
    # Save
    converted.save_to_disk(OUT_DIR)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
