
from datasets import load_dataset, DatasetDict, Features, Sequence, ClassLabel, Value
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
import re

DATASET_NAME = "bigbio/swedish_medical_ner"
CONFIG_NAME  = "swedish_medical_ner_1177_source"
OUT_DIR      = "swedish_medical_ner_1177_bio"  # local HF dataset folder

def whitespace_tokens_with_spans(text):
    # tokens as contiguous non-space runs; returns [(token, start, end_exclusive), ...]
    # tokens = word_tokenize(text, language='swedish')
    # return tokens
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def build_label_list(ds):
    types = set()
    for ex in ds:
        for ent in ex["entities"]:
            types.add(ent["type"])
    # BIO scheme (+ 'O')
    labels = ["O"]
    for t in sorted(types):
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels

def tag_sentence(tokens_spans, entities, label2id):
    """
    tokens_spans: list of (tok, start, end)
    entities: list of dicts with start, end, type
    """
    labels = ["O"] * len(tokens_spans)

    # sort entities for deterministic behavior; assume no overlaps
    ents = sorted(entities, key=lambda e: (e["start"], e["end"]))

    for e in ents:
        est, eend, etype = e["start"], e["end"], e["type"]
        # mark tokens whose span overlaps [est, eend)
        covered = []
        for i, (_, s, t) in enumerate(tokens_spans):
            if not (t <= est or s >= eend):  # overlap
                covered.append(i)
        if not covered:
            continue  # entity spans no token (rare; can happen with odd whitespace)
        labels[covered[0]] = f"B-{etype}"
        for j in covered[1:]:
            labels[j] = f"I-{etype}"

    return [label2id[l] for l in labels]

def main():
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, trust_remote_code=True)

    # Build the BIO label set from the whole training set (do train split only is fine)
    label_list = build_label_list(ds["train"])
    label2id = {l: i for i, l in enumerate(label_list)}

    def convert(example):
        text = example["sentence"]
        toks_spans = whitespace_tokens_with_spans(text)
        tokens = [t for t, _, _ in toks_spans]
        ner_tags = tag_sentence(toks_spans, example["entities"], label2id)
        return {"tokens": tokens, "ner_tags": ner_tags}

    converted = DatasetDict()
    for split in ds.keys():
        converted[split] = ds[split].map(convert, remove_columns=ds[split].column_names, desc=f"Converting {split}")

    # Cast to explicit features with ClassLabel, so your script can infer the label list cleanly
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list)),
    })
    for split in converted.keys():
        converted[split] = converted[split].cast(features)

    converted.save_to_disk(OUT_DIR)
    print(f"Saved to {OUT_DIR} with labels: {label_list}")

if __name__ == "__main__":
    main()
