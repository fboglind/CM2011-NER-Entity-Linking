# Build a BM25 index over ICD-10-SE TSV columns
import argparse
import json
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import pickle

COLUMNS_CANDIDATES = [
    "KOD", "KAPITEL", "RUBRIK", "TITEL", "LATIN", "BESKRIVNING",
    "INNEFATTAR", "EXEMPEL", "UTESLUTER", "ANVANDAREN",
    "RICH_TEXT", "RICH_TEXT_ENG"  # names vary; we'll keep flexible
]

def normalize_text(s: str) -> str:
    s = str(s) if s is not None else ""
    # Strip HTML if present
    s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd_tsv", type=str, required=True, help="Path to ICD-10-SE TSV file")
    ap.add_argument("--index_path", type=str, default="outputs/icd10se_bm25.index")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    df = pd.read_csv(args.icd_tsv, sep="\t", dtype=str, keep_default_na=False)

    # Keep commonly useful columns if present
    keep = [c for c in df.columns if c.upper() in COLUMNS_CANDIDATES or c in COLUMNS_CANDIDATES]
    if "KOD" not in df.columns and "Kod" in df.columns:
        df.rename(columns={"Kod": "KOD"}, inplace=True)
    if "TITEL" not in df.columns and "Titel" in df.columns:
        df.rename(columns={"Titel": "TITEL"}, inplace=True)

    # Build text field
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if col.upper() in ["RICH_TEXT", "RICH_TEXT_ENG"] or "RICH" in col.upper():
                parts.append(normalize_text(row[col]))
            elif col.upper() in ["TITEL", "LATIN", "BESKRIVNING", "INNEFATTAR", "EXEMPEL"]:
                parts.append(normalize_text(row[col]))
        texts.append(" ".join([p for p in parts if p]))

    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "codes": df.get("KOD", df.get("Kod", pd.Series([""] * len(df)))).fillna("").tolist(),
        "titles": df.get("TITEL", df.get("Titel", pd.Series([""] * len(df)))).fillna("").tolist(),
        "texts": texts,
        "bm25": bm25,
    }

    with open(args.index_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved BM25 index with {len(df)} entries to {args.index_path}")

if __name__ == "__main__":
    main()
