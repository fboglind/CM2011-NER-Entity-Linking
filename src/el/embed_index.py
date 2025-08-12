# Optional: build a multilingual embedding index for ICD-10-SE
# Uses sentence-transformers (paraphrase-multilingual-mpnet-base-v2 by default)
import argparse
import os
import pickle
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np

def normalize(s: str) -> str:
    s = BeautifulSoup(str(s), "html.parser").get_text(" ", strip=True)
    return " ".join(s.split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd_tsv", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--out_path", type=str, default="outputs/icd10se_embed.index")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df = pd.read_csv(args.icd_tsv, sep="\t", dtype=str, keep_default_na=False)

    texts = []
    titles = df.get("TITEL", df.get("Titel"))
    codes = df.get("KOD", df.get("Kod"))

    for _, row in df.iterrows():
        title = normalize(row.get("TITEL", row.get("Titel", "")))
        latin = normalize(row.get("LATIN", ""))
        desc = normalize(row.get("BESKRIVNING", ""))
        rich = normalize(row.get("RICH_TEXT", ""))
        texts.append(" ".join([title, latin, desc, rich]).strip())

    st_model = SentenceTransformer(args.model_name)
    embs = st_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    payload = {
        "codes": codes.tolist(),
        "titles": titles.tolist(),
        "texts": texts,
        "embeddings": embs.astype(np.float32),
        "model_name": args.model_name,
    }

    with open(args.out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved embedding index with {len(df)} entries to {args.out_path}")

if __name__ == "__main__":
    main()
