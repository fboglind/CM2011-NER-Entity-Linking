# Simple EL CLI: query BM25 index and return Top-K ICD-10-SE codes
import argparse
import pickle

def bm25_search(bm25, docs_tokens: list[list[str]], query: str, top_k: int = 10) -> list[tuple[int, float]]:
    scores = bm25.get_scores(query.split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    with open(args.index_path, "rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]

    ranked = bm25.get_top_n(args.query.split(), payload["texts"], n=args.top_k)
    # We also want indices & scores; rank-bm25 doesn't return scores for get_top_n, so recompute:
    scores = bm25.get_scores(args.query.split())
    idx_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.top_k]

    print(f"Query: {args.query}")
    print("="*60)
    for i, (idx, sc) in enumerate(idx_score, 1):
        code = payload["codes"][idx] if idx < len(payload["codes"]) else ""
        title = payload["titles"][idx] if idx < len(payload["titles"]) else ""
        print(f"{i:>2}. {code:10s}  score={sc:.3f}  {title}")

if __name__ == "__main__":
    main()
