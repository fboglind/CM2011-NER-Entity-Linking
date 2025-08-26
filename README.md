# CM2011 Group Project — NER & Entity Linking (Swedish biomedical)

This repo serves as a **working proof-of-concept** for:
1) **NER** on Swedish biomedical text (1177 subset) using Hugging Face Transformers
2) **Entity linking (EL)** from recognized mentions to **ICD‑10‑SE** using a **lexical baseline** (BM25) + optional **multilingual embeddings** reranker.

> Target: Get a clean, reproducible baseline in ~2 weeks using non-local GPU.

## Quick start

### 1) Clone & Python env
```bash
git clone https://github.com/fboglind/CM2011-NER-Entity-Linking.git
cd CM2011-NER-Entity-Linking
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install PyTorch as per https://pytorch.org/get-started/locally/
# (Select the CUDA version that matches your VM; CPU-only also works but is slow.)
```

### 2) Train NER (KB-BERT on 1177 subset data)
```bash
python -m src.ner.train_ner \
    --model_name KB/bert-base-swedish-cased \
    --dataset_name bigbio/swedish_medical_ner \
    --dataset_config swedish_medical_ner_1177_source \
    --output_dir outputs/ner_kbbert_1177\
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 1 \
    --seed 42
```

The script evaluates with **seqeval** on dev/test and saves the best checkpoint to `--output_dir`.

### 3) Prepare ICD‑10‑SE (Entity Linking)
Place the official TSV (e.g., `icd10se.tsv`) under `data/icd10se/` (filename is configurable):
```bash
python -m src.el.icd_index   --icd_tsv data/icd10se/icd10se.tsv   --index_path outputs/icd10se_bm25.index
```

### 4) Link mentions → ICD‑10‑SE
```bash
# Example: link a raw mention string
python -m src.el.linker   --index_path outputs/icd10se_bm25.index   --query "Appendicit" --top_k 10
```

Optional: Add multilingual embeddings reranker (see `src/el/embed_index.py`).

---

## Repo layout
```
configs/                 # YAML configs (optional)
data/icd10se/            # Place ICD‑10‑SE TSV here (not committed)
scripts/                 # Helper scripts (e.g., run/train wrappers)
src/ner/                 # NER training & eval
src/el/                  # Entity Linking (BM25 + embeddings rerank)
outputs/                 # (created at runtime) models, indices, logs
```

## Notes
- **Dataset**: The **1177** subset is used for gold evaluation. Other subsets are auto-annotated and can be used with care.
- **Repro**: Set seeds; enable `--report_to none` in the trainer if you don't use WandB.

## Next steps
- Run the baseline as-is.
- Add error analysis notebooks (qualitative examples).
- Consider domain-adaptive pretraining (MLM) if time/compute allows.
- (Stretch) Translate to English and compare BioBERT/SapBERT.
