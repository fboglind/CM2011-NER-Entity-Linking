# CM2011 Swedish Biomedical Named Entity Recognition & Entity Linking

A complete pipeline for extracting medical entities from Swedish clinical text and linking them to ICD-10-SE codes.

## Overview

This project fine-tunes [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased) (a Swedish BERT model from the National Library of Sweden) for Named Entity Recognition on Swedish medical text, then links extracted entities to the Swedish ICD-10-SE classification system.

### Key Results

| Task                    | Metric            | Score |
| ----------------------- | ----------------- | ----- |
| **NER** (1177 test set) | F1                | 0.95  |
|                         | Precision         | 0.96  |
|                         | Recall            | 0.94  |
| **Entity Linking**      | Coverage          | 100%  |
|                         | Median BM25 Score | 9.8   |

### Entity Types

| Type                 | Swedish          | Examples                         |
| -------------------- | ---------------- | -------------------------------- |
| Disorder and Finding | Sjukdom & Symtom | diabetes, smärta, stroke         |
| Pharmaceutical Drug  | Läkemedel        | ibuprofen, kortison, paracetamol |
| Body Structure       | Kroppsdel        | hjärnan, huden, knä              |

## Project Structure

```
├── notebooks/
│   └── cm2011_ner_entity_linking.ipynb   # Main notebook
├── ner_utility_functions.py               # Helper functions
├── models/
│   └── ner_kbbert_1177/                   # Trained NER model
└── data/
    └── icd10se/                           # ICD-10-SE classification
```

## Key Findings

### 1. Data Quality Analysis

The Swedish Medical NER dataset contains three subsets with significantly different annotation quality:

| Subset   | Size    | Annotation             | Quality                     |
| -------- | ------- | ---------------------- | --------------------------- |
| **1177** | 927     | Manual (gold standard) | ✓ Clean                     |
| **lt**   | 745,753 | Distant supervision    | ✗ ~10,000 systematic errors |
| **wiki** | 48,720  | Distant supervision    | ✗ ~1,000 systematic errors  |

**Finding**: The distantly-supervised subsets (lt, wiki) contain systematic mislabelings — body structure terms like "hjärtat" (the heart) are incorrectly labeled as disorders thousands of times.

### 2. Annotation Artifact: Bracket Markers

The original dataset uses brackets to mark entity boundaries in the text itself:

```
( Demens ) innebär att man på olika sätt får svårt att minnas...
```

Training on this raw data causes the model to rely on brackets for recognition, failing on natural text. **Our preprocessing removes brackets and adjusts entity offsets**, enabling the model to recognize entities in real clinical text.

### 3. Curriculum Learning: Negative Result

We hypothesized that pre-training on larger noisy datasets would improve performance. Results showed the opposite:

| Training Data                 | F1 on 1177 (Gold) |
| ----------------------------- | ----------------- |
| 1177 only                     | **0.95**          |
| lt → wiki → 1177 (curriculum) | 0.70              |

**Conclusion**: Data quality matters more than quantity. The incompatible annotation patterns in lt/wiki actively hurt performance.

## Methods

### NER Model

- **Base Model**: KB-BERT (`KB/bert-base-swedish-cased`)
- **Architecture**: BERT + Token Classification Head (7 labels: BIO scheme)
- **Training**: 4 epochs, lr=2e-5, batch_size=16
- **Data**: 1177 subset only (880 train / 47 validation)

### Entity Linking

- **Method**: BM25 lexical retrieval
- **Index**: ICD-10-SE titles + descriptions (~39,000 codes)
- **Scope**: Disorder/Finding entities → ICD-10-SE codes

## Usage

### NER Inference

```python
from transformers import pipeline

ner = pipeline("ner", model="./models/ner_kbbert_1177", aggregation_strategy="simple")

text = "Patienten har diagnosen diabetes och tar metformin dagligen."
entities = ner(text)
# [{'entity_group': 'disorder_finding', 'word': 'diabetes', 'score': 0.98},
#  {'entity_group': 'pharmaceutical_drug', 'word': 'metformin', 'score': 0.97}]
```

```
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

### Entity Linking

```python
from ner_utility_functions import ICD10Linker

linker = ICD10Linker("./data/icd10se/icd-10-se.tsv")
results = linker.link("diabetes", top_k=3)
# [{'code': 'E11', 'title': 'Diabetes mellitus typ 2', 'score': 15.2}, ...]
```

```
python -m src.el.icd_index   --icd_tsv data/icd10se/icd10se.tsv   --index_path outputs/icd10se_bm25.index

# Example: link a raw mention string
python -m src.el.linker   --index_path outputs/icd10se_bm25.index   --query "Appendicit" --top_k 10
```



## Requirements

```
transformers>=4.30.0
datasets>=2.14.0
seqeval>=1.2.2
rank_bm25>=0.2.2
torch>=2.0.0
pandas>=2.0.0
```

## Repo layout



```
configs/                 # YAML configs (optional)
data/icd10se/            # Place ICD‑10‑SE TSV here (not committed)
scripts/                 # Helper scripts (e.g., run/train wrappers)
src/ner/                 # NER training & eval
src/el/                  # Entity Linking (BM25 + embeddings rerank)
outputs/                 # (created at runtime) models, indices, logs
```

​                          

## References

- Almgren, S., & Pavlov, S. (2016). *Named Entity Recognition in Swedish Medical Journals*. Chalmers.
- Rosvall, E., & Paasonen, A. (2023). *Data Augmentation for Swedish Clinical NER*. Chalmers.
- Malmsten, M., et al. (2020). *Playing with Words at the National Library of Sweden*. arXiv:2007.01658.

## Future Work

1. **Embedding-based linking**: Replace BM25 with multilingual medical embeddings for semantic matching
2. **Multi-ontology support**: Add ATC codes for drugs, SNOMED-CT for body structures
3. **Context-aware linking**: Use sentence context to disambiguate entities
4. **Negation detection**: Identify negated entities ("ingen smärta" = no pain)

## License

MIT
