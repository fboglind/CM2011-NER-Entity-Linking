#!/usr/bin/env bash
set -euo pipefail

python -m src.ner.train_ner   --model_name KB/bert-base-swedish-cased   --dataset_name bigbio/swedish_medical_ner   --dataset_config 1177   --output_dir outputs/ner_kbbert_1177   --num_train_epochs 4   --per_device_train_batch_size 16   --per_device_eval_batch_size 32   --learning_rate 2e-5   --weight_decay 0.01   --warmup_ratio 0.1   --gradient_accumulation_steps 1   --seed 42
