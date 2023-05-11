#!/bin/bash

set +x

OUTPUT_DIR="klue_output"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"

# KLUE-NER
task="klue-ner"
for model_name in "klue/roberta-small" "klue/roberta-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path ${model_name} --num_train_epochs 3 --max_seq_length 510 --metric_key character_macro_f1 --gpus 0 --num_workers 4
done
