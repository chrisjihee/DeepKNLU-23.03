#!/bin/bash

#python run_klue.py train --task klue-ner --output_dir klue_output0 --data_dir data/klue-ner-mini --model_name_or_path pretrained-com/KLUE-RoBERTa-Small --num_train_epochs 1 --max_seq_length 64 --metric_key character_macro_f1 --gpus 0 --num_workers 4
#python run_klue.py valid --task klue-ner --output_dir klue_output0 --data_dir data/klue-ner-mini --model_name_or_path klue_ner_0/version_0/transformers --num_train_epochs 1 --max_seq_length 64 --metric_key character_macro_f1 --gpus 0 --num_workers 4

python run_klue.py train --task klue-ner --output_dir klue_output1 --data_dir data/klue-ner-v1.1 --model_name_or_path klue/roberta-small --num_train_epochs 3 --max_seq_length 510 --metric_key character_macro_f1 --gpus 0 --num_workers 4
