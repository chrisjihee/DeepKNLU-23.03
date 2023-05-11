#!/bin/bash

#python run_klue.py train --task klue-ner --output_dir klue_output0 --data_dir data/klue_benchmark/klue-ner-mini --model_name_or_path klue/roberta-small --num_train_epochs 1 --max_seq_length 32 --metric_key character_macro_f1 --gpus 0 --num_workers 4
python run_klue.py train --task klue-ner --output_dir klue_output1 --data_dir data/klue_benchmark/klue-ner-v1.1 --model_name_or_path klue/roberta-small --num_train_epochs 3 --max_seq_length 510 --metric_key character_macro_f1 --gpus 0 --num_workers 4
