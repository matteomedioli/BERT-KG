#!/bin/bash
source /data/medioli/env/bin/activate
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=64
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=6


echo "[Training ${1} outputdir /data/medioli/models/mlm/${1}_${3}_${5}_${6}_${7}/ with tokenizer config ${2}custom on dataset ${3} with config ${4} for ${5} epochs with regularization: ${7} and data_folder: ${8}]"
nohup python3 run_mlm.py \
--dataset_name $3 \
--tokenizer_name $2 \
--knowledge_base freebase \
--model_type $1 \
--data_folder $8 \
--dataset_config_name $4 \
--do_train \
--do_eval \
--per_device_train_batch_size 8 \
--eval_accumulation_steps 1 \
--learning_rate 1e-5 \
--num_train_epochs $5 \
--save_steps 5000 \
--output_dir $8/models/mlm/${1}_${3}_${5}_${6}_${7} \
--use_fast_tokenizer \
--logging_dir $8/models/mlm/${1}_${3}_${5}_${6}_${7}/runs \
--cache_dir $8/models/mlm/${1}_${3}_${5}_${6}_${7}/cache \
--max_seq_length 512 \
--line_by_line \
--overwrite_output_dir &