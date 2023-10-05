#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Expected 1 argument telling the GPUs to use. The argument is in
          the form of 0,1,2,... where 0, 1 and 2 are GPU Ids. nvidia-smi command
          provides more information about the GPUs in your device"
fi

GPUS=$1

function train_and_validate() {

    echo "=================================================="
    echo "= Training (with validation after each epoch)... ="
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=$GPUS python3.10 main.py \
    --src_vocab_size 50000 \
    --tgt_vocab_size 30000 \
    --max_src_length 150 \
    --max_tgt_length 50 \
    --freq_threshold 0 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --dropout 0.1 \
    --learning_rate 1.0 \
    --batch_size 128 \
    --num_workers 0 \
    --num_epochs 200 \
    --label_smoothing 0.1 \
    --init_type kaiming \
    --train_filename ../data/python/train_processed.json \
    --validation_filename ../data/python/validation_processed.json \
    --train_token_matrix ../data/python/train_processed_python_in_token.txt \
    --train_statement_matrix ../data/python/train_processed_python_in_statement.txt \
    --train_data_flow_matrix ../data/python/train_processed_python_data_flow.pkl \
    --train_control_flow_matrix ../data/python/train_processed_python_control_flow.pkl \
    --validation_token_matrix ../data/python/validation_processed_python_in_token.txt \
    --validation_statement_matrix ../data/python/validation_processed_python_in_statement.txt \
    --validation_data_flow_matrix ../data/python/validation_processed_python_data_flow.pkl \
    --validation_control_flow_matrix ../data/python/validation_processed_python_control_flow.pkl \
    --hyperparameter_hsva 6 \
    --hyperparameter_data_flow 5 \
    --hyperparameter_control_flow 5 \
    --mode greedy \
    --checkpoint True \
    --hyperparameter_tuning False \
    --optimizer adam \
    --dir_iteration $1

}

function test_model() {

    echo "=================================================="
    echo "=================== Testing... ==================="
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=$GPUS python3.10 test.py \
    --src_vocab_size 50000 \
    --tgt_vocab_size 30000 \
    --max_src_length 150 \
    --max_tgt_length 50 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --dropout 0.1 \
    --learning_rate 1.0 \
    --label_smoothing 0.1 \
    --init_type kaiming \
    --batch_size 128 \
    --num_workers 0 \
    --test_filename ../data/python/test_processed.json \
    --test_token_matrix ../data/python/test_processed_python_in_token.txt \
    --test_statement_matrix ../data/python/test_processed_python_in_statement.txt \
    --test_data_flow_matrix ../data/python/test_processed_python_data_flow.pkl \
    --test_control_flow_matrix ../data/python/test_processed_python_control_flow.pkl \
    --hyperparameter_hsva 6 \
    --hyperparameter_data_flow 5 \
    --hyperparameter_control_flow 5 \
    --mode greedy \
    --optimizer adam \
    --dir_iteration $1

}

train_and_validate test_2
test_model test_2