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

    CUDA_VISIBLE_DEVICES=$GPUS python3 main.py \
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
    --batch_size 32 \
    --num_workers 0 \
    --num_epochs 1 \
    --gradient_clipping 1.0 \
    --label_smoothing 0.1 \
    --train_filename ../data/python/train_processed.json \
    --validation_filename ../data/python/validation_processed.json \
    --mode translation \
    --checkpoint True \
    --debug_max_lines 32

}

function test_model() {

    echo "=================================================="
    echo "=================== Testing... ==================="
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=$GPUS python3 test.py \
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
    --batch_size 32 \
    --num_workers 0 \
    --test_filename ../data/python/test_processed.json \
    --debug_max_lines 32

}

train_and_validate
test_model