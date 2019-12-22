#!/usr/bin/env bash

data_file="data/query/model_data_train.query"
vocab_file="data/query/youhaohuo_and_query_vec.txt"
label_file="data/query/label_fine_grain_idx.query"
encoding="utf8"

max_seq_len=12
num_classes=59
vocab_size=855
embedding_dim=100

num_epochs=10
batch_size=64
train_frac=0.8
valid_frac=0.1
dropout_keep_prob=1.0
eval_step=1

max_grad_norm=5.0
learning_rate=0.005
l2_reg_lambda=0.0

lstm_dim=256
layer_size=2


python NER_train.py \
    --data_file=${data_file} \
    --vocab_file=${vocab_file} \
    --label_file=${label_file} \
    --encoding=${encoding} \
    --max_seq_len=${max_seq_len} \
    --num_classes=${num_classes} \
    --vocab_size=${vocab_size} \
    --embedding_dim=${embedding_dim} \
    --num_epochs=${num_epochs} \
    --batch_size=${batch_size} \
    --train_frac=${train_frac} \
    --valid_frac=${valid_frac} \
    --dropout_keep_prob=${dropout_keep_prob} \
    --eval_step=${eval_step} \
    --max_grad_norm=${max_grad_norm} \
    --learning_rate=${learning_rate} \
    --l2_reg_lambda=${l2_reg_lambda} \
    --lstm_dim=${lstm_dim} \
    --layer_size=${layer_size}