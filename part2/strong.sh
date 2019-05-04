#!/bin/bash
TEST_CSV_PATH=$1
PREDICTION_DIR_PATH=$2

# preprocess test.csv
python3.7 -m BERT.create_test_dataset $TEST_CSV_PATH dataset/classification

# predict
CUDA_VISIBLE_DEVICES="0" python3.7 -m BERT.predict ./BERT/models/strong0/ckpts/epoch-2.ckpt ./BERT/models/strong1/ckpts/epoch-2.ckpt $PREDICTION_DIR_PATH --batch_size 32