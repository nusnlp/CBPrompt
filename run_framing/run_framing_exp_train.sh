#!/bin/bash
# For framing effect experiments
export EXP_DIR=`pwd`
export ADAPET_ROOT="$(dirname "$EXP_DIR")/ADAPET-CB"
export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
export PYTHON_EXEC=python

cd $ADAPET_ROOT

# Train on original data at data/ddi/framing/ori/train.jsonl val.jsonl
CUDA_VISIBLE_DEVICES=3 python $ADAPET_ROOT/cli.py \
              -d $ADAPET_ROOT/data/ddi/framing/ori \
              -p '([LBL]) [TEXT1]' \
              -v '{"0": "false", "DDI-effect": "effect", "DDI-mechanism":"mechanism", "DDI-advise": "advice", "DDI-int": "interaction"}' \
              -w "bert-base-uncased" \
              -bs 1 \
              --grad_accumulation_factor 32 \
              --num_batches 6000 \
              --eval_every 100 \
              --max_text_length 300 --lr 5e-5 \
              --weight_decay 1e-2 \
              --warmup_ratio 0.06 \
              --pattern_idx 1 \
              --max_num_lbl_tok 1

# Saved models are located at $ADAPET_ROOT/exp_out/generic/bert-base-uncased/20XX-XX-XX-XX-XX-XX


# Train on toned-down paraphrase data/ddi/framing/framed/train.jsonl val.jsonl
CUDA_VISIBLE_DEVICES=3 python $ADAPET_ROOT/cli.py \
              -d $ADAPET_ROOT/data/ddi/framing/gpt_completion \
              -p '([LBL]) [TEXT1]' \
              -v '{"0": "false", "DDI-effect": "effect", "DDI-mechanism":"mechanism", "DDI-advise": "advice", "DDI-int": "interaction"}' \
              -w "bert-base-uncased" \
              -bs 1 \
              --grad_accumulation_factor 32 \
              --num_batches 6000 \
              --eval_every 100 \
              --max_text_length 300 --lr 5e-5 \
              --weight_decay 1e-2 \
              --warmup_ratio 0.06 \
              --pattern_idx 1 \
              --max_num_lbl_tok 1

# Saved models are located at $ADAPET_ROOT/exp_out/generic/bert-base-uncased/20XX-XX-XX-XX-XX-XX






