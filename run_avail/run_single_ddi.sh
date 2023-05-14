#!/bin/bash
# For availability bias experiments
export EXP_DIR=`pwd`
export ADAPET_ROOT="$(dirname "$EXP_DIR")/ADAPET-CB"
export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
export PYTHON_EXEC=python

cd $ADAPET_ROOT

# Train
CUDA_VISIBLE_DEVICES=3 python $ADAPET_ROOT/cli.py \
              -d $ADAPET_ROOT/data/ddi/framing/ori \
              -p '([LBL]) [TEXT1]' \
              -v '{"0": "false", "DDI-effect": "effect", "DDI-mechanism":"mechanism", "DDI-advise": "advice", "DDI-int": "interaction"}' \
              -w "bert-base-uncased" \
              -bs 1 \
              --grad_accumulation_factor 32 \
              --num_batches 8000 \
              --eval_every 100 \
              --max_text_length 300 --lr 5e-5 \
              --weight_decay 1e-2 \
              --warmup_ratio 0.06 \
              --pattern_idx 1 \
              --max_num_lbl_tok 1

# Saved models are located at $ADAPET_ROOT/exp_out/generic/bert-base-uncased/20XX-XX-XX-XX-XX-XX

# Evaluate on the original test set using the scripts below
# export EXP_DIR=`pwd`
# export ADAPET_ROOT="$(dirname "$EXP_DIR")/ADAPET-CB"
# export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
# export PYTHON_EXEC=python

# cd $ADAPET_ROOT
# all_args=("$@")
# pretrained="$1" # enter the pretrained filepath in command line, or specify it here

# # Get test preds 
# sh $ADAPET_ROOT/bin/test.sh $pretrained
# # Eval test
# python $EXP_DIR/get_test_scores.py -y $ADAPET_ROOT/data/ddi/original/ -pred $pretrained

