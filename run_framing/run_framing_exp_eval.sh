#!/bin/bash
# For framing effect experiments
export EXP_DIR=`pwd`
export ADAPET_ROOT="$(dirname "$EXP_DIR")/ADAPET-CB"
export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
export PYTHON_EXEC=python

cd $ADAPET_ROOT
all_args=("$@")
framing_ori="$1"
framing_gpt="$2"

# Evaluate on original test data at ./data/ddi/framing/ori/test.jsonl
# Get test preds 
sh $ADAPET_ROOT/bin/test.sh $framing_ori
# Eval test
python $EXP_DIR/get_test_scores.py -y $ADAPET_ROOT/data/ddi/framing/ori -pred $framing_ori


# Evaluate on toned-down paraphrase test data/ at ./data/ddi/framing/gpt_completion/test.jsonl
# Get test preds 
sh $ADAPET_ROOT/bin/test.sh $framing_gpt
# Eval test
python $EXP_DIR/get_test_scores.py -y $ADAPET_ROOT/data/ddi/framing/gpt_completion -pred $framing_gpt




