#!/usr/bin/env bash
export ADAPET_ROOT=`pwd`
export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
export PYTHON_EXEC=python


# Iterate through pretraiend few-shot models 10,100,1k,10k
arr1=([PRETRAINED_MODEL_PATH_FS_10])
for i in "${arr1[@]}"; do
	echo "$i"
	./bin/test_avail.sh $i
done

# pretrained model trained on full training set
./bin/test_avail.sh [PRETRAINED_MODEL_PATH_FULL]
