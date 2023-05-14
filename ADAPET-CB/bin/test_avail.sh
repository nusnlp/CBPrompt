#!/usr/bin/env bash

set -exu

exp_dir=$1


arr1=(100) # Change it to (100 200 300 400 500 600 700 800 900 1000) for Appendix B discussions
arr2=(0 1 2)
for i in "${arr1[@]}"; do
	echo "$i"
	ns=1
	((ns = i/5))
	for j in "${arr2[@]}"; do
		echo "$j"
		# Eval dummy test
		CUDA_VISIBLE_DEVICES=1 python -m src.test_avail -e $exp_dir -d $i -r $j
		# Compute availability bias scores
		python avail_bias.py -e $exp_dir --file_name test_dummy_"${i}"_"${j}".json --num_samples $ns
	done
done












