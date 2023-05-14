# Utility functions for measurements of availability bias scores

import numpy as np
import os
import re
import ast
import torch
import random
import json
import csv

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def fix_seed(): # For 3 repeats of sample drawing, we use seeds of [101,123,15], kindly change the following function accordingly
	""" Enable reproducibility """
	torch.manual_seed(101)
	torch.cuda.manual_seed_all(101)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(101)
	random.seed(101)
	os.environ['PYTHONHASHSEED'] = str(101)



def replace_content_with_dummy(fs_test_fp, kw_test_fp, out_fp, dummy):
	fs_test = open(fs_test_fp).readlines()
	fs_test = [ast.literal_eval(line.strip()) for line in fs_test]
	kw_test = open(kw_test_fp).readlines()
	kw_test = [ast.literal_eval(line.strip()) for line in kw_test]
	for fs_sent in fs_test:
		kw_sent = next(item for item in kw_test if item['ID'] == fs_sent['ID'])
		fs_text_split = re.split('[ ,.!?-]', fs_sent['TEXT1'])
		kw_text_split = re.split('[ ,.!?-]', kw_sent['TEXT1'])
		for i, x in enumerate(fs_text_split):
			if x in kw_text_split and x != '@DRUG$':
				fs_text_split[i] = dummy # e.g. 'N/A'
		fs_text_split=list(filter(len, fs_text_split)) # remove empty string from list
		dummy_sent = ' '.join(fs_text_split) # 'N/A Inhibitors: @DRUG$ an inhibitor of the drug metabolizing enzyme N/A significantly N/A N/A N/A of @DRUG$ when coadministered to N/A who were N/A N/A (see N/A N/A N/A in N/A and Drug Drug Interactions)'
		fs_sent['TEXT1'] = dummy_sent
	with open(out_fp, 'w') as f:
		for line in fs_test:
			f.write(json.dumps(line)+'\n')


def sample_uniform_test(test_data, out_fp, num_samples):
	all_test_dict = {}
	all_uniform_test = []
	all_test = open(test_data).readlines()
	all_test = [ast.literal_eval(line.strip()) for line in all_test]
	for x in all_test:
		if x['LBL'] in all_test_dict:
			all_test_dict[x['LBL']].append(x)
		else:
			all_test_dict[x['LBL']] = [x]

	print('len of each class in test data ', {k:len(v) for k, v in all_test_dict.items()})
	# ('len of each class in test data ', {'DDI-mechanism': 302, '0': 4737, 'DDI-advise': 221, 'DDI-int': 96, 'DDI-effect': 360})
	assert len(all_test) == sum([len(x) for _,x in all_test_dict.items()])

	# Get random sample of equal amount of test sents from each class
	for k, sents in all_test_dict.items():
		print('Ramdom sample {} sentences for class {}'.format(num_samples, k))
		if num_samples <= len(sents):
			idxs = np.random.choice(len(sents), size=num_samples, replace=False)
		else: # set repalce to True to enable upsampling for a class that has fewer instances than the drawing size
			idxs = np.random.choice(len(sents), size=num_samples, replace=True)
		uniform_test = [sents[i] for i in idxs]
		all_uniform_test.extend(uniform_test)
	random.shuffle(all_uniform_test)
	with open(out_fp, 'w') as f:
		for line in all_uniform_test:
			f.write(json.dumps(line)+'\n')


# Example usage

fix_seed()
rseed2run = {101: 0, 123: 1, 15: 2}
rseed = 15
dummy = 'N/A'
replace_content_with_dummy('./data/ddi/original/full_train/test.jsonl', './data/ddi/umls_mappings/test.jsonl', './data/ddi/availability/avail_bias_test_all.jsonl', dummy)

for num_samples in [20]: # 20 per class, run it one-by-one to avoid random seed error [20,40,60,80,100,120,140,160,180,200]:
	sample_uniform_test('./data/ddi/availability/avail_bias_test_all.jsonl', './data/ddi/availability/avail_test_%_%d.jsonl'%(num_samples*5, rseed2run(rseed)), num_samples)


