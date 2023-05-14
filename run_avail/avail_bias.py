# Run measurements for availability bias scores

# Method: feed equal amount of dummy prompts from each class to pretrained prompted model, 
# to measure the frequencey of each class prediction.
# The normalized deviation from 1/5=20% (including the negative class) is the bias for each predicted class.
# dummy prompt: replace medical UMLS keywords with N/A
# Reason to use euqal amount of dummy prompts from each class:
# to mitigate the effect that a class-specific content-free input 
# may correlate with surface class patterns


import argparse
import numpy as np
import os
import re
import ast
import torch
import random
import json



def fix_seed():
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
		idxs = np.random.choice(len(sents), size=num_samples, replace=False)
		uniform_test = [sents[i] for i in idxs]
		all_uniform_test.extend(uniform_test)
	random.shuffle(all_uniform_test)
	with open(out_fp, 'w') as f:
		for line in all_uniform_test:
			f.write(json.dumps(line)+'\n')

def compute_avail_bias(test_pred_fp, num_samples):
	avail_bias_scores = {}
	counts = {'0':0, 'DDI-mechanism':0, 'DDI-advise':0, 'DDI-effect':0, 'DDI-int':0}
	test_preds = open(test_pred_fp).readlines()
	test_preds = [ast.literal_eval(line.strip()) for line in test_preds]
	for inst in test_preds:
		if inst['label'] in counts:
			counts[inst['label']] += 1
		else:
			counts[inst['label']] = 1
	# compute availability bias scores
	num_test = len(test_preds)
	for k, v in counts.items():
		avail_bias_scores[k] = abs(v - num_samples)/num_test
	return avail_bias_scores



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', "--exp_dir", required=True)
	parser.add_argument("--file_name", required=True)
	parser.add_argument("--num_samples", required=True)
	args = parser.parse_args()

	fix_seed()

	test_pred_fp = os.path.join(args.exp_dir, args.file_name)
	num_samples = int(args.num_samples)
	# num_samples = 20
	res = compute_avail_bias(test_pred_fp, num_samples)
	print('Normalized availability bias for each class ', res)

	# Save to file
	out_fn = 'avail_bias_'+args.file_name.split('.')[0]+'.txt'
	with open(os.path.join(args.exp_dir, out_fn), 'w') as f:
		f.write(str(res)+'\n')











