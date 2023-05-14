"""
	**** Framing effect measurements ****
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text.bert import BERTScore
import ast
import numpy as np
import os
import copy
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


bertscore = BERTScore()
toneddown_word2idx = {189: 'may', 64: 'can', 2288: 'reportedly', 1106: 'if', 1779: 'when'} # use roberta bpe tokenized embedding lookup

class FramingEffectDataset(Dataset):
	""" Return a tuple of source and paraphrase encodings. """
	def __init__(self, src_encodings, par_encodings):
		self.src_encodings = src_encodings
		self.par_encodings = par_encodings
	def __getitem__(self, idx):
		encoded_src = {k : torch.tensor(v[idx]) for k,v in self.src_encodings.items()} 
		encoded_par = {k : torch.tensor(v[idx]) for k,v in self.par_encodings.items()} 
		return encoded_src, encoded_par
	def __len__(self):
		return len(self.src_encodings['input_ids'])


def read_sents(in_fp):
	data = open(in_fp).readlines()
	data = [ast.literal_eval(line.strip()) for line in data]
	sents = [x['TEXT1'] for x in data]
	labels = [x['LBL'] for x in data]
	return sents, labels


def get_max_sent_len(source, paraphrase, tokenizer):
	src_enc = tokenizer(source, padding=True)
	par_enc = tokenizer(paraphrase, padding=True)
	len_src = len(src_enc['input_ids'][0])
	len_par = len(par_enc['input_ids'][0])
	return len_src if len_src >= len_par else len_par


def compute_bertscore(source, paraphrase):
	# roberta-large is used to generate contextual embeddings
	# Args: source, an input list of source sents
	#		paraphrase, an input list of paraphrase sents
	score = bertscore(paraphrase, source)
	rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}
	return rounded_score['f1'] 	# return f1 scores

def compute_bertscore_new(source, paraphrase):
	# roberta-large is used to generate contextual embeddings
	# Args: source, an input list of source sents
	#		paraphrase, an input list of paraphrase sents
	score = bertscore(paraphrase, source)
	# if type(score['f1']) != list:
	# 	# single element input
	# 	rounded_score = {k: round(v, 3) for k, v in score.items()}
	# else:
	# 	rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}
	# return rounded_score
	return score

def load_roberta(device='cpu'):
	from transformers import RobertaTokenizer, RobertaModel
	tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
	model = RobertaModel.from_pretrained('roberta-large')
	return model.to(device), tokenizer

def comptue_rescaling_score(source_batches, paraphrase_batches, model, device):
	model.eval()
	with torch.no_grad():
		src_input_ids = source_batches['input_ids'].to(device)
		src_attention_mask = source_batches['attention_mask'].to(device)
		src_out = model(src_input_ids, attention_mask=src_attention_mask, output_hidden_states=False)
		src_emb = src_out[0]
		par_input_ids = paraphrase_batches['input_ids'].to(device)
		par_attention_mask = paraphrase_batches['attention_mask'].to(device)
		par_out = model(par_input_ids, attention_mask=par_attention_mask, output_hidden_states=False)
		par_emb = par_out[0]	
	# Pre-normalize the embeddings, following the BERTScore code
	# Code reference: https://github.com/Tiiiger/bert_score
	src_emb.div_(torch.norm(src_emb, dim=-1).unsqueeze(-1))
	par_emb.div_(torch.norm(par_emb, dim=-1).unsqueeze(-1))	
	# Compute bmm of par_emb to src_emb, transpose par_emb for computation, and pick max along dim 1 for P and dim 2 for R
	bmm_scores = torch.bmm(src_emb, par_emb.transpose(1,2))
	masks = torch.bmm(src_attention_mask.unsqueeze(2).float(), par_attention_mask.unsqueeze(1).float())
	# masks = masks.expand(src_emb.shape[0], -1, -1).contiguous().view_as(sim)
	masks = masks.float().to(device)
	bmm_scores_masked = bmm_scores * masks
	word_precision = bmm_scores_masked.max(dim=1)[0]
	word_recall = bmm_scores_masked.max(dim=2)[0]	
	# Compute resacling scores (add-on scores) for revised P, R
	addon_word_p = copy.deepcopy(word_precision)
	for i in range(len(addon_word_p)):
		for j in range(len(addon_word_p[i])):
			if paraphrase_batches['input_ids'][i, j].item() in toneddown_word2idx:
				print(paraphrase_batches['input_ids'][i, j])
				addon_word_p[i, j] = 1.0 - word_precision[i, j]
			else:
				addon_word_p[i, j] = 0.0	
	addon_word_r = copy.deepcopy(word_recall)
	for i in range(len(addon_word_r)):
		for j in range(len(addon_word_r[i])):
			if source_batches['input_ids'][i, j].item() in toneddown_word2idx:
				print(source_batches['input_ids'][i, j])
				addon_word_r[i, j] = 0.0 - word_recall[i, j]
			else:
				addon_word_r[i, j] = 0.0	
	addon_P = addon_word_p.sum(dim=1)/torch.count_nonzero(par_attention_mask, dim=1)
	addon_R = addon_word_r.sum(dim=1)/torch.count_nonzero(src_attention_mask, dim=1)
	return addon_P, addon_R


def compute_paraphrase_score(source_sents, paraphrase_sents, base_P, base_R):
	# roberta-large is used to generate contextual embeddings
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # torch.device('cuda:1')
	model, tokenizer = load_roberta(device)
	# Encode input word pairs, padded to max length of all src, par sents
	max_len = get_max_sent_len(source_sents, paraphrase_sents, tokenizer)
	src_encodings = tokenizer(source_sents, padding='max_length', max_length=max_len)
	par_encodings = tokenizer(paraphrase_sents, padding='max_length', max_length=max_len)
	# Create a dataset and dataloader
	framing_dataset = FramingEffectDataset(src_encodings, par_encodings)
	data_loader = DataLoader(framing_dataset, batch_size=16, shuffle=False)
	# Compute score
	rescale_P, rescale_R = [], []
	for batch in data_loader:
		# print('Batch: ', batch)
		source_batches, paraphrase_batches = batch[0], batch[1]
		res_p, res_r = comptue_rescaling_score(source_batches, paraphrase_batches, model, device)
		rescale_P.extend(res_p)
		rescale_R.extend(res_r)
	rescale_P = [x.detach().cpu().item() for x in rescale_P]
	rescale_R = [x.detach().cpu().item() for x in rescale_R]
	assert len(rescale_P) == len(base_P)
	assert len(rescale_R) == len(base_R)
	revised_P = [a+b for _, (a, b) in enumerate(zip(base_P, rescale_P))]
	revised_R = [a+b for _, (a, b) in enumerate(zip(base_R, rescale_R))]
	revised_F = [2*revised_P[i]*revised_R[i]/(revised_P[i]+revised_R[i]) for i in range(len(revised_P))]
	revised_score = {'precision': revised_P, 'recall': revised_R, 'f1': revised_F}
	rounded_score = {k: [round(v, 3) for v in vv] for k, vv in revised_score.items()}
	return rounded_score['f1'] 	# return f1 scores


def measure_base(sents_ori, sents_par, res_fp):
	"""compute the bertscore of valid source and paraphrase sentences"""
	all_bertscore = compute_bertscore_new(sents_ori, sents_par)
	all_P, all_R, all_F = all_bertscore['precision'], all_bertscore['recall'], all_bertscore['f1']
	avg_bertscore = np.mean(np.array(all_F))
	# print('all_bertscore ', all_bertscore)
	print('avg bertscore f1 ', avg_bertscore)
	with open(res_fp, 'w') as f:
		f.write('Ori_sent\tPara_sent\tBertscore_P\tBertscore_R\tBertscore_F\n')
		for i in range(len(sents_ori)):
			f.write('{}\t{}\t{}\t{}\t{}\n'.format(sents_ori[i], sents_par[i], all_P[i], all_R[i], all_F[i]))
		f.write('Average bertscore F1:{}\n'.format(avg_bertscore))


def compute_prob_given_paraphrase_score(test_fp):

	all_dict = {}
	results = open(test_fp).readlines()
	results = results[1:]

	for idx, line in enumerate(results):
		line = line.strip().split('\t')
		all_dict[idx] = {'ori_sent':line[0].strip('\"'), 
		'par_sent':line[1].strip('\"'), 'label':line[2], 
		'ori_pred': line[3], 'par_pred':line[4],
		'bert_score': float(line[5]), 'paraphrase_score':float(line[-1])}

	# Set paraphrase score bins
	par_scores = [all_dict[idx]['paraphrase_score'] for idx in all_dict]

	# print(max(par_scores), np.mean(par_scores), min(par_scores))
	# 1.0 0.9763894230769231 0.939

	# plt.hist(par_scores, bins=5)
	# plt.gca().set(title='Paraphrase Score Frequency Histogram', ylabel='Frequency');
	# plt.show()

	# According to the histogram, we set 7 par_scores bins with bin size 0.01
	# Calculate conditional probabilites and delta correct rate
	# E.g. delta correct rate: for par_scores in [0.99, 1.0], ori wrong test inst. count: a, ori wrong par correct count: b, return b/a


	res = {'0.93-0.94':{'ori_pred':[], 'par_pred':[], 'label':[]}, 
			'0.94-0.95':{'ori_pred':[], 'par_pred':[], 'label':[]}, 
			'0.95-0.96':{'ori_pred':[], 'par_pred':[], 'label':[]}, 
			'0.96-0.97':{'ori_pred':[], 'par_pred':[], 'label':[]}, 
			'0.97-0.98':{'ori_pred':[], 'par_pred':[], 'label':[]}, 
			'0.98-0.99':{'ori_pred':[], 'par_pred':[], 'label':[]}, 
			'0.99-1.00':{'ori_pred':[], 'par_pred':[], 'label':[]}}

	for i, v in all_dict.items():
		if v['paraphrase_score'] >= 0.93 and v['paraphrase_score'] < 0.94:
			res['0.93-0.94']['ori_pred'].append(v['ori_pred'])
			res['0.93-0.94']['par_pred'].append(v['par_pred'])
			res['0.93-0.94']['label'].append(v['label'])
		elif v['paraphrase_score'] >= 0.94 and v['paraphrase_score'] < 0.95:
			res['0.94-0.95']['ori_pred'].append(v['ori_pred'])
			res['0.94-0.95']['par_pred'].append(v['par_pred'])
			res['0.94-0.95']['label'].append(v['label'])
		elif v['paraphrase_score'] >= 0.95 and v['paraphrase_score'] < 0.96:
			res['0.95-0.96']['ori_pred'].append(v['ori_pred'])
			res['0.95-0.96']['par_pred'].append(v['par_pred'])
			res['0.95-0.96']['label'].append(v['label'])
		elif v['paraphrase_score'] >= 0.96 and v['paraphrase_score'] < 0.97:
			res['0.96-0.97']['ori_pred'].append(v['ori_pred'])
			res['0.96-0.97']['par_pred'].append(v['par_pred'])
			res['0.96-0.97']['label'].append(v['label'])
		elif v['paraphrase_score'] >= 0.97 and v['paraphrase_score'] < 0.98:
			res['0.97-0.98']['ori_pred'].append(v['ori_pred'])
			res['0.97-0.98']['par_pred'].append(v['par_pred'])
			res['0.97-0.98']['label'].append(v['label'])
		elif v['paraphrase_score'] >= 0.98 and v['paraphrase_score'] < 0.99:
			res['0.98-0.99']['ori_pred'].append(v['ori_pred'])
			res['0.98-0.99']['par_pred'].append(v['par_pred'])
			res['0.98-0.99']['label'].append(v['label'])
		else:
			res['0.99-1.00']['ori_pred'].append(v['ori_pred'])
			res['0.99-1.00']['par_pred'].append(v['par_pred'])
			res['0.99-1.00']['label'].append(v['label'])

	print ('# instances ', {k:len(v['label']) for k, v in res.items()})
	# instances  {'0.93-0.94': 2, '0.94-0.95': 14, '0.95-0.96': 31, '0.96-0.97': 40, '0.97-0.98': 16, '0.98-0.99': 19, '0.99-1.00': 86}

	# Get delta correct rate for all test pairs
	pred_for_conf_mat = []
	label_for_conf_mat = []
	with open('./framing_results/ffep_based_cond_probs_for_vaild.txt', 'w') as f:
		f.write('FFEP_range\tCount_wrong_ori\tCount_wrong_ori_correct_paraphrase\n')
		for k, v in res.items():
			wrong_ori_ct = 0
			wrong_ori_correct_par_ct = 0
			for i in range(len(v['label'])):
				# if v['label'][i] != '0':
				if v['ori_pred'][i] != v['label'][i]:
					wrong_ori_ct += 1
					if v['par_pred'][i] == v['label'][i]:
						wrong_ori_correct_par_ct += 1
						if k == '0.99-1.00':
							pred_for_conf_mat.append(v['par_pred'][i])
							label_for_conf_mat.append(v['label'][i])
			print(k, wrong_ori_ct, wrong_ori_correct_par_ct)
			f.write('{}\t{}\t{}\n'.format(k, str(wrong_ori_ct), str(wrong_ori_correct_par_ct)))


def load_valid(in_fp):
	data = open(in_fp).readlines()
	data = [x.strip().split(',') for x in data][1:]
	tgt = [x[0] for x in data]
	pred_ori = [x[1] for x in data]
	pred_par = [x[2] for x in data]
	return tgt, pred_ori, pred_par



def compute_metrics(test_y, test_preds, res_fp):
	test_acc = accuracy_score(test_y, test_preds)
	print('Overall test acc: ', test_acc)
	# precision_mi, recall_mi, f1score_mi, _ = precision_recall_fscore_support(test_y, test_preds, average='micro')
	precision_mi, recall_mi, f1score_mi, _ = precision_recall_fscore_support(test_y, test_preds, average='micro', labels=["DDI-effect", "DDI-mechanism", "DDI-advise", "DDI-int"])
	print('4 DDI types test precision_mi, recall_mi, f1score_mi', precision_mi, recall_mi, f1score_mi)
	score_report = classification_report(test_y, test_preds)
	print(score_report)
	# Get confusion matrix
	# conf_matrix = confusion_matrix(test_y, test_preds, labels=["0", "DDI-effect", "DDI-mechanism", "DDI-advise", "DDI-int"])
	# print(conf_matrix)
	conf_matrix = pd.DataFrame(
		confusion_matrix(test_y, test_preds, labels=["0", "DDI-effect", "DDI-mechanism", "DDI-advise", "DDI-int"]), 
		index=['true:0', 'true:eff', 'true:mech', 'true:adv', 'true:int'], 
		columns=['pred:0', 'pred:eff', 'pred:mech', 'pred:adv', 'pred:int']
	)
	print(conf_matrix)
	# Save test scores
	with open(res_fp, 'w') as fout:
		fout.write('Test acc:\t{}\n'.format(test_acc))
		fout.write('4 DDI types Test precision_mi:\t{}\trecall_mi\t{}\tf1score_mi\t{}\n'.format(precision_mi, recall_mi, f1score_mi))
		fout.write(str(score_report)+'\n')
		fout.write(str(conf_matrix)+'\n')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-ori_train', required=True)
	parser.add_argument('-ori_val', required=True)
	parser.add_argument('-ori_test', required=True)
	parser.add_argument('-gpt_train', required=True)
	parser.add_argument('-gpt_val', required=True)
	parser.add_argument('-gpt_test', required=True)
	args = parser.parse_args()

	input_train, _ = read_sents(args.ori_train)
	input_val, _ = read_sents(args.ori_val)
	input_test, labels_test = read_sents(args.ori_test)
	
	gpt_par_train, _ = read_sents(args.gpt_train)
	gpt_par_val, _ = read_sents(args.gpt_val)
	gpt_par_test, _ = read_sents(args.gpt_test)

	ori_all = input_train + input_val + input_test
	gpt_all = gpt_par_train + gpt_par_val + gpt_par_test 
	gpt_par_score = compute_bertscore(ori_all, gpt_all)
	avg_gpt_par_score = np.mean(np.array(gpt_par_score))
	print('Mean bertscore for gpt paraphases (using original sentences as references) ', avg_gpt_par_score)
	# 0.9675152941176471


	# We manually check valid paraphrases in the test set, and store valid instances in valid_par_and_results.csv, 
	# valid_par_and_results_full.txt, valid_par_and_results_prf.txt, and we compute F1 
	# for valid paraphrases and the corresponding ori. inputs
	tgt, pred_ori, pred_par = load_valid('./framing_results/check_for_valid/valid_par_and_results.csv')
	compute_metrics(tgt, pred_ori, './framing_results/test_ori_sents_of_the_valid_pairs_scores.txt')
	# Overall test acc:  0.32211538461538464
	# 4 DDI types test f1: 0.08974358974358974
	compute_metrics(tgt, pred_par, './framing_results/test_valid_gpt_par_scores.txt')
	# Overall test acc:  0.8798076923076923
	# 4 DDI types test f1: 0.5573770491803278


	# Compute paraphrase score
	text_data = open('./framing_results/check_for_valid/valid_par_and_results_full.txt').readlines()
	text_data = text_data[1:]
	source_sents, paraphrase_sents = [], []
	for line in text_data:
		line = line.strip().split('\t')
		source_sents.append(line[0].strip('\"'))
		paraphrase_sents.append(line[1].strip('\"'))
		# base_scores.append(float(line[-1]))

	# Measure base score (uncomment below if not done)
	# measure_base(source_sents, paraphrase_sents, './framing_results/check_for_valid/valid_par_and_results_prf.txt')
	prf_data = open('./framing_results/check_for_valid/valid_par_and_results_prf.txt').readlines()
	prf_data = prf_data[1:-1]
	base_P, base_R = [], []
	for line in prf_data:
		line = line.strip().split('\t')
		base_P.append(float(line[-3]))
		base_R.append(float(line[-2]))

	# Measure F_FEP scores, copy valid_par_and_results_full.txt as valid_par_and_results_full_revised.txt,
	# and append the F_FEP scores to the last column in valid_par_and_results_full_revised.txt
	final_scores = compute_paraphrase_score(source_sents, paraphrase_sents, base_P, base_R)
	print('F_fep scores', final_scores)
	with open('./framing_results/ffep_score.txt', 'w') as f:
		for x in final_scores:
			f.write(str(x)+'\n')

	# Get conditional probabilities based on F_fep bins
	test_fp = './framing_results/check_for_valid/valid_par_and_results_full_revised.txt'
	compute_prob_given_paraphrase_score(test_fp)



