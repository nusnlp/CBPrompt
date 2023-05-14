""" Get UMLS mappings using MetaMap
	Usage: find UMLS concepts in a biomedical context
	Example usage: 

	python umls_mapping.py -in_fp ./DDIExtractionCorpus/ddi2013-dataset/test.jsonl \
						  -out_fp ./data/ddi/umls_mappings/test_umls_replaced.jsonl \
						  -out_ordered_fp ./data/ddi/umls_mappings/test.jsonl


"""



import os
import argparse
import json
import re
from pymetamap import MetaMap
import ast
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize

# Start the MetaMap server first
mm = MetaMap.get_instance('./public_mm/bin/metamap18')

# Negation word lists
negation_words = ['no', 'not', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'scarcely', 'barely']



def get_umls_mappings(text):
	umls_dict = {}
	concepts, error_msg = mm.extract_concepts([text])
	for idx, c in enumerate(concepts):
		if type(c).__name__=='ConceptMMI':
			umls_dict[idx] = {'trigger':c.trigger, 'preferred_name':c.preferred_name, 'score':c.score, 'cui':c.cui, 'semtypes':c.semtypes, 'location':c.location, 'pos_info':c.pos_info, 'tree_codes':c.tree_codes}
	return umls_dict



def create_mapping_datasets_for_ddi(in_fp, out_fp):

	lines = open(in_fp).readlines()
	lines = [ast.literal_eval(x.strip()) for x in lines]
	print('Full num instances: ', len(lines))

	with open(out_fp, 'w') as f_out:
		for i, example_json in enumerate(lines):
			# example_json = lines[i]
			print(i)
			idx = example_json['ID']
			text = example_json['TEXT1']
			drugA = example_json['DRUGA']
			drugB = example_json['DRUGB']
			label = example_json['LBL']
			# processed_text = clean(text).lower()
			# get potential umls keywords from the text
			umls_dict = get_umls_mappings(text)
			dict_json = {"TEXT1": text, "LBL": label, "UMLS": umls_dict, "ID": idx, "DRUGA": drugA, "DRUGB": drugB}
			# write to output file
			f_out.write(json.dumps(dict_json) + '\n')


def extract_trigger_word(trigger, text, thres=0.7):
	raw_phrase = trigger.split('-')[-3][1:-1]
	trigger_split = raw_phrase.split()
	if len(trigger_split)==0:	return ''
	text_split = word_tokenize(text)
	if len(trigger_split) == 1: # trigger is a single word
		return raw_phrase if raw_phrase in text_split else ''
	else: # trigger is a phrase, do fuzzy matching
		# Get the substring in text that starts with the first word in trigger and ends with the last word in trigger
		# Calculate string matching score (levenshtein distance fuzzy matching ratio) of the substring and trigger
		# Return the trigger as keyword if matching score is above a threshold 
		s_tr, e_tr = trigger_split[0], trigger_split[-1]
		if s_tr in text_split and e_tr in text_split:
			text_substr = text_split[text_split.index(s_tr):text_split.index(e_tr)+1]
			text_substr = ' '.join(text_substr)
			match_ratio = fuzz.token_sort_ratio(raw_phrase, text_substr)/100
			return raw_phrase if match_ratio > thres else ''
		else:
			return ''

def extract_trigger_word_with_idx(trigger, text, thres=0.7):
	raw_phrase = trigger.split('-')[-3][1:-1]
	trigger_split = raw_phrase.split()
	if len(trigger_split)==0:	return ''
	text_split = word_tokenize(text)
	if len(trigger_split) == 1: # trigger is a single word
		return (text_split.index(raw_phrase), raw_phrase) if raw_phrase in text_split else ''
	else: # trigger is a phrase, do fuzzy matching
		# Get the substring in text that starts with the first word in trigger and ends with the last word in trigger
		# Calculate string matching score (levenshtein distance fuzzy matching ratio) of the substring and trigger
		# Return the trigger as keyword if matching score is above a threshold 
		s_tr, e_tr = trigger_split[0], trigger_split[-1]
		if s_tr in text_split and e_tr in text_split:
			text_substr = text_split[text_split.index(s_tr):text_split.index(e_tr)+1]
			text_substr = ' '.join(text_substr)
			match_ratio = fuzz.token_sort_ratio(raw_phrase, text_substr)/100
			return (text_split.index(s_tr), raw_phrase) if match_ratio > thres else ''
		else:
			return ''

def extract_trigger_word_with_idx_keep_drug(trigger, text, thres=0.7):
	raw_phrase = trigger.split('-')[-3][1:-1]
	trigger_split = raw_phrase.split()
	if len(trigger_split)==0:	return ''
	text_split = re.split('[ ,.!?-]', text) #word_tokenize(text)
	if len(trigger_split) == 1: # trigger is a single word
		return (text_split.index(raw_phrase), raw_phrase) if raw_phrase in text_split else ''
	else: # trigger is a phrase, do fuzzy matching
		# Get the substring in text that starts with the first word in trigger and ends with the last word in trigger
		# Calculate string matching score (levenshtein distance fuzzy matching ratio) of the substring and trigger
		# Return the trigger as keyword if matching score is above a threshold 
		s_tr, e_tr = trigger_split[0], trigger_split[-1]
		if s_tr in text_split and e_tr in text_split:
			text_substr = text_split[text_split.index(s_tr):text_split.index(e_tr)+1]
			text_substr = ' '.join(text_substr)
			match_ratio = fuzz.token_sort_ratio(raw_phrase, text_substr)/100
			return (text_split.index(s_tr), raw_phrase) if match_ratio > thres else ''
		else:
			return ''

def find_repeated_sequences(s):
	match = re.findall(r'((\b.+?\b)(?:\s\2)+)', s)
	return [(m[1], int((len(m[0]) + 1) / (len(m[1]) + 1))) for m in match]


def find_negation_words(lst):
	res = []
	for x in negation_words:
		if x in lst:
			res.append((lst.index(x), x))
	return res

def find_drug_words(lst, special_word):
	res = []
	for idx, x in enumerate(lst):
		if x == special_word:
			res.append((idx, x))
	return res



def create_keyword_datasets_for_ddi(in_data_fp, out_ordered_fp):
	# Retain only UMLS mapped keywords in the original text, NOT keeping @DRUG$ words
	lines = open(in_data_fp).readlines()
	
	for i, line in enumerate(lines):
		line = ast.literal_eval(line.strip())
		text = line['TEXT1']
		pair_id = line['ID']
		drugA = line['DRUGA']
		drugB = line['DRUGB']
		label = line['LBL']
		mapped_dict = line['UMLS']
		# find negation words in the input text
		ne_words = find_negation_words(word_tokenize(text))	
		
		# create ordered keyword only sequence
		mapped_seq_list = [extract_trigger_word_with_idx(v['trigger'], text, 0.7) for k, v in mapped_dict.items()]
		mapped_seq_list=list(filter(len, mapped_seq_list)) # remove empty string from list
		mapped_seq_list = list(set(mapped_seq_list)) # remove repeated words
		mapped_seq_list += ne_words # add negated words back (if exists)
		mapped_seq_list.sort() # sort ascendlingly according to idx of the first word
		mapped_seq_ordered = ' '.join([x[1] for x in mapped_seq_list])
		repeated_words = find_repeated_sequences(mapped_seq_ordered)
		for w in repeated_words:
			mapped_seq_ordered = mapped_seq_ordered.replace(w[0], '', 1)
		mapped_seq_ordered_clean = ' '.join(mapped_seq_ordered.split())
		mapped_seq_ordered_clean = re.sub(r' +', ' ', mapped_seq_ordered_clean)
		# replace drugA and drugB with @DRUG$ (done separately, not here)
		# mapped_seq_ordered_clean = mapped_seq_ordered_clean.replace(drugA[0])
		# mapped_seq_ordered_clean = mapped_seq_ordered_clean.replace(drugB[0])
		# save to dict
		dict_json_ordered = {"TEXT1": mapped_seq_ordered_clean, "LBL": label, "ID": pair_id, "DRUGA": drugA, "DRUGB": drugB}
		out_dir = '/'.join(out_ordered_fp.split('/')[:-1])
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		with open(out_ordered_fp, 'a') as f2:
			f2.write(json.dumps(dict_json_ordered) + '\n')


def create_keyword_datasets_for_ddi_keep_drug(in_data_fp, out_ordered_fp):
	# Retain only UMLS mapped keywords in the original text, KEEPING @DRUG$ words	
	lines = open(in_data_fp).readlines()
	
	for i, line in enumerate(lines):
		line = ast.literal_eval(line.strip())
		text = line['TEXT1']
		pair_id = line['ID']
		drugA = line['DRUGA']
		drugB = line['DRUGB']
		label = line['LBL']
		mapped_dict = line['UMLS']
		# Find negation words in the input text
		ne_words = find_negation_words(re.split('[ ,.!?-]', text))	

		# Find drugA and drugB words as @DRUG$ in the input text
		drug_words = find_drug_words(re.split('[ ,.!?-]', text), '@DRUG$')
		
		# create ordered keyword only sequence
		mapped_seq_list = [extract_trigger_word_with_idx_keep_drug(v['trigger'], text, 0.7) for k, v in mapped_dict.items()]
		mapped_seq_list=list(filter(len, mapped_seq_list)) # remove empty string from list
		mapped_seq_list = list(set(mapped_seq_list)) # remove repeated words

		mapped_seq_list += drug_words # add @DRUG$ back
		mapped_seq_list += ne_words # add negated words back (if exists)

		mapped_seq_list.sort() # sort ascendlingly according to idx of the first word
		mapped_seq_ordered = ' '.join([x[1] for x in mapped_seq_list])
		repeated_words = find_repeated_sequences(mapped_seq_ordered)
		for w in repeated_words:
			mapped_seq_ordered = mapped_seq_ordered.replace(w[0], '', 1)

		mapped_seq_ordered_clean = ' '.join(mapped_seq_ordered.split())
		mapped_seq_ordered_clean = re.sub(r' +', ' ', mapped_seq_ordered_clean)

		dict_json_ordered = {"TEXT1": mapped_seq_ordered_clean, "LBL": label, "ID": pair_id, "DRUGA": drugA, "DRUGB": drugB}
		out_dir = '/'.join(out_ordered_fp.split('/')[:-1])
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		with open(out_ordered_fp, 'a') as f2:
			f2.write(json.dumps(dict_json_ordered) + '\n')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-in_fp', "--input_filepath", required=True)
	parser.add_argument('-out_fp', "--output_filepath", required=True)
	parser.add_argument('-out_ordered_fp', "--output_ordered_filepath", help="Output path to ordered-keyword dataset")
	args = parser.parse_args()

	# Map inputs to UMLS concepts/terms (which we refer them as keywords) For DDI corpus
	if os.path.exists(args.output_filepath):
		print('UMLS keywords datasets already exist!') 
	else:
		print('start generating UMLS mappings...')
		create_mapping_datasets_for_ddi(args.input_filepath, args.output_filepath)
		print('done')


	# Create ordered keyword datasets, keeping @drug$ terms
	# IMPORTANT NOTE: drug names may not appear in UMLS, so we simply ignore the names if nonexistent in UMLS
	if os.path.exists(args.output_ordered_filepath):
		print('Ordered datasets already exist!')
	else:
		print('start creating keyword-only datasets...')
		create_keyword_datasets_for_ddi_keep_drug(args.output_filepath, 
				args.output_ordered_filepath)
		print('done')


