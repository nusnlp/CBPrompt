"""Framing data generation"""
import os
import json
import argparse
import ast
import copy
import random
from data_utils import gpt_completion


def fix_seed():
	""" Enable reproducibility """
	# torch.manual_seed(101)
	# torch.cuda.manual_seed_all(101)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	# np.random.seed(101)
	random.seed(101)
	os.environ['PYTHONHASHSEED'] = str(101)


def read_jsonl(in_fp):
	data = open(in_fp).readlines()
	data = [ast.literal_eval(line.strip()) for line in data]
	return data


# For GPT3 text generaton
def load_full_sent_template(input_sent, priming_examples):
	concat_examples = ("\n\n===\n\n".join([""] + priming_examples + [""]) if priming_examples else "\n")
	prompt = "Paraphrase the following drug interaction description.{}{}\nRephrase the above description to sound soft. Write the description in a warm tone.\nDescription:\n".format(concat_examples, input_sent)
	return prompt


def load_keyword_template(input_sent, priming_examples):
	concat_examples = ("\n\n===\n\n".join([""] + priming_examples + [""]) if priming_examples else "\n")
	prompt = "Elaborate on the following points to write a drug interaction description.{}{}\nRephrase the above points to sound soft. Not list them as points. Write the description in a warm tone.\nDescription:\n".format(concat_examples, input_sent)
	return prompt


def generate_prompt(input_sent, priming_examples, prompt_mode):
	if prompt_mode == "full":
		prompt_str = load_full_sent_template(input_sent, priming_examples)
	elif prompt_mode == "keyword":
		prompt_str = load_keyword_template(input_sent, priming_examples)
	else:
		print("The prompt mode is not supported! Please specify a prompt mode from [full, keyword].")
		return ""
	return prompt_str


def create_data(input_data, example_data, out_fp, num_shots=1, complete=gpt_completion, prompt_mode="full"):
	with open(out_fp, 'w') as f:
		for i in range(len(input_data)):
			if i % 100 == 0:
				print("Finished creating {} instances...".format(i))
			priming_data = random.sample(example_data, num_shots)
			input_sent = input_data[i]['TEXT1']
			prompt = generate_prompt(input_sent, priming_data, prompt_mode)
			print('===prompt=== ', prompt)
			framed_sent = complete(prompt)
			print('===gpt output=== ', framed_sent)
			framed_inst = copy.deepcopy(input_data[i])
			framed_inst['TEXT1'] = framed_sent
			# save to file
			f.write(json.dumps(framed_inst) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', "--input_data_fp", required=True)
	parser.add_argument("-e", "--example_data_fp", required=True)
	parser.add_argument("-o", "--out_fp", required=True)
	parser.add_argument("--num_shots", required=True, type=int, default=1, help="Number of priming examples to be used for few-shot prompting.")
	parser.add_argument("--complete", required=True, type=str, default="gpt_completion", help="Text completion method.")
	parser.add_argument("--prompt_mode", required=True, type=str, default="full", choices=["full", "keyword"], help="Prompt generation method.")
	
	args = parser.parse_args()
	fix_seed()

	input_data = read_jsonl(args.input_data_fp)
	example_data = read_jsonl(args.example_data_fp)
	example_data = [x['example'] for x in example_data]

	out_fp = args.out_fp
	num_shots = args.num_shots
	complete = args.complete
	prompt_mode = args.prompt_mode
	
	if complete == "gpt_completion":
		complete = gpt_completion
	else:
		print("Please re-enter a valid completion method.")

	# Run GPT3 generation
	create_data(input_data, example_data, out_fp, num_shots, complete, "full")




