cd run_framing

# GPT paraphrase generation
# Example usage
python get_framing_data.py -i ./data/ddi/framing/ori/train.jsonl \
	-e ./data/ddi/framing/gpt_priming_examples.jsonl \
	-o ./data/ddi/framing/gpt_completion/train.jsonl \
	--num_shots 1 \
	--complete gpt_completion \
	--prompt_mode full 

