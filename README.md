# Mind the Biases: Quantifying Cognitive Biases in Language Model Prompting

This repository contains the datasets, code, and scripts to conduct the analysis in paper in [Lin and Ng (2023)](#reference).

## Reference
Ruixi Lin and Hwee Tou Ng (2023). 
[Mind the Biases: Quantifying Cognitive Biases in Language Model Prompting](https://TBD).  Proceedings of the Findings-ACL 2023. 

Please cite: 
```
TBD

```

**Table of contents**

[Prerequisites](#prerequisites)

[Run availability bias experiments](#run-availability-bias-experiments)

[Run framing effect experiments](#run-framing-effect-experiments)

[License](#license)


## Prerequisites
Go to /your/home/CBPrompt, run
```
	source setup_env.sh
```
Activate your environment,
```
	source env/bin/activate
```
For software, we build upon the open source code of [ADAPET (Tam et al., 2021)](https://aclanthology.org/2021.emnlp-main.407/) and added our scripts and code for the experiments in this work. For the DDI dataset used in this work, we preprocess the DDIExtraction dataset (Segura-Bedmar et al., 2013) for our experiments, and the processed datasets can be found at ADAPET-CB/data/ddi.


## Run availability bias experiments
Go to /your/home/CBPrompt/run_avail, copy ADAPET-CB to the current directory

1. You can train your own models, for full training and few-shot training settings
```
	./run_single_ddi.sh
```

2. Get test predictions for the dummy prompts on the pretrained models,
   and compute availability bias scores. Specify paths to pretrained models in run_avail_test.sh, and in the pretrained model directory, add config files for 3 repeated runs (for 3 repeats of sample drawing) according to config_avail_100_0_example.json, then
	
cd /your/home/CBPrompt/ADAPET-CB
```
	./run_avail_test.sh
```

If you want to create your own dummy test prompts on biomedical texts, functions in avail_bias_util.py and umls_mapping.py may be useful.


#### Run framing effect experiments ####
Go to /your/home/CBPrompt/run_framing, copy ADAPET-CB to the current directory

1. Get the original and GPT-3 generated toned-down paraphrase datasets (including train, val, test sets) at

/your/home/CBPrompt/ADAPET-CB/data/ddi/framing/ori/
	train.jsonl
	val.jsonl
	test.jsonl

/your/home/CBPrompt/ADAPET-CB/data/ddi/framing/gpt_completion/
	train.jsonl
	val.jsonl
	test.jsonl

If you want to generate your own GPT-3 paraphrases with custom data, kindly refer to get_data.sh


2. You can train your own models on the original data and paraphrase data
```
	./run_framing_exp_train.sh
```

3. Get pretrained models, evaluate the models on original and paraphrase test data, where
$framing_ori and $framing_gpt are your paths to trained model directories, e.g., /your/home/CBPrompt/ADAPET-CB/exp_out/generic/bert-base-uncased/20XX-XX-XX-XX-XX-XX
```
	./run_framing_exp_eval.sh $framing_ori $framing_gpt
```

4. Get Framing Effect Paraphrase (FEP) scores, and compute conditional probabilies, given FEP ranges
```
	python measure_framing_effect.py \
		-ori_train /your/home/CBPrompt/ADAPET-CB/data/ddi/framing/ori/train.jsonl \
		-ori_val /your/home/CBPrompt/ADAPET-CB/data/ddi/framing/ori/val.jsonl \
		-ori_test /your/home/CBPrompt/ADAPET-CB/data/ddi/framing/ori/test.jsonl \
		-gpt_train /your/home/CBPrompt/ADAPET-CB/data/ddi/framing/gpt_completion/train.jsonl \
		-gpt_val /your/home/CBPrompt/ADAPET-CB/data/ddi/framing/gpt_completion/val.jsonl \
		-gpt_test /your/home/CBPrompt/ADAPET-CB/data/ddi/framing/gpt_completion/test.jsonl \
```


## License
The source code and models in this repository are licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). For commercial use of this code and models, separate commercial licensing is also available. Please contact Ruixi Lin (ruixi@u.nus.edu) and Prof. Hwee Tou Ng (nght@comp.nus.edu.sg).



