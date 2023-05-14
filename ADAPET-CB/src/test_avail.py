import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.utils.Config import Config
from src.utils.util import device
from src.adapet import adapet
from src.eval.eval_model_avail import test_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    parser.add_argument('-d', "--dummy_test_size", required=True)
    parser.add_argument('-r', "--repeat", required=True)
    args = parser.parse_args()

    dummy_test_size = args.dummy_test_size
    repeat = args.repeat

    config_file = os.path.join(args.exp_dir, "config_avail_{}_{}.json".format(dummy_test_size, repeat))
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))
    test_eval(config, model, batcher, dummy_test_size, repeat)

