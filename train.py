###################################################################################################
#                                                                                                 #
# Some of this Code refering to: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py  #
#                                                                                                 #
###################################################################################################
import sys
import pdb
import time
import json
import copy
import utils
import torch
import logging
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence
HERE = Path(__file__).parent
sys.path.append(str(HERE / "meta_llama2_7b"))

from meta_llama2_7b.llama.model import ModelArgs, Transformer
from meta_llama2_7b.llama.tokenizer import Tokenizer
from meta_llama2_7b.llama import Llama

DATA_PATH="./alpaca_data.json"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class CustomNLPDataset(Dataset):
    def __init__(self, json_path, tokenizer,max_length):
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {json_path}: {e}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        combined_input = item['instruction'] + ' ' + item['input']
        output_text = item['output']

        # Use the custom tokenizer's encode method
        input_ids = self.tokenizer.encode(combined_input, bos=True, eos=True)
        output_ids = self.tokenizer.encode(output_text, bos=True, eos=True)

        # Truncate the sequences to the maximum length
        input_ids = input_ids[:self.max_length]
        output_ids = output_ids[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }


def train(
    ckpt_dir: str="./meta_llama2_7b/checkpoint/",
    tokenizer_path: str="./meta_llama2_7b/tokenizer/tokenizer.model",
    max_seq_len: int = 128*4,
    max_batch_size: int = 4
):
    llama_instance = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    """list_data_dict = utils.jload(DATA_PATH)
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
    targets = [f"{example['output']}{11111122222}" for example in list_data_dict]
    examples = [s + t for s, t in zip(sources, targets)]
    x=[strings for strings in (examples, sources)]
    pdb.set_trace()"""
    dataset = CustomNLPDataset(json_path=DATA_PATH, tokenizer=llama_instance.tokenizer,max_length=max_seq_len)
    pdb.set_trace()

if __name__ == "__main__":
    train()