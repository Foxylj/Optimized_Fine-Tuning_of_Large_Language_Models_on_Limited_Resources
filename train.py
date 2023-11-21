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
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from typing import Dict, Optional, Sequence
HERE = Path(__file__).parent
sys.path.append(str(HERE / "meta_llama2_7b"))

from meta_llama2_7b.llama.model import ModelArgs, Transformer
from meta_llama2_7b.llama.tokenizer import Tokenizer
from meta_llama2_7b.llama import Llama

IGNORE_INDEX = -100
DATA_PATH="./alpaca_data_dummy.json"

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

def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    """Tokenize a list of strings."""
    
    tokenized_list = [tokenizer.encode(text, bos=True, eos=True) for text in strings]
    input_ids = labels = [torch.tensor(tokenized) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized) for tokenized in tokenized_list]
    pdb.set_trace()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    x=[strings  for strings in (examples, sources)]
    pdb.set_trace()
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class CustomNLPDataset(Dataset):
    def __init__(self, json_path, tokenizer,max_length,device='cuda'):
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {json_path}: {e}")
        self.tokenizer = tokenizer

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in self.data
        ]
        targets = [f"{example['output']}" for example in self.data]
        data_dict = preprocess(sources, targets, self.tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        # self.input_ids=self.input_ids.to(device)
        # self.labels=self.labels.to(device)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def train(
    ckpt_dir: str="./meta_llama2_7b/checkpoint/",
    tokenizer_path: str="./meta_llama2_7b/tokenizer/tokenizer.model",
    max_seq_len: int = 128*4,
    max_batch_size: int = 2,
    epochs=3,
    learning_rate=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    generator = torch.Generator()

    llama_instance = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model=(llama_instance.model).to(device)
    tokenizer=llama_instance.tokenizer
    # 转成list看看
    dataset = CustomNLPDataset(json_path=DATA_PATH, tokenizer=tokenizer,max_length=max_seq_len)
    pdb.set_trace()
    train_dataloader = DataLoader(dataset, batch_size=max_batch_size, shuffle=True, generator=generator)

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            pdb.set_trace()

if __name__ == "__main__":
    train()