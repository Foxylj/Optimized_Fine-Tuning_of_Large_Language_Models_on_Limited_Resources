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
from dataclasses import dataclass
from torch.utils.data import Dataset,DataLoader
from typing import Dict, Optional, Sequence
HERE = Path(__file__).parent
sys.path.append(str(HERE / "meta_llama2_7b"))

from meta_llama2_7b.llama.model import ModelArgs, Transformer,LoRA
from meta_llama2_7b.llama.tokenizer import Tokenizer
from meta_llama2_7b.llama import Llama,ModelArgs

IGNORE_INDEX = -1
DATA_PATH="./dataset/alpaca_data_dummy_2.json"
torch.manual_seed(1024)

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
    
    tokenized_list = [torch.tensor(tokenizer.encode(text, bos=True, eos=True)) for text in strings]
    #input_ids = labels = [torch.tensor(tokenized) for tokenized in tokenized_list]
    input_ids = labels = [tokenized for tokenized in tokenized_list]
    #input_ids_lens = labels_lens = [len(tokenized) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.ne(-1).sum().item() for tokenized in tokenized_list
    ]
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
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        super(SupervisedDataset, self).__init__()
        try:
            with open(json_path, 'r') as file:
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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=-1
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(-1),
        )

def make_supervised_data_module(tokenizer, data_path):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, json_path=data_path)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def replace_with_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    for name, module in model.named_children():
        
        if isinstance(module, nn.Linear):
            # Replace only Q and V projection layers, which can be identified by their dimensions
            # Replace with your criterion, e.g., by name or by size
            if name=="wq" or name=="wv":  # Define this function based on your model's architecture
                in_features, out_features = module.in_features, module.out_features
                model_weight=module.weight
                lora_linear = LoRA(model_weight,in_features, out_features, lora_r, lora_alpha, lora_dropout)
                setattr(model, name, lora_linear)
        else:
            replace_with_lora(module, lora_r, lora_alpha, lora_dropout)
 
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

def save_checkpoint(model,epoch,loss):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint/consolidated.00.pth')

def train(
    ckpt_dir: str="./meta_llama2_7b/checkpoint/",
    tokenizer_path: str="./meta_llama2_7b/tokenizer/tokenizer.model",
    max_seq_len: int = 128*4,
    max_batch_size: int = 1,
    epochs=50,
    learning_rate=1e-3,
    use_amp=False,
    accumulation_steps = 8
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device="cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(Path(ckpt_dir) / "consolidated.00.pth", map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f: params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    #torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=DATA_PATH)
    dataloader = DataLoader(
        data_module["train_dataset"],
        batch_size=max_batch_size,
        collate_fn=data_module["data_collator"],
        shuffle=True
    )

    replace_with_lora(model)
    freeze_parameters(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epoch_loss=[]
    for epoch in range(epochs):
        sum_loss=0
        for i,batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model.forward(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, 32000)
                shift_labels = shift_labels.view(-1)
                loss = criterion(shift_logits, shift_labels)
                loss = loss / accumulation_steps

            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            sum_loss+=loss
            if (i+1)%accumulation_steps==0 or (i+1)==len(dataloader):
                # Unscales the gradients of optimizer's assigned parameters in-place
                scaler.unscale_(optimizer)
                # Clips gradient norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        epoch_loss.append(sum_loss/len(dataloader))
        #if (min(epoch_loss)==epoch_loss[-1]): save_checkpoint(model,epoch,epoch_loss)
        if epoch%2==0: print(f"Epoch [{epoch+1}/{epochs}], Loss: {sum_loss}")

if __name__ == "__main__":
    train()