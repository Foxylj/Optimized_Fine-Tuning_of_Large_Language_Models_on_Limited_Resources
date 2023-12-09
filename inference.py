import pdb
import sys
import fire
import torch
import json
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
HERE = Path(__file__).parent
sys.path.append(str(HERE / "meta_llama2_7b"))

from meta_llama2_7b.llama import Llama
from meta_llama2_7b.llama.model import ModelArgs, Transformer
from meta_llama2_7b.llama.tokenizer import Tokenizer

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128*1,
    max_gen_len: int = 64*1,
    max_batch_size: int = 4,
): 
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    ckpt_path = Path(ckpt_dir) / "consolidated.00.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f: params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    

    generator=Llama(model=model,tokenizer=tokenizer)

    prompts: List[str] = [
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\
        Write a response that appropriately completes the request.\n\n ### Instruction:\n Give three tips for staying healthy.\n\n ### Response:",
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\
        Write a response that appropriately completes the request.\n\n ### Instruction:\n What are the three primary colors? \n\n ### Response:",
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\
        Write a response that appropriately completes the request.\n\n ### Instruction:\n How am I supposed to get a better grade? \n\n ### Response:",
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\
        Write a response that appropriately completes the request.\n\n ### Instruction:\n How to be happy every day? \n\n ### Response:"
    ]

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print("\n",prompt)
        print(f"> {result['generation']}")
        print("\n============================================================================================================================\n")

    loss=[i.detach().numpy() for i in checkpoint["loss"]]
    epoch_len=len(loss)
    plt.plot(range(epoch_len),loss)
    plt.title("Loss VS. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('./figures/Epoch_loss.png')
    

if __name__ == "__main__":
    fire.Fire(main)