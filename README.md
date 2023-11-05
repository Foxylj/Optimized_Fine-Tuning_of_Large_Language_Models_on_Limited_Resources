# Efficient_Fine_Tuning_LLM_on_Single_GPU

## Problem 4: LLaMA Inference

Run follow code in terninal:

```shell
$ python -m torch.distributed.run --nproc_per_node 1 ../llama/example_text_completion.py --ckpt_dir ./model/llama2-7b --tokenizer_path ./model/llama2-7b/tokenizer.model
```
Result:
```
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 77.05 seconds
I believe the meaning of life is
> to be happy. I believe we are all born with the potential to be happy. The meaning of life is to be happy and the way to be happy is to live life to the fullest.
I believe that happiness is a choice. I believe that we can choose to be happy every day. I believe that

==================================

Simply put, the theory of relativity states that 
> 1) time, space, and mass are relative, and 2) the speed of light is constant, regardless of the relative motion of the observer.
What I'm not clear on is the relationship between the two.
My understanding is that the speed of light is constant, and that it is also invariant

==================================

A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just 
> wanted to take a moment to congratulate the team on the launch of the new website.
        I think it looks great!
        
        I'm so proud to be part of this team.
        
        You guys rock!
        
        Best,
        [Bill]

==================================

Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>
> fromage
        olive => olives
        lamb => mouton
        lamb => mouton
        tuna => thon
        pear => pêche
        onion => oignon
        apple => pomme
        pear => pêche
        st

==================================
```

