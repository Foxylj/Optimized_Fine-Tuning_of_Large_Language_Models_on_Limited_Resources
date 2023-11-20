# Efficient_Fine_Tuning_LLM_on_Single_GPU

## Problem 4: LLaMA Inference

Run follow code in terninal:

```shell
$ python -m torch.distributed.run --nproc_per_node 1 ./meta_llama2_7b/example_text_completion.py --ckpt_dir ./meta_llama2_7b/checkpoint --tokenizer_path ./meta_llama2_7b/tokenizer/tokenizer.model
```
Result:
```
Loaded in 33.73 seconds
I believe the meaning of life is
> to create a better world for those who come after us. I believe in the power of the individual to create a better world. I believe that if we each do our part, we can change the world. I believe that we are all connected and that what we do to one, we do to all.
##

==================================

Simply put, the theory of relativity states that 
> 1) the speed of light is the same for all observers (independent of the relative motion of the source and observer), and 2) the laws of physics are the same for all observers (independent of their relative motion).
2.1 The Speed of Light is the Same for All Obser

==================================

A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just 
> wanted to take a moment to congratulate the team on the launch of the 

        The product looks great and I am very happy with the results.

        I hope that the product continues to grow and prosper.

        Have a great day.

        Best regards,



==================================

Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>
> fromage
        pizza => pizza
        pepper => poivre
        pepperoni => pizza
        pizza => pizza
        pizza => pizza
        pizza => pizza
        pizza => pizza
        pizza => pizza
        pizza

==================================
```

