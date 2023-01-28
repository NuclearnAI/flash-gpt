# flash-huggingface (WIP)
Minimal reproducible implementations of [Huggingface Transformers](https://github.com/huggingface/transformers) equipped with the [Triton](https://github.com/openai/triton) version of [Flash-Attention](https://github.com/HazyResearch/flash-attention).

## Acknowledgement:
Big thanks to @zphang ((Jason Phang)[https://jasonphang.com/]) of EleutherAI for his great work in implementing T5, @lucidrains ((Phil Wang)[https://github.com/lucidrains]) for his implementations of numerous transformer architectures and taking the time to review my work, and @ptillet ((Phil Tillet)[https://github.com/ptillet]) for his help resolving issues I had with the Triton language. 

## Supported Architectures:
- T5 (including flan-T5)

## In-progress:
- GPT-2
- GPT-J
- GPT-NEOX
- OPT
- Bloom