<p align="center">
  <img src="./assets/nexus_logo.png" alt="Nexus logo" width="140">
</p>
<h1 align="center">Nexus</h1>

<p align="center">
An autonomous English AI agent built from scratch, designed to complete multi-step tasks
using reasoning, tools, memory, web research, code execution, and document retrieval.
</p>

> Nexus is currently under active development.

## Overview

Nexus is built end to end: its own tokenizer, its own pretraining data, and its own model
trained from scratch, then fine-tuned to use tools. The goal is an agent anyone can run.

- **Model:** decoder-only transformer, 794.5M parameters, 4,096-token context
- **Tokenizer:** custom byte-level BPE, 50,000 tokens, built for English, code, math and tool calls
- **Pretraining data:** about 16B tokens of English web text, math, Wikipedia and permissively
  licensed Python and C code
- **Training:** Custom PyTorch training loop with AdamW, gradient accumulation, gradient clipping, learning-rate warmup, cosine decay, validation, and checkpointing.
- **Agent:** tool use through a structured call format (`<|tool_call|>` ... `<|end_tool_call|>`)

## Status

| Stage | Status |
|-------|--------|
| Tokenizer | Done: 50,000-token byte-level BPE with 32 special tokens (see [`tokenizer/`](tokenizer/)) |
| Pretraining data | Done: tokenized shards of web, math, Wikipedia and code (see [`data_prep/`](data_prep/)) |
| Data loader | Done: mixes sources by weight and serves training batches (see [`data_loader.py`](data_loader.py)) |
| Model layers | Done: token embedding and RoPE (see [`layers/`](layers/)) |
| Attention block | Done: grouped-query attention with KV cache (see [`attention_block.py`](attention_block.py)) |
| Transformer model | Done: 20 blocks, tied head, verified parameter count (see [`model.py`](model.py)) |
| Training loop | Implemented: AdamW, gradient accumulation, gradient clipping, learning-rate scheduling, validation, and checkpointing (see [`train.py`](train.py)) |
| Pipeline testing | Passed: core training components tested with a small model; full-scale training not yet verified |
| Pretraining | Planned |
| Supervised fine-tuning for tool use | Planned |
| Agent runtime (tools, memory, planner) | Planned |
| Evaluation | Planned |

## Architecture

Nexus uses the design of modern open LLMs (Llama, Mistral, Qwen) rather than the 2017
transformer. The attention block and the full model differ from the classic version in these ways.

**Attention (`attention_block.py`)**

- **Grouped-query attention.** 16 query heads share 8 key/value heads, halving the KV cache
  without a noticeable quality loss.
- **Pre-norm residual blocks** with RMSNorm before each sublayer, which keeps training stable at depth.
- **SwiGLU MLP** in each block: `down(silu(gate(x)) * up(x))`, with hidden width 3584 instead of
  the classic 4 × d_model, since three matrices replace two.
- **No biases** in any linear layer.
- **Fused attention** through `scaled_dot_product_attention`, which uses FlashAttention on supported GPUs.
- **KV cache** for generation. Keys and values are cached before the head repeat, and the mask
  depends on the case: no cache, one new token, or several new tokens after a cache.

**Model (`model.py`)**

- **Tied output head.** The head reuses the embedding table, saving about 102M parameters.
- **Scaled initialization.** Linear weights start from a normal distribution with std 0.02. The two
  layers that write into the residual stream (attention output and MLP down-projection) use
  `0.02 / sqrt(2 × num_layers)`, so the residual stream doesn't grow with depth.
- **Final RMSNorm** before the output head, with logits returned in the compute dtype.
- **Verified size:** 794,544,128 parameters with the Nexus config.

## Training

Nexus uses a custom PyTorch training loop implemented in `train.py` to train the
Transformer on tokenized text.

Each training step follows this process:

1. **Batch loading:** `PretrainLoader` provides input token sequences and their next-token targets.
2. **Forward pass:** The Transformer produces logits for next-token prediction.
3. **Loss calculation:** Cross-entropy measures the difference between predictions and target tokens.
4. **Gradient accumulation:** Gradients are accumulated across multiple micro-batches before updating model parameters.
5. **Gradient clipping:** Gradients are clipped to the configured maximum norm.
6. **Optimization:** AdamW updates the model parameters.
7. **Learning-rate scheduling:** Linear warmup is followed by cosine decay toward the minimum learning rate.
8. **Evaluation:** Validation loss and perplexity are calculated at configured intervals.
9. **Checkpointing:** Model weights, optimizer state, training step, best validation loss, and training-loader state are saved.

Training hyperparameters are configured in `config.json`, including batch size,
gradient accumulation, learning rate, warmup steps, maximum training steps,
evaluation frequency, and checkpoint frequency.

The core pipeline has passed a small-model test. Full-scale pretraining and
`torch.compile` execution remain to be verified.

## Project Structure

```
Nexus/
├── assets/            logo and images
├── tokenizer/         tokenizer code and the trained tokenizer
├── data_prep/         pretraining data pipeline
├── layers/            token embedding and RoPE
├── attention_block.py grouped-query causal attention with KV cache, and transformer block
├── model.py           full Transformer: embedding, 20 blocks, final norm, tied head
├── dataloader.py      batch loader for pretraining
├── config.json        model and training settings
├── training/          (next)
├── agent/             (planned)
│   ├── planner/
│   ├── executor/
│   ├── memory/
│   └── tools/
├── rag/               (planned)
├── evaluation/        (planned)
├── inference/         (planned)
└── scripts/           (planned)
```

## License

MIT, see [LICENSE](LICENSE). The datasets used for training are not redistributed here;
see each dataset's page for its license.