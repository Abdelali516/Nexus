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

- **Model:** decoder-only transformer, about 800M parameters, 4,096-token context (planned)
- **Tokenizer:** custom byte-level BPE, 50,000 tokens, built for English, code, math and tool calls
- **Pretraining data:** about 16B tokens of English web text, math, Wikipedia and permissively
  licensed Python and C code
- **Agent:** tool use through a structured call format (`<|tool_call|>` ... `<|end_tool_call|>`)

## Status

| Stage | Status |
|-------|--------|
| Tokenizer | Done: 50,000-token byte-level BPE with 32 special tokens (see [`tokenizer/`](tokenizer/)) |
| Pretraining data | Done: tokenized shards of web, math, Wikipedia and code (see [`data_prep/`](data_prep/)) |
| Data loader | Done: mixes sources by weight and serves training batches (see [`dataloader.py`](dataloader.py)) |
| Model layers | Done: token embedding and RoPE (see [`layers/`](layers/)) |
| Attention block | Next |
| Pretraining | Planned |
| Supervised fine-tuning for tool use | Planned |
| Agent runtime (tools, memory, planner) | Planned |
| Evaluation | Planned |

## Data Loader

[`dataloader.py`](dataloader.py) reads the tokenized shards and produces training batches.

1. **Sources.** Each source in `pretrain_dataset.json` gets its own list of windows. A window is a
   non-overlapping slice of `SEQ_LEN + 1` tokens inside one shard, so each window gives an input and
   a shifted target.
2. **Mixing.** For each batch, every row picks a source at random, weighted by `target_tokens`.
   This keeps all sources in the batch at all times, instead of training on one source at a time.
3. **Reading order.** Within a source, windows are read in a shuffled order. When a source runs out
   of windows, it starts a new pass (epoch) with a new order. Each source tracks its own epoch and
   position.
4. **Reproducibility.** All randomness comes from `config["seed"]`, so the same seed gives the same
   batches.

## Project Structure

```
Nexus/
├── assets/          logo and images
├── tokenizer/       tokenizer code and the trained tokenizer
├── data_prep/       pretraining data pipeline
├── dataloader.py    batch loader for pretraining
├── layers/          token embedding and RoPE
├── config.json      model and training settings
├── model/           (planned)
├── training/        (planned)
├── agent/           (planned)
│   ├── planner/
│   ├── executor/
│   ├── memory/
│   └── tools/
├── rag/             (planned)
├── evaluation/      (planned)
├── inference/       (planned)
└── scripts/         (planned)
```

## License

MIT, see [LICENSE](LICENSE). The datasets used for training are not redistributed here;
see each dataset's page for its license.