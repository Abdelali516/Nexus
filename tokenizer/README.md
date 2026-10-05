# Nexus Tokenizer

A custom Byte-Level BPE tokenizer built specifically for Nexus,
designed to handle natural English, programming languages,
mathematics, and structured tool interactions.

## Pipeline

```mermaid
flowchart TD
    A[Dataset Configuration<br/>dataset.json] --> B

    subgraph B[Corpus Sources]
        direction LR
        B1[General Text<br/>FineWeb, FineWeb-Edu, Wikipedia]
        B2[Technical Content<br/>Python, C, Mathematics]
        B3[Structured Tool Data<br/>Nexus Tools, JSON/APIs, Paths/URLs]
    end

    B --> C[Corpus Sampling<br/>1.11B characters]
    C --> D[Combined Training Corpus<br/>corpus_file.txt]
    D --> E[Byte-Level BPE Training<br/>50K vocabulary]
    E --> F[Special Tokens<br/>User, Assistant, System]
    F --> G[Final Tokenizer<br/>tokenizer.json]
```

## Corpus

The tokenizer is trained on a 1.11B-character corpus composed of:

- General English web text
- Educational text
- Mathematical content
- Python code
- C code
- Wikipedia
- Nexus tool interactions

## Configuration

- Algorithm: Byte-Level BPE
- Vocabulary: 50,000 tokens
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<|user|>`, `<|assistant|>`, `<|system|>`

## Files

| File | Purpose |
|------|---------|
| dataset.json | Defines tokenizer datasets |
| sample_corpus.py | Samples the training corpus |
| create_nexus_tools_corpus.py | Generates tool-use text |
| train_tokenizer.py | Trains the Byte-Level BPE tokenizer |