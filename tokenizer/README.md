# Nexus Tokenizer

A custom Byte-Level BPE tokenizer built specifically for Nexus,
designed to handle natural English, programming languages,
mathematics, and structured tool interactions.

## Pipeline
## Pipeline

```mermaid
flowchart TB
    A["Dataset Configuration<br/>dataset.json"]

    A --> B["General Text"]
    A --> C["Technical Content"]
    A --> D["Structured Tool Data"]

    B --> B1["FineWeb<br/>FineWeb-Edu<br/>Wikipedia"]
    C --> C1["Python<br/>C<br/>Mathematics"]
    D --> D1["Nexus Tools<br/>JSON / APIs<br/>Paths / URLs"]

    B1 --> E["Corpus Sampling"]
    C1 --> E
    D1 --> E

    E --> F["Combined Corpus<br/>1.11B Characters"]

    F --> G["Byte-Level BPE<br/>Training"]

    G --> H["45K Vocabulary<br/>+ Special Tokens"]

    H --> I["tokenizer.json<br/>Nexus Tokenizer"]

    classDef source fill:#f6f8fa,stroke:#8b949e,stroke-width:1px
    classDef process fill:#ddf4ff,stroke:#0969da,stroke-width:1px
    classDef output fill:#dafbe1,stroke:#1a7f37,stroke-width:1px

    class A,B,C,D,B1,C1,D1 source
    class E,F,G,H process
    class I output
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
- Vocabulary: 45,000 tokens
- Context length: 2,048
- Special tokens: ...


## Files

| File | Purpose |
|------|---------|
| dataset.json | Defines tokenizer datasets |
| sample_corpus.py | Samples the training corpus |
| create_nexus_tools_corpus.py | Generates tool-use text |
| train_tokenizer.py | Trains the Byte-Level BPE tokenizer |
