from pathlib import Path
import json

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = Path(__file__).resolve().parent

CONFIG_FILE = ROOT_DIR / "config.json"


with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


SAMPLE_FILE = TOKENIZER_DIR / "data" / "corpus_file.txt"
OUTPUT_DIR = TOKENIZER_DIR / "trained"


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|end_tool_call|>",
    "<|tool_result|>",
    "<|end_tool_result|>",
    "<|end_turn|>",
]

SPECIAL_TOKENS += [
    f"<|reserved_{i}|>"
    for i in range(16)
]


SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def corpus_chunks(chunk_chars=1_000_000):

    buffer = []
    size = 0

    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:

        for line in f:

            buffer.append(line)
            size += len(line)

            if size >= chunk_chars:

                yield "".join(buffer)

                buffer = []
                size = 0

    if buffer:
        yield "".join(buffer)


def train_tokenizer():

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.normalizer = None

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex(SPLIT_PATTERN),
            behavior="isolated",
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=False,
        ),
    ])

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        corpus_chunks(),
        trainer=trainer,
    )

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))

    special_ids = {
        token: tokenizer.token_to_id(token)
        for token in SPECIAL_TOKENS
    }

    with open(
        OUTPUT_DIR / "special_tokens.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            special_ids,
            f,
            indent=2,
        )

    print(
        f"Vocab size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    return tokenizer


if __name__ == "__main__":
    train_tokenizer()