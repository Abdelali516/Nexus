from pathlib import Path
import json
from tokenizers import (Regex, Tokenizer, decoders, models, pre_tokenizers,trainers)


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
    "<|end_turn|>"
]

SPECIAL_TOKENS += [
    f"<|reserved_{i}|>" # reserved slots for future features !
    for i in range(20)
]


SPLIT_PATTERN = (
    r"|[^\r\n\p{L}\p{N}]?\p{L}+" # so words like don't, it's, we'll stay as one token
    r"|\p{N}" # split numbers into separate digits 
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*" # groups consecutive punctuation/symbols together
    r"|\s*[\r\n]+" #  matches whitespace followed by one or more line breaks
    r"|\s+(?!\S)" #  matches trailing whitespace after a word 
    r"|\s+" #  matches any remaining whitespace
) 


def corpus_chunks(chunk_chars=1_000_000):

    buffer = []
    size = 0

    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:

        for line in f:

            buffer.append(line)
            size += len(line)

            if size >= chunk_chars:

                yield "".join(buffer) # hands the chunk to the Tokenizer

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

    tokenizer.normalizer = None  # don't modify the original text before tokenization

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex(SPLIT_PATTERN),
            behavior="isolated", # keep my custom regex pieces separate 
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False, # don't add space at the beginning of every input !
            use_regex=False, # don't use the default regex since i already created my own splitting rules !
        ),
    ])

    tokenizer.decoder = decoders.ByteLevel() # the reverse side of ByteLevel !

    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # make sure the basic ByteLevel characters are available as starting pieces !
        show_progress=True # show a progress bar while training !
    )

    tokenizer.train_from_iterator(
        corpus_chunks(),
        trainer=trainer
    )

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))

    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()