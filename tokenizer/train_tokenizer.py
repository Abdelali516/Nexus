from pathlib import Path # is python's built in way of working with file and folder paths.
import json
from tokenizers import ByteLevelBPETokenizer


ROOT_DIR=Path(__file__).resolve().parent.parent
TOKENIZER_DIR=Path(__file__).resolve().parent

CONFIG_FILE = ROOT_DIR / "config.json"

with open(CONFIG_FILE,"r") as f:
    config=json.load(f)


SAMPLE_FILE = TOKENIZER_DIR / "data" / "corpus_file.txt"

OUTPUT_DIR = TOKENIZER_DIR / "trained"

SPECIAL_TOKENS=["<pad>", "<bos>", "<eos>", "<unk>"]
SPECIAL_TOKENS+=["<|user|>", "<|assistant|>", "<|system|>"]

def train_tokenizer():
    
    tokenizer=ByteLevelBPETokenizer()

    tokenizer.train(
        files=[SAMPLE_FILE],
        vocab_size=config["vocab_size"],
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_DIR/"tokenizer.json"))

    return tokenizer

if __name__=="__main__":
    train_tokenizer()