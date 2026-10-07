import os
import json
from pathlib import Path

from datasets import load_dataset


os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


TOKENIZER_DIR = Path(__file__).resolve().parent

DATASET_FILE = TOKENIZER_DIR / "dataset.json"
SAMPLE_FILE = TOKENIZER_DIR / "data" / "corpus_file.txt"


with open(DATASET_FILE, "r", encoding="utf-8") as f:
    datasets = json.load(f)


SEED = datasets["seed"]
SHUFFLE_BUFFER = datasets["shuffle_buffer"]
MIN_DOC_CHARS = datasets["min_doc_chars"]


def sample_local(dataset_config, output):

    written_chars = 0
    local_file = TOKENIZER_DIR / dataset_config["file"]

    with open(local_file, "r", encoding="utf-8") as source:
        for line in source:
            output.write(line)
            written_chars += len(line)

    print(f"{dataset_config['name']}: {written_chars:,} characters collected")


def sample_huggingface(dataset_config, output):

    load_args = {
        "path": dataset_config["dataset"],
        "split": dataset_config["split"],
        "streaming": True,
    }

    if "config" in dataset_config:
        load_args["name"] = dataset_config["config"]

    dataset = load_dataset(**load_args)

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER,
    )

    target_chars = dataset_config["target_chars"]
    written_chars = 0

    for example in dataset:

        if written_chars >= target_chars:
            break

        text = example.get(dataset_config["text_field"])


        remaining = target_chars - written_chars
        text = text[:remaining]

        output.write(text+"\n\n")
        written_chars += len(text)

    print(f"{dataset_config['name']}: {written_chars:,} characters collected")


def sample_corpus():

    SAMPLE_FILE.parent.mkdir(parents=True,exist_ok=True)

    if SAMPLE_FILE.exists():
        SAMPLE_FILE.unlink()

    with open(SAMPLE_FILE,"w",encoding="utf-8") as output:

        for dataset_config in datasets["datasets"]:
            
            if dataset_config["type"] == "local":
                    sample_local(dataset_config,output)

            elif dataset_config["type"] == "huggingface":
                sample_huggingface(dataset_config,output)

            else:
                raise ValueError(
                    f"Unknown dataset type: "
                    f"{dataset_config['type']}"
                )

if __name__ == "__main__":
    sample_corpus()