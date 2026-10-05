import os
import json
from pathlib import Path

from datasets import load_dataset


os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


TOKENIZER_DIR = Path(__file__).resolve().parent

DATASET_FILE = TOKENIZER_DIR / "dataset.json"
SAMPLE_FILE = TOKENIZER_DIR / "data" / "corpus_file.txt"


with open(DATASET_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)


SEED = config.get("seed", 42)
SHUFFLE_BUFFER = config.get("shuffle_buffer", 10_000)
MIN_DOC_CHARS = config.get("min_doc_chars", 200)


def sample_local(dataset_config, output):

    target_chars = dataset_config["target_chars"]
    written_chars = 0

    local_file = TOKENIZER_DIR / dataset_config["file"]

    with open(local_file, "r", encoding="utf-8") as source:

        for line in source:

            if written_chars >= target_chars:
                break

            remaining = target_chars - written_chars
            text = line[:remaining]

            output.write(text)
            written_chars += len(text)

    print(
        f"{dataset_config['name']}: "
        f"{written_chars:,} characters collected"
    )


def sample_huggingface(dataset_config, output):

    load_args = {
        "path": dataset_config["dataset"],
        "split": dataset_config["split"],
        "streaming": True
    }

    if "config" in dataset_config:
        load_args["name"] = dataset_config["config"]

    dataset = load_dataset(**load_args)

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER
    )

    target_chars = dataset_config["target_chars"]
    written_chars = 0

    for example in dataset:

        if written_chars >= target_chars:
            break

        text = example.get(dataset_config["text_field"])

        if not isinstance(text, str):
            continue

        if len(text) < MIN_DOC_CHARS:
            continue

        remaining = target_chars - written_chars
        text = text[:remaining]

        output.write(text)
        output.write("\n\n")

        written_chars += len(text)

    print(
        f"{dataset_config['name']}: "
        f"{written_chars:,} characters collected"
    )


def sample_corpus():

    SAMPLE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if SAMPLE_FILE.exists():
        SAMPLE_FILE.unlink()

    with open(
        SAMPLE_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for dataset_config in config["datasets"]:

            if dataset_config["type"] == "local":

                sample_local(
                    dataset_config,
                    output
                )

            elif dataset_config["type"] == "huggingface":

                sample_huggingface(
                    dataset_config,
                    output
                )

            else:

                raise ValueError(
                    f"Unknown dataset type: "
                    f"{dataset_config['type']}"
                )


if __name__ == "__main__":
    sample_corpus()