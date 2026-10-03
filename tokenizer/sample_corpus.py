from pathlib import Path
import json
from datasets import load_dataset

TOKENIZER_DIR = Path(__file__).resolve().parent

DATASET_FILE = TOKENIZER_DIR / "dataset.json"
SAMPLE_FILE = TOKENIZER_DIR / "data" / "corpus_file.txt"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    datasets = json.load(f)

def sample_corpus():
    
    if SAMPLE_FILE.exists():
        SAMPLE_FILE.unlink()
    

    for dataset_type in datasets["datasets"]:

        if dataset_type["type"] == "local":

            local_file = TOKENIZER_DIR / dataset_type["file"]
            target_chars = dataset_type["target_chars"]

            written_chars = 0
            with open(local_file, "r", encoding="utf-8") as source:
                with open(SAMPLE_FILE, "a", encoding="utf-8") as output:

                    for line in source:
                        remaining = target_chars - written_chars
                        
                        if remaining <= 0:
                            break

                        text = line[:remaining]
                        output.write(text)
                        written_chars += len(text)

            print(f"{dataset_type['name']}: {written_chars:,} characters collected!")
            continue

        # Hugging Face Dataset !
        load_args= {
            "path":dataset_type["dataset"],
            "split":dataset_type["split"],
            "streaming":True
            }

        if "config" in dataset_type:
            load_args["name"]=dataset_type["config"]
        
        dataset= load_dataset(
            **load_args
        )

        written_chars=0
        with open (SAMPLE_FILE,"a",encoding="utf-8") as f:
            for example in dataset:

                if "language" in dataset_type:
                    if example["metadata"]["language"] != dataset_type["language"]:
                        continue

                text=example["text"]
                remaining = dataset_type["target_chars"] - written_chars
                if remaining <= 0:
                    break
                
                text = text[:remaining]
                f.write(text+"\n")
                written_chars+=len(text)

        print(
            f"{dataset_type['name']}: {written_chars:,} characters collected !"
        )

if __name__=="__main__":
    sample_corpus()