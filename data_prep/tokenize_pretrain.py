import gzip
import json
import re
from array import array
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

DATASET=HERE / "pretrain_dataset.json"
TOKENIZER_PATH= ROOT /"tokenizer" /"trained" / "tokenizer.json"

with open (DATASET, "r", encoding="utf-8") as f:
    cfg=json.load(f)

OUT = ROOT / "data_prep" / "shards"
OUT.mkdir(parents=True, exist_ok=True)

tok = Tokenizer.from_file(str(TOKENIZER_PATH))

BOS = tok.token_to_id("<bos>")
EOS = tok.token_to_id("<eos>")

# Text like "<|user|>" inside a document would become a real special token, so we skip it.
special_re = re.compile("|".join(re.escape(t.content) for t in tok.get_added_tokens_decoder().values()))

allowed = set(cfg["allowed_licenses"])


def documents(src):
    if src["type"] == "huggingface":  
        args = {"path": src["dataset"], "split": src["split"], "streaming": True}
        if "config" in src:
            args["name"] = src["config"]
        dataset = load_dataset(**args).shuffle(seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"])
        for example in dataset:
            yield example.get(src["text_field"])
    else:  
        files = sorted((ROOT / src["path"]).glob("chunk_*.jsonl.gz"))
        for _ in range(1):
            for path in files:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        licenses = record["detected_licenses"]
                        if record["language"] == src["language"] and all(lic in allowed for lic in licenses): # double check the languages too 
                            yield record["text"]


def write(path, tokens):
    with open(path, "wb") as f:
        tokens.tofile(f)  
    del tokens[:]         


for src in cfg["sources"]:
    name = src["name"]
    target= src["target_tokens"]
    done_marker = OUT / f"{name}.done"
    if done_marker.exists():
        print(f"[{name}] already done, skipping")
        continue

    train, val = array("H"), array("H")  
    total = shard = count = 0
    batch, docs, finished = [], documents(src), False

    while not finished:
        # collect up to 1000 good documents
        for txt in docs:
            if isinstance(txt, str) and len(txt) >= cfg["min_doc_chars"] and not special_re.search(txt):
                batch.append(txt)
                if len(batch) == 1000:
                    break
        else:
            finished = True  # the source ran out of documents

        # tokenize them together (much faster than one by one)
        for encoding in tok.encode_batch(batch):
            ids = [BOS] + encoding.ids + [EOS]
            count += 1
            (val if count % 200 == 0 else train).extend(ids)  # every 200th document is validation
            total += len(ids)
        batch = []

        if total >= target:
            finished = True

        if len(train) >= cfg["tokens_per_shard"] or finished:
            write(OUT / f"{name}_train_{shard:04d}.bin", train)
            if len(val):
                write(OUT / f"{name}_val_{shard:04d}.bin", val)
            print(f"[{name}] shard {shard:04d} saved | {total:,} / {target:,} tokens")
            shard += 1

    done_marker.write_text(str(total))
    if total < 0.98 * target:
        print(f"[{name}] WARNING: ran out of data at {total:,} of {target:,} tokens")

print("Done.")