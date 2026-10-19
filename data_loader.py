from pathlib import Path
import json
import glob

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
SHARDS_DIR = ROOT_DIR / "data_prep" / "shards"

with open(ROOT_DIR / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open(ROOT_DIR / "data_prep" / "pretrain_dataset.json", "r", encoding="utf-8") as f:
    pretrain_cfg = json.load(f)

SEQ_LEN = config["max_seq_len"]  
SEED = config["seed"]           


def list_shards(name, split):
    # find all shard files for one source and split, e.g. fineweb_train_0000.bin, fineweb_train_0001.bin
    paths = sorted(glob.glob(str(SHARDS_DIR / f"{name}_{split}_*.bin")))
    if not paths:
        raise FileNotFoundError(f"No {name} {split} shards. Run tokenize_pretrain.py first.")
    return paths


class PretrainLoader:
    def __init__(self, split="train", batch_size=None, device="cpu"):
        self.batch_size = batch_size or config["batch_size"]
        self.device = device
        # chooses which source each row of a batch comes from, seeded so runs are reproducible
        self.rng = np.random.default_rng(SEED)

        self.sources = []  # one state dictionary per source (math, fineweb, ...)
        for src in pretrain_cfg["sources"]:
            # memory-map each shard: the file is read lazily, not loaded into RAM
            shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in list_shards(src["name"], split)]
            windows = [(i, s) for i, arr in enumerate(shards)
                       for s in range(0, len(arr) - SEQ_LEN, SEQ_LEN)]
            self.sources.append({"name": src["name"], "shards": shards, "windows": windows,
                                 "order": None, "pos": 0, "epoch": 0})
            # order: shuffled reading plan (set by _reshuffle)
            # pos: bookmark, index of the next item to read from order


        weights = np.array([s["target_tokens"] for s in pretrain_cfg["sources"]], dtype=np.float64)
        self.probs = weights / weights.sum()
        # give every source its first shuffled reading plan before any batch is built
        for idx in range(len(self.sources)):
            self._reshuffle(idx, epoch=0)

    def _reshuffle(self, idx, epoch):
        # idx: which source (0 is the first one in pretrain_dataset.json)
        # epoch: which pass through that source (0 is the first, 1 the second, ...)
        src = self.sources[idx]  # same dictionary as self.sources[idx], so changes are saved
        # the same [SEED, idx, epoch] always gives the same order, so a pass can be reproduced
        rng = np.random.default_rng([SEED, idx, epoch])
        src["order"] = rng.permutation(len(src["windows"]))
        src["pos"] = 0  
        src["epoch"] = epoch  

    def _next_window(self, idx):
        src = self.sources[idx]
        if src["pos"] >= len(src["order"]):
            next_epoch = src["epoch"] + 1
            self._reshuffle(idx, epoch=next_epoch)  # note: epoch_of does not exist yet
        shard_id, start = src["windows"][src["order"][src["pos"]]]
        src["pos"] += 1  
        return src["shards"][shard_id], start

    def next_batch(self):
        # pick a source for each row, according to the weights
        picks = self.rng.choice(len(self.sources), size=self.batch_size, p=self.probs)
        rows = np.empty((self.batch_size, SEQ_LEN + 1), dtype=np.int64)
        for r, idx in enumerate(picks):
            # get the next window from that row's source, then copy its tokens into row r
            shard, start = self._next_window(idx)
            rows[r] = shard[start : start + SEQ_LEN + 1]
        x = torch.from_numpy(rows[:, :-1].copy()).to(self.device)
        # targets: every token except the first, so y is x shifted one position left
        y = torch.from_numpy(rows[:, 1:].copy()).to(self.device)
        return x, y