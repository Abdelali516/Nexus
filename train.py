from pathlib import Path
import json
from huggingface_hub import HfApi
from model import Transformer
import torch
from data_loader import PretrainLoader
import math
import torch.nn.functional as F
import time

ROOT_DIR = Path(__file__).resolve().parent

CONFIG_FILE = ROOT_DIR / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)

api = HfApi()
REPO_ID = "Abdelali06/nexus-pretrain"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = Transformer(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    init_std=cfg["init_std"],
    num_heads=cfg["num_heads"],
    num_kv_heads=cfg["num_kv_heads"],
    max_seq_len=cfg["max_seq_len"],
    base=cfg["rope_theta"],
    norm_eps=cfg["norm_eps"],
    d_ff=cfg["d_ff"],
    num_layers=cfg["num_layers"]
)
model = model.to(device)

# norms and embeddings skip weight decay; linear weights get it
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.ndim < 2 or "embedding" in name:
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0}
    ],
    lr=cfg["learning_rate"],
    betas=(cfg["beta1"], cfg["beta2"]),
    eps=cfg["adam_eps"]
)

# the compiled model runs the forward pass; the original is kept for clipping and saving
if cfg["compile"]:
    model_compiled = torch.compile(model)
else:
    model_compiled = model

loader = PretrainLoader(split="train", batch_size=cfg["batch_size"], device=device)
eval_loader = PretrainLoader(split="val", batch_size=cfg["batch_size"], device=device)
eval_batches = [eval_loader.next_batch() for _ in range(20)]
best_val = float("inf")

step = 0


def get_lr(step):
    # linear warmup, then cosine decay from learning_rate down to min_learning_rate
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * (step + 1) / cfg["warmup_steps"]

    progress = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg["min_learning_rate"] + (cfg["learning_rate"] - cfg["min_learning_rate"]) * cosine


@torch.no_grad()
def evaluate():
    model.eval()
    total = 0.0
    for x, y in eval_batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
        total += F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1)).item()
    model.train()
    return total / len(eval_batches)


def save_checkpoint(path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "loader": loader.state_dict(),
    }, path)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="checkpoint.pt",
        repo_id=REPO_ID,
        commit_message=f"checkpoint at step {step}",
    )


accum = cfg["gradient_accumulation_steps"]
vocab_size = cfg["vocab_size"]
tokens_per_step = cfg["batch_size"] * cfg["gradient_accumulation_steps"] * cfg["max_seq_len"]
t0 = time.time()

model.train()

while step < cfg["max_steps"]:
    # set this step's learning rate before any micro-batch runs
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # accumulate gradients over `accum` micro-batches of 8 sequences each
    total_loss = 0.0
    for _ in range(accum):
        x, y = loader.next_batch()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model_compiled(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
        (loss / accum).backward()
        total_loss += loss.item()

    # rescale the full gradient if its size exceeds grad_clip, then update the weights
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    step += 1
    if step % 10 == 0:
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_per_sec = tokens_per_step * 10 / dt
        avg_loss = total_loss / accum
        print(f"step {step:>6} | train loss {avg_loss:.4f} | train ppl {math.exp(avg_loss):.2f} | "
              f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s")
        t0 = time.time()

    if step % cfg["eval_every"] == 0:
        val_loss = evaluate()
        print(f"step {step:>6} | val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.2f} | "
              f"gap {val_loss - avg_loss:+.4f}")
        if cfg["save_best"] and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ROOT_DIR / "best.pt")
        t0 = time.time()

    if step % cfg["save_every"] == 0:
        save_checkpoint(ROOT_DIR / "checkpoint.pt")
        t0=time.time()