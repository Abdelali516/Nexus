import gzip
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config


DATA_DIR = Path(__file__).resolve().parent / "data" / "stack_v2"
MANIFEST_FILE = DATA_DIR / "python_manifest.jsonl"  # the list made by filter_stack_v2.py
OUTPUT_DIR = DATA_DIR / "clean_python"                       # clean code is saved here

BUCKET = "softwareheritage"  # where The Stack v2 keeps the actual code
REGION = "us-west-2"
NUM_THREADS = 128            # parallel downloads
CHUNK_SIZE = 10_000          # manifest lines per output file


# A file that matches any of these patterns is skipped completely.
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                       # AWS access key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),     # private keys
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                    # GitHub token
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}"),  # API keys starting with sk-
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),           # Slack token
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),                  # Google API key
]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")


def clean_source(text): # Return the code with email addresses masked, or None if it contains a secret.
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return None
    return EMAIL.sub("user@example.com", text)


def looks_like_code(text): # Reject minified files, data dumps and binary-looking text.
    lines = text.split("\n")
    longest = max(len(line) for line in lines)           # no line longer than 1000 characters
    mean = len(text) / len(lines)                        # average line under 100 characters
    alnum = sum(c.isalnum() for c in text) / len(text)   # at least 25% letters and digits
    return longest <= 1000 and mean <= 100 and alnum >= 0.25


# One shared connection to the bucket, unsigned means no AWS account is needed.
s3 = boto3.client(
    "s3",
    region_name=REGION,
    config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=NUM_THREADS * 2,
        retries={"max_attempts": 5, "mode": "standard"},
    ),
)


def process(item): # Download and check one file. Returns (status, record).
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"content/{item['blob_id']}")
        raw = gzip.decompress(obj["Body"].read())
        try:
            text = raw.decode(item.get("src_encoding") or "utf-8", errors="ignore")
        except LookupError:  # unknown encoding name
            text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return "failed", None

    if not text or "\x00" in text or len(text) < 100 or not looks_like_code(text):
        return "filtered", None

    cleaned = clean_source(text)
    if cleaned is None:
        return "secret", None

    return "kept", {
        "language": item["language"],
        "repo_name": item.get("repo_name"),
        "path": item.get("path"),
        "blob_id": item["blob_id"],
        "detected_licenses": item.get("detected_licenses", []),  # keeps the license record
        "text": cleaned,
    }


def write_chunk(index, records, files, size):
    data_path = OUTPUT_DIR / f"chunk_{index:06d}.jsonl.gz"
    stats_path = OUTPUT_DIR / f"chunk_{index:06d}.stats.json"
    tmp_path = data_path.with_suffix(".tmp")

    # Small stats file, so later runs can add up the totals without reading the code again.
    stats_path.write_text(json.dumps({"files": dict(files), "bytes": dict(size)}), encoding="utf-8")

    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(tmp_path, data_path)  # the .jsonl.gz file appearing means "chunk finished"


def load_totals(): # Files and bytes kept per language by chunks finished in earlier runs.
    totals = {"files": Counter(), "bytes": Counter()}
    for stats_path in OUTPUT_DIR.glob("chunk_*.stats.json"):
        data_path = stats_path.with_name(stats_path.name.replace(".stats.json", ".jsonl.gz"))
        if data_path.exists():  # ignore a chunk that was interrupted before it finished
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            totals["files"].update(stats["files"])
            totals["bytes"].update(stats["bytes"])
    return totals


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    totals = load_totals()   # what earlier runs already kept
    run_status = Counter()   # what happened in this run
    start = time.time()

    with MANIFEST_FILE.open("r", encoding="utf-8") as manifest, \
            ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:

        index = 0
        while True:
            lines = list(islice(manifest, CHUNK_SIZE))  # next 10,000 manifest lines
            if not lines:
                break

            if (OUTPUT_DIR / f"chunk_{index:06d}.jsonl.gz").exists():
                index += 1  # this chunk was finished in an earlier run
                continue

            results = list(pool.map(process, (json.loads(line) for line in lines)))

            records, status, files, size = [], Counter(), Counter(), Counter()
            for state, record in results:
                status[state] += 1
                if record:
                    records.append(record)
                    files[record["language"]] += 1
                    size[record["language"]] += len(record["text"].encode("utf-8"))

            write_chunk(index, records, files, size)
            totals["files"].update(files)
            totals["bytes"].update(size)
            run_status.update(status)

            print(
                f"chunk {index:06d} | kept {status['kept']:,} | secrets {status['secret']:,} "
                f"| filtered {status['filtered']:,} | failed {status['failed']:,} "
                f"| total Python {totals['files']['Python']:,} "
                f"| {(time.time() - start) / 60:.1f} min"
            )
            index += 1

    print("\nDone.")
    print("This run:", dict(run_status))
    print("Files kept:", dict(totals["files"]))
    print("Bytes kept:", {lang: f"{n:,}" for lang, n in totals["bytes"].items()})
    print("Output:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()