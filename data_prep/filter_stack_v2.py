import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "bigcode/the-stack-v2-train-smol-ids"
OUTPUT_DIR = Path("data/stack_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "python_c_manifest.jsonl"

TARGET_LANGUAGES = {"Python", "C"}

def main():

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    counts = Counter()
    selected_files = 0
    selected_repositories = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for repository in dataset:
            matching_files = []

            for file in repository.get("files", []):
                language = file.get("language")

                if language not in TARGET_LANGUAGES:
                    continue

                # Exclude vendor and generated files.
                if file.get("is_vendor", False) or file.get("is_generated", False):
                    continue

                matching_files.append({
                    "repo_name": repository.get("repo_name"),
                    "path": file.get("path"),
                    "language": language,
                    "blob_id": file.get("blob_id"),
                    "src_encoding": file.get("src_encoding"),
                    "length_bytes": file.get("length_bytes"),
                    "detected_licenses": file.get("detected_licenses", []),
                    "license_type": file.get("license_type"),
                })

            if not matching_files:
                continue

            selected_repositories += 1

            for item in matching_files:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[item["language"]] += 1
                selected_files += 1

            if selected_repositories % 1000 == 0:
                print(
                    f"Repositories selected: {selected_repositories:,} | "
                    f"Files selected: {selected_files:,}"
                )

    print("\nFiltering complete.")
    print(f"Python files: {counts['Python']:,}")
    print(f"C files:      {counts['C']:,}")
    print(f"Total files:  {selected_files:,}")
    print(f"Manifest:     {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
