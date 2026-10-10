import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "bigcode/the-stack-v2-train-smol-ids"

DATA_PREP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_PREP_DIR / "data" / "stack_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "python_c_manifest.jsonl"


TARGET_LANGUAGES = {"Python", "C"}

MAX_FILES = {
    "Python": 70_000,
    "C": 30_000,
}

MAX_FILES_PER_REPO = 50

def main():

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    counts = Counter()
    selected_files = 0
    selected_repositories = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for repository in dataset:
            
            if all(counts[lang] >= MAX_FILES[lang] for lang in TARGET_LANGUAGES):
                break
            
            matching_files = []

            for file in repository.get("files", []):
                language = file.get("language")

                if language not in TARGET_LANGUAGES:
                    continue

                if counts[language] >= MAX_FILES[language]:
                    continue

                if file.get("license_type") != "permissive":
                    continue

                if file.get("is_vendor", False) or file.get("is_generated", False):
                    continue

                length = file.get("length_bytes") or 0
                if length < 200 or length > 100_000:
                    continue

                matching_files.append({
                    "repo_name": repository.get("repo_name"),
                    "path": file.get("path"),
                    "language": language,
                    "blob_id": file.get("blob_id"),
                    "src_encoding": file.get("src_encoding"),
                    "length_bytes": length,
                    "detected_licenses": file.get("detected_licenses", []),
                    "license_type": file.get("license_type"),
                })

                if len(matching_files) >= MAX_FILES_PER_REPO:
                    break

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
