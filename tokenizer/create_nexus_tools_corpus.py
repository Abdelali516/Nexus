import json
import random
from pathlib import Path


TOKENIZER_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = TOKENIZER_DIR / "data" / "nexus_tools.txt"

TARGET_CHARS = 10_000_000
SEED = 42

random.seed(SEED)


TOOLS = [
    "web_search",
    "web_open",
    "web_extract",
    "calculator",
    "python",
    "terminal",
    "file_search",
    "file_read",
    "file_write",
    "file_edit",
    "pdf_read",
    "document_extract",
    "memory_search",
    "memory_save",
    "memory_update",
    "memory_delete",
    "retrieve",
    "rerank",
    "git",
    "code_search",
    "run_tests",
    "calendar",
    "email"
]


TOPICS = [
    "Python decorators",
    "machine learning",
    "transformer architecture",
    "C programming",
    "Linux commands",
    "Git branches",
    "neural networks",
    "retrieval augmented generation",
    "database indexing",
    "REST APIs",
    "JSON parsing",
    "CUDA programming",
    "Docker",
    "data structures",
    "operating systems"
]


URLS = [
    "https://docs.python.org/3/",
    "https://github.com/",
    "https://arxiv.org/",
    "https://pytorch.org/",
    "https://huggingface.co/",
    "https://developer.mozilla.org/"
]


FILE_PATHS = [
    "/home/user/project/main.py",
    "/workspace/Nexus/config.json",
    "/content/data/train.json",
    "src/model.py",
    "README.md",
    "data/dataset.json",
    "logs/training.log"
]


COMMANDS = [
    "python train.py",
    "python inference.py",
    "git status",
    "git pull",
    "git checkout main",
    "pytest tests/",
    "ls -la",
    "pwd",
    "nvidia-smi",
    "pip install torch"
]


ERRORS = [
    "FileNotFoundError",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "SyntaxError",
    "ModuleNotFoundError"
]


PYTHON_SNIPPETS = [
    "print('Hello, world!')",
    "x = sum(numbers)",
    "model.train()",
    "optimizer.zero_grad()",
    "loss.backward()",
    "result = json.loads(data)",
    "with open('data.json') as f: data = json.load(f)"
]


SEARCH_QUERIES = [
    "how does attention work in transformers",
    "Python list comprehension documentation",
    "C memory management",
    "PyTorch mixed precision training",
    "Git rebase tutorial",
    "BPE tokenizer implementation"
]


def tool_call(name, arguments):
    return json.dumps(
        {
            "name": name,
            "arguments": arguments
        },
        separators=(",", ":")
    )


def tool_result(result):
    return json.dumps(
        {
            "status": "success",
            "result": result
        },
        separators=(",", ":")
    )


def make_example(user_text, tool_name, arguments, result, response):
    return (
        f"<|user|> {user_text} "
        f"<|assistant|> <tool_call> {tool_call(tool_name, arguments)} </tool_call> "
        f"<tool_result> {tool_result(result)} </tool_result> "
        f"<|assistant|> {response}"
    )


def generate_example():

    tool = random.choice(TOOLS)

    if tool == "web_search":
        query = random.choice(SEARCH_QUERIES)
        return make_example(
            f"Search the web for {query}.",
            tool,
            {"query": query, "max_results": random.randint(3, 10)},
            {"results": [{"title": "Documentation", "url": random.choice(URLS)}]},
            "I found relevant results."
        )

    if tool == "web_open":
        url = random.choice(URLS)
        return make_example(
            f"Open this webpage: {url}",
            tool,
            {"url": url},
            {"url": url, "title": "Documentation page"},
            "The webpage was opened successfully."
        )

    if tool == "web_extract":
        url = random.choice(URLS)
        return make_example(
            f"Extract the useful information from {url}.",
            tool,
            {"url": url, "query": random.choice(TOPICS)},
            {"text": "Relevant information was extracted from the page."},
            "I extracted the relevant information."
        )

    if tool == "calculator":
        a = random.randint(1, 500)
        b = random.randint(1, 500)
        operation = random.choice(["+", "-", "*"])
        expression = f"{a}{operation}{b}"

        result = eval(expression)

        return make_example(
            f"Calculate {expression}.",
            tool,
            {"expression": expression},
            {"value": result},
            f"The result is {result}."
        )

    if tool == "python":
        code = random.choice(PYTHON_SNIPPETS)

        return make_example(
            "Run this Python code.",
            tool,
            {"code": code},
            {"stdout": "Execution completed successfully."},
            "The Python code ran successfully."
        )

    if tool == "terminal":
        command = random.choice(COMMANDS)

        return make_example(
            f"Run the command `{command}`.",
            tool,
            {"command": command},
            {"stdout": "Command completed successfully.", "exit_code": 0},
            "The command completed successfully."
        )

    if tool == "file_search":
        path = random.choice(FILE_PATHS)

        return make_example(
            f"Search for information related to {random.choice(TOPICS)} in {path}.",
            tool,
            {"query": random.choice(TOPICS), "path": path},
            {"matches": [path]},
            "I found a relevant file."
        )

    if tool == "file_read":
        path = random.choice(FILE_PATHS)

        return make_example(
            f"Read the file {path}.",
            tool,
            {"path": path},
            {"content": "Example file contents."},
            "The file was read successfully."
        )

    if tool == "file_write":
        path = random.choice(FILE_PATHS)

        return make_example(
            f"Write data to {path}.",
            tool,
            {"path": path, "content": "Example generated content."},
            {"bytes_written": random.randint(50, 500)},
            "The file was written successfully."
        )

    if tool == "file_edit":
        path = random.choice(FILE_PATHS)

        return make_example(
            f"Edit {path}.",
            tool,
            {
                "path": path,
                "old_text": "old value",
                "new_text": "new value"
            },
            {"modified": True},
            "The file was updated successfully."
        )

    if tool == "pdf_read":
        path = random.choice([
            "documents/research.pdf",
            "papers/transformers.pdf",
            "reports/report.pdf"
        ])

        return make_example(
            f"Read the PDF {path}.",
            tool,
            {"path": path},
            {"pages": 12, "text": "Extracted PDF text."},
            "The PDF was read successfully."
        )

    if tool == "document_extract":
        path = random.choice([
            "documents/report.docx",
            "documents/notes.docx",
            "documents/specification.docx"
        ])

        return make_example(
            f"Extract text from {path}.",
            tool,
            {"path": path},
            {"text": "Extracted document content."},
            "The document content was extracted."
        )

    if tool == "memory_search":
        query = random.choice(TOPICS)

        return make_example(
            f"Search memory for information about {query}.",
            tool,
            {"query": query},
            {"matches": [{"memory": "Previous relevant information."}]},
            "I found a relevant memory."
        )

    if tool == "memory_save":
        memory = random.choice(TOPICS)

        return make_example(
            "Save this information to memory.",
            tool,
            {"content": f"User is working on {memory}."},
            {"saved": True},
            "The information was saved."
        )

    if tool == "memory_update":
        return make_example(
            "Update the stored project information.",
            tool,
            {
                "query": "current project",
                "content": "The project is actively being developed."
            },
            {"updated": True},
            "The memory was updated."
        )

    if tool == "memory_delete":
        return make_example(
            "Delete the outdated memory.",
            tool,
            {"query": "outdated project information"},
            {"deleted": True},
            "The memory was deleted."
        )

    if tool == "retrieve":
        query = random.choice(TOPICS)

        return make_example(
            f"Retrieve documents about {query}.",
            tool,
            {"query": query, "top_k": random.randint(3, 10)},
            {"documents": ["document_1", "document_2", "document_3"]},
            "I retrieved the most relevant documents."
        )

    if tool == "rerank":
        query = random.choice(TOPICS)

        return make_example(
            f"Rerank these documents for {query}.",
            tool,
            {
                "query": query,
                "documents": ["document_1", "document_2", "document_3"]
            },
            {"ranked_documents": ["document_2", "document_1", "document_3"]},
            "The documents were reranked."
        )

    if tool == "git":
        command = random.choice([
            "status",
            "log",
            "diff",
            "branch",
            "pull",
            "checkout main"
        ])

        return make_example(
            f"Run git command: {command}.",
            tool,
            {"command": command},
            {"output": "Git command completed successfully."},
            "The Git operation completed."
        )

    if tool == "code_search":
        query = random.choice([
            "find the attention implementation",
            "find the tokenizer class",
            "find the training loop",
            "find the optimizer setup",
            "find the model configuration"
        ])

        return make_example(
            f"Search the codebase for {query}.",
            tool,
            {"query": query},
            {"matches": ["src/model.py", "src/train.py"]},
            "I found the relevant code."
        )

    if tool == "run_tests":
        return make_example(
            "Run the project tests.",
            tool,
            {"path": "tests/"},
            {"passed": random.randint(5, 20), "failed": 0},
            "All tests passed."
        )

    if tool == "calendar":
        return make_example(
            "Check my calendar for tomorrow.",
            tool,
            {"date": "tomorrow"},
            {"events": ["Project meeting at 10:00"]},
            "You have a project meeting tomorrow."
        )

    if tool == "email":
        return make_example(
            "Check my recent emails.",
            tool,
            {"action": "list", "limit": 5},
            {"messages": [{"subject": "Project update"}]},
            "I found your recent emails."
        )


def create_corpus():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    examples = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        while total_chars < TARGET_CHARS:

            example = generate_example()

            remaining = TARGET_CHARS - total_chars
            example = example[:remaining]

            f.write(example + "\n")

            total_chars += len(example) + 1
            examples += 1

            if examples % 10000 == 0:
                print(
                    f"{examples:,} examples | "
                    f"{total_chars:,}/{TARGET_CHARS:,} characters"
                )

    print()
    print("Nexus tools corpus created!")
    print(f"Examples: {examples:,}")
    print(f"Characters: {total_chars:,}")
    print(f"File: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_corpus()