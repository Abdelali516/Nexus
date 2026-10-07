import hashlib
import json
import random
from pathlib import Path


TOKENIZER_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = TOKENIZER_DIR / "data" / "nexus_tools.txt"

TARGET_CHARS = 10_000_000
SEED = 42
ERROR_RATE = 0.15  # fraction of tool calls that fail

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
    "logs/training.log",
    "C:\\Users\\abdel\\Projects\\nexus\\train.py",
    "D:\\work\\notes\\todo.txt",
    "tests/test_model.py",
    "configs/settings.yaml",
    "src/utils.c"
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

ERROR_MESSAGES = {
    "FileNotFoundError": "[Errno 2] No such file or directory: 'data/train.json'",
    "ValueError": "invalid literal for int() with base 10: 'abc'",
    "TypeError": "unsupported operand type(s) for +: 'int' and 'str'",
    "RuntimeError": "CUDA out of memory. Tried to allocate 2.00 GiB",
    "KeyError": "'learning_rate'",
    "SyntaxError": "invalid syntax",
    "ModuleNotFoundError": "No module named 'torch'"
}


SEARCH_QUERIES = [
    "how does attention work in transformers",
    "Python list comprehension documentation",
    "C memory management",
    "PyTorch mixed precision training",
    "Git rebase tutorial",
    "BPE tokenizer implementation",
    "Docker volumes explained",
    "SQL join types",
    "how to read a file in Python",
    "CUDA kernel launch parameters"
]


PRE_TEXT = [
    "",
    "",
    "I'll check that now.",
    "Let me do that.",
    "Sure, working on it.",
    "One moment."
]


# Settings used for file_edit examples: (key, old value, new value)
SETTINGS = [
    ("learning_rate", "0.001", "0.0003"),
    ("batch_size", "8", "16"),
    ("num_epochs", "3", "10"),
    ("max_length", "512", "1024"),
    ("dropout", "0.1", "0.0"),
    ("seed", "0", "42"),
    ("debug", "True", "False")
]


# Texts used for file_write examples
WRITE_TEXTS = [
    "print('Hello, world!')",
    "TODO: finish the tokenizer tests",
    "# Notes\n\nRemember to update the README.",
    "def add(a, b):\n    return a + b",
    "batch_size = 16\nlearning_rate = 0.0003"
]


EVENTS = [
    "Project meeting",
    "Dentist appointment",
    "Code review",
    "Lunch with Sara",
    "Study session",
    "Call with the client"
]


EMAIL_SUBJECTS = [
    "Project update",
    "Meeting notes",
    "Invoice for October",
    "Question about the report",
    "Schedule for next week"
]


def hexid(n=4):
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def tool_call(name, arguments):
    return dumps(
        {
            "name": name,
            "arguments": arguments
        }
    )


def tool_result(result):
    return dumps(
        {
            "status": "success",
            "result": result
        }
    )


def tool_error(message):
    return dumps(
        {
            "status": "error",
            "error": message
        }
    )


def make_example(user_text, tool_name, arguments, result, response,error=None, error_response=None):
    """Builds one example in the exact format the model will use.

    If `error` is given, the call fails ERROR_RATE of the time and the
    assistant answers with `error_response` instead of `response`.
    """
    assert tool_name in TOOLS

    if error is not None and random.random() < ERROR_RATE:
        result_text = tool_error(error)
        response = error_response
    else:
        result_text = tool_result(result)

    return (
        f"<|user|>{user_text}<|end_turn|>"
        f"<|assistant|>{random.choice(PRE_TEXT)}"
        f"<|tool_call|>{tool_call(tool_name, arguments)}<|end_tool_call|>"
        f"<|tool_result|>{result_text}<|end_tool_result|>"
        f"<|assistant|>{response}<|end_turn|>"
    )


def error_for(prefix=""):
    """Picks one error type from ERRORS and returns (message, short text)."""
    name = random.choice(ERRORS)
    message = f"{name}: {ERROR_MESSAGES[name]}"
    return message, f"That failed with {name}: {ERROR_MESSAGES[name]}."


def file_content(path):
    ext = path.rsplit(".", 1)[-1]
    key, old, new = random.choice(SETTINGS)

    if ext == "py":
        return f"import json\n\n{key} = {old}\n\nprint({key})\n"
    if ext == "c":
        return "#include <stdio.h>\n\nint main(void) {\n    printf(\"hello\\n\");\n    return 0;\n}\n"
    if ext in ("json",):
        return json.dumps({key: float(old) if old.replace('.', '').isdigit() else old, "name": "nexus"}, indent=2)
    if ext == "yaml":
        return f"{key}: {old}\nname: nexus\n"
    if ext == "md":
        return f"# {random.choice(TOPICS).title()}\n\nNotes on {random.choice(TOPICS)}.\n"
    if ext == "log":
        return (
            f"2026-10-0{random.randint(1, 7)} 12:{random.randint(10, 59)}:01 INFO started\n"
            f"2026-10-0{random.randint(1, 7)} 12:{random.randint(10, 59)}:07 {random.choice(['INFO', 'WARNING', 'ERROR'])} {random.choice(['loaded config', 'step 100 loss 2.31', 'connection reset'])}\n"
        )
    return f"{random.choice(TOPICS)} notes.\n"


def command_output(command):
    if command == "python train.py":
        losses = sorted((round(random.uniform(0.5, 4.0), 4) for _ in range(3)), reverse=True)
        return "\n".join(f"epoch {i + 1}/3 loss: {l}" for i, l in enumerate(losses))
    if command == "python inference.py":
        return f"Loaded model.\nGenerated {random.choice([64, 128, 256])} tokens in {round(random.uniform(0.8, 6.0), 2)}s"
    if command == "git status":
        return "On branch main\nYour branch is up to date with 'origin/main'.\n\nnothing to commit, working tree clean"
    if command == "git pull":
        return "Already up to date."
    if command == "git checkout main":
        return "Already on 'main'"
    if command == "pytest tests/":
        n, secs = random.randint(3, 40), round(random.uniform(0.2, 9.0), 2)
        return f"collected {n} items\n\n{'.' * n}\n\n===== {n} passed in {secs}s ====="
    if command == "ls -la":
        names = random.sample(["main.py", "README.md", "config.json", "data", "src", "tests", "train.py"], 4)
        return "total 24\n" + "\n".join(
            f"-rw-r--r-- 1 user user {random.randint(120, 9000):>5} Oct  {random.randint(1, 7)} 1{random.randint(0, 8)}:{random.randint(10, 59)} {n}"
            for n in names)
    if command == "pwd":
        return random.choice(["/home/user/project", "/workspace/Nexus", "/content"])
    if command == "nvidia-smi":
        gpu = random.choice(["NVIDIA A100-SXM4-40GB", "NVIDIA RTX 4090", "NVIDIA T4"])
        return f"{gpu} | {random.randint(0, 900)}MiB / 40960MiB | {random.randint(0, 100)}% util"
    ver = f"2.{random.randint(0, 6)}.{random.randint(0, 3)}"
    return f"Collecting torch\nSuccessfully installed torch-{ver}"


def py_snippet():
    n, w = random.randint(3, 30), random.choice(["python", "tokenizer", "agent", "neural"])
    return random.choice([
        (f"print(sum(range({n})))", str(sum(range(n)))),
        (f"print([i**2 for i in range({min(n, 8)})])", str([i ** 2 for i in range(min(n, 8))])),
        (f"print('{w}'.upper())", w.upper()),
        (f"print(len('{w}'))", str(len(w))),
        (f"print('{w}'[::-1])", w[::-1]),
        ("print('Hello, world!')", "Hello, world!")
    ])


def generate_example():

    tool = random.choice(TOOLS)

    if tool == "web_search":
        query = random.choice(SEARCH_QUERIES)
        url = random.choice(URLS)
        return make_example(
            f"Search the web for {query}.",
            tool,
            {"query": query, "max_results": random.randint(3, 10)},
            {"results": [{"title": f"{query.capitalize()} - Documentation", "url": url}]},
            f"I found a result about {query}: {url}",
            error="Search service unavailable",
            error_response="The search failed because the service is unavailable."
        )

    if tool == "web_open":
        url = random.choice(URLS)
        site = url.split("/")[2]
        return make_example(
            random.choice([f"Open this webpage: {url}", f"Can you open {url}?"]),
            tool,
            {"url": url},
            {"url": url, "title": f"{site} home page"},
            f"I opened {url}. The page title is \"{site} home page\".",
            error="HTTP 404 Not Found",
            error_response="The page returned a 404 error, so it doesn't exist."
        )

    if tool == "web_extract":
        url = random.choice(URLS)
        topic = random.choice(TOPICS)
        return make_example(
            f"Extract what {url} says about {topic}.",
            tool,
            {"url": url, "query": topic},
            {"text": f"This page explains {topic} with examples."},
            f"The page explains {topic} with examples.",
            error="Connection timed out",
            error_response="The page took too long to respond."
        )

    if tool == "calculator":
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        operation = random.choice(["+", "-", "*", "/"])
        expression = f"{a}{operation}{b}"

        result = {
            "+": a + b,
            "-": a - b,
            "*": a * b,
            "/": round(a / b, 4)
        }[operation]

        return make_example(
            random.choice([f"Calculate {expression}.", f"What is {expression}?"]),
            tool,
            {"expression": expression},
            {"value": result},
            f"{expression} = {result}."
        )

    if tool == "python":
        code, output = py_snippet()
        error, error_response = error_for()
        return make_example(
            f"Run this Python code: `{code}`",
            tool,
            {"code": code},
            {"stdout": output},
            f"It printed: {output}",
            error=error,
            error_response=error_response
        )

    if tool == "terminal":
        command = random.choice(COMMANDS)
        output = command_output(command)
        error, error_response = error_for()
        return make_example(
            random.choice([f"Run the command `{command}`.", f"Please run {command}."]),
            tool,
            {"command": command},
            {"stdout": output, "exit_code": 0},
            f"The command finished. Output:\n{output}" if output else "The command finished with no output.",
            error=error,
            error_response=error_response
        )

    if tool == "file_search":
        path = random.choice(FILE_PATHS)
        topic = random.choice(TOPICS)
        return make_example(
            f"Search for information related to {topic} in {path}.",
            tool,
            {"query": topic, "path": path},
            {"matches": [{"path": path, "line": random.randint(1, 200)}]},
            f"I found a match for {topic} in {path}.",
            error=f"Path not found: {path}",
            error_response=f"I couldn't search because {path} doesn't exist."
        )

    if tool == "file_read":
        path = random.choice(FILE_PATHS)
        content = file_content(path)
        return make_example(
            random.choice([f"Read the file {path}.", f"What's in {path}?"]),
            tool,
            {"path": path},
            {"content": content},
            f"Here is the content of {path}:\n{content.strip()}",
            error=f"FileNotFoundError: No such file or directory: '{path}'",
            error_response=f"I couldn't read {path} because it doesn't exist."
        )

    if tool == "file_write":
        path = random.choice(FILE_PATHS)
        text = random.choice(WRITE_TEXTS)
        size = len(text.encode("utf-8"))
        return make_example(
            f"Write this to {path}:\n{text}",
            tool,
            {"path": path, "content": text},
            {"bytes_written": size},
            f"Saved {size} bytes to {path}.",
            error=f"PermissionError: Permission denied: '{path}'",
            error_response=f"I couldn't write to {path}: permission denied."
        )

    if tool == "file_edit":
        path = random.choice(FILE_PATHS)
        key, old, new = random.choice(SETTINGS)
        old_text, new_text = f"{key} = {old}", f"{key} = {new}"
        return make_example(
            f"In {path}, replace `{old_text}` with `{new_text}`.",
            tool,
            {
                "path": path,
                "old_text": old_text,
                "new_text": new_text
            },
            {"modified": True, "replacements": 1},
            f"Done. I replaced `{old_text}` with `{new_text}` in {path}.",
            error=f"old_text not found in {path}",
            error_response=f"I couldn't find `{old_text}` in {path}."
        )

    if tool == "pdf_read":
        topic = random.choice(TOPICS)
        name = topic.lower().replace(" ", "-")
        path = random.choice([f"documents/{name}.pdf", f"papers/{name}.pdf", f"reports/{name}.pdf"])
        pages = random.randint(2, 40)
        return make_example(
            f"Read {path} and tell me what it covers.",
            tool,
            {"path": path},
            {"pages": pages, "text": f"This document is an introduction to {topic}."},
            f"The PDF has {pages} pages and is an introduction to {topic}.",
            error=f"FileNotFoundError: No such file or directory: '{path}'",
            error_response=f"I couldn't find {path}."
        )

    if tool == "document_extract":
        topic = random.choice(TOPICS)
        name = topic.lower().replace(" ", "-")
        path = random.choice([f"documents/{name}.docx", f"reports/{name}.docx"])
        return make_example(
            f"Extract the text from {path}.",
            tool,
            {"path": path},
            {"text": f"These notes cover {topic}."},
            f"The document says: these notes cover {topic}.",
            error=f"FileNotFoundError: No such file or directory: '{path}'",
            error_response=f"I couldn't find {path}."
        )

    if tool == "memory_search":
        topic = random.choice(TOPICS)
        return make_example(
            f"What do you remember about {topic}?",
            tool,
            {"query": topic},
            {"matches": [{"id": f"mem_{hexid()}", "content": f"User is working on {topic}."}]},
            f"I remember that you are working on {topic}.",
            error="Memory store is unavailable",
            error_response="I couldn't access memory right now."
        )

    if tool == "memory_save":
        topic = random.choice(TOPICS)
        return make_example(
            f"Remember that I'm working on {topic}.",
            tool,
            {"content": f"User is working on {topic}."},
            {"id": f"mem_{hexid()}", "saved": True},
            f"Saved. I'll remember that you are working on {topic}.",
            error="Memory store is unavailable",
            error_response="I couldn't save that because memory is unavailable."
        )

    if tool == "memory_update":
        memory_id = f"mem_{hexid()}"
        topic = random.choice(TOPICS)
        return make_example(
            f"Update memory {memory_id}: I'm now working on {topic}.",
            tool,
            {
                "id": memory_id,
                "content": f"User is working on {topic}."
            },
            {"id": memory_id, "updated": True},
            f"Updated memory {memory_id}.",
            error=f"Memory not found: {memory_id}",
            error_response=f"I couldn't find memory {memory_id}."
        )

    if tool == "memory_delete":
        memory_id = f"mem_{hexid()}"
        return make_example(
            f"Delete memory {memory_id}.",
            tool,
            {"id": memory_id},
            {"id": memory_id, "deleted": True},
            f"Deleted memory {memory_id}.",
            error=f"Memory not found: {memory_id}",
            error_response=f"I couldn't find memory {memory_id}."
        )

    if tool == "retrieve":
        topic = random.choice(TOPICS)
        top_k = random.randint(3, 10)
        n = random.randint(2, 3)
        scores = sorted((round(random.uniform(0.5, 0.98), 2) for _ in range(n)), reverse=True)
        return make_example(
            f"Retrieve the top {top_k} documents about {topic}.",
            tool,
            {"query": topic, "top_k": top_k},
            {"documents": [{"id": f"doc_{hexid()}", "score": s, "text": f"A guide to {topic}."} for s in scores]},
            f"I retrieved {n} documents. The best match scored {scores[0]}.",
            error="Knowledge base is unavailable",
            error_response="I couldn't reach the knowledge base."
        )

    if tool == "rerank":
        topics = random.sample(TOPICS, 3)
        query = random.choice(topics)
        documents = [f"A guide to {t}." for t in topics]
        best = topics.index(query)
        order = [best] + [i for i in range(3) if i != best]
        return make_example(
            f"Rank these documents by relevance to {query}: " + " | ".join(documents),
            tool,
            {
                "query": query,
                "documents": documents
            },
            {"ranked": [{"index": i, "score": s} for i, s in zip(order, [0.93, 0.41, 0.22])]},
            f"Document {best + 1} is the most relevant to {query}."
        )

    if tool == "git":
        command = random.choice(["status", "pull", "checkout main"])
        output = command_output(f"git {command}")
        return make_example(
            f"Run git {command}.",
            tool,
            {"command": command},
            {"output": output},
            f"Result:\n{output}",
            error="fatal: not a git repository (or any of the parent directories): .git",
            error_response="This folder isn't a git repository."
        )

    if tool == "code_search":
        query = random.choice([
            "find the attention implementation",
            "find the tokenizer class",
            "find the training loop",
            "find the optimizer setup",
            "find the model configuration"
        ])
        path = random.choice(["src/model.py", "src/tokenizer.py", "src/train.py", "src/config.py"])

        return make_example(
            f"Search the codebase to {query}.",
            tool,
            {"query": query},
            {"matches": [{"path": path, "line": random.randint(1, 300)}]},
            f"I found it in {path}.",
            error="Code index not found",
            error_response="I couldn't search because the code index is missing."
        )

    if tool == "run_tests":
        secs = round(random.uniform(0.2, 9.0), 2)
        if random.random() < ERROR_RATE:
            failed = random.randint(1, 3)
            passed = random.randint(2, 30)
            names = [f"test_{random.choice(['model', 'tokenizer', 'loader'])}_{random.choice(['basic', 'empty', 'large'])}" for _ in range(failed)]
            return make_example(
                "Run the project tests.",
                tool,
                {"path": "tests/"},
                {"passed": passed, "failed": failed, "failed_tests": names, "duration": secs},
                f"{failed} test(s) failed: {', '.join(names)}. {passed} passed."
            )
        passed = random.randint(3, 40)
        return make_example(
            "Run the project tests.",
            tool,
            {"path": "tests/"},
            {"passed": passed, "failed": 0, "duration": secs},
            f"All {passed} tests passed in {secs}s."
        )

    if tool == "calendar":
        day = random.choice(["today", "tomorrow", "Monday", "Friday"])
        events = random.sample(EVENTS, random.randint(0, 2))
        listing = [{"title": e, "time": f"{random.randint(8, 18):02d}:{random.choice(['00', '30'])}"} for e in events]
        response = (
            "You have " + ", ".join(f"{e['title']} at {e['time']}" for e in listing) + f" {day}."
            if listing else f"You have nothing scheduled {day}."
        )
        return make_example(
            f"What's on my calendar {day}?",
            tool,
            {"date": day},
            {"events": listing},
            response,
            error="Calendar service unavailable",
            error_response="I couldn't reach your calendar."
        )

    if tool == "email":
        limit = random.choice([3, 5, 10])
        messages = [{"id": f"msg_{hexid()}", "subject": random.choice(EMAIL_SUBJECTS)} for _ in range(random.randint(1, 3))]
        return make_example(
            f"Show my last {limit} emails.",
            tool,
            {"action": "list", "limit": limit},
            {"messages": messages},
            f"You have {len(messages)} recent emails. The newest is \"{messages[0]['subject']}\".",
            error="Mail server connection failed",
            error_response="I couldn't connect to your mail server."
        )


def create_corpus():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    examples = 0
    duplicates = 0
    streak = 0
    seen = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        while total_chars < TARGET_CHARS:

            example = generate_example()

            key = hashlib.md5(example.encode("utf-8")).digest()
            if key in seen:
                duplicates += 1
                streak += 1
                if streak > 5000:
                    print("Stopping: too many duplicates in a row. Add more variety.")
                    break
                continue
            streak = 0
            seen.add(key)

            # Stop on a whole example, never cut one in the middle.
            if total_chars + len(example) + 1 > TARGET_CHARS:
                break

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
    print(f"Examples: {examples:,} (duplicates skipped: {duplicates:,})")
    print(f"Characters: {total_chars:,}")
    print(f"File: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_corpus()