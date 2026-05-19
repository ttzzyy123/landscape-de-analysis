from pathlib import Path
import ast
import re
import json
import textwrap


# ============================================================
# New Function Task B
# Inspect generation entry points in the LLaMEA package
#
# Goal:
# 1. Identify files/classes/functions related to generation
# 2. Find how LLaMEA is initialized and run
# 3. Find whether OpenAI/API keys are required
# 4. Find logger/output JSONL writing logic
# 5. Find generated-function code format
# 6. Prepare for Task C: generate one function
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "external"
    / "LLaMEA-paper-ela"
    / "XAI-liacs-LLaMEA-6d8b3c1"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "task_B_generation_entrypoints_summary.txt"


TARGET_FILES = [
    "README.md",
    "pyproject.toml",
    "ELA.py",
    "test_functions.py",
    "plot.py",
    "llamea/llamea.py",
    "llamea/solution.py",
    "llamea/llm.py",
    "llamea/loggers.py",
    "llamea/utils.py",
    "tests/test_evolution.py",
    "tests/test_algorithm_generation.py",
    "tests/test_individual.py",
    "tests/test_initialization.py",
    "tests/test_llm.py",
    "tests/test_niching.py",
]


SEARCH_PATTERNS = [
    "OpenAI",
    "OPENAI_API_KEY",
    "api_key",
    "gpt",
    "LLM",
    "LLaMEA",
    "run",
    "evolve",
    "generate",
    "mutate",
    "recombine",
    "evaluate",
    "fitness",
    "jsonl",
    "output",
    "logger",
    "code",
    "name",
    "class",
    "problem",
    "landscape",
    "ELA",
    "sharing",
    "niching",
    "dim",
    "bounds",
]


def log(lines, text=""):
    print(text)
    lines.append(text)


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR reading file: {e}]"


def preview(text, n=1000):
    text = text.replace("\n", "\\n")
    return text[:n]


def get_ast_summary(path):
    result = {
        "classes": [],
        "functions": [],
        "imports": [],
        "errors": [],
    }

    text = read_text(path)

    try:
        tree = ast.parse(text)
    except Exception as e:
        result["errors"].append(str(e))
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = [a.arg for a in item.args.args]
                    methods.append({
                        "name": item.name,
                        "args": args,
                        "lineno": item.lineno,
                    })

            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    bases.append(str(type(base)))

            result["classes"].append({
                "name": node.name,
                "bases": bases,
                "lineno": node.lineno,
                "methods": methods,
            })

        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            result["functions"].append({
                "name": node.name,
                "args": args,
                "lineno": node.lineno,
            })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                result["imports"].append(f"{module}.{alias.name}")

    return result


def grep_context(text, pattern, context=350, max_hits=8):
    hits = []
    try:
        iterator = re.finditer(re.escape(pattern), text, flags=re.IGNORECASE)
    except Exception:
        return hits

    for match in iterator:
        start = max(0, match.start() - context)
        end = min(len(text), match.end() + context)
        snippet = text[start:end]
        hits.append(snippet)
        if len(hits) >= max_hits:
            break

    return hits


def inspect_file(lines, rel_path):
    path = EXTRACTED_DIR / rel_path

    log(lines, "\n" + "=" * 80)
    log(lines, f"FILE: {rel_path}")
    log(lines, "=" * 80)

    if not path.exists():
        log(lines, "[MISSING]")
        return

    log(lines, f"Path: {path}")
    log(lines, f"Size: {path.stat().st_size / 1024:.1f} KB")

    text = read_text(path)

    log(lines, "\n--- Text preview ---")
    log(lines, preview(text, 1500))

    if path.suffix == ".py":
        ast_summary = get_ast_summary(path)

        log(lines, "\n--- AST imports ---")
        imports = ast_summary["imports"]
        if imports:
            for imp in imports[:80]:
                log(lines, f"  - {imp}")
        else:
            log(lines, "  No imports found or parse failed.")

        if ast_summary["errors"]:
            log(lines, "\n--- AST parse errors ---")
            for err in ast_summary["errors"]:
                log(lines, f"  - {err}")

        log(lines, "\n--- AST classes ---")
        if ast_summary["classes"]:
            for cls in ast_summary["classes"]:
                log(lines, f"Class {cls['name']} at line {cls['lineno']}, bases={cls['bases']}")
                for m in cls["methods"][:30]:
                    log(lines, f"  method {m['name']}({', '.join(m['args'])}) at line {m['lineno']}")
        else:
            log(lines, "  No classes found.")

        log(lines, "\n--- AST top-level functions ---")
        funcs = ast_summary["functions"]
        if funcs:
            for f in funcs[:80]:
                log(lines, f"  function {f['name']}({', '.join(f['args'])}) at line {f['lineno']}")
        else:
            log(lines, "  No functions found.")

    log(lines, "\n--- Pattern hits ---")
    for pattern in SEARCH_PATTERNS:
        count = len(re.findall(re.escape(pattern), text, flags=re.IGNORECASE))
        if count > 0:
            log(lines, f"{pattern:<20} {count}")

    important_context_patterns = [
        "OPENAI_API_KEY",
        "OpenAI",
        "jsonl",
        "logger",
        "write",
        "code",
        "name",
        "generate",
        "mutate",
        "evaluate",
        "fitness",
        "class",
        "dim",
    ]

    log(lines, "\n--- Important contexts ---")
    for pattern in important_context_patterns:
        hits = grep_context(text, pattern, context=300, max_hits=3)
        if not hits:
            continue

        log(lines, "\n" + "-" * 60)
        log(lines, f"Pattern: {pattern}")
        log(lines, "-" * 60)

        for i, h in enumerate(hits, start=1):
            log(lines, f"Hit {i}: {preview(h, 900)}")


def scan_repository_for_generation_clues(lines):
    log(lines, "\n" + "=" * 80)
    log(lines, "REPOSITORY-WIDE SEARCH FOR GENERATION CLUES")
    log(lines, "=" * 80)

    candidate_suffixes = {".py", ".md", ".toml", ".ipynb", ".json", ".jsonl"}
    all_files = [
        p for p in EXTRACTED_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in candidate_suffixes
    ]

    repo_patterns = [
        "OPENAI_API_KEY",
        "OpenAI",
        "AsyncOpenAI",
        "ChatOpenAI",
        "client.chat",
        "jsonl",
        "log",
        "logger",
        "code",
        "entry[\"code\"]",
        "class ",
        "def f(",
        ".f",
        "fitness",
        "evaluate",
        "mutate",
        "evolve",
        "run(",
        "LLaMEA(",
        "Individual",
        "Solution",
        "prompt",
        "system_prompt",
        "problem",
        "landscape",
        "ELA",
    ]

    for pattern in repo_patterns:
        log(lines, "\n" + "-" * 80)
        log(lines, f"Pattern: {pattern}")
        log(lines, "-" * 80)

        hits = []
        for p in all_files:
            try:
                text = read_text(p)
            except Exception:
                continue

            if pattern in text:
                rel = p.relative_to(EXTRACTED_DIR)
                count = text.count(pattern)
                hits.append((rel, count))

        if not hits:
            log(lines, "No hits.")
            continue

        for rel, count in hits[:50]:
            log(lines, f"- {rel} | count={count}")


def inspect_pyproject(lines):
    log(lines, "\n" + "=" * 80)
    log(lines, "PYPROJECT DEPENDENCY / ENTRY POINT CHECK")
    log(lines, "=" * 80)

    path = EXTRACTED_DIR / "pyproject.toml"
    if not path.exists():
        log(lines, "[MISSING] pyproject.toml")
        return

    text = read_text(path)
    log(lines, text[:5000])


def main():
    lines = []

    log(lines, "=" * 80)
    log(lines, "NEW FUNCTION TASK B: INSPECT GENERATION ENTRYPOINTS")
    log(lines, "=" * 80)
    log(lines, f"Project root:  {PROJECT_ROOT}")
    log(lines, f"Extracted dir: {EXTRACTED_DIR}")
    log(lines, f"Summary file:  {SUMMARY_FILE}")

    if not EXTRACTED_DIR.exists():
        log(lines, "[ERROR] Extracted directory not found.")
        SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
        return

    inspect_pyproject(lines)

    log(lines, "\n" + "=" * 80)
    log(lines, "TARGET FILE INSPECTION")
    log(lines, "=" * 80)

    for rel_path in TARGET_FILES:
        inspect_file(lines, rel_path)

    scan_repository_for_generation_clues(lines)

    log(lines, "\n" + "=" * 80)
    log(lines, "TASK B MANUAL INTERPRETATION CHECKLIST")
    log(lines, "=" * 80)

    checklist = [
        "Which file defines the main LLaMEA class?",
        "Which class/function starts the evolutionary loop?",
        "Which class stores candidate code/name/fitness?",
        "Which file calls the LLM API?",
        "Which environment variable/API key is required?",
        "Which logger writes jsonl outputs?",
        "What is the generated function class format?",
        "Can we instantiate a generated class as cls(dim=dim).f?",
        "Which test file provides the shortest runnable example?",
        "Can Task C run without actual API by using a mock LLM or existing test?",
    ]

    for item in checklist:
        log(lines, f"- {item}")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    log(lines, "\nSummary written to:")
    log(lines, str(SUMMARY_FILE))


if __name__ == "__main__":
    main()