from pathlib import Path
import json


# ============================================================
# New Function Task A2
# Scan all JSONL files in the LLaMEA package
#
# Goal:
# Find which JSONL files contain generated function definitions/code.
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

SUMMARY_FILE = OUTPUT_DIR / "task_A2_jsonl_scan_summary.txt"


SEARCH_DIRS = [
    EXTRACTED_DIR / "outputs",
    EXTRACTED_DIR / "landscapes",
]

CODE_LIKE_KEYWORDS = [
    "code",
    "function",
    "solution",
    "algorithm",
    "phenotype",
    "genotype",
    "response",
    "content",
    "text",
    "class",
    "imports",
]


def preview_value(value, max_len=300):
    if isinstance(value, str):
        return value.replace("\n", "\\n")[:max_len]
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:20]})"
    return repr(value)[:max_len]


def inspect_jsonl(path):
    result = {
        "path": str(path),
        "size_kb": path.stat().st_size / 1024,
        "n_preview_rows": 0,
        "keys": [],
        "code_like_keys": [],
        "preview": [],
        "error": None,
    }

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break

                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                result["n_preview_rows"] += 1

                if isinstance(obj, dict):
                    keys = list(obj.keys())
                    if not result["keys"]:
                        result["keys"] = keys

                    code_like = [
                        k for k in keys
                        if any(keyword.lower() in k.lower() for keyword in CODE_LIKE_KEYWORDS)
                    ]
                    result["code_like_keys"] = sorted(set(result["code_like_keys"] + code_like))

                    row_preview = {}
                    for k, v in obj.items():
                        row_preview[k] = {
                            "type": type(v).__name__,
                            "preview": preview_value(v),
                        }
                    result["preview"].append(row_preview)
                else:
                    result["preview"].append({
                        "__non_dict__": {
                            "type": type(obj).__name__,
                            "preview": preview_value(obj),
                        }
                    })

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("=" * 80)
    log("NEW FUNCTION TASK A2: SCAN JSONL FILES")
    log("=" * 80)
    log(f"Project root:  {PROJECT_ROOT}")
    log(f"Extracted dir: {EXTRACTED_DIR}")
    log(f"Summary file:  {SUMMARY_FILE}")

    all_jsonl_files = []

    for search_dir in SEARCH_DIRS:
        log("\n" + "=" * 80)
        log(f"Searching directory: {search_dir}")
        log("=" * 80)

        if not search_dir.exists():
            log(f"[MISSING] {search_dir}")
            continue

        jsonl_files = sorted(search_dir.glob("*.jsonl"))
        all_jsonl_files.extend(jsonl_files)

        log(f"Found {len(jsonl_files)} .jsonl files")
        for p in jsonl_files:
            log(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    log("\n" + "=" * 80)
    log("JSONL FIELD INSPECTION")
    log("=" * 80)

    candidate_files = []

    for path in all_jsonl_files:
        result = inspect_jsonl(path)

        rel_path = path.relative_to(EXTRACTED_DIR)

        log("\n" + "-" * 80)
        log(f"File: {rel_path}")
        log(f"Size: {result['size_kb']:.1f} KB")

        if result["error"]:
            log(f"[ERROR] {result['error']}")
            continue

        log(f"Preview rows parsed: {result['n_preview_rows']}")
        log(f"Keys: {result['keys']}")
        log(f"Code-like keys: {result['code_like_keys']}")

        if result["code_like_keys"]:
            candidate_files.append((rel_path, result["code_like_keys"]))

        for row_idx, row in enumerate(result["preview"], start=1):
            log(f"\n  Row {row_idx} preview:")
            for k, meta in row.items():
                log(f"    - {k}: {meta['type']} | {meta['preview']}")

    log("\n" + "=" * 80)
    log("LIKELY CANDIDATE FILES FOR GENERATED FUNCTION DEFINITIONS")
    log("=" * 80)

    if not candidate_files:
        log("No obvious candidate JSONL files found based on code-like keys.")
        log("Next step should inspect function-definitions.ipynb more deeply.")
    else:
        for rel_path, keys in candidate_files:
            log(f"- {rel_path}")
            log(f"  code-like keys: {keys}")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    log("\nSummary written to:")
    log(str(SUMMARY_FILE))


if __name__ == "__main__":
    main()