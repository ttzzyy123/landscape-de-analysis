from pathlib import Path
import json
import re
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "external"
    / "LLaMEA-paper-ela"
    / "XAI-liacs-LLaMEA-6d8b3c1"
)

NOTEBOOK_PATH = EXTRACTED_DIR / "function-definitions.ipynb"

OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "task_A4_verify_generated_files.txt"


def log(lines, text=""):
    print(text)
    lines.append(text)


def main():
    lines = []

    log(lines, "=" * 80)
    log(lines, "NEW FUNCTION TASK A4: VERIFY GENERATED OUTPUT FILES")
    log(lines, "=" * 80)
    log(lines, f"Project root:  {PROJECT_ROOT}")
    log(lines, f"Extracted dir: {EXTRACTED_DIR}")
    log(lines, f"Notebook:      {NOTEBOOK_PATH}")
    log(lines, f"Summary file:  {SUMMARY_FILE}")

    if not EXTRACTED_DIR.exists():
        log(lines, "[ERROR] Extracted directory not found.")
        SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
        return

    # ------------------------------------------------------------
    # 1. List all files and extensions
    # ------------------------------------------------------------
    all_files = [p for p in EXTRACTED_DIR.rglob("*") if p.is_file()]
    ext_counter = Counter(p.suffix.lower() if p.suffix else "<no_ext>" for p in all_files)

    log(lines, "\n" + "=" * 80)
    log(lines, "1. FILE EXTENSION SUMMARY")
    log(lines, "=" * 80)
    log(lines, f"Total files: {len(all_files)}")
    for ext, count in ext_counter.most_common():
        log(lines, f"{ext:<12} {count}")

    # ------------------------------------------------------------
    # 2. List all jsonl files
    # ------------------------------------------------------------
    jsonl_files = sorted(EXTRACTED_DIR.rglob("*.jsonl"))

    log(lines, "\n" + "=" * 80)
    log(lines, "2. ALL JSONL FILES")
    log(lines, "=" * 80)
    log(lines, f"Found {len(jsonl_files)} jsonl files")
    for p in jsonl_files:
        rel = p.relative_to(EXTRACTED_DIR)
        log(lines, f"- {rel} ({p.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------
    # 3. Extract expected exp-*.jsonl files from notebook
    # ------------------------------------------------------------
    expected_files = []

    if NOTEBOOK_PATH.exists():
        nb_text = NOTEBOOK_PATH.read_text(encoding="utf-8", errors="replace")
        expected_files = sorted(set(re.findall(r'exp-[^"]+?\.jsonl', nb_text)))

    log(lines, "\n" + "=" * 80)
    log(lines, "3. EXPECTED EXP-*.JSONL FILES REFERENCED BY NOTEBOOK")
    log(lines, "=" * 80)
    log(lines, f"Expected files found in notebook: {len(expected_files)}")

    for f in expected_files:
        expected_path = EXTRACTED_DIR / "outputs" / f
        status = "FOUND" if expected_path.exists() else "MISSING"
        log(lines, f"[{status}] outputs/{f}")

    # ------------------------------------------------------------
    # 4. Search actual file names containing exp-11
    # ------------------------------------------------------------
    exp_files = sorted([p for p in all_files if "exp-11" in p.name])

    log(lines, "\n" + "=" * 80)
    log(lines, "4. ACTUAL FILES CONTAINING exp-11")
    log(lines, "=" * 80)
    log(lines, f"Found {len(exp_files)} files")
    for p in exp_files:
        log(lines, f"- {p.relative_to(EXTRACTED_DIR)}")

    # ------------------------------------------------------------
    # 5. Search text files for generated-code indicators
    # ------------------------------------------------------------
    indicators = [
        '"code"',
        'entry["code"]',
        'cls = globals()',
        'func = cls(dim=dim).f',
        'def f(',
        'class ',
    ]

    candidate_suffixes = {
        ".py", ".ipynb", ".json", ".jsonl", ".txt", ".md"
    }

    log(lines, "\n" + "=" * 80)
    log(lines, "5. TEXT SEARCH FOR CODE INDICATORS")
    log(lines, "=" * 80)

    for indicator in indicators:
        log(lines, "\n" + "-" * 80)
        log(lines, f"Indicator: {indicator}")
        log(lines, "-" * 80)

        hits = []
        for p in all_files:
            if p.suffix.lower() not in candidate_suffixes:
                continue

            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if indicator in text:
                hits.append(p)

        log(lines, f"Hits: {len(hits)}")
        for p in hits[:30]:
            log(lines, f"- {p.relative_to(EXTRACTED_DIR)}")

    # ------------------------------------------------------------
    # 6. Conclusion
    # ------------------------------------------------------------
    missing_count = sum(
        1 for f in expected_files
        if not (EXTRACTED_DIR / "outputs" / f).exists()
    )

    log(lines, "\n" + "=" * 80)
    log(lines, "TASK A4 CONCLUSION")
    log(lines, "=" * 80)

    if expected_files and missing_count == len(expected_files):
        log(lines, "All exp-*.jsonl files referenced by function-definitions.ipynb are missing.")
        log(lines, "The current package contains the visualization notebook but not the generated function output jsonl files.")
        log(lines, "Next step: obtain the missing exp-*.jsonl files or use another package/repository version that includes them.")
    elif missing_count > 0:
        log(lines, f"{missing_count}/{len(expected_files)} expected exp-*.jsonl files are missing.")
        log(lines, "Some generated functions may still be available.")
    else:
        log(lines, "All expected exp-*.jsonl files are present. We can proceed to Task B extraction.")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    log(lines, "\nSummary written to:")
    log(lines, str(SUMMARY_FILE))


if __name__ == "__main__":
    main()