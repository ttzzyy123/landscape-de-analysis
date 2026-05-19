from pathlib import Path
import json
import zipfile


# ============================================================
# New Function Task A
# Inspect LLaMEA-paper-ela reproducibility package
#
# Goal:
# 1. Check whether the package has been downloaded and extracted
# 2. Check important files
# 3. Inspect jsonl files
# 4. Write a summary file for the next extraction step
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PACKAGE_DIR = PROJECT_ROOT / "external" / "LLaMEA-paper-ela"
ZIP_FILE = PACKAGE_DIR / "LLaMEA-paper-ela.zip"

# Expected extracted folder name from Zenodo package
EXTRACTED_DIR = PACKAGE_DIR / "XAI-liacs-LLaMEA-6d8b3c1"

OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "task_A_inspection_summary.txt"


IMPORTANT_FILES = [
    "README.md",
    "ELA.py",
    "function-definitions.ipynb",
    "ela.ipynb",
    "test_functions.py",
    "pyproject.toml",
    "outputs/bbob.jsonl",
    "landscapes/bbob.jsonl",
    "llamea/llamea.py",
    "llamea/solution.py",
    "llamea/llm.py",
    "llamea/utils.py",
]


def write_line(lines, text=""):
    print(text)
    lines.append(text)


def inspect_zip(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. ZIP FILE CHECK")
    write_line(lines, "=" * 80)

    if ZIP_FILE.exists():
        write_line(lines, f"[OK] Zip file found: {ZIP_FILE}")
        write_line(lines, f"     Size: {ZIP_FILE.stat().st_size / 1024 / 1024:.2f} MB")

        try:
            with zipfile.ZipFile(ZIP_FILE, "r") as zf:
                names = zf.namelist()
                write_line(lines, f"     Number of files inside zip: {len(names)}")
                write_line(lines, "     First 10 entries:")
                for name in names[:10]:
                    write_line(lines, f"       - {name}")
        except Exception as e:
            write_line(lines, f"[WARNING] Could not inspect zip content: {e}")
    else:
        write_line(lines, f"[MISSING] Zip file not found: {ZIP_FILE}")


def inspect_extracted_dir(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. EXTRACTED DIRECTORY CHECK")
    write_line(lines, "=" * 80)

    if EXTRACTED_DIR.exists():
        write_line(lines, f"[OK] Extracted directory found: {EXTRACTED_DIR}")

        top_items = sorted(EXTRACTED_DIR.iterdir(), key=lambda p: p.name.lower())
        write_line(lines, f"     Number of top-level items: {len(top_items)}")
        write_line(lines, "     Top-level items:")
        for item in top_items:
            item_type = "DIR " if item.is_dir() else "FILE"
            write_line(lines, f"       [{item_type}] {item.name}")
    else:
        write_line(lines, f"[MISSING] Extracted directory not found: {EXTRACTED_DIR}")
        write_line(lines, "          Please run:")
        write_line(lines, f"          cd {PACKAGE_DIR}")
        write_line(lines, "          unzip LLaMEA-paper-ela.zip")


def inspect_important_files(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "3. IMPORTANT FILES CHECK")
    write_line(lines, "=" * 80)

    if not EXTRACTED_DIR.exists():
        write_line(lines, "[SKIPPED] Extracted directory does not exist.")
        return

    for rel_path in IMPORTANT_FILES:
        path = EXTRACTED_DIR / rel_path
        if path.exists():
            size_kb = path.stat().st_size / 1024
            write_line(lines, f"[OK]      {rel_path:<35} {size_kb:>10.1f} KB")
        else:
            write_line(lines, f"[MISSING] {rel_path}")


def inspect_readme(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "4. README PREVIEW")
    write_line(lines, "=" * 80)

    readme_path = EXTRACTED_DIR / "README.md"

    if not readme_path.exists():
        write_line(lines, "[MISSING] README.md not found.")
        return

    try:
        text = readme_path.read_text(encoding="utf-8", errors="replace")
        preview_lines = text.splitlines()[:40]

        write_line(lines, f"[OK] README found: {readme_path}")
        write_line(lines, "First 40 lines:")
        for line in preview_lines:
            write_line(lines, f"  {line}")
    except Exception as e:
        write_line(lines, f"[ERROR] Could not read README.md: {e}")


def inspect_jsonl_file(lines, rel_path, max_rows=3):
    write_line(lines, "\n" + "-" * 80)
    write_line(lines, f"JSONL INSPECTION: {rel_path}")
    write_line(lines, "-" * 80)

    jsonl_path = EXTRACTED_DIR / rel_path

    if not jsonl_path.exists():
        write_line(lines, f"[MISSING] {rel_path}")
        return

    write_line(lines, f"[OK] Found: {jsonl_path}")
    write_line(lines, f"     Size: {jsonl_path.stat().st_size / 1024:.1f} KB")

    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            rows = []
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except json.JSONDecodeError as e:
                    write_line(lines, f"[ERROR] Could not parse line {i + 1}: {e}")
                    write_line(lines, f"        Raw line preview: {line[:300]}")
                    continue

        write_line(lines, f"     Parsed preview rows: {len(rows)}")

        for idx, obj in enumerate(rows, start=1):
            write_line(lines, f"\n     Row {idx}:")
            write_line(lines, f"       Type: {type(obj)}")

            if isinstance(obj, dict):
                keys = list(obj.keys())
                write_line(lines, f"       Keys ({len(keys)}): {keys}")

                for key in keys:
                    value = obj.get(key)
                    value_type = type(value).__name__

                    if isinstance(value, str):
                        preview = value.replace("\n", "\\n")[:300]
                        write_line(lines, f"       - {key}: str, len={len(value)}, preview={preview}")
                    elif isinstance(value, (list, tuple)):
                        write_line(lines, f"       - {key}: {value_type}, len={len(value)}")
                    elif isinstance(value, dict):
                        write_line(lines, f"       - {key}: dict, keys={list(value.keys())[:20]}")
                    else:
                        write_line(lines, f"       - {key}: {value_type}, value={value}")
            else:
                write_line(lines, f"       Value preview: {str(obj)[:300]}")

    except Exception as e:
        write_line(lines, f"[ERROR] Could not inspect jsonl file: {e}")


def inspect_notebook_metadata(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "6. NOTEBOOK METADATA CHECK")
    write_line(lines, "=" * 80)

    notebook_path = EXTRACTED_DIR / "function-definitions.ipynb"

    if not notebook_path.exists():
        write_line(lines, "[MISSING] function-definitions.ipynb not found.")
        return

    write_line(lines, f"[OK] Notebook found: {notebook_path}")
    write_line(lines, f"     Size: {notebook_path.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        with notebook_path.open("r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        write_line(lines, f"     Number of notebook cells: {len(cells)}")

        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]

        write_line(lines, f"     Code cells: {len(code_cells)}")
        write_line(lines, f"     Markdown cells: {len(markdown_cells)}")

        write_line(lines, "\n     First 10 code cell previews:")
        shown = 0
        for i, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            source_preview = source.replace("\n", "\\n")[:500]
            write_line(lines, f"       Cell {i}: {source_preview}")

            shown += 1
            if shown >= 10:
                break

    except Exception as e:
        write_line(lines, f"[ERROR] Could not inspect notebook: {e}")


def main():
    lines = []

    write_line(lines, "=" * 80)
    write_line(lines, "NEW FUNCTION TASK A: INSPECT LLAMEA PACKAGE")
    write_line(lines, "=" * 80)

    write_line(lines, f"Project root:     {PROJECT_ROOT}")
    write_line(lines, f"Package dir:      {PACKAGE_DIR}")
    write_line(lines, f"Zip file:         {ZIP_FILE}")
    write_line(lines, f"Extracted dir:    {EXTRACTED_DIR}")
    write_line(lines, f"Summary output:   {SUMMARY_FILE}")

    inspect_zip(lines)
    inspect_extracted_dir(lines)
    inspect_important_files(lines)
    inspect_readme(lines)

    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "5. JSONL FILE CHECK")
    write_line(lines, "=" * 80)

    inspect_jsonl_file(lines, "outputs/bbob.jsonl", max_rows=3)
    inspect_jsonl_file(lines, "landscapes/bbob.jsonl", max_rows=3)

    inspect_notebook_metadata(lines)

    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "TASK A CONCLUSION")
    write_line(lines, "=" * 80)
    write_line(lines, "Use this summary to decide where generated function definitions are stored.")
    write_line(lines, "Likely candidates:")
    write_line(lines, "  1. function-definitions.ipynb")
    write_line(lines, "  2. outputs/bbob.jsonl")
    write_line(lines, "  3. landscapes/bbob.jsonl")
    write_line(lines, "  4. test_functions.py")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\nSummary written to:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()