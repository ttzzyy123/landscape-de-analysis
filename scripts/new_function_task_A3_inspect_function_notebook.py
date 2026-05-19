from pathlib import Path
import json
import re


# ============================================================
# New Function Task A3
# Deep inspect function-definitions.ipynb
#
# Goal:
# Find where generated function definitions/code are stored.
# ============================================================


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

SUMMARY_FILE = OUTPUT_DIR / "task_A3_function_notebook_inspection.txt"

KEYWORDS = [
    "def ",
    "class ",
    "evaluate",
    "objective",
    "fitness",
    "problem",
    "solution",
    "code",
    "phenotype",
    "genotype",
    "exp-11",
    "LLaMEA",
    "landscape",
    "Separable",
    "Multimodality",
    "Basins",
    "GlobalLocal",
    "Homogeneous",
    "sharing",
]


def safe_join_source(source):
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return str(source)


def extract_output_text(output):
    parts = []

    if "text" in output:
        text = output["text"]
        parts.append(safe_join_source(text))

    if "data" in output:
        data = output["data"]

        for mime_key in [
            "text/plain",
            "text/html",
            "application/json",
            "application/vnd.jupyter.widget-view+json",
        ]:
            if mime_key in data:
                parts.append(safe_join_source(data[mime_key]))

    if "ename" in output or "evalue" in output:
        parts.append(str(output.get("ename", "")))
        parts.append(str(output.get("evalue", "")))

    return "\n".join(parts)


def find_keyword_context(text, keyword, context=600, max_hits=10):
    hits = []
    pattern = re.escape(keyword)

    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(match.start() - context, 0)
        end = min(match.end() + context, len(text))
        snippet = text[start:end]
        hits.append(snippet)

        if len(hits) >= max_hits:
            break

    return hits


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("=" * 80)
    log("NEW FUNCTION TASK A3: INSPECT FUNCTION-DEFINITIONS NOTEBOOK")
    log("=" * 80)
    log(f"Project root:   {PROJECT_ROOT}")
    log(f"Notebook path:  {NOTEBOOK_PATH}")
    log(f"Summary file:   {SUMMARY_FILE}")

    if not NOTEBOOK_PATH.exists():
        log(f"[MISSING] Notebook not found: {NOTEBOOK_PATH}")
        SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
        return

    with NOTEBOOK_PATH.open("r", encoding="utf-8", errors="replace") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    log(f"\nNumber of cells: {len(cells)}")

    all_sources = []
    all_outputs = []

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = safe_join_source(cell.get("source", ""))

        all_sources.append(f"\n\n# ===== CELL {idx} SOURCE ({cell_type}) =====\n{source}")

        outputs = cell.get("outputs", [])
        for out_idx, output in enumerate(outputs):
            out_text = extract_output_text(output)
            if out_text.strip():
                all_outputs.append(
                    f"\n\n# ===== CELL {idx} OUTPUT {out_idx} =====\n{out_text}"
                )

    source_text = "\n".join(all_sources)
    output_text = "\n".join(all_outputs)
    full_text = source_text + "\n\n" + output_text

    log("\n" + "=" * 80)
    log("BASIC SIZE INFO")
    log("=" * 80)
    log(f"Source text length: {len(source_text):,} characters")
    log(f"Output text length: {len(output_text):,} characters")
    log(f"Full text length:   {len(full_text):,} characters")

    log("\n" + "=" * 80)
    log("CELL SUMMARY")
    log("=" * 80)

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = safe_join_source(cell.get("source", ""))
        outputs = cell.get("outputs", [])

        log(f"\nCell {idx}")
        log(f"  Type: {cell_type}")
        log(f"  Source length: {len(source):,}")
        log(f"  Outputs: {len(outputs)}")

        preview = source.replace("\n", "\\n")[:1000]
        log(f"  Source preview: {preview}")

        for out_idx, output in enumerate(outputs[:5]):
            out_text = extract_output_text(output)
            log(f"  Output {out_idx} length: {len(out_text):,}")
            output_preview = out_text.replace("\n", "\\n")[:1000]
            log(f"  Output {out_idx} preview: {output_preview}")

    log("\n" + "=" * 80)
    log("KEYWORD COUNTS")
    log("=" * 80)

    keyword_counts = []
    for kw in KEYWORDS:
        count = len(re.findall(re.escape(kw), full_text, flags=re.IGNORECASE))
        keyword_counts.append((kw, count))
        log(f"{kw:<20} {count}")

    log("\n" + "=" * 80)
    log("KEYWORD CONTEXTS")
    log("=" * 80)

    for kw, count in keyword_counts:
        if count == 0:
            continue

        log("\n" + "-" * 80)
        log(f"Keyword: {kw} | Count: {count}")
        log("-" * 80)

        hits = find_keyword_context(full_text, kw, context=500, max_hits=5)

        for i, snippet in enumerate(hits, start=1):
            cleaned = snippet.replace("\n", "\\n")
            log(f"\nHit {i}:")
            log(cleaned[:1500])

    log("\n" + "=" * 80)
    log("REGEX SEARCH: FUNCTION-LIKE DEFINITIONS")
    log("=" * 80)

    function_patterns = [
        r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
        r"class\s+[A-Za-z_][A-Za-z0-9_]*\s*[\(:]",
        r"```python",
        r"from\s+llamea",
        r"import\s+llamea",
    ]

    for pattern in function_patterns:
        matches = list(re.finditer(pattern, full_text))
        log(f"\nPattern: {pattern}")
        log(f"Matches: {len(matches)}")

        for i, match in enumerate(matches[:20], start=1):
            start = max(match.start() - 300, 0)
            end = min(match.end() + 700, len(full_text))
            snippet = full_text[start:end].replace("\n", "\\n")
            log(f"  Match {i}: {snippet[:1200]}")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    log("\nSummary written to:")
    log(str(SUMMARY_FILE))


if __name__ == "__main__":
    main()