#!/usr/bin/env python3
"""Regenerate dataset and derived graph indexes from Markdown.

Paper README metadata and glossary footnotes are the source of truth. This
script deliberately never reads or writes the local MCP `.aim` cache.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
NOTE_GLOB = "20*/*/README.md"
LEGACY_FIELD_RE = re.compile(r"^- \*\*(Train Data|Evaluation/Validation Data):\*\*\s*(.*)$", re.M)
DATA_SECTION_RE = re.compile(
    r"^#### (Train Data|Evaluation/Validation Data)\n\n\| (?:Name|Dataset) \| Usage \| Proposed by \|\n\|---\|---\|---\|\n((?:\|[^\n]+\|\n?)+)", re.M
)
TERM_RE = re.compile(r"^\[\^[^]]+\]: \*\*([^*]+)\*\*.*\[glossary\]\(\.\./\.\./common/terms/\)", re.M)
INDEX_ENTRY_RE = re.compile(r'<a href="(20[^\"]+?)/?">([^<]+)</a>')


def title_and_fields(readme: Path) -> tuple[str, dict[str, list[tuple[str, str, str]]], list[str]]:
    text = readme.read_text()
    title = re.search(r"^#\s+(.+)$", text, re.M).group(1).strip()
    fields: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for match in DATA_SECTION_RE.finditer(text):
        category = match.group(1)
        for row in match.group(2).splitlines():
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if len(cells) == 3:
                fields[category].append((cells[0], cells[1], cells[2]))
    terms = sorted({term.strip() for term in TERM_RE.findall(text)})
    return title, fields, terms


def validate_data_section_position(readme: Path) -> None:
    text = readme.read_text()
    detailed = text.find("### Detailed Technical Summary\n")
    datasets = text.find("### Datasets\n")
    hidden = text.find("### Hidden Assumptions\n")
    # Current paper-reading notes place datasets in Pass 3, immediately after
    # the detailed summary and before hidden assumptions. A few legacy notes
    # have no pass headings; their presence and table format are still checked.
    if detailed >= 0 or hidden >= 0:
        if not (0 <= detailed < datasets < hidden):
            raise SystemExit(
                f"Data sections must follow Detailed Technical Summary and precede Hidden Assumptions: {readme.relative_to(ROOT)}"
            )
        return
    if datasets < 0:
        raise SystemExit(f"Missing Datasets section: {readme.relative_to(ROOT)}")


def datasets(entries: list[tuple[str, str, str]]) -> list[str]:
    return [name for name, _, _ in entries if name and name.casefold() != "none stated"]


def migrate_legacy_data_sections() -> None:
    """One-time migration from header metadata to Method/Results data sections."""
    for readme in ROOT.glob(NOTE_GLOB):
        text = readme.read_text()
        legacy = {name: value.strip() for name, value in LEGACY_FIELD_RE.findall(text)}
        if not legacy:
            continue
        if set(legacy) != {"Train Data", "Evaluation/Validation Data"}:
            raise SystemExit(f"Incomplete legacy dataset metadata: {readme.relative_to(ROOT)}")
        sections = []
        for category in ("Train Data", "Evaluation/Validation Data"):
            rows = []
            for item in legacy[category].split(";"):
                name, _, usage = item.strip().partition(" — ")
                rows.append(f"| {name.strip()} | {usage.strip()} | — |")
            sections.append(
                f"#### {category}\n\n| Dataset | Usage | Proposed by |\n|---|---|---|\n" + "\n".join(rows)
            )
        text = LEGACY_FIELD_RE.sub("", text).replace("\n\n\n", "\n\n")
        marker = "### Key Results\n"
        if "### Method / Approach\n" not in text or marker not in text:
            raise SystemExit(f"Cannot position data sections: {readme.relative_to(ROOT)}")
        before, after = text.split(marker, 1)
        text = before.rstrip() + "\n\n" + "\n\n".join(sections) + "\n\n" + marker + after
        readme.write_text(text)


def is_public_dataset(name: str) -> bool:
    """Internal datasets remain in note metadata but are omitted from the public catalog."""
    return not name.casefold().startswith("internal ")


def replace_block(path: Path, name: str, body: str) -> None:
    text = path.read_text()
    begin, end = f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def paper_short_names() -> dict[str, str]:
    """Read the display abbreviations from the root paper index."""
    names = {}
    for path, label in INDEX_ENTRY_RE.findall((ROOT / "README.md").read_text()):
        match = re.match(r"\(([^)]+)\)", label.strip())
        names[path.rstrip("/")] = match.group(1) if match else label.strip()
    return names


def main() -> None:
    train, evaluation, proposed_by, term_usage = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    short_names = paper_short_names()
    checked = 0
    for readme in sorted(ROOT.glob(NOTE_GLOB)):
        title, fields, terms = title_and_fields(readme)
        source = readme.parent / "resources/paper.md"
        if source.exists():
            checked += 1
            if set(fields) != {"Train Data", "Evaluation/Validation Data"}:
                raise SystemExit(f"Missing standardized dataset fields: {readme.relative_to(ROOT)}")
            validate_data_section_position(readme)
        paper_name = short_names.get(str(readme.parent.relative_to(ROOT)), title)
        for dataset in datasets(fields.get("Train Data", [])):
            train[dataset].add(paper_name)
        for dataset in datasets(fields.get("Evaluation/Validation Data", [])):
            evaluation[dataset].add(paper_name)
        for entries in fields.values():
            for dataset, _, proposer in entries:
                if dataset.casefold() != "none stated" and proposer not in {"", "—"}:
                    proposed_by[dataset].add(proposer)
        for term in terms:
            term_usage[term].add(title)

    all_datasets = sorted(set(train) | set(evaluation), key=str.casefold)
    conflicts = {dataset: proposers for dataset, proposers in proposed_by.items() if len(proposers) > 1}
    if conflicts:
        details = "; ".join(
            f"{dataset}: {', '.join(sorted(proposers))}"
            for dataset, proposers in sorted(conflicts.items(), key=lambda item: item[0].casefold())
        )
        raise SystemExit(f"Conflicting Proposed by values: {details}")
    dataset_lines = [
        "# Dataset Catalog", "",
        "Auto-generated from each paper note's `Train Data` and `Evaluation/Validation Data` metadata.",
        "Dataset names are canonical identifiers used by the knowledge-graph rebuild.", "",
        "| Dataset | Train Data papers | Evaluation/Validation papers | Proposed by |",
        "|---|---|---|---|",
    ]
    for dataset in filter(is_public_dataset, all_datasets):
        dataset_lines.append(
            f"| {dataset} | {', '.join(sorted(train[dataset])) or '—'} | {', '.join(sorted(evaluation[dataset])) or '—'} | {', '.join(sorted(proposed_by[dataset])) or '—'} |"
        )
    (ROOT / "common/datasets.md").write_text("\n".join(dataset_lines) + "\n")

    graph_datasets = ["## Dataset Index", "", "| Dataset | Train Data papers | Evaluation/Validation papers | Proposed by |", "|---|---|---|---|"]
    for dataset in all_datasets:
        graph_datasets.append(f"| {dataset} | {', '.join(sorted(train[dataset])) or '—'} | {', '.join(sorted(evaluation[dataset])) or '—'} | {', '.join(sorted(proposed_by[dataset])) or '—'} |")
    graph_terms = ["## Term Usage Index", "", "Only glossary-linked footnotes create `uses_term` relationships.", "", "| Term | Papers |", "|---|---|"]
    for term in sorted(term_usage, key=str.casefold):
        graph_terms.append(f"| {term} | {', '.join(sorted(term_usage[term]))} |")
    graph = ROOT / "KNOWLEDGE_GRAPH.md"
    replace_block(graph, "DATASET_INDEX", "\n".join(graph_datasets))
    replace_block(graph, "TERM_USAGE_INDEX", "\n".join(graph_terms))
    print(
        f"validated {checked} sourced notes; generated {len(all_datasets)} datasets "
        f"({sum(bool(proposed_by[name]) for name in all_datasets)} with known proposers), "
        f"and {len(term_usage)} terms"
    )


if __name__ == "__main__":
    if "--migrate-data-sections" in sys.argv:
        migrate_legacy_data_sections()
    main()
