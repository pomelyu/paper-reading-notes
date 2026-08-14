---
name: update-graph
description: >
  Rebuild or incrementally synchronize the paper knowledge graph, dataset
  catalog, and term usage index from Markdown notes.
---

# Update Knowledge Graph Skill

Use this skill when the user asks to update, rebuild, or sync the knowledge
graph. Markdown is authoritative; the local MCP `.aim` JSONL is a disposable
cache and must never be used as the source for rebuilding the graph.

## Source schema

Each sourced paper note (`20*/*/resources/paper.md` exists) must have a
`### Datasets` section in Pass 3, immediately after `### Detailed Technical
Summary` and before `### Hidden Assumptions`. It must contain these two
subsections:

```markdown
#### Train Data

| Dataset | Usage | Proposed by |
|---|---|---|
| Dataset A | purpose | Paper abbreviation or — |

#### Evaluation/Validation Data

| Dataset | Usage | Proposed by |
|---|---|---|
| Dataset C | purpose | Paper abbreviation or — |
```

`Train Data` covers pretraining, fine-tuning, calibration, synthetic, and
self-generated data. `Evaluation/Validation Data` covers validation, test,
and benchmark data. Names before ` — ` are canonical dataset entity names.
`None stated — reason` creates no dataset relation. A glossary footnote that
links to `../../common/terms/` creates a `uses_term` relation.

`Proposed by` names the paper that originally introduced the dataset. Use the
repository's canonical paper abbreviation when that paper has a note; use a
recognizable paper title for an external work, and `—` only when the origin is
not established by the reviewed source. Repeated uses of the same dataset must
agree on this value.

## Step 1 — Determine scope

For a normal incremental update, inspect changed `20*/*/README.md`,
`common/terms/README.md`, and `resources/paper.md` files with git. Use full
rebuild when requested with `--full`, when the MCP cache is empty, or after
changing the schema.

## Step 2 — Validate and generate repository views

Run:

```bash
python3 .agents/skills/update-graph/scripts/sync_knowledge_graph.py
```

The command fails if any sourced note lacks either required dataset field. It
regenerates:

- `common/datasets.md` — public canonical dataset index and paper usage;
  internal datasets remain in paper metadata and the MCP graph but are omitted.
- marked Dataset and Term Usage sections of `KNOWLEDGE_GRAPH.md`.

Review its output before patching MCP state.

## Step 3 — Synchronize MCP cache

The configured `mcp-knowledge-graph` v1.3.2 server exposes these tool names:

- `aim_memory_read_all`
- `aim_memory_store`
- `aim_memory_add_facts`
- `aim_memory_link`
- `aim_memory_search`
- `aim_memory_get`
- `aim_memory_forget`
- `aim_memory_unlink`

Use the project-local default store. For each paper, upsert a `paper` entity
with title, year, authors, venue, keywords, path, and `Has note: true`. Upsert
each glossary term as `term` and each canonical catalog entry as `dataset`.

Create only missing directed relations:

```text
paper -[builds_on|competes_with|succeeded_by]-> paper
paper -[trains_on]-> dataset
paper -[evaluates_on]-> dataset
paper -[uses_term]-> term
```

Extract paper-to-paper edges only from the three `Comparison Papers` tables.
Extract datasets only from the two data sections. Extract terms only from
glossary-linked footnotes. For a full rebuild, clear the project cache with
`aim_memory_forget`, then recreate every entity and relation from Markdown.

## Step 4 — Regenerate and report

Run `python3 .agents/skills/update-graph/scripts/sync_knowledge_graph.py`
once more after cache sync.
Keep existing paper relationship/topic sections in `KNOWLEDGE_GRAPH.md`; the
script replaces only its marked derived indexes. Report processed notes,
datasets and terms created or updated, and new relations.

## Invariants

- Do not version `.aim/*.jsonl`; versioned Markdown fully reconstructs it.
- Do not use `KNOWLEDGE_GRAPH.md` as a rebuild input.
- The workflow is idempotent: a second run must not duplicate catalog rows or
  MCP relations.
