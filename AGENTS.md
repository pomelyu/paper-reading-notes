# Paper Reading Notes

This repository stores structured notes for research papers.

- Reusable workflows live in `.agents/skills/`. Use the matching skill for
  paper reading, document-to-Markdown conversion, or knowledge-graph updates.
- Keep paper notes under `{year}/{Paper_Title}/README.md` with supporting
  material in the note's `resources/` directory.
- Preserve the existing folder naming and note format unless the user requests
  a migration.
- Every new paper note must include `#### Train Data` and `####
  Evaluation/Validation Data` sections, each with `Name` and `Usage` columns,
  immediately after `### Method / Approach` and before `### Key Results`. Use
  the source paper, use canonical dataset names, and run
  `python3 scripts/sync_knowledge_graph.py` after changing paper metadata.
