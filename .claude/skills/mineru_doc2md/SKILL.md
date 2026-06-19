---
name: mineru_doc2md
description: Convert a document (PDF, DOCX, PPTX, image, etc.) to Markdown using the MinerU Precision Extract API. Use this skill whenever the user provides a local file path or a URL to a document and wants it converted to Markdown — even if they say "extract text", "parse this paper", "turn this into markdown", or "read this PDF". Triggers on any document-to-markdown conversion request that should use MinerU.
---

# MinerU Document → Markdown

Convert a local document or a URL to Markdown by calling the MinerU Precision Extract API.

## Prerequisites

A `MINERU_TOKEN` must be set in a `.env` file in the **current working directory**:

```
MINERU_TOKEN=your_token_here
```

If the token is missing, the script will print setup instructions and exit. Tokens can be obtained at https://mineru.net/apiManage/token.

## How to run

The script lives next to this file. Get its absolute path relative to this SKILL.md's location, then run:

```bash
python <skill_dir>/scripts/mineru_doc2md.py <input> [options]
```

Where `<skill_dir>` is the directory containing this SKILL.md.

**Input** can be:
- A local file path: `paper.pdf`, `/abs/path/to/doc.docx`
- A direct URL: `https://example.com/paper.pdf`

**Output** defaults to the same location as the input with a `.md` extension. Override with `-o <path>`.

## Common invocations

```bash
# Convert a local PDF (output saved next to the input)
python <skill_dir>/scripts/mineru_doc2md.py paper.pdf

# Save to a specific path
python <skill_dir>/scripts/mineru_doc2md.py paper.pdf -o notes/paper.md

# Convert from URL
python <skill_dir>/scripts/mineru_doc2md.py https://example.com/paper.pdf

# Force English language parsing
python <skill_dir>/scripts/mineru_doc2md.py paper.pdf --language en

# Use VLM model with OCR for scanned documents
python <skill_dir>/scripts/mineru_doc2md.py scan.pdf --model-version vlm --is-ocr

# Parse only specific pages
python <skill_dir>/scripts/mineru_doc2md.py paper.pdf --page-ranges 1-10
```

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `-o PATH` | `<input>.md` | Output file path |
| `--language LANG` | ch | Language code: `en`, `ch`, `ja`, etc. |
| `--model-version` | pipeline | `pipeline`, `vlm`, or `MinerU-HTML` |
| `--is-ocr` | off | Enable OCR (for scanned documents) |
| `--no-formula` | — | Disable formula recognition |
| `--no-table` | — | Disable table recognition |
| `--page-ranges RANGES` | all | E.g. `1-5`, `1,3,5-10` |

## Workflow

1. Determine `<skill_dir>` — the directory containing this SKILL.md.
2. Check that the user's `.env` in the current working directory contains `MINERU_TOKEN`. If not, tell them to add it.
3. Confirm the output path with the user if not obvious.
4. Run the script via Bash and stream output so the user sees upload/poll progress.
5. Report the saved file path when done.

## Local file flow (background)

For local files, the script:
1. Requests a presigned upload URL from MinerU (`POST /api/v4/file-urls/batch`)
2. Uploads via `curl` (suppressing Content-Type to avoid OSS signature mismatch)
3. Polls `GET /api/v4/extract-results/batch/{batch_id}` until extraction completes
4. Downloads the result ZIP and extracts `full.md`

For URL inputs, it submits directly to `POST /api/v4/extract/task` and polls `GET /api/v4/extract/task/{task_id}`.
