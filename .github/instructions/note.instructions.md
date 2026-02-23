---
name: "Paper Note Standard"
description: "Create consistent paper-reading notes focused on method, benchmarks, and limitations."
applyTo: "**/note.md"
---

# Paper Note Instructions

Use these rules whenever creating or updating a paper note in this repository.

## Goal

Read a target research paper PDF, extract key text and figures, and produce a structured Markdown note in English, with emphasis on:
- algorithm design
- performance benchmarks
- limitations

## Output Location and Naming

Create notes in this structure:

```text
ROOTFOLDER/
	{year}/
		{paper_title_sanitized}/
			resources/
			note.md
```

Rules:
- `{year}` is the paper publication year.
- `{paper_title_sanitized}` uses `_` instead of spaces.
- Replace `:` in titles with `-`.
- Store all extracted figure/equation screenshots in `resources/`.
- Main note content must be in `note.md`.

Example:

```text
2023/
	Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis/
		resources/
		note.md
```

## Writing Requirements

- Write the note in English.
- Explain in plain language.
- Avoid unexplained paper-specific abbreviations or jargon; at first mention, add a brief explanation in parentheses.
- Prefer concise bullet points for methods, highlights, and limitations.
- Include precise numbers when reporting speed/quality gains (with hardware/setup when available).

## Required Note Structure

Use this exact section order in `note.md`:

```markdown
## Introduction
- project: <url or N/A>
- code: <url or N/A>

Summarize abstract + introduction in plain language, including core idea and problem setting.
Include a key overview figure when available (usually Figure 1 / teaser), saved under `resources/`.

List:
- first author
- notable co-authors and affiliations
- keywords

## Method
If an overview/pipeline figure exists, show it before the method text.

Use bullet points to cover:
1. How training data is collected/generated
2. Method overview
3. Key model components/modules (and clearly mark reused modules from prior work)
4. Loss functions (itemized)

For complex equations, include screenshots from the paper in `resources/` and explain them briefly.

## Highlight
List the paper's strongest contributions/results.

- If speed/performance is emphasized: include exact metrics and hardware.
- If visual quality is emphasized: include comparison targets/baselines.
- Include representative comparison figures when available.

## Limitation
List limitations in bullets, including:
- data-related limitations
- method/design limitations
- limitations relative to related work

## Comments
Optional additional notes. Leave blank if no extra remarks.
```

## Figure and Resource Conventions

- Keep filenames descriptive and stable, e.g. `fig1_teaser.png`, `method_overview.png`, `eq_loss_total.png`.
- Reference images with relative paths from `note.md`.
- Do not embed large raw extracts; summarize and cite with concise context.

## Agent Behavior

- Do not change the required section order.
- Do not skip benchmark numbers when the paper reports them.
- Do not omit hardware details when speed/performance claims are included.
- Use `N/A` for unavailable project/code links instead of removing fields.

## Reference Example

For style reference, see:
- `archived/my_skills/paper_note/references/Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis/note.md`
