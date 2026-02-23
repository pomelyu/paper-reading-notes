---
name: instructions-creator
description: Use this skill when the user asks to create or update VS Code custom instructions files (for example .github/copilot-instructions.md, AGENTS.md, or *.instructions.md). It helps choose the correct file type/location, apply valid YAML frontmatter fields, and write concise instruction bodies that are easy for coding agents to follow.
license: Proprietary. LICENSE.txt has complete terms
---

# Instructions Creator

## Purpose

Use this skill to create or update custom instruction files for VS Code Copilot customization with correct structure and minimal ambiguity.

## Supported Instruction Types

Choose the file type that matches the user goal:

1. Always-on workspace instructions:
   - `.github/copilot-instructions.md`
   - `AGENTS.md`
2. File-based conditional instructions:
   - `*.instructions.md`
   - usually stored under `.github/instructions/`

If the user does not specify, default to:
- `.github/copilot-instructions.md` for project-wide rules
- `.github/instructions/<topic>.instructions.md` for scoped/language/framework rules

## Instructions File Format

For `*.instructions.md`, use Markdown with optional YAML frontmatter:

```markdown
---
name: "Python Standards"
description: "Coding conventions for Python files"
applyTo: "**/*.py"
---

# Python coding standards
- Follow PEP 8.
- Use type hints for function signatures.
```

Frontmatter fields:
- `name` (optional): display name in UI
- `description` (optional): short summary shown in Chat UI
- `applyTo` (optional): glob pattern relative to workspace root
  - If omitted, file is not auto-applied and must be attached manually
  - Use `**` to apply to all files

## File Location Rules

Default locations:
- Workspace-scoped file-based instructions: `.github/instructions/`
- Workspace always-on instructions: `.github/copilot-instructions.md` or `AGENTS.md`

Important:
- Use `.instructions.md` extension for file-based instructions.
- Keep `applyTo` patterns explicit and testable (`**/*.ts`, `src/**/*.py`, `docs/**/*.md`).

## Authoring Guidelines

When writing the instruction body:
- Keep instructions short, specific, and actionable.
- Prefer one rule per bullet.
- Include rationale for non-obvious rules.
- Use concrete preferred/avoided patterns when needed.
- Avoid duplicating lint/formatter defaults unless project-specific.

## Creation Workflow

1. Determine scope:
   - all files (always-on) or selected files (file-based)
2. Choose correct file path and name
3. Add YAML frontmatter (for `*.instructions.md`) with `applyTo` when auto-application is needed
4. Write concise Markdown instructions
5. Verify glob pattern matches intended files
6. Save under version control when workspace-specific

## Quality Checklist

Before finishing, confirm:
- File path and filename are correct for the intended scope
- Frontmatter is valid YAML
- `applyTo` exists when automatic matching is expected
- Instruction text is concise and non-conflicting
- No extra documentation files are created unless requested

## Response Pattern for Agent

When asked to create instructions:
1. Infer most likely instruction type and location
2. Create/update the file directly
3. Include minimal but complete frontmatter/body
4. Summarize what was created and where
5. Offer to add additional scoped instruction files if useful
