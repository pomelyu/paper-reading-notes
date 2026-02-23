---
applyTo: "**"
---
# Project Guidelines
This project aims to automatically create paper notes that focus on algorithms, performance benchmarks, and limitations.

1. Ignore content under `./tmp/` unless the user explicitly specifies a file or folder there.
2. When creating or modifying paper notes, follow `.agents/skills/paper_note`.
3. Before implementation, review available skills in `.agents/skills` and `~/.agents/skills`.
4. If environment issues occur during execution (e.g., missing packages, incompatible package versions), pause execution and list concrete remediation options. Continue only after the user decides how to proceed.
5. After creating a new paper note, update `README.md` with a link in this format:
```markdown
	### {year}
	- [paper_title](link—to-note)
```
