---
name: paper-reading
description: >
  Summarize a research paper using the three-pass method from S. Keshav's
  "How to Read a Paper", then add a modern review step. Use this skill whenever
  the user provides a paper (PDF path, arXiv link, pasted text, or title) and
  asks to read, summarize, review, analyze, or take notes on it — even if they
  don't mention "three-pass" explicitly. Also trigger when the user says things
  like "help me understand this paper", "what does this paper do", or "give me
  notes on this paper".
---

# Paper Reading Skill

You are helping the user deeply understand a research paper using a structured
reading methodology. Work through the four steps below in order. For each pass,
explicitly label the section with the pass name so the user can follow along.

## Before starting — obtain the paper content

Identify what the user gave you, then get the content:

**If a local file path or a URL pointing to a PDF/document:**

Before running the conversion, determine the paper's title and year (from
the filename, a quick skim of the abstract, or the URL). Use those to derive
the output folder path (`{year}/{Paper_Title}/`), create it along with the
`resources/` subdirectory.

Then convert the PDF to Markdown. First check whether `{year}/{Paper_Title}/resources/paper.md` already exists — if it does, skip conversion and read it directly. If not, use the `mineru_doc2md` skill with `-o {year}/{Paper_Title}/resources/paper.md` so the converted Markdown lands inside the note's folder.
Read that `.md` file for all four passes below.

**If an arXiv link:** Fetch the abstract page to get metadata, then use
`mineru_doc2md` to convert the PDF URL (e.g. `https://arxiv.org/pdf/<id>`),
saving to `{year}/{Paper_Title}/resources/paper.md` as above.

**If pasted text or a title only:** Read/search directly without conversion.

---

## Find associated links

Before starting Pass 1, do a quick web search for links associated with this
paper. Search for the paper title plus terms like "project page", "GitHub",
"HuggingFace". Collect whichever of the following exist:

- **Webpage** — official project page (e.g. `https://somelab.github.io/project/`)
- **GitHub** — authors' official code repository
- **HuggingFace** — model weights, datasets, or demo spaces

Only include links you are confident belong to this paper. Skip any that don't
exist rather than guessing. Record the results — they go into the README header
(see Output Format below).

---

## Check knowledge graph for existing references

Before starting Pass 4 (Comparison Papers), query the knowledge graph to find
papers that already reference or is mentioned in this paper. This ensures bidirectional consistency.

**If `aim_` MCP tools are available:**

1. Call `aim_search_nodes` with the paper's title (or key terms from it).
2. If the paper already exists as an entity, call `aim_open_nodes` to see all
   its relations — specifically look for papers where this paper appears as a
   predecessor, competitor, or successor.
3. Collect these "reverse references" — papers that cite or build on this paper
   but that this paper's own comparison tables might not mention.

**If `aim_` tools are not available:** skip this step (the `update-graph` skill
will handle reverse linking later).

**How to use the results in Pass 4:**

- In the `#### Predecessors` table: if the knowledge graph shows that this paper
  is listed as a predecessor by Paper X (i.e., X `builds_on` this paper), then
  Paper X should appear in `#### Successors / Extensions` of this note.
- In the `#### Successors / Extensions` table: include any paper from the graph
  that has a `builds_on` relation pointing to this paper, even if your web
  search didn't find it.
- Add a column note `(from knowledge graph)` in the Relation column for entries
  discovered this way, so the user can distinguish them from web-search results.

---

## Pass 1 — Bird's-Eye View (5–10 minutes equivalent)

Goal: Decide whether the paper deserves deeper reading and understand its
high-level shape.

Read only these parts:
- Title, abstract, and introduction
- All section and sub-section headings (just the headings, not the body)
- Conclusions section
- Glance at the references list

Answer the **Five Cs** in a table, then write a 30-second summary paragraph.
The Five Cs table uses "Assessment" as the column header (not a question —
write the actual answer, not a prompt):

| C | Assessment |
|---|-----------|
| **Category** | What type of paper is this? |
| **Context** | Related work and theoretical background it builds on |
| **Correctness** | Whether the assumptions appear valid; any red flags |
| **Contributions** | The main claims or novel contributions |
| **Clarity** | Whether the paper is well written |

The 30-second summary should be dense enough that a knowledgeable colleague
can get the gist without reading anything else.

---

## Pass 2 — Careful Read (up to 1 hour equivalent)

Goal: Grasp the content without getting lost in the details.

Read the paper carefully but skip proofs and deep derivations for now. Pay
attention to figures, tables, and references worth following up on.

Produce these named sub-sections:

**### Core Idea in One Sentence** — the simplest possible statement of what
the paper does, in a single sentence.

**### Method / Approach** — 2–4 bullet points, each with a **bold label**
followed by a colon and the explanation.

**### Key Results** — a markdown table comparing the paper against baselines
on the most important metrics. Add a few bullet points below for ablation
findings if present.

**### Strengths** — bullet list with **bold labels** per point.

**### Weaknesses / Open Questions** — numbered list; each item starts with a
**bold label** for the issue, then the explanation.

**### References to Follow Up** — numbered list of 3–5 papers. Format each
as: `**Title** — Authors, Venue Year: one sentence on why it matters here.`

---

## Pass 3 — Virtual Re-implementation (1–5 hours equivalent)

Goal: Deeply understand the paper to the point you could reconstruct it.

Work through the full paper including proofs and appendices. Identify every
assumption (stated or unstated), note hand-wavy steps, and think about
whether the presentation could be clearer.

Produce these named sub-sections:

**### Detailed Technical Summary** — narrative prose with **bold section
headers** for each major component (e.g., the representation, the training
procedure, the loss function). Use LaTeX math notation for all equations and
mathematical symbols: `$...$` for inline expressions and `$$...$$` on their
own line for standalone equations. This should be dense enough that a reader
could implement the method from this section alone.

**### Hidden Assumptions** — numbered list of assumptions that the work
depends on but that are never stated explicitly.

**### Reproducibility Notes** — bullet list with **bold labels** covering
what data, code, compute, and hyperparameters are needed to reproduce the
main results, and what is missing or underspecified.

**### Ideas for Future Work** — numbered list with **bold labels** per idea.

---

## Pass 4 — Modern Perspective Review

Goal: Situate the paper in today's research landscape, which may have moved
significantly since publication.

The section heading should include the current date:
`## Pass 4 — Modern Perspective Review (as of Month Year)`

Produce these sub-sections:

**### What Has Changed Since Publication** — bullet list covering shifts in
datasets, compute, dominant architectures, evaluation standards, or community
assumptions since the paper appeared.

**### Has the Community Accepted the Claims?** — a prose paragraph on whether
the contributions were validated, refined, or challenged by follow-on work.

Then a `---` divider, followed by:

**### Comparison Papers** — three sub-tables, each a 4-column markdown table
with headers `| Paper | Authors | Year | Relation |`:

- **#### Predecessors** — papers this work builds on directly
- **#### Contemporaries / Competitors** — papers solving the same problem at
  roughly the same time with a different approach
- **#### Successors / Extensions** — later papers that extend, improve, or
  challenge this work

Then another `---` divider, followed by:

**### Bottom Line** — a prose paragraph answering: given everything published
since, is this paper still worth reading? Is it a foundational classic, a
useful reference, or largely superseded?

---

## Output Format

### Folder structure

Create the following directory structure rooted at the current working
directory:

```
{year}/
  {Paper_Title}/
    resources/
    README.md
```

**Determining the year**: Search for the paper's publication year (venue
acceptance year, not arXiv submission year when both exist). If uncertain,
use the arXiv submission year.

**Naming the folder**: Take the full paper title and apply these rules:
- Replace every space with `_`
- Replace every `:` with `-`
- Keep all other characters as-is (preserve capitalisation, hyphens, etc.)

Example: `Real-Time Radiance Fields for Single-Image Portrait View Synthesis`
→ `Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis`

The `resources/` subdirectory holds the MinerU-converted `paper.md` (if
conversion was run) plus any screenshots or figures extracted from the PDF.
Create it before running `mineru_doc2md` so the output lands there directly.

### README.md content

The note goes into `{year}/{Paper_Title}/README.md`. It starts with a
document header block:

```
# Full Paper Title

- **Authors:** First Last, First Last, ...
- **Affiliations:** Institution A, Institution B
- **Published:** Venue/arXiv ID, Date
- **Keywords:** keyword1, keyword2, keyword3
- **Webpage:** https://...        ← omit if not found
- **GitHub:** https://...         ← omit if not found
- **HuggingFace:** https://...    ← omit if not found

---
```

Then the four passes follow with `## Pass N — Name` headings and
`### Sub-section` headings exactly as described above.

Aim for depth over brevity — the user is building a durable reference they
will return to, not a quick abstract. Use markdown tables and bold-labeled
lists to keep the output scannable.

**Math formatting:** Use LaTeX notation for equations and mathematical
symbols throughout all four passes:

- **Inline math:** Use `$...$` for inline expressions (e.g., `$L_{ground}$`,
  `$\mu_{i,t}$`).
- **Block math:** Use fenced code blocks with the `math` language tag
  instead of `$$...$$`:
  ````
  ```math
  C = \sum_{i \in N} c_i \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j)
  ```
  ````

**GitHub rendering rules:**
- Do NOT use `\mathcal`, `\mathbf`, `\mathbb`, or `\text` — these render
  incorrectly in GitHub markdown. Use plain letters instead (e.g.,
  `$L_{ground}$` not `$\mathcal{L}_{ground}$`).
- For bold/calligraphic distinction, rely on subscripts, context, or naming
  conventions rather than font commands.
- **Punctuation after inline math:** GitHub fails to render inline math when
  the closing `$` is immediately followed by any punctuation character
  (`, . ) ] ; : ! ?` etc.). Always add a space between the closing `$` and
  the following punctuation: write `$x$ ,` not `$x$,`, `$x$ .` not `$x$.`,
  `$x$ )` not `$x$)`, and so on.

The MinerU-converted `paper.md` in `resources/` may use these font
commands — simplify them when writing the README.

If the paper is long and you are missing key sections (e.g., the appendix or
supplementary), say so explicitly and ask the user to provide those pages
rather than guessing.

### Updating the index

After writing the note, register it in the root `README.md` index
(at the working directory root). The index groups papers by year:

```markdown
# Paper Notes
### 2023
- [Paper Title](2023/Paper_Title_Folder/)

### 2024
- [Another Paper](2024/Another_Paper_Folder/)
```

Steps:
1. Read the current `README.md`.
2. Find the `### {year}` section that matches this paper's year.
   - If the section exists, append a new bullet at the end of it.
   - If the section does not exist yet, create it in the correct
     chronological position among the other year sections.
3. The new bullet format is:
   `- [{Full Paper Title}]({year}/{folder_name}/)`
   Use the paper's full original title (with spaces and colons) as the
   link text, and the sanitised folder name (underscores, dashes) in the
   path.
4. Write the updated `README.md` back.
