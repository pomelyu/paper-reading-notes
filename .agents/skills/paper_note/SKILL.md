---
name: paper_note
description: 從指定的research paper pdf提取重點產生摘要筆記。
---

# Paper Note
瀏覽指定的pdf文件，提取文件摘要和圖片，形成指定格式的markdown圖文筆記。用英文書寫。

# Instruction
假設論文名稱為 Real-Time Radiance Fields for Single-Image Portrait View Synthesis，網路搜尋這篇論文發表時間是2023後，建立資料夾格式如下。
```
2023/
  Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis/
    resources/
    note.md
```
- 文件夾名稱以underscore取代space，colon以dash取代
- resources存放需要pdf截圖
- note.md為主要的筆記內容
- 引用到其他論文的作法或是比較時，加上註腳

## Note structure
```markdown
## Introduction
<!-- 如果作者提供程式碼和project page，分別列出連結 -->
- **Project**: <url or N/A>
- **Code**: <url or N/A>

<!--
整理論文的abstract和introduction，以白話的方式說明這篇論文的主旨。在使用論文自定義的縮寫字或名詞時加上用（）加上一句話的簡單解釋。
這部份通常會包含一張圖片來清楚表達這篇的內容。此圖片通常是論文的首頁標頭圖片(Figure1)。
-->

<!-- 列出第一作者和其他著名的作者和單位 -->
**Authors**:
- **Tao Lu** (First Author) - Insitiutation

<!-- 列出keywords -->
**Keywords**: Neural Rendering, 3D Gaussian Splatting,

<!-- 如果有的話，列出出版在哪個 Conference -->
**Publication**: CVPR 2024 (Highlight)

## Method
<!--
說明論文的實作方法，如果論文提供overview的圖片，顯示在文字說明之前。
內容以條列式的表達。包含
1. 訓練資料如何產生的
2. 方法的overview
3. 各個模型的具體，如果使用到另一篇論文的模型，要特別標注。並在最後建立參照。
4. 損失函數，以條列表示
遇到方程式的部分可以以截圖來說明
-->

## Highlight
<!--
這篇論文突出的地方，如果強調速度和效能，列出精確的數據和使用的硬體
如果強調視覺品質，請列出比較的對象
如果有比較圖，列出比較圖
-->

## Limitation
<!--
條列式的列出論文的限制，可能包含訓練資料帶來的限制，方法上的限制，和其他相關研究比較的限制。
-->

## Comments
<!-- 其他說明，通常可以留白 -->

<!-- 前面提及的相關論文 -->
[^1]: reference paper name
```

# Figure and Resource Conventions

- Keep filenames descriptive and stable, e.g. `fig_01_teaser.jpg`, `fig_02_method_overview.jpg`, `eq_loss_total.jpg`.
- Reference images with relative paths from `note.md`.
- Do not embed large raw extracts; summarize and cite with concise context.

# PDF Cropping Guideline
## Exact Crop Only

- Crop only the target figure/table; do not export full-page images.
- Keep padding minimal (`<= 12px`).
- Include caption only when it is needed for context.

## Output Quality and Format

- Output format must be `.jpg` (not `.png`).
- JPEG quality must be `>= 92`.
- Prefer direct extraction of embedded image objects at native resolution.
- If rasterizing pages, render at `>= 350 DPI` before cropping.
- Keep long side `>= 1800px` when the source supports it.

## Naming Convention
Use deterministic, stable names:
- `fig_01_teaser.jpg`
- `fig_02_method_overview.jpg`
- `table_01_main_results.jpg`

## Rejection Criteria
Do not accept outputs that are:
- full-page captures,
- blurry or low-resolution,
- wrong figure/table,
- PNG format,
- overly padded with irrelevant whitespace

# Agent Behavior
- Do not change the required section order.
- Do not skip benchmark numbers when the paper reports them.
- Do not omit hardware details when speed/performance claims are included.
- Use `N/A` for unavailable project/code links instead of removing fields.

# Examples
參考 references/Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis.note.md 當作範例
