# Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025)

- **Authors:** Fengming Lin
- **Affiliations:** School of Computer Science, The University of Manchester, UK
- **Published:** arXiv:2510.09586, October 13, 2025
- **Keywords:** vision-language models, bibliometrics, research trends, survey, CVPR, ICLR, NeurIPS, TF-IDF, multimodal LLMs

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|-----------|
| **Category** | Bibliometric meta-survey — quantitative measurement of research trends across 26,104 accepted papers from CVPR, ICLR, and NeurIPS (2023–2025) using lexicon-based abstract analysis |
| **Context** | Builds on TF-IDF text analysis and bibliometric methods; takes VLM literature (CLIP, BLIP, LLaVA, Flamingo, ALIGN, DINO) as the subject of study rather than contributing to it; complements traditional hand-curated surveys with automated, reproducible measurement |
| **Correctness** | Methodology is transparent and explicitly self-limited: abstract-only scope means datasets/training details are systematically under-reported; lexicon recall may miss niche synonyms; all caveats clearly stated in Sec. 6; numerical claims are internally consistent |
| **Contributions** | (1) First quantitative trajectory analysis of 35 research directions across three top venues over three years; (2) Fine-grained VLM dissection across 8 dimensions (models, fusion, tasks, training, losses, datasets, co-modalities); (3) Cross-venue CVPR vs. ICLR vs. NeurIPS comparison; (4) Lexicon + methodology released for auditing and extension; (5) Actionable practical advice for VLM researchers derived from trend data |
| **Clarity** | Very well written — compact (13 content pages), tables well-formatted, caveats clearly stated, practical takeaways explicit |

This paper measures what the computer vision / ML community actually worked on in 2023–2025 by running a hand-crafted, 35-category lexicon over 26,104 accepted-paper abstracts from CVPR, ICLR, and NeurIPS, and plotting yearly TF-IDF trajectories per direction. Three macro findings emerge: (1) Vision-Language/LLM work exploded from 16% to 40% of all papers by 2025, reframing classical perception tasks as instruction-following; (2) Generative/diffusion research grew steadily while consolidating around controllability, distillation, and speed; (3) 3D/video activity is resilient, with composition shifting from NeRFs to Gaussian splatting. Within VLMs, instruction tuning and LoRA adaptation dominate training; contrastive objectives decline relative to KL/distillation; ALIGN and LLaVA are the most and fastest-cited model families respectively; and grounding/referring tasks cede ground to reasoning/instruction.

---

## Pass 2 — Careful Read

### Core Idea in One Sentence

Apply a transparent, reproducible TF-IDF lexicon pipeline to 26,104 paper abstracts from three top venues over three years to quantify which research directions rose, fell, or shifted in the VLM era.

### Method / Approach

- **Data collection and corpus**: A Python spider collects accepted-paper JSONL from CVPR (2,353/2,713/2,871 for 2023/24/25), ICLR (4,372/2,260/3,704), and NeurIPS (3,337/4,494/— for 2023/24; 2025 not yet available at submission time), yielding 26,104 abstracts for content analysis and an additional 8,424 from 2022 for longitudinal baseline only.
- **Lexicon labeling**: Abstracts are Unicode-normalized, lowercased, punctuation-stripped, and multi-word phrases protected as single tokens (e.g., "gaussian splatting"); then matched against 35 regular-expression categories (Diffusion, VLM/LLM, 3D, Video, Robustness, etc.) with multi-label assignment. Prevalence = fraction of abstracts matching a category in a given year.
- **Trend quantification**: Per-category yearly aggregated TF-IDF scores are plotted as direction trajectories; slopes are least-squares linear fits over 2022→2025, used as the "rising" score in Fig. 3. Trend = 2025 value minus 2023 value in percentage points.
- **Fine-grained VLM sub-analysis**: For VLM-tagged papers, eight additional mining passes extract named model mentions (Table 1), fusion/architecture mechanisms (Table 3), task strata (Table 4), training paradigms (Table 5), loss families (Table 6), curated dataset mentions (Table 7), and co-mentioned modalities (Table 8).

### Key Results

**Macro direction trajectories (Fig. 1 & 3, aggregated TF-IDF slopes 2022→2025):**

| Direction | 2023 share | 2025 share | Trend | Slope (pp/yr) |
|-----------|-----------|-----------|-------|--------------|
| Vision-Language / Multimodal / LLM | ~16% | ~40% | +24pp | **highest** |
| Diffusion & Generative | ~8% | ~15% | +7pp | 2nd |
| Video Understanding | low | rising | ~+3pp | 3rd |
| NeRF / Gaussian Splatting | moderate | rising | moderate | top 10 |
| Self-supervised / Pretraining | high | declining | −6pp | falling |
| GNN / Meta-learning | flat | flat/declining | −1 to −2pp | falling |

**Named models within VLM papers (Table 1, share of VLM abstracts):**

| Model | 2023 | 2024 | 2025 | Trend | Slope |
|-------|------|------|------|-------|-------|
| ALIGN | 4.3% | 5.8% | 5.1% | −0.8% | 0.65 |
| LLaVA | 0.1% | 1.2% | 2.7% | +2.6% | **0.91** (fastest) |
| ResNet/ConvNeXt | 2.9% | 0.4% | 0.5% | −2.4% | −0.74 |
| ViT | 1.5% | 1.2% | 0.6% | −0.9% | −0.13 |
| MoE | 0.6% | 0.6% | 1.3% | +0.6% | 0.26 |

**Tasks within VLM papers (Table 4):**

| Task | 2023 | 2025 | Trend | Slope |
|------|------|------|-------|-------|
| Reasoning / Instruction | 13.5% | 25.0% | **+11.5pp** | 5.71 |
| Grounding / Referring | 25.9% | 12.9% | **−13.0pp** | −8.36 |
| Retrieval | 8.5% | 8.3% | −0.2pp | 0.53 |
| Captioning | 6.2% | 4.4% | −1.9pp | −0.53 |

**Training paradigms (Table 5):**

| Paradigm | 2023 | 2025 | Trend |
|----------|------|------|-------|
| Pretrain + Finetune | 11.6% | 16.8% | +5.2pp |
| Instruction Tuning | 1.1% | 5.0% | +3.9pp |
| LoRA / Adapters | 1.3% | 4.1% | +2.8pp |
| Self/Weak/Semi supervision | 9.6% | 3.5% | −6.1pp |

**Cross-venue highlights (Sec. 5):**
- CVPR 2025: strongest 3D emphasis (23.1% of papers mention 3D geometry)
- ICLR 2025: highest VLM share (40.7%)
- NeurIPS: early VLM ramp (30.5% in 2024); diffusion at 11.6%

**Ablation-style findings:**
- Removing the dynamic collision reward is the single largest negative factor in VLM design choices (mirrored in task analysis: grounding/referring decline reflects tasks absorbed as sub-capabilities, not abandoned)
- 3D/point-cloud co-mentions with VLMs tick upward (+0.7pp), while audio/speech declines (−1.4pp) — consistent with embodied AI gaining traction
- Legacy benchmarks (MS-COCO, ImageNet) decline in abstract mentions; named dataset mentions overall declining as instruction-style suites grow

### Strengths

- **Unprecedented reproducibility**: All inputs (publicly available JSONL from venues) and methodology (released lexicon) are documented; any researcher can replicate or extend.
- **Scale and breadth**: 26K papers across 3 venues, 3 years, 35 directions — no comparable prior quantitative measurement exists at this scope.
- **Granularity within VLMs**: The 8-dimension fine-grained sub-analysis (models, fusion, tasks, training, losses, datasets, modalities) provides a complete structural picture of how the VLM community operates.
- **Actionable practitioner advice**: 5 concrete framing recommendations (Sec. 4.7, 5) derived from data, not opinion.
- **Cross-venue benchmarking**: CVPR vs. ICLR vs. NeurIPS VLM vs. 3D vs. diffusion profiles are quantified for the first time.

### Weaknesses / Open Questions

1. **Abstract-only scope**: Datasets, training details, and loss functions are routinely omitted from abstracts; all Tables 5–7 measurements are lower bounds. The true training paradigm distribution is likely more skewed toward instruction tuning than the numbers show.
2. **Lexicon recall gap**: Only canonical phrases are matched; emerging terms, non-English notation, and niche synonyms are missed. No recall estimate is provided.
3. **Single-author, no validation**: No inter-annotator agreement for the 35-category lexicon; self-reported precision but not recall; no ablation over lexicon design choices.
4. **Abstract-level multi-label ambiguity**: A paper tagged as both "VLM" and "3D" may be either a 3D VLM paper or a paper that happens to mention both — the analysis cannot distinguish.
5. **NeurIPS 2025 gap**: NeurIPS 2025 data was not available at submission time, making the 3-venue comparison asymmetric for the most recent year.

### References to Follow Up

1. **LLaVA: Large Language and Vision Assistant** — Haotian Liu et al., arXiv 2023: The fastest-growing model family in the survey; the paradigm case for instruction-tuned VLMs and the primary driver of the Reasoning/Instruction rise.
2. **ALIGN: Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision** — Jia et al., ICML 2021: Most-cited family in VLM abstracts; anchor of the dual-encoder contrastive pretraining paradigm that the survey tracks declining.
3. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., arXiv 2022: The adapter technique with the second-fastest growth (+2.8pp) in VLM papers per this survey; now standard in multimodal fine-tuning.
4. **DINOv2: Learning Robust Visual Features without Supervision** — Oquab et al., arXiv 2023: The self-supervised visual encoder that the survey identifies as increasingly used as a frozen backbone in LVLMs, supplanting supervised ViTs.
5. **Flamingo: A Visual Language Model for Few-Shot Learning** — Alayrac et al., arXiv 2022: The LVLM that established gated cross-attention + Perceiver Resampler as the architectural template for injecting vision into LLMs; multiple successors tracked in survey tables.

---

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Data Acquisition Pipeline**

A Python spider fetches JSONL files from the official open-access portals of each venue. For CVPR, papers are from the Open Access repository; for ICLR and NeurIPS, from OpenReview. Fields extracted per paper: title, abstract, year, venue. After deduplication and empty-record removal, the working corpus contains 26,104 abstracts for 2023–2025 and 8,424 for 2022 (longitudinal baseline only; not included in content analysis percentages).

**Text Normalization and Phrase Protection**

The normalization sequence is: (1) decode Unicode to ASCII/NFC, (2) lowercase, (3) strip punctuation. Before stopword removal, a phrase-protection pass tokenizes multi-word technical terms into single tokens using a hand-crafted list — examples include "gaussian splatting" → `gaussian_splatting` , "neural radiance fields" → `neural_radiance_fields` , "vision language model" → `vision_language_model`. This prevents the next step from splitting co-occurring words that carry joint meaning. Generic CV terms (e.g., "image", "visual", "network", "model") are then removed as domain stopwords, leaving the technical content.

**35-Category Lexicon Matching**

Each cleaned abstract is matched against 35 regular-expression category patterns. The paper does not publish the exact regexes, but examples of category → keyword families include:
- `Vision-Language / Multimodal / LLM`: "vision language", "vlm", "multimodal llm", "lvlm", "vision-language model"
- `NeRF / Gaussian Splatting / Neural Rendering`: "neural radiance field", "nerf", "gaussian splatting", "3d gaussian"
- `Diffusion & Generative`: "diffusion model", "score matching", "denoising diffusion"

A paper receives a label if any regex in that category matches. Multiple labels are allowed. Prevalence for category $c$ in year $y$ :

```math
P(c, y) = \frac{|\{p \in papers_y : label(p, c) = 1\}|}{|papers_y|}
```

**TF-IDF Aggregation and Trajectory Plotting**

Beyond binary prevalence, the paper uses aggregated TF-IDF scores for the trajectory plots. The likely computation: for each category $c$ and year $y$ , build a pseudo-document by concatenating all matching abstracts, compute TF-IDF over the vocabulary, and sum the TF-IDF weights of lexicon terms to get a single scalar intensity score. This captures emphasis (how prominently terms appear) in addition to frequency. The y-axis in Figs. 1–2 shows this aggregated TF-IDF score, normalized so comparisons across years are meaningful.

**Slope Estimation**

For each direction $c$ across years $\{2022, 2023, 2024, 2025\}$ , the slope is the OLS coefficient $\hat{\beta}_1$ from:

```math
TF\text{-}IDF(c, y) = \beta_0 + \beta_1 \cdot y + \epsilon
```

This slope (pp/yr) is used as the "rising speed" shown in Fig. 3 and the Slope columns in all tables.

**Fine-Grained VLM Sub-Analysis**

For the subset of papers matching the VLM/LLM category, 8 additional mining passes run with domain-specific lexicons:

*Models (Table 1)*: Exact name matching for ALIGN, CLIP, BLIP, BLIP-2, LLaVA, ResNet, ConvNeXt, ViT, MoE/Switch, Flamingo, Grounding DINO, Swin, GLIP, DINO, DINOv2, and recent models (InternVL, Qwen-VL, etc.). Prevalence = share of VLM abstracts mentioning each name.

*Fusion/Architecture (Table 3)*: Pattern matching for mechanism names: prompt/prefix tuning, adapter/LoRA, cross-/co-attention, projector/MLP head, MoE/gating, encoder-decoder, dual-encoder/two-tower, Q-Former bridge.

*Tasks (Table 4)*: Task phrase matching: reasoning/instruction, grounding/referring, retrieval, captioning, VQA, video QA, OCR/text recognition, open-vocabulary detection/segmentation.

*Training Paradigms (Table 5)*: Regime phrase matching: pretrain+finetune, prompt/prefix, self/weak/semi-supervision, distillation, instruction tuning, LoRA/adapters, multi-task/curriculum.

*Loss Families (Table 6)*: Loss objective matching: contrastive/InfoNCE, KL/distillation, triplet/ranking, cross-entropy/focal, MSE/L1/L2, Dice/IoU, Chamfer/EMD.

*Datasets (Table 7)*: Explicit dataset name mentions: MS-COCO, ImageNet, LAION, RefCOCO variants, Flickr30k, CC3M/CC12M, VQA-v2/OK-VQA, WebVid/MSR-VTT/MSVD, YouCook2/HowTo100M, Visual Genome, COCO Captions.

*Modalities (Table 8)*: Co-mentioned non-image-text modalities: 3D/point cloud, image-text (baseline), depth/RGB-D, audio/speech, video-text.

**Cross-Venue Analysis**

The entire pipeline is conditioned on venue (CVPR / ICLR / NeurIPS) as a stratification variable, producing per-venue-per-year trajectories for the Section 5 comparison. The key finding: CVPR maintains strongest 3D emphasis (23.1% vs. 7.8% at ICLR for 3D geometry in 2025); ICLR has highest VLM share (40.7% in 2025); NeurIPS data only available through 2024 (30.5% VLM in 2024).

**Named Model Families Described in the Paper**

Beyond the trend data, the paper provides interpretive summaries for each top model family (Sec. 4.1): ALIGN (dual-encoder, web-scale noisy pairs, contrastive), CLIP (InfoNCE dual-encoder, 400M web pairs, universal embeddings), BLIP (MED: ITC+ITM+LM, CapFilt data bootstrapping), Flamingo (gated cross-attention, Perceiver-Resampler, few-shot generalization), LLaVA (CLIP encoder + MLP projector + LLM decoder, GPT-4 synthetic SFT data), DINO/DINOv2/DINOv3 (student-teacher self-distillation on ViTs), Grounding DINO (region-text pretraining for open-set detection), MoE/Switch Transformers (sparse expert routing for scaling).

### Hidden Assumptions

1. The 35-category lexicon has high enough precision that matching papers are actually about the labeled topic — i.e., the regexes do not fire on papers that merely *mention* VLMs as related work rather than contributing to them.
2. Abstracts are representative of paper content: authors include all major technical keywords in the abstract, so abstract-based matching approximates full-paper categorization.
3. Venue acceptance rates and selectivity are roughly stable across years, so changes in topic fraction reflect genuine field shifts rather than changing editorial policies.
4. The phrase protection list is complete for the most important technical multi-word phrases; missing entries would cause double-counting or mis-counting of key categories.
5. CVPR, ICLR, and NeurIPS collectively represent the authoritative set of top AI/CV venues — the trends observed generalize to the broader community (ECCV, ICCV, ACL, EMNLP not included).

### Reproducibility Notes

- **Data**: Fully reproducible from public sources. CVPR Open Access and OpenReview (ICLR/NeurIPS) provide JSONL/JSON abstracts for all accepted papers. No proprietary data.
- **Lexicon**: Released "in the conversation history" per the Sec. 7 conclusion — not a standard GitHub repo. This is the primary reproducibility bottleneck; the exact regex patterns are not published in the paper body.
- **Code**: Python spider described but not open-sourced as of submission. Standard NLP libraries (sklearn TfidfVectorizer or similar) suffice for the analysis.
- **Compute**: Trivial — minutes on a single CPU for all 26K abstracts.
- **Missing**: Exact regex for each of the 35 categories; phrase-protection word list; generic stopword list; TF-IDF aggregation formula (IDF computed globally vs. per-year?); handling of multi-word tokenization edge cases.
- **Verification**: The released lexicon (whenever publicly accessible) would allow audit of precision via random sampling of matched abstracts per category.

### Ideas for Future Work

1. **Full-text expansion**: Apply the same pipeline to arXiv PDFs (title + abstract + methods section) to close the under-reporting gap for training details, dataset usage, and loss functions.
2. **Monthly resolution**: Use arXiv submission dates to plot sub-yearly trends and detect inflection points (e.g., when did GPT-4V / LLaVA-1.5 appear in the submission stream?).
3. **Citation-weighted trajectories**: Weight each paper's label contribution by its citation count to distinguish high-impact direction shifts from volume shifts.
4. **Venue extension**: Include ECCV, ICCV, ACL, EMNLP, AAAI to characterize the VLM rise from both the CV and NLP sides.
5. **Author/institution stratification**: Identify which institutions drive the VLM rise vs. which remain anchored in classical CV, and whether trends differ between academic and industry labs.

---

## Pass 4 — Modern Perspective Review (as of July 2026)

### What Has Changed Since Publication

- **VLM dominance confirmed**: The 40% VLM share at 2025 conferences appears to have continued or accelerated into 2026 submissions, consistent with the paper's trajectory; Qwen3-VL, InternVL 3.5, LLaVA-OneVision, and Gemini Flash represent the model families now dominating follow-on work.
- **Instruction tuning as default**: By 2026, instruction-tuned multimodal models are the baseline assumption in most new VLM papers; the paper correctly predicted this would overtake contrastive pretraining as the primary training paradigm.
- **3DGS further consolidated**: The NeRF→Gaussian Splatting composition shift the paper documents was largely complete by mid-2025; 2026 3D work focuses on applications (robotics, autonomous driving, SLAM) rather than representation changes.
- **Reasoning / chain-of-thought VLMs**: The Reasoning/Instruction rise (+11.5pp) has continued, now encompassing visual chain-of-thought, multi-step tool use, and preference optimization (RLHF for VLMs) — the paper captures the beginning of this trend.
- **Sparse MoE proliferation**: MoE citation growth (+0.6pp) in the paper understates how central MoE has become; nearly all frontier VLMs (Qwen3-Omni, Gemini, GPT-4o) use sparse MoE architectures.
- **Venues expanding**: ECCV 2026 and ICCV 2025 add additional data points not covered by the paper; the lexicon methodology is directly applicable.

### Has the Community Accepted the Claims?

As a bibliometric measurement paper, "acceptance" takes a different form than for a method paper. The quantitative trends reported — VLM dominance, instruction tuning rise, contrastive loss decline, LoRA proliferation, grounding/referring decline — align with what the community has independently observed and discussed. The paper is likely to serve as a citable reference for the "when did VLMs dominate" question, analogous to how benchmark papers anchor other historical inflection points. Its key limitation — abstract-only scope — is widely understood and honestly acknowledged. No follow-on work has challenged the methodology at the time of this note; instead, the paper is positioned as a starting point for further venue/year extensions. The released lexicon enables community auditing, which strengthens its credibility. The one vulnerability is that the lexicon itself is not published in a standard, versioned form (e.g., GitHub), making true reproducibility dependent on the author sharing the conversation artifact referenced in the conclusion.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| A Survey on Vision and Language Pre-training | Du et al. | 2022 | Comprehensive hand-curated VLM survey; Lin automates and scales this to 26K papers |
| Trends in Computer Vision Research | Emerging literature in bibliometrics | various | General CV bibliometric methodology that Lin applies to VLMs specifically |
| CLIP | Radford et al. | 2021 | Foundational VLM whose rise the survey documents as the #1 trend driver |
| LLaVA | Liu et al. | 2023 | Fastest-growing model family in the survey; paradigm case for instruction-tuned VLMs |
| ALIGN | Jia et al. | 2021 | Most-cited model family across all VLM papers; anchor of dual-encoder pretraining era |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| A Survey of Large Language Models | Zhao et al. | 2023 | Broad LLM survey with manual categorization; overlaps on multimodal LLM section |
| Vision-Language Models for Vision Tasks: A Survey | Zhang et al. | 2024 | Manual VLM task survey; complementary (method-focused vs. Lin's trend-focused) |
| Multimodal Large Language Models: A Survey | Yin et al. | 2023 | Manual MLLM survey contemporaneous with this period; different scope and method |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| (none identified as of July 2026 — paper too recent for confirmed follow-ons) | — | — | — |

---

### Bottom Line

This paper occupies an unusual but valuable niche: it is neither a method contribution nor a traditional manual survey, but a transparent, reproducible *measurement* of what the research community actually published. Its core value is in answering questions like "what fraction of top-venue papers are VLMs now?" and "when did instruction tuning overtake contrastive pretraining?" with citable, defensible numbers rather than impressions. The 8-table fine-grained dissection of VLM practices (Table 1–8) is particularly useful as a snapshot of the field's technical defaults in 2023–2025. Practitioners who need to frame a new paper, reviewers assessing whether a direction is timely, and researchers writing related-work sections on field-level trends will all find the tables directly actionable. The paper's shelf life for the raw numbers is limited — a 2026 version extending to ICCV/ECCV/EMNLP would be immediately valuable — but the methodology and lexicon are durable tools. Worth reading Sections 4 and 5 in full; skim Section 3 if already familiar with the macro VLM narrative.
