# Tracking Anything with Decoupled Video Segmentation

- **Authors:** Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, Joon-Young Lee
- **Affiliations:** University of Illinois Urbana-Champaign, Adobe Research
- **Published:** ICCV 2023, arXiv:2309.03903 (7 Sep 2023)
- **Keywords:** video segmentation, decoupled segmentation, temporal propagation, tracking, open-world, SAM
- **Webpage:** https://hkchengrex.github.io/Tracking-Anything-with-DEVA
- **GitHub:** https://github.com/hkchengrex/Tracking-Anything-with-DEVA

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|-----------|
| **Category** | Video segmentation system paper; proposes a modular, decoupled framework for "tracking anything" without per-task video-level training |
| **Context** | Builds on SAM (universal image segmentation), XMem (memory-based VOS), and Mask2Former; extends "tracking-by-detection" with a bi-directional temporal propagation mechanism and in-clip consensus denoising |
| **Correctness** | Core assumptions are well-supported: image-level labels are indeed cheaper than video labels, and temporal propagation naturally generalizes across classes. Results validated across four distinct benchmarks (VIPSeg, BURST, Ref-Davis, DAVIS). |
| **Contributions** | (1) DEVA: decoupled video segmentation = swappable task-specific image model + universal class-agnostic propagation module; (2) In-clip consensus denoising via integer programming; (3) Bi-directional merging of propagated and new image segmentations; (4) SOTA on VIPSeg and BURST, competitive on referring and unsupervised VOS |
| **Clarity** | Well-written with clear formulations, good illustrative figures, and thorough ablations |

**30-second summary.** DEVA (DEcoupled Video segmentation Algorithm) decouples video segmentation into two components: a task-specific image segmentation model (e.g., Mask2Former, SAM) and a single class-agnostic temporal propagation model trained once on generic mask-propagation data. The key innovation is bi-directional propagation: segmentations from future frames are back-aligned into the current frame via an in-clip consensus (which denoises per-frame errors using integer programming), and this consensus is periodically merged with the forward-running propagated result via bipartite matching. This design enables "tracking anything" by plugging in any image model per task — no video-level training required per task — while achieving SOTA results on large-vocabulary and open-world video segmentation benchmarks.

---

## Pass 2 — Careful Read

### Core Idea in One Sentence

DEVA "tracks anything" by pairing a swappable task-specific image segmentation model with a single class-agnostic temporal propagation model, connected through a bi-directional propagation pipeline that denoises per-frame errors via in-clip consensus and merges propagated and new image segmentations gracefully.

### Method / Approach

- **Decoupled two-module design:** Image segmentation model $Seg(I)$ outputs task-specific per-frame masks; temporal propagation model $Prop(H, I)$ (based on XMem) takes a set of past segmented frames as memory $H$ and propagates to the query frame. The propagation model is trained once on class-agnostic data and reused across all tasks; only the image model changes per task.
- **In-clip consensus (denoising):** For a clip of $n$ frames, segmentations from future frames are spatially aligned back to the current frame using the propagation model (single-frame memory). All resulting mask proposals are pooled and filtered via an integer program that selects proposals supported by other proposals (high pairwise IoU) while rejecting lone noisy detections and overlapping selections.
- **Merging propagation with consensus:** Every 5 frames, the running propagation result $R_t$ (from the past) is merged with the new in-clip consensus $C_t$ (from the near future) via greedy bipartite matching on mask IoU. Matched pairs are unioned; unmatched masks from both sides are preserved, allowing new objects to enter and existing ones to persist.
- **Online vs. semi-online:** Online mode uses $n=1$ (no lookahead); semi-online uses $n=3$ frames, which improves accuracy at the cost of a small latency (7.8 FPS vs. 10.3 FPS).

### Key Results

| Task / Benchmark | Metric | DEVA (best) | Best prior SOTA | Improvement |
|---|---|---|---|---|
| VIPSeg (large-scale VPS) | VPQ (avg) | 52.2 (Mask2Former+SwinB, semi-online) | 37.5 (Video-K-Net+SwinB, end-to-end) | +14.7 VPQ |
| VIPSeg | STQ | 52.2 | 45.2 | +7.0 STQ |
| BURST open-world (test) | OWTA_all | 70.5 (Mask2Former, semi-online) | 57.5 (Mask2Former+STCN) | +13.0 |
| Ref-DAVIS | J&F | 66.3 | 61.6 (VLT) | +4.7 |
| Ref-YouTubeVOS | J&F | 66.0 | 63.8 (VLT) | +2.2 |
| DAVIS-16 val | J&F | 88.9 | 85.9 (PMN) | +3.0 |
| DAVIS-17 val | J&F | 73.4 | 70.4 (Propose-Reduce) | +3.0 |

**Ablation findings:**
- Spatial alignment within in-clip consensus is critical: removing it drops VPQ from 38.3 → 32.8 (−5.5).
- Bi-directional propagation is essential: tracking-by-detection baselines (IoU, flow, query association) all stay near VPQ 28, while DEVA reaches 36.4.
- Clip size $n=3$ and merge frequency every 5 frames give best performance/speed trade-off.
- Gains over end-to-end are largest (>60% relative) for rare classes when only 10% of target training data is available.

### Strengths

- **Plug-and-play modularity:** Any new image segmentation model (SAM, Grounding-DINO+SAM, Mask2Former, EntitySeg) can be dropped in without modifying or retraining the propagation module.
- **Data efficiency:** The propagation model never sees target-task data; task-specific data is only needed for the cheaper image model.
- **Denoising beyond tracking-by-detection:** In-clip consensus can improve upon the image model's per-frame predictions, unlike pure tracking-by-detection approaches that treat detections as immutable.
- **Generality across tasks:** A single framework covers four distinct video segmentation settings with competitive performance in each.
- **Long-range tracking:** Built on XMem's long-term memory, DEVA maintains coherent tracks across long videos, unlike clip-based end-to-end models.

### Weaknesses / Open Questions

1. **New object detection delay:** New objects are only detected at merge intervals (every 5 frames by default), so a newly appearing object can be invisible for up to 5 frames.
2. **End-to-end still better with sufficient data:** On small-vocabulary benchmarks like YouTube-VIS (40 classes, ample annotations), end-to-end methods outperform DEVA — the decoupled approach is not a universal replacement.
3. **ILP scalability:** The in-clip consensus integer program grows with the number of proposals per frame; performance at very high object counts is not analyzed.
4. **Propagation model quality cap:** The ceiling of DEVA's performance is bounded by XMem's propagation quality; failure modes of XMem (e.g., fast motion, large appearance change) cascade into DEVA.
5. **No explicit re-identification:** After long occlusions, there is no explicit re-ID; the merge step's IoU matching may fail to re-associate a re-appearing object.

### References to Follow Up

1. **XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model** — Cheng & Schwing, ECCV 2022: Core temporal propagation backbone; understanding its memory model is essential to understanding DEVA's propagation module.
2. **Segment Anything** — Kirillov et al., ICCV 2023: Universal promptable image model that DEVA integrates as a drop-in image segmentation backbone for open-world tracking.
3. **Video-K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation** — Li et al., CVPR 2022: Main end-to-end baseline on VIPSeg; directly compared against in ablations.
4. **VIPSeg: Large-scale Video Panoptic Segmentation** — Miao et al., CVPR 2022: Primary large-vocabulary benchmark used to motivate and evaluate the decoupled approach.
5. **BURST: A Benchmark for Unifying Object Recognition, Segmentation and Tracking in Video** — Athar et al., WACV 2023: Open-world tracking dataset central to the open-world evaluation.

---

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Problem Setup**

A video $\{I_t\}$ must be segmented into a set of per-object binary masks at each frame, with consistent object IDs across time. Formally, the final segmentation at time $t$ is $M_t = \{m_i\}$ , a set of non-overlapping binary segments.

**Module 1 — Image Segmentation Model**

$Seg(I_t) = Seg_t = \{s_i\}$ : any image model that produces a set of non-overlapping masks per frame. The model is task-specific and trained with image-level supervision only (no video data needed). In experiments, Mask2Former, Video-K-Net (image head), EntitySeg, ReferFormer (for referring), and DIS (for saliency) all serve as drop-in choices.

**Module 2 — Temporal Propagation Model**

$Prop(H, I_t)$ : a modified XMem that takes a memory set $H = \{(I_\tau, M_\tau)\}$ of past segmented frames and produces a segmentation of query frame $I_t$ . The model is trained in a class-agnostic fashion on image segmentation datasets (COCO, LVIS, SA-1B) plus video object segmentation datasets (DAVIS, YouTube-VOS). This training means the model can propagate arbitrary object masks, regardless of semantic class — the key to task-agnostic generalization. Top-k filtering with $k=30$ is applied at each step.

**In-clip Consensus (Section 3.2.1)**

For a clip of $n$ frames starting at time $t$ :

**Step 1 — Spatial Alignment.** For each future frame $t+i$ ($0 < i < n$), back-propagate its segmentation to frame $t$:

```math
\widehat{Seg}_{t+i} = Prop(\{I_{t+i}, Seg_{t+i}\}, I_t), \quad 0 < i < n
```

This temporary single-frame memory is discarded immediately after alignment; it does not update the global memory $H$.

**Step 2 — Proposal Pool.** Collect all aligned segments as proposals:

```math
P = \bigcup_{i=0}^{n-1} \widehat{Seg}_{t+i} = \{p_i\}
```

**Step 3 — Integer Programming.** Select a subset $v^* \in \{0,1\}^{|P|}$ maximizing:

```math
v^* = \arg\max_v \sum_i (Supp_i + Penal_i) \quad \text{s.t.} \sum_{i,j} Overlap_{ij} = 0
```

where:

```math
Supp_i = v_i \sum_j \begin{cases} IoU_{ij}, & \text{if } IoU_{ij} > 0.5 \text{ and } i \neq j \\ 0, & \text{otherwise} \end{cases}
```

```math
Overlap_{ij} = \begin{cases} v_i v_j, & \text{if } IoU_{ij} > 0.5 \text{ and } i \neq j \\ 0, & \text{otherwise} \end{cases}
```

```math
Penal_i = -\alpha v_i, \quad \alpha = 0.5
```

Intuitively: lone proposals with no IoU > 0.5 support score only $-\alpha$ and are rejected; supported proposals gain positive Supp and are selected; overlapping proposals cannot both be selected (Overlap constraint).

The consensus output is:

```math
C_t = \{p_i \mid v_i^* = 1\}
```

Overlapping consensus segments are resolved by rendering smaller segments first (they are more fragile to being overwritten).

**Merging Propagation and Consensus (Section 3.2.2)**

Every 5 frames, merge $R_t = Prop(H, I_t)$ with $C_t$ :

**Step 1 — Association.** Find $a_{ij} \in \{0,1\}$ via greedy bipartite matching:

```math
e_{ij} = \begin{cases} IoU(r_i, c_j), & \text{if } IoU(r_i, c_j) > 0.5 \\ -1, & \text{otherwise} \end{cases}
```

Set $a_{ij} = 1$ if $e_{ij} > 0$, else $0$. (Uniqueness guaranteed when IoU threshold > 0.5 for non-overlapping masks.)

**Step 2 — Merge.** Construct final segmentation:

```math
M_t = \{r_i \cup c_j \mid a_{ij} = 1\} \cup \{r_i \mid \forall_j\, a_{ij} = 0\} \cup \{c_j \mid \forall_i\, a_{ij} = 0\}
```

Matched pairs are unioned (consensus refines the propagation mask). Unmatched propagated segments persist (objects that briefly failed the image model). Unmatched consensus segments are added (new objects entering the scene).

**Segment Deletion.** Each propagated segment $r_i$ has a counter $cnt_i$ , incremented when $r_i$ is unmatched in a merge step, reset when matched. When $cnt_i \geq L=5$ , $r_i$ is removed from memory (likely gone out of view or was a false positive).

**Offline Variant (Referring / Unsupervised VOS)**

For referring video segmentation, 10 uniformly spaced frames are run through the image model; the highest-confidence frame is the "key frame". In-clip consensus is computed at the key frame only (aligning other candidate frames to it). Temporal propagation then runs forward and backward from the key frame. No additional image model queries after initialization.

### Hidden Assumptions

1. The temporal propagation model, trained on class-agnostic generic datasets, generalizes sufficiently to novel object categories and domains without fine-tuning.
2. $n=3$ frames of lookahead is sufficient to catch most noisy detections; very brief false positives (shorter than the clip window) may not be denoised.
3. The IoU > 0.5 threshold is a reliable criterion for mask correspondence across adjacent frames; it can fail under fast motion or rapid scale changes.
4. Image segmentation outputs are non-overlapping binary masks — downstream modules assume this, so models that produce overlapping soft masks require additional post-processing.
5. Object appearance changes slowly enough that a merged mask ($r_i \cup c_j$) is still a valid representation; very fast deformation or scale change could cause union masks to be unnecessarily large.

### Reproducibility Notes

- **Propagation model training data:** COCO, LVIS, SA-1B (image), DAVIS-17, YouTube-VOS (video) — all publicly available. Training procedure follows XMem's protocol with class-agnostic masks.
- **Image models:** Off-the-shelf open-sourced weights used for most experiments (Mask2Former, Video-K-Net, EntitySeg, ReferFormer, DIS). No proprietary models.
- **Hyperparameters:** Clip size $n=3$, merge frequency every 5 frames, $\alpha=0.5$, $L=5$, top-k $k=30$. All reported in the paper and ablated in Table 5.
- **Evaluation:** Official evaluation codebases and servers used — results directly comparable to reported baselines.
- **Missing details:** Exact training schedule and data mixing ratios for the class-agnostic propagation model are deferred to an appendix (not fully visible in arXiv main paper); the appendix covers XMem modification specifics and per-task protocol variants (referring, unsupervised).
- **Code:** Available at https://github.com/hkchengrex/Tracking-Anything-with-DEVA — includes training scripts, evaluation, and demo.

### Ideas for Future Work

1. **Learnable merging:** Replace the hand-crafted IoU bipartite matching with a learned association module that considers appearance, motion, and semantic features for more robust re-identification after long occlusions.
2. **Adaptive merge frequency:** Dynamically decide when to merge based on estimated confidence of the propagation (e.g., merge more frequently during fast motion or scene changes).
3. **In-clip consensus without the propagation model:** The spatial alignment step uses the propagation model, making it computationally expensive. Optical flow or a lightweight warping network could reduce overhead.
4. **Extending to 3D or multi-view:** The decoupled design naturally extends to 3D video segmentation by replacing the image model with a 3D image model and the propagation with a 3D spatial-temporal propagation.
5. **Applying DEVA's bi-directional design to SAM 2:** SAM 2 uses a streaming memory but still runs the image encoder every frame. DEVA's in-clip consensus could be applied on top of SAM 2's per-frame predictions to denoise its outputs for open-world tasks.

---

## Pass 4 — Modern Perspective Review (as of June 2026)

### What Has Changed Since Publication

- **SAM 2 (2024):** Meta's direct sequel to SAM extends segment-anything to video via a streaming memory (similar to XMem) built into a unified model. It largely supersedes the need for the DEVA pipeline for SAM-powered video segmentation, though DEVA's modular design still allows swapping any image model — something SAM 2 does not natively support.
- **Grounded-SAM 2 and open-vocabulary tracking:** The open-source community built Grounded-SAM 2 shortly after SAM 2, combining Grounding-DINO + SAM 2 in a similar spirit to DEVA's plug-and-play philosophy.
- **SAMURAI (2024):** Builds on SAM 2's memory mechanism with Kalman-filter-based motion modeling for single-object tracking, showing that the community continued exploring the "propagation + appearance model" design space.
- **Evaluation benchmark shifts:** BURST and VIPSeg remain relevant; newer benchmarks for long-video and in-the-wild evaluation have emerged (e.g., LVOS for long-video VOS).
- **Universal video models:** Works like VideoGPT, Video-LLaVA, and video foundation models have started unifying video understanding tasks differently from DEVA's modular approach — through large-scale pre-training.

### Has the Community Accepted the Claims?

Yes. The core DEVA claim — that decoupling video segmentation into an image model and a class-agnostic temporal propagation model is a competitive and scalable alternative to end-to-end training — was validated by the community through follow-on work. SAM 2 can be seen as the ultimate validation: Meta's own approach to video segmentation adopts a memory-based propagation model conceptually aligned with DEVA's design. DEVA's specific bi-directional propagation with in-clip consensus was not directly adopted but influenced the thinking around temporal denoising. The practical plug-and-play advantage — any image model, no video retraining — continues to be relevant in 2026 for tasks where no large-scale video training data exists.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|---|---|---|---|
| XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model | Cheng & Schwing | 2022 | Temporal propagation backbone; DEVA modifies XMem's memory architecture for class-agnostic use |
| Segment Anything (SAM) | Kirillov et al. | 2023 | Universal image segmentation model integrated as DEVA's image module |
| Mask2Former | Cheng et al. | 2022 | Primary image segmentation backbone for panoptic experiments |
| Video-K-Net | Li et al. | 2022 | End-to-end VPS baseline; main comparison and image model source on VIPSeg |
| STCN: Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation | Cheng et al. | 2021 | Earlier propagation model; baseline tracker in BURST evaluation |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|---|---|---|---|
| OWTB: Opening Up Open World Tracking | Liu et al. | 2022 | Competing open-world tracking approach; uses IoU + flow + Re-ID; outperformed by DEVA on BURST |
| SAM-PT: Segment Anything Meets Point Tracking | Rajivc et al. | 2023 | Concurrent; combines point tracking with SAM for VOS; DEVA tracks masks directly rather than points |
| Grounding DINO | Liu et al. | 2023 | Concurrent; open-vocabulary detection; integrated with SAM inside DEVA for text-prompted tracking |
| UNINEXT | Yan et al. | 2023 | Concurrent end-to-end unified tracker across multiple tasks; different philosophy (joint training) vs. DEVA (decoupled) |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|---|---|---|---|
| SAM 2: Segment Anything in Images and Videos | Ravi et al. | 2024 | Direct successor in spirit; unified streaming-memory model for video segmentation; integrates image and propagation into one model |
| SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory | Yang et al. | 2024 | Builds on SAM 2 with Kalman-filter motion priors; same decoupled-tracking spirit |
| Grounded-SAM 2 | Community / IDEA Research | 2024 | Directly extends DEVA's integration idea (Grounding-DINO + SAM) to SAM 2's video backbone |

---

### Bottom Line

DEVA is a foundational paper that crystallized the "decoupled video segmentation" paradigm and demonstrated it could match or beat end-to-end approaches on large-vocabulary and open-world benchmarks. It is worth reading as a design reference — the bi-directional propagation logic and the in-clip consensus integer programming are clearly explained and directly reproducible. The paper's practical influence is evidenced by SAM 2 adopting a similar memory-propagation architecture. For readers in 2026, SAM 2 largely supersedes DEVA for SAM-based video tracking, but DEVA remains the clearest exposition of why decoupling works and how to engineer the connection between image and temporal modules — making it a useful reference for anyone building modular video understanding systems or working in data-scarce video settings.
