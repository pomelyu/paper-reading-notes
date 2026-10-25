# DSDrive: Distilling Large Language Model for Lightweight End-to-End Autonomous Driving with Unified Reasoning and Planning

- **Authors:** Wenru Liu, Pei Liu, Jun Ma
- **Affiliations:** The Hong Kong University of Science and Technology, China (as stated in the supplied manuscript)
- **Published:** arXiv:2505.05360v1, May 8, 2025; no accepted venue confirmed in the checked official record
- **Keywords:** autonomous driving, reasoning distillation, lightweight language model, waypoint prediction, closed-loop evaluation
- **Webpage:** https://personal.hkust-gz.edu.cn/junma/Reasearch.html
- **Paper:** https://arxiv.org/abs/2505.05360
- **Video:** https://youtu.be/op8PzQurugY

**Source and scope.** Read the complete supplied 13-page v1 PDF, including Algorithms 1–2, Tables I–IV, figures, and references. The retained [source extraction](resources/paper.md) was produced locally with `pdftotext -layout`; it preserves page breaks and two-column layout rather than being a semantically reconstructed Markdown conversion. There is no appendix in this PDF. Paper-specific claims below refer to this version; later-work links were checked on October 25, 2026. No official DSDrive code repository, weights, or Hugging Face release was verified from the paper and author research page, so no such release is asserted.

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|---|
| **Category** | An empirical end-to-end driving architecture and training paper: distill textual driving reasoning into a nominally 1B LLaMA backbone and train reasoning and numerical waypoint outputs together. |
| **Context** | The direct foundation is **LMDrive: Closed-Loop End-to-End Driving with Large Language Models**, with BLIP-2-style visual querying and DeepSeek-R1-inspired think-and-answer supervision. It connects language explanation to the planning-oriented tradition of [Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/). |
| **Correctness** | Closed-loop CARLA evaluation supports improved task completion over the compact baseline. It does not establish universally safer behavior, causal faithfulness of explanations, or embedded deployment. Main-table and ablation results differ without a clear reconciliation. |
| **Contributions** | Offline teacher-generated reasoning targets; ground-truth waypoints embedded in textual answers; a compact shared backbone with reasoning and waypoint heads; closed-loop LangAuto comparisons and memory/runtime measurements. |
| **Clarity** | The motivation and training flow are understandable. Exact model identifiers, data splits, fusion details, output dimensions, loss weights, and inference timing boundaries remain underspecified. The abstract's broad efficiency and reliability language needs qualification against the tables. |

**30-second summary.** DSDrive uses Qwen2.5-VL-max as an offline VLM[^vlm] teacher to describe driving scenes and suggest actions, then appends navigation instructions and ground-truth waypoints to create think-and-answer targets. A small LLaMA-based driving model shares multimodal hidden states between an autoregressive CoT[^cot] decoder and a direct waypoint predictor. Joint supervision improves LangAuto-Short driving score from 45.03 for LMDrive with LLaVA-7B to 62.05, while reported peak memory falls from 14,263 to 8,082 MB. Long-route gains are modest and Tiny-route score is slightly lower; several infraction measures worsen. The useful idea is task-aligned reasoning supervision for a compact closed-loop policy, with incomplete evidence about explanation faithfulness and full reasoning latency.

## Pass 2 — Careful Read

### Core Idea in One Sentence

Teach a compact multimodal driving model to explain scenes and predict trajectories together by supervising both textual answers and numerical planning with the same ground-truth waypoints.

### Method / Approach

- **Generate driving reasoning offline:** Prompt Qwen2.5-VL-max with driving images, navigation, and structured scene/object/decision questions; collect explanations and combine them with ground-truth navigation and waypoints.
- **Compress multimodal input:** Reuse a pretrained LMDrive visual encoder, align visual features through learnable Q-Former queries and a projection, and combine them with navigation/question token embeddings for a LLaMA-1B backbone.
- **Coordinate outputs through supervision:** Decode shared hidden states with a numerical waypoint head, a Qwen2-style textual reasoning decoder, and a navigation-completion classifier. The same trajectory appears in the textual target and waypoint target.
- **Train jointly and execute closed loop:** Optimize waypoint absolute error plus answer-token and completion classification losses; use predicted waypoints to produce steering, throttle, and brake in CARLA.

#### Train Data

| Name | Usage |
|---|---|
| Internal DSDrive reasoning dataset | 11,067 driving clips with teacher-generated reasoning, navigation, and ground-truth waypoint supervision. The paper does not give a canonical upstream training-set name or precise split construction. |

#### Evaluation/Validation Data

| Name | Usage |
|---|---|
| LangAuto | CARLA closed-loop Long, Short, and Tiny routes; no separately named validation dataset or train/test overlap audit is reported. |

### Key Results

Table II, reproduced below. DS is driving score, RC is route completion (%), and IS is infraction score; higher is better for all three. Incident columns are normalized by simulated scenarios, with lower better. Long routes exceed 500 m; Short routes span 150–500 m; Tiny routes are shorter than 150 m.

| Route | Method | DS ↑ | RC ↑ | IS ↑ | Collision ↓ | Red-light violation ↓ | Outside lanes ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Long | Vanilla LLaMA-1B | 21.27 | 23.03 | 0.93 | 0.00 | 0.03 | 0.16 |
| Long | LMDrive LLaMA-1B | 21.59 | 27.52 | 0.87 | 0.19 | 0.06 | 0.13 |
| Long | LMDrive LLaVA-7B | 28.51 | 37.87 | 0.80 | 0.13 | 0.22 | 0.25 |
| Long | DSDrive LLaMA-1B | 29.57 | 39.31 | 0.77 | 0.12 | 0.24 | 0.32 |
| Short | Vanilla LLaMA-1B | 38.75 | 42.88 | 0.87 | 0.07 | 0.14 | 0.21 |
| Short | LMDrive LLaMA-1B | 41.86 | 48.71 | 0.88 | 0.07 | 0.14 | 0.21 |
| Short | LMDrive LLaVA-7B | 45.03 | 52.90 | 0.84 | 0.07 | 0.20 | 1.00 |
| Short | DSDrive LLaMA-1B | 62.05 | 76.18 | 0.81 | 0.07 | 0.27 | 1.00 |
| Tiny | Vanilla LLaMA-1B | 47.76 | 53.70 | 0.84 | 0.06 | 0.00 | 0.19 |
| Tiny | LMDrive LLaMA-1B | 49.97 | 53.57 | 0.88 | 0.25 | 0.00 | 0.06 |
| Tiny | LMDrive LLaVA-7B | 61.91 | 73.19 | 0.84 | 0.19 | 0.06 | 0.13 |
| Tiny | DSDrive LLaMA-1B | 60.67 | 72.53 | 0.84 | 0.06 | 0.00 | 0.25 |

Compared with LMDrive LLaVA-7B, DSDrive changes DS by +1.06, +17.02, and −1.24 points on Long, Short, and Tiny respectively. Its main gain is completion, especially on Short routes. Lower IS on Long/Short and higher red-light/outside-lane values prevent interpreting this as an across-the-board safety improvement. Zero collisions for Vanilla on Long also illustrate why incident counts alone cannot establish a strong policy when completion is only 23.03%.

**Dual-head data ablation (Table III; a separate result set).** CoT only excludes appended ground-truth navigation/waypoints from the answer target; GT only excludes teacher-generated reasoning. These are target-construction ablations, not clearly documented parameter-matched removals of each head.

| Route | CoT only DS / RC / IS | GT only DS / RC / IS | Dual-task DS / RC / IS |
|---|---|---|---|
| Long | 23.56 / 27.08 / 0.88 | 21.91 / 26.51 / 0.85 | 27.35 / 29.52 / 0.89 |
| Short | 38.42 / 44.99 / 0.83 | 40.90 / 49.72 / 0.86 | 42.91 / 51.76 / 0.85 |
| Tiny | 46.14 / 54.46 / 0.87 | 48.50 / 55.83 / 0.88 | 49.35 / 54.50 / 0.89 |

- **What ablation supports:** Dual-task gives the highest DS in each of these three comparisons.
- **What it does not support:** Dual-task does not win every metric: Short IS trails GT only (0.85 vs 0.86), and Tiny RC trails GT only (54.50 vs 55.83). The prose overstates uniform dominance.
- **Unreconciled results:** Dual-task Short DS is 42.91 here versus 62.05 in Table II. The paper does not explain an altered protocol or checkpoint sufficiently to combine these tables into one controlled estimate.

**Efficiency (Table IV).**

| Model | Inference time (s) | Peak memory (MB) |
|---|---:|---:|
| LMDrive LLaVA-7B | 0.06 | 14,263 |
| LMDrive LLaMA-1B | 0.05 | 6,682 |
| DSDrive LLaMA-1B | 0.05 | 8,082 |

DSDrive uses about 43.3% less memory and 16.7% less reported time than the 7B baseline, but 21.0% more memory than compact LMDrive. The paper does not specify generated answer length or clearly establish whether complete autoregressive reasoning is included in the 0.05 s figure. The 1B label describes the language backbone, not an audited total-system parameter count.

### Strengths

- **Closed-loop evidence:** Evaluates actual feedback-driven simulation rather than relying entirely on logged future-path similarity.
- **Planning-grounded target design:** Reuses waypoint supervision in the textual answer, creating a shared task endpoint instead of unrelated generic explanation generation.
- **Relevant compact baseline:** Includes both vanilla and trained LLaMA-1B variants, showing that a smaller backbone alone does not explain the gain.
- **Visible resource trade-off:** Reports the extra memory cost over compact LMDrive alongside savings over its larger version.

### Weaknesses / Open Questions

1. **Explanation faithfulness:** Shared features and similar waypoint visualizations do not demonstrate that the emitted explanation causally determines the numerical plan. No counterfactual explanation intervention is reported.
2. **Confounded adaptation:** A specialized student receives waypoint targets while the qualitative Qwen2.5-VL-7B comparison does not receive equivalent adaptation. Better example answers do not establish superior general reasoning or teacher-level capability.
3. **Incomplete reproducibility:** Data collection, split definitions, exact model checkpoints, query/head sizes, loss coefficients, and inference settings are missing.
4. **Narrow safety evidence:** One simulator benchmark, no real-vehicle results, no reported uncertainty bars, and mixed infraction metrics limit reliability claims.
5. **Ablation consistency:** Table III differs substantially from Table II; architecture, extra labels, target format, and training budget are not fully isolated.
6. **Runtime boundary:** Autoregressive explanation length and its scheduling relative to the 10 Hz controller are not resolved by a single forward-time number. Orin deployment is proposed, not measured.

### References to Follow Up

1. **LMDrive: Closed-Loop End-to-End Driving with Large Language Models** — Hao Shao et al., CVPR 2024: the visual-encoder, closed-loop baseline, and LangAuto foundation needed to reconstruct DSDrive's inherited system.
2. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — Junnan Li et al., ICML 2023: explains the learnable-query interface between visual features and a language model.
3. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI / Daya Guo et al., arXiv 2025: motivates think-and-answer distillation; DSDrive does not thereby inherit its reinforcement-learning training pipeline.
4. **WiseAD: Knowledge Augmented End-to-End Autonomous Driving with Vision-Language Model** — Songyan Zhang et al., arXiv 2024: a nearby knowledge-transfer approach for separating language-model supervision from deployment cost.
5. **DriveLM: Driving with Graph Visual Question Answering** — Chonghao Sima et al., ECCV 2024: relates structured driving questions to perception, prediction, and planning reasoning.

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**1. Separate the offline teacher from the deployed policy.** Section III-A defines a reasoning model $V$ that receives image sequences $I$ , navigation $X_{navi}$ , and structured prompts. The manuscript calls the teacher Qwen2.5-VL-max; its exact parameter count, checkpoint/API revision, and reproducible access configuration are not specified. Its description as an open-sourced model should not be treated as verification that this exact “max” configuration has public weights. Qwen2.5-VL-7B is used separately for qualitative answer comparison; it is not the stated data-generation teacher.

```math
A_{vlm} = V(I, X_{navi}, X_{prompt}),
\qquad X_{prompt} = [S,Q_1,Q_2,Q_3].
```

The system prompt specifies interaction and output structure. Questions progress from scene comprehension (weather, time, road type/condition), to key objects (identity, location, characteristics, risks), to strategy and human-readable explanation. Although four considerations are listed, the symbolic template has three question prompts; the complete exact prompt strings are not supplied in the prose. Teacher output is concatenated with navigation and ground-truth waypoints into an explicit think-and-answer target. This is sequence-level supervision from generated answers, not teacher-logit matching, feature matching, or an online teacher in the control loop.

**2. Encode sensor history and instructions.** The model consumes front/left/right/rear camera images and LiDAR in CARLA. The symbolic image tensor is $I\in R^{T\times H\times W\times C}$ ; the paper does not explicitly give a separate camera axis or a precise LiDAR fusion equation. It reuses a pretrained visual encoder from LMDrive, with feature shape $F_v\in R^{T\times N_v\times D_v}$ . Figure 2 shows a vision encoder, a BEV[^bev] encoder, and navigation/question/scene query groups. This indicates an inherited multimodal perception stack rather than a demonstrated camera-only policy.

Navigation and the student question (such as “think and answer”) are tokenized with a LLaMA tokenizer into $H_l\in R^{N_l\times D_l}$ and repeated across $T$ frames. Learnable queries $Q\in R^{K\times D_l}$ attend to image features and textual context through a BLIP-2-inspired Q-Former. A linear projector maps its outputs into the backbone embedding width. The number $K$ and exact masking, fusion, and temporal aggregation recipes are not fixed in the source. In particular, this is not evidence for a newly specified cross-camera/time compression algorithm.

**3. Share the backbone representation across outputs.** A nominal LLaMA-1B processes the aligned visual and text embeddings. Its hidden states $H$ condition three outputs:

```math
(W_{pred}, A_{pred}, C_{pred}) = E(I,X_{navi},X_{qn}),
\qquad W_{pred}=f_{wp}(H),\qquad C_{pred}=f_{end}(H).
```

The waypoint and completion predictors are described as two-MLP constructions with an intervening ReLU. The waypoint head predicts future coordinates directly; the completion head classifies whether the current navigation instruction has ended. A custom Qwen2-style decoder produces textual reasoning from the same context. A practical implementation must choose how $H$ is pooled or indexed and specify waypoint count, time horizon, coordinate frame, and head dimensions; these are not adequately defined here.

“Dual-head” refers to reasoning and planning, while the full model additionally has the completion classifier. Coordination consists of shared features, joint optimization, and ground-truth trajectories represented in both targets. No explicit numerical-versus-textual trajectory consistency loss, feedback from generated reasoning into the waypoint head, or deterministic reconciliation mechanism is given. Thus the safe reconstruction is parallel prediction with aligned supervision, rather than a proven serial reason-then-act computation graph.

**4. Train the reasoning decoder with teacher forcing.** Select hidden states from the last few keyframes, concatenate the embeddings of the answer prefix, and apply the answer decoder with causal attention and position IDs. Algorithm 1 appends an end-of-sequence token when needed, shifts answer inputs and labels by one token, extracts only answer-position outputs, and projects to vocabulary logits. A minimum-length truncation handles mismatched lengths. Schematically, the answer objective is:

```math
L_{ans} = -\sum_{j=1}^{m}\log p(a_j\mid H,a_{<j}).
```

This equation expresses the intended token objective; exact start-token conventions, context masking, padding rules, truncation length, and handling of multi-frame answer boundaries need implementation decisions. Only the student driving model is trained in the distillation stage. The paper does not state the frozen/trainable status of every inherited perception component precisely enough to reproduce it unchanged.

**5. Joint task objective and schedule.** Waypoints use absolute error; answers and instruction completion use cross-entropy:

```math
L = \lambda_1 L_{wp}+\lambda_2 L_{ans}+\lambda_3 L_{end},
\qquad L_{wp}=\lVert W_{pred}-W_{gt}\rVert_1.
```

Coefficients are said to lie between zero and one, but their values and loss reductions are not reported. The dataset has 11,067 clips. Train for 50 epochs, batch size 1, weight decay 0.06, on four NVIDIA RTX 3090 GPUs with 24 GB each. Warm up for 2,000 iterations from $10^{-6}$ to $10^{-4}$ , then cosine-decay to $10^{-5}$ . The optimizer type, effective global batch size, distributed setup, gradient accumulation, precision, and wall-clock training cost remain unknown; four GPUs should not be silently translated into a batch size of four.

**6. Execute planning and optionally decode an explanation.** Waypoints are converted into steering, throttle, and brake, and the simulator evaluates a 10 Hz closed loop. The exact controller settings and coordination of language generation with control updates are not provided. Algorithm 2 starts from a beginning token, repeatedly combines context with token embeddings, applies an attention mask and positions, obtains last-token logits, applies a repetition penalty, and greedily selects tokens until an end token or a token limit.

A critical pseudocode gap is that Algorithm 2 does not explicitly update the current token/embedding or construct the growing generated prefix after appending a token. The accompanying prose clearly intends ordinary autoregression. A working reconstruction must feed back the accumulated generated prefix (or an equivalent cached decoding state); implementing the printed loop literally risks repeatedly using the same initial input. The repetition penalty, token limit, caching strategy, and full answer latency are unspecified.

**7. Interpret the evaluation at the correct level.** LangAuto provides routes across eight CARLA towns and varying environmental/weather conditions. RC measures progress, IS starts at 1 and is penalized for infractions, and DS combines progress and compliance. Scenario-normalized collision counts do not normalize exposure by distance traveled. Table II is the central closed-loop comparison; Figures 3–8 provide selected reasoning/control examples and visually similar text-head/planning-head waypoints. There is no quantitative text/trajectory agreement distribution or independent broad reasoning benchmark. No real-world open-loop dataset result or real-vehicle closed-loop trial is supplied.

### Datasets

The Pass 2 tables satisfy the repository's requested immediate Method/Results placement. The following canonical three-column tables provide the graph metadata and dataset provenance; model-pretraining corpora are not reverse-engineered from backbone citations.

#### Train Data

| Dataset | Usage | Proposed by |
|---|---|---|
| Internal DSDrive reasoning dataset | 11,067 driving clips used for 50-epoch joint student training. Teacher-generated scene/object/action reasoning is combined with navigation and ground-truth waypoints. Exact collection source, route split, filtering, and annotation release are not stated. | DSDrive |

#### Evaluation/Validation Data

| Dataset | Usage | Proposed by |
|---|---|---|
| LangAuto | Closed-loop CARLA Long (>500 m), Short (150–500 m), and Tiny (<150 m) routes across eight towns and varying conditions. Includes baseline and target-construction ablation comparisons. A separate validation split is not named. | LMDrive: Closed-Loop End-to-End Driving with Large Language Models |

CARLA is the simulator used to instantiate these routes, not an additional independent dataset row. The source does not explicitly establish that the 11,067 training clips are a named subset of LangAuto or of LMDrive's training corpus; those identities must not be assumed.

### Hidden Assumptions

1. **Textual supervision transfers useful decision structure:** Teacher explanations must contain actionable information beyond labels already present in navigation and trajectory targets, rather than merely adding correlated descriptive text.
2. **Semantic and numerical targets agree:** Appending an expert trajectory does not automatically make a teacher's independently generated maneuver description correct or consistent. No systematic conflict filtering is specified.
3. **Shared context yields faithful explanations:** Similar output trajectories are assumed to imply explanation/planning alignment, although parallel heads can exploit different information in the same hidden state.
4. **Teacher-forcing robustness transfers to rollout:** At test time, imperfect earlier tokens and unseen closed-loop states can shift the reasoning decoder away from its training distribution.
5. **Progress and incident exposure are comparable:** Higher completion changes time and distance exposed to infractions; aggregate scenario-normalized counts alone cannot resolve risk per kilometer.
6. **Small backbone means practical deployment:** Additional perception, query, reasoning-decoder, controller, and sensor-processing costs must fit the deployed hardware budget; the paper does not measure all these on an embedded system.

### Reproducibility Notes

- **Source coverage:** Complete 13-page v1 was available; no appendix appears in it. Local extraction and visual inspection of the architecture figure were used together because two-column extraction does not preserve diagram connectivity.
- **Code and weights:** No verified official DSDrive implementation or released checkpoint was found in the checked paper/author sources. LMDrive code is an inherited foundation, not a DSDrive release.
- **Teacher:** Need the exact Qwen2.5-VL-max version, image formatting, all prompts, sampling settings, filtering, generation cost, and the actual teacher-generated dataset.
- **Student:** Need the precise “LLaMA-1B” checkpoint, tokenizer, visual-encoder checkpoint, sensor calibration, BEV fusion, query counts, decoder size, and sequence/horizon settings. The citation to the Llama 3 report is insufficient to choose a unique checkpoint.
- **Training:** Reported schedule and four 24-GB RTX 3090 GPUs are useful, but optimizer, loss coefficients, trainable modules, random seeds, effective batch size, precision, and training duration are missing.
- **Evaluation:** Need exact simulator/version, routes, scenario seeds/counts, train/test town overlap, weather schedule, control gains, and the explanation for Table II/III differences.
- **Timing:** Need measurement hardware, batching, warmup, synchronization, whether sensor preprocessing and full language generation are included, answer lengths, and end-to-end latency percentiles. An Orin feasibility statement is not an Orin benchmark.

### Ideas for Future Work

1. **Test causal explanation alignment:** Swap or contradict language supervision and measure whether waypoint predictions change appropriately; compare textual and numerical trajectories quantitatively across all test scenes.
2. **Isolate the supervision gain:** Match data, steps, parameter count, and control settings across trajectory-only, reasoning-only, appended-waypoint, and explicit consistency-loss variants.
3. **Schedule reasoning around control:** Measure asynchronous explanations or on-demand decoding while keeping a fixed-rate waypoint controller, including worst-case latency and stale-context failures.
4. **Audit the teacher dataset:** Release prompts, clip provenance, train/test splits, target conflict checks, and failures in rare hazards; test whether teacher mistakes are amplified by the student.
5. **Evaluate transfer and safety exposure:** Add held-out towns, reactive-agent stress tests, infraction rates per kilometer, multiple seeds, and embedded hardware measurements before making deployment claims.

## Pass 4 — Modern Perspective Review (as of October 2026)

### What Has Changed Since Publication

- **Adaptive reasoning is a concrete later alternative:** [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](https://arxiv.org/abs/2506.13757) (NeurIPS 2025) combines direct trajectory generation with optional language reasoning and reinforcement fine-tuning. This broadens the design question from “can a small model learn explanations?” to “when should it spend compute generating them?” It is later related work, not an established DSDrive extension.
- **Asynchronous integration addresses another bottleneck:** [NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.14589) extracts early reasoning representations and adapts specialized planners operating at their native rates. It is a related later direction with overlapping authors, not evidence that DSDrive's dual-head mechanism has been independently replicated.
- **Backbone scale alone remains insufficient:** The September 2026 revision of [LightEMMA: A Longitudinal Evaluation of Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2505.00284v3) evaluates 15 models and reports non-monotonic driving improvements across generations. This supports the relevance of domain adaptation while not validating DSDrive's particular distillation recipe. Its original May 2025 title was **LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving**; version and title should be distinguished.
- **Reviewer's implication:** Given these alternatives, a convincing modern study should report full language/control timing, matched-data ablations, multi-seed performance, and explanation interventions. This is an assessment derived from the gaps above, not a claim that one universally adopted evaluation standard has replaced LangAuto.

### Has the Community Accepted the Claims?

The checked primary sources establish a continuing research interest in efficient reasoning and action generation, but do not establish independent replication of DSDrive's numerical gains or a consensus that its explanations are causally faithful. AutoVLA and NetRoller pursue related trade-offs with materially different architectures and evaluation settings; their results cannot be read as confirmations or direct refutations of DSDrive. The defensible conclusion is that planning-aligned reasoning distillation is a useful hypothesis supported by one simulator study, with mixed safety metrics and unresolved reproducibility details. This review is a scoped comparison of verified sources, not an exhaustive citation census through October 2026.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|---|---|---|---|
| LMDrive: Closed-Loop End-to-End Driving with Large Language Models | Hao Shao et al. | 2024 | Direct system/visual-encoder foundation and LangAuto source. Both experimental baselines are included: released LMDrive LLaVA-7B and the authors' retrained LMDrive LLaMA-1B variant. |
| BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | Junnan Li et al. | 2023 | Source of the visual-query alignment architecture. |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI / Daya Guo et al. | 2025 | Motivation for structured think-and-answer supervision and compact-model reasoning transfer; not evidence that DSDrive uses reinforcement learning. |
| Qwen2.5-VL Technical Report | Shuai Bai et al. | 2025 | Teacher-model family citation; exact max model configuration remains unspecified. Qwen2.5-VL-7B is additionally compared qualitatively in Fig. 7. |
| The Llama 3 Herd of Models | Aaron Grattafiori et al. | 2024 | Backbone/tokenizer family citation. Vanilla LLaMA-1B is an experimental baseline variant, not a separately identified driving paper. |
| [Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/) | Yihan Hu et al. | 2023 | Planning-oriented multi-task context cited in related work; no direct UniAD benchmark comparison here. |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|---|---|---|---|
| WiseAD: Knowledge Augmented End-to-End Autonomous Driving with Vision-Language Model | Songyan Zhang et al. | 2024 | Nearby knowledge-transfer approach; discussed in Table I, not a measured Table II closed-loop baseline. |
| Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning | R. Zhao et al. | 2025 | Scene-to-driving language-model approach discussed in Table I; not evaluated numerically against DSDrive. |
| Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving | B. Jiang et al. | 2024 | Related language/planning integration in Table I, not a DSDrive closed-loop baseline. |
| DriveLM: Driving with Graph Visual Question Answering | Chonghao Sima et al. | 2024 | Structured scene reasoning context; not a matched closed-loop experiment. |
| LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving | Zhijie Qiao, Haowei Li, Zhong Cao, Henry X. Liu | 2025 | Contemporary lightweight driving-model study; the substantially revised 2026 evaluation version is discussed above. |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|---|---|---|---|
| [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](https://arxiv.org/abs/2506.13757) | Zewei Zhou et al. | 2025 | Later related approach: unified autoregressive action generation, fast/slow modes, and reinforcement fine-tuning; no verified direct extension of DSDrive. |
| [NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving](https://arxiv.org/abs/2506.14589) | Ren Xin et al. | 2025 | Later related work with overlapping authors; asynchronous feature adaptation rather than DSDrive's compact dual-head policy. |

No direct DSDrive successor was verified in this review. Knowledge-graph MCP tools were unavailable in this session, so reverse-reference cache queries were skipped; local Markdown was searched and contained no pre-existing DSDrive mentions.

---

### Bottom Line

DSDrive is worth reading as a compact example of using task-aligned teacher-generated text to improve a closed-loop driving policy. Its strongest contribution is the supervision design: make an expert trajectory the shared endpoint of explanation and planning. Treat it as a useful architectural reference rather than a settled safety or deployment result. The memory saving is tangible in the reported comparison, but safety is mixed, full reasoning latency is unclear, and missing implementation details prevent exact reconstruction without additional author material.

[^vlm]: **VLM** — Vision-Language Model. See the [glossary](../../common/terms/).
[^cot]: **CoT** — Chain-of-Thought. See the [glossary](../../common/terms/).
[^bev]: **BEV** — Bird's-Eye-View. See the [glossary](../../common/terms/).
