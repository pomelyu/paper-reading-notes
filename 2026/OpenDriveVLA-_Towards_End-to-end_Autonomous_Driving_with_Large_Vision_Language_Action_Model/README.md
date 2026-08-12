# OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model

- **Authors:** Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, Volker Tresp, Alois C. Knoll
- **Affiliations:** Technical University of Munich; Ludwig Maximilian University of Munich
- **Published:** AAAI 2026; arXiv:2503.23463v2, November 2025
- **Keywords:** end-to-end autonomous driving, Vision-Language-Action, trajectory planning, 3D perception, instruction tuning, nuScenes
- **Webpage:** https://drivevla.github.io/
- **GitHub:** https://github.com/DriveVLA/OpenDriveVLA
- **HuggingFace:** https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|---|
| **Category** | A camera-only, end-to-end autonomous-driving policy that treats driving as language-conditioned autoregressive trajectory generation. |
| **Context** | Extends planning-oriented structured perception in [Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/) and large-VLM driving work such as [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](../../2024/DriveVLM-_The_Convergence_of_Autonomous_Driving_and_Large_Vision-Language_Models/). Its BEV[^1] encoder inherits the query-based multi-camera design of [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](../../2022/BEVFormer-_Learning_Bird's-Eye-View_Representation_from_Multi-Camera_Images_via_Spatiotemporal_Transformers/). |
| **Correctness** | The staged design makes its claimed spatial grounding plausible, and the ablations support contributions from ego state, commands, alignment, and interaction prediction. However, every planning result is offline/open-loop on nuScenes; it does not establish closed-loop safety, recovery behavior, or deployability. |
| **Contributions** | (1) structured scene, agent, and map tokens rather than generic image patches; (2) per-token visual-language alignment; (3) an auxiliary agent–environment–ego forecasting task; and (4) unified question answering and waypoint prediction from an open Qwen2.5 backbone. |
| **Clarity** | The pipeline and supplement are unusually concrete: token interfaces, prompts, data sources, learning rates, model scales, and qualitative failures are reported. The distinction between language fluency and safety-critical planning validity needs stronger empirical treatment. |

OpenDriveVLA is best understood as a structured adapter around an LLM: a strong driving perception stack turns six camera views into global-scene, tracked-agent, and map tokens; small projectors translate them into the LLM embedding space; then a Qwen2.5-Instruct model answers driving questions and emits six future waypoints. Its key bet is that explicit 3D, instance-level tokenization plus auxiliary prediction of other agents makes language-conditioned planning more grounded than directly feeding image tokens into a VLM[^2]. It is a compelling open-source research baseline, but its open-loop metrics and approximately 1.36 s best-case planning latency leave the central closed-loop, real-time safety question unresolved.

## Pass 2 — Careful Read

### Core Idea in One Sentence

Convert a pre-trained 3D driving perception system into a compact, typed token interface for an autoregressive LLM, then train that LLM to align perception with language, forecast surrounding agents, and produce command-conditioned ego trajectories.

### Method / Approach

- **Structured visual interface:** Six images pass through a ResNet-101 + FPN backbone and a BEVFormer encoder; a global sampler produces $90$ scene tokens, TrackQFormer produces up to $900$ dynamic-agent tokens, and MapQFormer produces up to $300$ map tokens.
- **Hierarchical alignment (stage 1):** Separate two-layer MLP projectors map scene, agent, and map tokens into the Qwen language-embedding space. Frozen vision and language backbones learn from scene captions, map captions, and object captions augmented with BEV coordinates.
- **Driving instruction tuning (stage 2):** With the visual encoder still frozen, the projectors and all Qwen parameters learn multi-task driving question answering from nuCaption, nuScenes-QA, and nuX; the input includes visual tokens, textualized ego state, history, and a command/question.
- **Interaction then planning (stages 2.5–3):** An auxiliary conditional future-trajectory task predicts each tracked agent given scene, map, ego, and agent tokens. The final objective tokenizes six $0.5$-s ego displacements over a $3$-s horizon and autoregressively decodes them; all components except the 2D image backbone are tuned.

### Key Results

The authors evaluate 0.5B, 3B, and 7B Qwen2.5 variants on nuScenes open-loop planning. Values below are averaged over the three-second horizon; lower is better.

| Method | ST-P3 L2 (m) | ST-P3 collision (%) | UniAD L2 (m) | UniAD collision (%) |
|---|---:|---:|---:|---:|
| UniAD | 0.69 | 0.12 | 1.03 | 0.31 |
| DriveVLM | 0.40 | 0.27 | — | — |
| GPT-Driver | 0.44 | 0.17 | 0.84 | 0.44 |
| RDA-Driver | 0.40 | 0.10 | 0.80 | 0.32 |
| OpenDriveVLA-0.5B | 0.35 | **0.09** | 0.68 | 0.26 |
| OpenDriveVLA-7B | **0.33** | 0.10 | **0.66** | **0.25** |

On driving visual question answering (VQA[^3]), 7B obtains nuCaption BLEU-4 $27.6$ and BERTScore $92.2$; on nuScenes-QA it reaches $58.2$ overall accuracy, while 0.5B reaches the best nuX CIDEr score ($32.3$). This lack of monotonic scaling is important: the paper argues that the available driving language data is insufficient for the 7B model to consistently exploit its capacity.

- **Input ablation:** Visual + ego + two-second history + command is the best tested combination: UniAD/ST-P3 collision $0.26 / 0.09$ % and L2 $0.68 / 0.35$ m for 0.5B. Removing the visual input is especially damaging ($0.77 / 0.39$ m L2 on the two metrics even with ego, history, and command).
- **Training ablation:** The final agent–environment–ego stage lowers UniAD collision from $0.31$ % after stages 1–2–3 to $0.26$ %, while ST-P3 collision falls from $0.11$ % to $0.09$ %.
- **Efficiency:** On one A100 in bf16, 0.5B uses 1.56 GB and takes 1.36 s/sample; 3B and 7B take 1.85 s and 1.74 s. These are offline benchmark measurements, not an end-to-end vehicle control-loop latency measurement.

### Strengths

- **Grounded decomposition:** Scene, agent, and map token types expose dynamic objects, road structure, and global context separately instead of asking an image-only VLM to discover all three implicitly.
- **Useful auxiliary task:** Forecasting each agent's future trajectory gives the LLM a direct, spatially meaningful interaction objective before ego planning.
- **A genuinely unified interface:** The same structured context supports language questions and trajectory output, making semantic inspection more natural than with a waypoint-only policy.
- **Small-model competitiveness:** The 0.5B version is often near the larger variants and beats prior autoregressive baselines, making it a useful experimental starting point.
- **Reproducibility evidence:** The paper provides source code, a released 0.5B checkpoint, input prompts, tokenizer markers, data composition, and most hyperparameters.

### Weaknesses / Open Questions

1. **Open-loop safety proxy:** Low displacement and collision against logged future trajectories do not test interventions, compounding error, other agents reacting to the ego plan, traffic-law compliance, or recovery from off-distribution states.
2. **Latency and control rate:** A $1.36$-s autoregressive plan is hard to reconcile with frequent replanning in dynamic traffic, and measured throughput omits perception, sensor I/O, and vehicle integration costs.
3. **Grounding is not guaranteed:** The supplementary qualitative examples acknowledge a missed pedestrian, a wrong camera-view attribution, and an implausible agent turn. Token structure reduces, but does not eliminate, hallucination or geometric error.
4. **Training-data entanglement:** All stages and all principal evaluations derive from nuScenes or nuScenes-derived annotations. The reported VQA/planning gains may partly reflect dataset-specific language and map conventions rather than broad transfer.
5. **Counterfactual commands are qualitative:** The left/right/forward demonstrations are persuasive illustrations, but there is no systematic command-feasibility, legality, or safety evaluation for commands conflicting with the route and scene.

### References to Follow Up

1. **[Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/)** — Hu et al., CVPR 2023: the structured multi-task perception-and-planning stack that supplies OpenDriveVLA's visual-query foundation.
2. **[DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](../../2024/DriveVLM-_The_Convergence_of_Autonomous_Driving_and_Large_Vision-Language_Models/)** — Tian et al., arXiv 2024: an earlier VLM-driving formulation with interpretable, hierarchical reasoning and a practical slow-fast design.
3. **GPT-Driver: Learning to Drive with GPT** — Mao et al., arXiv 2023: casts planning as language generation and provides the coordinate/prompt convention adapted here.
4. **RDA-Driver: Better Planners with Reasoning-Decision Alignment** — Huang et al., arXiv 2024: a close autoregressive visual-language planning baseline that aligns reasoning and decisions.
5. **Bench2Drive-VL: Benchmarks for Closed-Loop Autonomous Driving with Vision-Language Models** — 2026: directly addresses the VLM-specific closed-loop evaluation gap left by this paper.

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Perception and token extraction.** Start from six synchronized camera images. Apply a shared ResNet-101/FPN to create multi-scale 2D features, aggregate them with a six-layer BEVFormer encoder into a $200 \times 200$ BEV grid with hidden dimension $256$, and retain a visual module pre-trained on detection, tracking, and panoptic map segmentation. Derive three complementary interfaces:

```math
V_{env} = \{V_{scene}, V_{agent}, V_{map}\}.
```

The scene sampler uses adaptive max pooling to produce $6 \times 3 \times 5 = 90$ global image-context tokens. A six-layer TrackQFormer decodes dynamic-object tokens from the BEV grid and filters detections by confidence (maximum $900$ queries). A six-layer MapQFormer emits map-element tokens (maximum $300$ queries). Thus the LLM is never asked to parse unrestricted image-patch streams: it receives a typed, bounded set of pretrained perception features.

**Stage 1: token-specific vision-language alignment.** Keep the visual encoder and Qwen2.5-Instruct frozen. Train a distinct GeLU two-layer projector $\Phi_k$ for $k \in \{scene, agent, map\}$ so each projected token conditions caption generation. Agent captions include an object's appearance and BEV location; scene captions summarize multiple views; map captions describe road geometry. In notation, the alignment target is the caption $X_k$ generated from the projected feature:

```math
\hat{X}_k = LLM(\Phi_k(v_k)).
```

This is an ordinary causal language loss, but applied separately by semantic token type; it is the main mechanism intended to bridge numeric driving features and the LLM's textual semantic space.

**Stage 2: instruction tuning.** Construct a sequence with special spans for scene, tracks, map, ego state, command, and optionally a question. Substitute projected visual embeddings at the visual placeholders; serialize velocity, yaw rate, acceleration, CAN-bus state, speed, steering, and two seconds of history into the ego text span. Train the projectors and all LLM weights on nuCaption, nuScenes-QA, and nuX answers. This injects driving-language supervision without requiring explicit chain-of-thought output at inference.

**Stage 2.5: agent–environment–ego interaction.** For each detected agent, condition on the global scene/map tokens, textual ego state, and that agent's token, then autoregressively predict its future relative displacement tokens. The auxiliary likelihood is:

```math
\max \prod_{t=1}^{T} p(w_t^i \mid w_{1:t-1}^i, V_{env}, S_{ego}, \Phi_{agent}(v_{agent}^i)).
```

This does not introduce an explicit collision checker or world model. Rather, it encourages the LLM hidden state to encode likely multi-agent evolution before it learns ego actions.

**Stage 3: ego trajectory tuning.** Represent the target plan as six 2D ego-relative waypoints at 0.5-s intervals. Quantize/tokenize the coordinates and train causal next-token prediction conditioned on $V_{env}$, the ego state, and a natural-language mission command:

```math
\hat{T}_{traj} = \arg\max_{T_{traj}} \prod_{t=1}^{T} p(w_t \mid w_{1:t-1}, V_{env}, S_{ego}, X_{drive}).
```

Decode the generated token sequence to numerical coordinates. In stage 3, tune the projectors, LLM, and 3D visual modules jointly, while freezing the 2D image backbone. The system prompt fixes a coordinate convention (right is $x$, forward is $y$) and requires a $3$-s, six-waypoint, collision-free route.

**Training recipe.** For the 0.5B variant, use four H100s, bf16, gradient checkpointing, per-GPU batch size 1, and one epoch per stage. Stage 1 uses only 3.1 MB trainable projector parameters with learning rate $10^{-4}$; stages 2 and 2.5 tune 496.9 MB at $10^{-5}$; stage 3 tunes 552.6 MB (except the 2D backbone) at $10^{-5}$. The full process reportedly takes about two days. At test time use temperature zero, then decode the coordinate sequence deterministically.

### Hidden Assumptions

1. **Perception is trusted:** The LLM only sees what the frozen/partly frozen visual stack proposes. Missed objects, wrong map queries, or poor confidence calibration cannot be repaired reliably by language reasoning.
2. **Captions teach geometry:** Supervision assumes text augmented with BEV coordinates is enough to align continuous spatial relations with discrete language-model embeddings.
3. **Token order and count are benign:** Variable agent sets are filtered by confidence, but the method does not establish robustness to crowded scenes, identity swaps, false positives, or the ordering of many agent tokens.
4. **Logged futures are appropriate targets:** Imitating nuScenes trajectories assumes the recorded human plan is safe and optimal for every language-conditioned counterfactual, even though alternative instructions may require different legal/feasible futures.
5. **Agent forecasting transfers to ego safety:** Improved auxiliary prediction is treated as evidence of better interaction-aware planning, but no causal test separates its effect from more training signal or model capacity.
6. **Language instructions are well formed:** High-level commands are assumed available, unambiguous, and consistent with route, traffic rules, and perception; command conflict resolution is unspecified.

### Reproducibility Notes

- **Code and weights:** The public repository includes environment setup and inference code; the official page/repository report an OpenDriveVLA-0.5B checkpoint. The paper's main 3B/7B results should not be assumed reproducible from that released checkpoint alone.
- **Data:** Prepare nuScenes plus TOD3Cap, nuCaption, nuScenes-QA, and nuX annotations. The reported stage sample counts are 536k alignment, 566k instruction, 459k interaction, and 28k planning examples; licenses and preprocessing for every derived dataset must be checked separately.
- **Model dependencies:** Use the specified ResNet-101/FPN + BEVFormer perception model, TrackQFormer/MapQFormer heads, LLaVA-NeXT-style multimodal integration, and Qwen2.5-Instruct. The repository carries customized mmcv/mmdet3d compatibility code, so a generic modern stack may not reproduce results.
- **Compute:** The reported 0.5B recipe needs four H100s for roughly two days; inference reporting uses one A100 and 6,019 nuScenes validation samples. Batch size 1 makes training sensitive to optimizer, accumulation, and distributed-training details.
- **Missing/underspecified:** The paper does not provide closed-loop evaluation, exact quantization bins/tokenizer serialization for waypoints in the main text, a full end-to-end latency breakdown, robustness protocols, or confidence calibration criteria for retained agent tokens.

### Ideas for Future Work

1. **Closed-loop language-aware evaluation:** Port the policy to Bench2Drive-VL or a comparable simulator with reactive agents, off-route recovery, traffic-rule checks, and adversarial/conflicting instructions.
2. **Constrained decoding:** Combine autoregressive candidate plans with an explicit kinematic, map, collision, and rule validator; evaluate whether constraints catch language-induced but unsafe trajectories.
3. **Uncertainty-aware token interface:** Propagate detector/track/map uncertainty and agent alternatives into the language model rather than discarding low-confidence tokens with a hard threshold.
4. **Latency-aware architecture:** Replace or shorten waypoint token decoding using parallel trajectory heads, speculative decoding, or a fast low-level controller conditioned on a slower semantic VLA plan.
5. **Cross-domain transfer:** Test across cities, sensor layouts, weather, maps, and datasets rather than only nuScenes-derived supervision; separate vision-language generalization from benchmark leakage.

## Pass 4 — Modern Perspective Review (as of August 2026)

### What Has Changed Since Publication

- **The paper has become a published baseline:** The original arXiv preprint is now an AAAI 2026 paper, its public repository contains inference/environment support, and a 0.5B checkpoint is available. That makes its open, structured VLA interface more useful as a baseline than it was at initial release.
- **Closed-loop VLM evaluation is now directly addressable:** Bench2Drive-VL (2026) adds closed-loop evaluation and behavior-grounded VQA generation for VLM driving, including off-route/off-road states absent from human logs. This turns the paper's own stated limitation into a concrete next experiment.
- **Robustness and long-tail assessment have intensified:** HiDrive and Bench2Drive-Robust (2026) emphasize rare, safety-critical objects, deployment perturbations, rule compliance, and emergency behavior—capabilities that displacement/collision evaluation on logged nuScenes clips does not measure.
- **The research frontier is moving from imitation alone to interactive optimization:** Recent closed-loop work such as PerlAD and CLEAR uses pseudo-simulation or reinforcement learning to address distribution shift after the ego policy changes the scene. OpenDriveVLA supplies useful semantic conditioning but not this feedback-aware optimization.
- **Architectural alternatives have broadened:** UniDriveVLA reports both open-loop and Bench2Drive closed-loop evaluation with expert decoupling for perception/reasoning/action. This challenges the premise that a single fully tuned autoregressive LLM is necessarily the best trade-off.

### Has the Community Accepted the Claims?

The narrower claim is well supported: structured 3D scene, agent, and map tokens plus staged language alignment can make an open model competitive on nuScenes open-loop planning and driving VQA. The official project has released code/inference support and a small checkpoint, which makes the result inspectable. The stronger implication—reliable end-to-end autonomous driving—has not yet been established. Subsequent benchmarks and methods concentrate on exactly the omissions the authors acknowledge: closed-loop interactions, deployment perturbations, long-tail hazards, legality, and real-time response. This is therefore a valuable architecture and training recipe, not evidence that language-mediated planning is ready for vehicle deployment.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|---|---|---:|---|
| [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](../../2022/BEVFormer-_Learning_Bird's-Eye-View_Representation_from_Multi-Camera_Images_via_Spatiotemporal_Transformers/) | Li et al. | 2022 | Multi-camera BEV feature encoder used in the visual stack. |
| [Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/) | Hu et al. | 2023 | Supplies the query-based, planning-oriented perception design and is a principal planning baseline. |
| GPT-Driver: Learning to Drive with GPT | Mao et al. | 2023 | Earlier language-token trajectory generation and coordinate prompt format. |
| [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](../../2024/DriveVLM-_The_Convergence_of_Autonomous_Driving_and_Large_Vision-Language_Models/) | Tian et al. | 2024 | Vision-language planning precursor and autoregressive baseline. |
| RDA-Driver: Better Planners with Reasoning-Decision Alignment | Huang et al. | 2024 | Direct VLM planner baseline aligning reasoning with decisions. |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|---|---|---:|---|
| EMMA: End-to-End Multimodal Model for Autonomous Driving | Hwang et al. | 2024 | Autoregressive multimodal driving planner; compared on ST-P3 metrics. |
| OminiDrive: A Holistic Vision Language Model for Autonomous Driving | Wang et al. | 2025 | LLaVA-based autoregressive open-loop planner; strong ST-P3 baseline. |
| DME-Driver: Integrating Human Decision Logic and 3D Scene Perception | Han et al. | 2025 | LLaVA-based planner evaluated with UniAD metrics. |
| InsightDrive: Insight Scene Representation for End-to-End Autonomous Driving | Song et al. | 2025 | Non-autoregressive scene-representation planner used in the paper's result summary. |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|---|---|---:|---|
| UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving | Xiaomi Research et al. | 2026 | Explores expert decoupling and reports both open- and closed-loop VLA results. |
| Bench2Drive-VL: Benchmarks for Closed-Loop Autonomous Driving with Vision-Language Models | Bench2Drive-VL authors | 2026 | Provides the language-aware closed-loop benchmark that OpenDriveVLA lacked. |
| PerlAD: Towards Enhanced Closed-Loop End-to-End Autonomous Driving With Pseudo-Simulation-Based Reinforcement Learning | PerlAD authors | 2026 | Addresses the imitation-learning/open-loop gap through pseudo-simulation and reinforcement learning. |
| CLEAR: Closed-Loop Reinforcement Learning at Scale for End-to-End Autonomous Driving | CLEAR authors | 2026 | Scales closed-loop RL fine-tuning for VLA waypoint policies. |

---

### Bottom Line

OpenDriveVLA is worth reading as a clear, reproducible bridge between structured camera-only driving perception and open VLM/VLA policies. Its most durable contribution is not simply a lower nuScenes L2 error: it is the decision to give an LLM typed, instance-aware scene tokens and to teach interaction prediction before trajectory decoding. Read it alongside UniAD and DriveVLM to understand that design space. Treat its reported state of the art as an open-loop benchmark result, however—not a closed-loop safety claim or a deployment-ready control stack.

[^1]: **BEV** — Bird's-Eye-View. See the [glossary](../../common/terms/).
[^2]: **VLM** — Vision-Language Model. See the [glossary](../../common/terms/).
[^3]: **VQA** — Visual Question Answering. See the [glossary](../../common/terms/).
