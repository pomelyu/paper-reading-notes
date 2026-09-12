# NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning

- **Authors:** Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan
- **Affiliations:** Applied Intuition; Texas A&M University; UC Berkeley
- **Published:** CVPR 2026; arXiv:2602.21172v3, June 2026
- **Keywords:** autonomous driving, Vision-Language-Action, reinforcement learning, data efficiency, trajectory planning, GRPO
- **Webpage:** https://nord-vla-ai.github.io/
- **GitHub:** https://github.com/Applied-Intuition-Open-Source/nord
- **HuggingFace:** https://huggingface.co/AppliedIntuitionResearch/nord

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|---|
| **Category** | A camera-based, end-to-end driving policy paper. It tests whether a VLA[^1] can directly emit trajectory tokens without natural-language reasoning traces. |
| **Context** | It follows reasoning-heavy driving VLAs such as AutoVLA, but also the direct trajectory-prediction line represented by [Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/). Its RL stage imports Dr. GRPO from LLM reasoning research. |
| **Correctness** | The central diagnosis is empirically plausible: weak SFT policies produce high-variance rewards on difficult scenes, and removing GRPO's per-group standard-deviation normalization substantially improves NAVSIM. The evidence is nevertheless limited to two benchmark data distributions and offline/simulation metrics, not on-road deployment. |
| **Contributions** | (1) identifies GRPO difficulty bias for weak, data-efficient driving policies; (2) applies Dr. GRPO to remove that variance normalization; (3) trains a reasoning-free Qwen2.5-VL policy from trajectory supervision; and (4) reports competitive NAVSIM and WaymoE2E performance with less data and fewer output tokens. |
| **Clarity** | Strong. The paper gives the reward analysis, data splits, model inputs, objective, comparison tables, and failure cases. Its headline “without reasoning” needs careful reading: the model still uses a pretrained multimodal language-model backbone; it only omits generated CoT[^2] traces. |

**30-second summary.** NoRD argues that the apparent need for language reasoning in driving VLAs is partly an optimization artifact. A Qwen2.5-VL-3B model learns to map three surround-view RGB frames, ego history/state, and a high-level command directly to discrete future-trajectory tokens. When its deliberately small SFT model is post-trained with ordinary GRPO, difficult scenes have high rollout-reward variance and receive weak gradients. Replacing the normalized GRPO advantage with Dr. GRPO lets those scenes contribute, lifting NAVSIM PDMS from 76.66 to 85.62 and reaching RFS 7.709 on WaymoE2E without reasoning traces or ensembles. This is a valuable efficiency result, but not evidence that explicit reasoning never helps or that the policy is ready for closed-loop deployment.

## Pass 2 — Careful Read

### Core Idea in One Sentence

Train a small-data, reasoning-free Qwen2.5-VL driving policy to predict discretized trajectories, then recover useful RL learning on hard scenes by replacing GRPO's variance-normalized advantage with Dr. GRPO.

### Method / Approach

- **Direct action-token policy:** Fine-tune Qwen2.5-VL-3B-Instruct on front-left, front, and front-right RGB images, past ego trajectory, current velocity/acceleration, and a driving command; the output is a sequence of discrete trajectory tokens rather than a textual explanation plus plan.
- **k-disc trajectory representation:** Interpolate future paths to 10 Hz, split them into 0.5-s segments, cluster segments by contour distance into a 2,048-entry codebook, and add the resulting tokens to Qwen's vocabulary. NAVSIM uses eight tokens for 4 s; WaymoE2E uses ten for 5 s.
- **Small SFT then RL:** Build NoRD-base with limited trajectory-only SFT, then run eight rollouts per input. The reward combines format and length rewards (0.25 each) with PDM score on NAVSIM or normalized RFS on WaymoE2E.
- **Dr. GRPO for hard scenes:** Standard GRPO divides each centered group reward by its standard deviation, shrinking updates from the high-variance groups in which weak policies sometimes succeed and sometimes fail. Dr. GRPO removes that denominator; DAPO-style asymmetric clipping stabilizes the update without KL regularization.

### Key Results

| Benchmark / setting | Method | Main result | What it shows |
|---|---|---:|---|
| NAVSIM navtest | NoRD-base | PDMS 76.66 | A small, trajectory-only SFT policy is weak but functional. |
| NAVSIM navtest | NoRD-base + GRPO | PDMS 77.18 (+0.67%) | Ordinary GRPO barely improves the weak starting policy. |
| NAVSIM navtest | NoRD (Dr. GRPO) | PDMS 85.62 (+11.68%) | The proposed optimizer change, not merely more RL steps, drives the main improvement. |
| NAVSIM navtest | NoRD | PDMS 85.6; collision 97.6; DAC 94.9 | Competitive camera-only, single-sample result with no reasoning data or LiDAR/HD map input. |
| NAVSIM navtest, oracle best-of-6 | NoRD-BoN | PDMS 92.4 | Exceeds AutoVLA-BoN 92.1, but relies on an oracle selection protocol rather than a deployable chooser. |
| WaymoE2E test | NoRD | RFS 7.709; ADE@3 s 1.2504 | Third-highest RFS among the listed VLAs, and the best ADE@3 s in its comparison table without reasoning or ensembling. |

- **Vocabulary ablation:** A 512-token codebook gives NAVSIM PDMS 83.07; 2,048 tokens give 85.62, consistent with finer codes being needed for sharp turns and complex manoeuvres.
- **Difficulty-bias analysis:** GRPO improves low-variance, already-easy samples but leaves the dominant mid-reward/high-variance region largely unchanged; Dr. GRPO improves medium- and high-variance tertiles.
- **Efficiency claim:** The authors report at least 3× fewer generated tokens and at least 2× lower runtime than their compared top VLA baselines. This is a benchmark comparison, not a full vehicle-system latency measurement.

### Strengths

- **Specific optimization diagnosis:** It does more than assert that “reasoning is unnecessary”; the GRPO-versus-Dr. GRPO experiment isolates a concrete failure mode of RL from weak SFT policies.
- **Well-matched representation:** Discrete trajectory tokens keep an autoregressive VLM interface while avoiding verbose intermediate text at training and inference.
- **Useful efficiency frontier:** Comparing performance against reported training-sample counts makes the data-cost trade-off visible instead of only chasing the best raw score.
- **Reproducibility progress:** The official repository supplies code, NAVSIM evaluation instructions, Apache-2.0 licensing, and released `nord` / `nord-base` checkpoints.

### Weaknesses / Open Questions

1. **Narrow causality claim:** The experiments show that NoRD works without explicit CoT on these benchmarks; they do not establish that reasoning is generally useless, or separate latent knowledge inherited from Qwen2.5-VL from language-reasoning supervision.
2. **Closed-loop gap:** NAVSIM scores a non-reactive simulation and WaymoE2E evaluates preference similarity. Neither result establishes safety under ego-induced distribution shift, interactive agents, or recovery after a mistake.
3. **Oracle best-of-N:** The headline 92.4 BoN NAVSIM score assumes the evaluator selects the best of six sampled paths using PDM. A real vehicle needs an online, non-oracle safety/ranking mechanism.
4. **Data accounting ambiguity:** “Less data” aggregates supervised and RL samples differently across baselines, while NoRD reuses validation splits for RL after filtering. The paper describes this, but the comparisons are not a controlled equal-data ablation.
5. **Residual difficulty bias:** The authors' failure cases include aggressive driving, missed rear traffic, and overly cautious stopping; they explicitly note that Dr. GRPO mitigates rather than eliminates the problem.

### References to Follow Up

1. **Understanding R1-Zero-Like Training: A Critical Perspective** — Liu et al., 2025: introduces Dr. GRPO and the difficulty-bias critique that NoRD transfers into driving.
2. **AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning** — Zhou et al., NeurIPS 2025: the principal reasoning-based VLA comparison and a useful counterpoint to NoRD's premise.
3. **NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking** — Dauner et al., 2024: defines the simulation and PDM score that carry NoRD's NAVSIM conclusion.
4. **Waymo Vision-Based End-to-End Driving Dataset** — Waymo LLC, 2025: defines the long-tail preference-trajectory benchmark and RFS used for the second principal result.
5. **[Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/)** — Hu et al., CVPR 2023: a structured end-to-end camera-driving baseline that helps distinguish direct planning from language-mediated VLA policies.

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Inputs and action vocabulary.** Start with Qwen2.5-VL-3B-Instruct. For NAVSIM, encode three RGB cameras, three past trajectory tokens covering 1.5 s, current velocity and acceleration, and a high-level `left`/`straight`/`right` command. Use six past trajectory tokens over 3 s for WaymoE2E. Interpolate each target path to 10 Hz, cut it into 0.5-s pieces, and cluster all training pieces by contour distance into $K=2048$ centres. Append one token per centre to the language-model vocabulary, initializing new embeddings from a multivariate normal distribution fitted to the original embedding mean and covariance. Decoding eight (NAVSIM) or ten (WaymoE2E) tokens reconstructs a 4-s or 5-s sequence of $(x, y, yaw)$ waypoints.

**SFT stage.** Fine-tune the visual encoder, multimodal MLP, and language model with causal next-token loss, conditioned on observations and ego state but with no textual rationale target. This intentionally data-limited stage produces NoRD-base. The paper uses a 16-A100 setup, bf16, gradient checkpointing, DeepSpeed ZeRO-3 for WaymoE2E, batch size 8 per device with four accumulation steps, AdamW, $5 \times 10^{-5}$ learning rate, cosine decay, 0.03 warm-up ratio, and evaluation every 50 steps. Select the checkpoint with minimum validation loss.

**Why GRPO fails here.** For an input $x$, draw $G=8$ sampled trajectories $o_i$ and score them with $r(o_i \mid x)$. Standard GRPO centres a reward by its group mean and divides by group standard deviation:

```math
\hat{A}^{GRPO}_{i,t} = \frac{r(o_i \mid x) - \frac{1}{G}\sum_{j=1}^{G} r(o_j \mid x)}{\operatorname{std}_{j=1,\ldots,G}(r(o_j \mid x))}.
```

For a weak policy, mundane scenes are consistently right or wrong and have low variance, while difficult turns/collision-avoidance scenarios have mixed successes and high variance. The denominator therefore magnifies easy groups and attenuates precisely the hard groups with potentially informative successes.

**Dr. GRPO post-training.** Replace the advantage with only its mean-centred numerator:

```math
\hat{A}^{DrGRPO}_{i,t} = r(o_i \mid x) - \frac{1}{G}\sum_{j=1}^{G} r(o_j \mid x).
```

Optimize the usual clipped policy ratio using asymmetric clipping $(\epsilon_l,\epsilon_h)=(-0.2,0.1)$, no KL penalty, and group size eight. Each reward is the normalized sum of a format reward, a length reward, and the task reward; the first two contribute 0.25 each. The task reward is PDM on NAVSIM and normalized RFS on WaymoE2E. Train 160 NAVSIM steps on 30 A100s at $5 \times 10^{-6}$, or 150 WaymoE2E steps on 32 A100s at $10^{-6}$; retain the checkpoint with highest validation reward. The implementation uses verl/FSDP for RL and vLLM for rollouts.

**Evaluation and interpretation.** Evaluate NoRD separately per dataset: NAVSIM executes predicted paths under PDM's safety, drivable-area, progress, TTC, comfort, and direction components; WaymoE2E compares against human preference trajectories using RFS and reports ADE. Keep single-sample scores separate from BoN: BoN samples six paths at different random seeds and retrospectively chooses the highest PDM path, so it measures candidate diversity plus an oracle rather than an executable policy.

### Datasets

#### Train Data

| Dataset | Usage | Proposed by |
|---|---|---|
| NAVSIM | `navtrain` supplies trajectory-only SFT. Its SFT split is 80/20 train/validation; RLFT reuses and filters the SFT-validation portion, dropping trivial constant-velocity trajectories and balancing straight/left/right intents. The main text reports 80,000 NAVSIM SFT samples. | NAVSIM |
| Waymo Vision-Based End-to-End Driving Dataset (WaymoE2E) | Official training data is filtered for sufficient history, uniformly sampled at 20% of valid sequences, then split 85/15 for trajectory-only SFT; the main text reports 12,000 SFT samples. Official validation scenes supply preference-annotated RLFT samples, again split 85/15; the main text reports 450 RLFT samples. | Waymo Vision-Based End-to-End Driving Dataset |

#### Evaluation/Validation Data

| Dataset | Usage | Proposed by |
|---|---|---|
| NAVSIM | SFT validation selects the lowest-loss checkpoint; filtered validation scenes also select the highest-reward RL checkpoint. The `navtest` subset is the reported independent NAVSIM test benchmark. | NAVSIM |
| Waymo Vision-Based End-to-End Driving Dataset (WaymoE2E) | Held-out portions of the authors' filtered SFT and RLFT subsets are used for validation; the official 1,505-segment test split provides the reported RFS and ADE. | Waymo Vision-Based End-to-End Driving Dataset |

### Hidden Assumptions

1. **Reward alignment:** PDM/RFS and formatting rewards are assumed to order candidate trajectories in the same way as safe, legal driving; the shaping terms could instead encourage syntactically valid but brittle behaviour.
2. **No language trace means no needed reasoning:** Direct action emission may exploit implicit planning inside the pretrained model. The paper does not test whether selectively generated reasoning helps rare situations when data and latency are held fixed.
3. **k-disc sufficiency:** A 2,048-centre codebook is treated as a sufficiently fine and smooth action space, even though a quantization error at a sharp manoeuvre can have safety consequences.
4. **Stable reuse of validation data:** Reusing filtered validation scenes for RL assumes the resulting model-selection process does not overfit the benchmark distribution.
5. **Benchmark transfer:** Three forward cameras, simple commands, and logged/offline scores are assumed to transfer to broader sensor suites, route contexts, reactive traffic, and real vehicle constraints.

### Reproducibility Notes

- **Code and models:** The official repository provides installation, an inference API, NAVSIM v1.1 evaluation instructions, and two Hugging Face model IDs: `AppliedIntuitionResearch/nord` and `AppliedIntuitionResearch/nord-base`.
- **Data:** Obtain NAVSIM and WaymoE2E under their own access terms. Recreate the stated history filters, 20% Waymo sampling, split ratios, NAVSIM RL trajectory filters, and intent balancing; otherwise “same sample count” will not mean the same training distribution.
- **Compute:** SFT uses 16 A100 GPUs. RL needs 30 A100s for NAVSIM or 32 for WaymoE2E; rollout generation uses vLLM and distributed training uses verl/FSDP. The released repository additionally states a CUDA GPU with at least 16 GB VRAM for inference.
- **Critical hyperparameters:** Use group size 8, temperature 1.0 for RL rollouts, temperature 0.01 for validation, the asymmetric clip bounds, no KL regularizer, the respective learning rates and 160/150 RL steps. The vocabulary size is not a cosmetic setting: 512 versus 2,048 changes PDMS by 2.55 points.
- **Missing/underspecified:** The paper does not give a full wall-clock training budget, exact clustering seed/preprocessing implementation, a deployment-grade candidate selector for BoN, a reactive closed-loop protocol, or uncertainty calibration for unsafe outputs.

### Ideas for Future Work

1. **Equal-budget reasoning test:** Compare direct NoRD, concise reasoning, and long-CoT policies under identical base model, data, action tokens, RL reward, and latency limits; this would test reasoning rather than pipeline confounds.
2. **Difficulty-aware driving RL:** Combine Dr. GRPO with curriculum/sampling methods that target high-variance scenes without needing prohibitive simulation rollouts for every candidate.
3. **Deployable selection:** Replace oracle BoN with a learned or rule-constrained risk selector, and report the safety/performance trade-off it introduces.
4. **Interactive validation:** Evaluate in closed-loop simulation with reactive pedestrians/vehicles, distribution shifts, map errors, sensor faults, and recovery after an unsafe initial plan.
5. **Continuous or hybrid actions:** Compare k-disc tokens with diffusion or continuous trajectory heads under the same encoder and RL objective to locate the cost of discretization.

## Pass 4 — Modern Perspective Review (as of September 2026)

### What Has Changed Since Publication

- **From preprint to published and inspectable system:** The February arXiv preprint is now a CVPR 2026 paper, and Applied Intuition has released an official project page, Apache-2.0 repository, NAVSIM evaluator integration, and `nord`/`nord-base` checkpoints. This makes the claim easier to audit than a result table alone.
- **The main design tension is clearer:** Driving VLA work now includes both explicit-reasoning systems such as AutoVLA and direct-action approaches such as NoRD. Rather than resolving the debate, NoRD narrows it: reasoning traces are not necessary to get competitive scores when trajectory coding and RL optimization are well matched.
- **Evaluation still dominates the open question:** NoRD's strongest outputs remain benchmark-level NAVSIM and WaymoE2E metrics. The research and engineering bar for vehicle use remains interactive closed-loop safety, calibration, rule compliance, and runtime under the complete perception-and-control stack.
- **Dr. GRPO is a reusable hypothesis:** The transfer of an LLM-RL normalization critique into autonomous driving is likely the paper's most portable idea. It suggests that a reward optimizer's variance behaviour must be analysed alongside model capacity and dataset size.

### Has the Community Accepted the Claims?

The narrow claim has good direct support: within the reported camera-only, trajectory-token setup, Dr. GRPO turns a negligible GRPO gain into a large NAVSIM improvement, and the released code/models make that experiment testable. The broader slogan needs restraint. NoRD has not demonstrated that explicit reasoning hurts driving in general, nor that direct trajectory generation is safer than a reasoning-equipped policy outside its two benchmarks. A fair reading is that the paper convincingly exposes an optimizer/data-regime failure that can masquerade as a need for reasoning—not that it closes the question of reasoning for autonomous driving.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|---|---|---:|---|
| [Planning-oriented Autonomous Driving](../../2023/Planning-oriented_Autonomous_Driving/) | Hu et al. | 2023 | Camera-based end-to-end planning baseline; NoRD's NAVSIM table compares against UniAD. |
| NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking | Dauner et al. | 2024 | Supplies the simulation, dataset redistribution, and PDM metric that NoRD optimizes. |
| Understanding R1-Zero-Like Training: A Critical Perspective | Liu et al. | 2025 | Introduces Dr. GRPO's removal of reward-standard-deviation normalization. |
| AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning | Zhou et al. | 2025 | Principal reasoning-based VLA baseline and contrast for NoRD's trajectory-only supervision. |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|---|---|---:|---|
| Poutine: Vision-Language-Trajectory Pre-training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving | Poutine authors | 2025 | WaymoE2E leader in NoRD's comparison; uses reasoning and an ensemble. |
| HMVLM: Hierarchical Multimodal Vision-Language Model for Autonomous Driving | HMVLM authors | 2025 | WaymoE2E reasoning-based, ensemble competitor. |
| DiffusionLTF: Truncated Diffusion Model for End-to-End Autonomous Driving | DiffusionLTF authors | 2025 | Reasoning-free, non-ensemble WaymoE2E competitor with lower RFS and higher ADE@3 s in NoRD's table. |
| RecogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving | RecogDrive authors | 2025 | Reasoning-based NAVSIM VLA comparator requiring substantially more reported training data. |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|---|---|---:|---|
| No direct successor identified in the paper's official materials | — | — | The official project/repository presently provides code and models rather than a named follow-on research paper; reassess as later work appears. |

---

### Bottom Line

NoRD is worth reading for a precise lesson that reaches beyond driving: when a small SFT policy fails under RL, first inspect the reward normalization and the distribution of learning signal before concluding that the representation or supervision type is inadequate. Its direct trajectory-token interface is practical and its Dr. GRPO ablation is unusually sharp. Read it alongside AutoVLA and NAVSIM, but treat its no-reasoning result as an efficiency-oriented benchmark finding—not a final verdict on language reasoning or a closed-loop safety case.

[^1]: **VLA** — Vision-Language-Action model. See the [glossary](../../common/terms/).
[^2]: **CoT** — Chain-of-Thought. See the [glossary](../../common/terms/).
