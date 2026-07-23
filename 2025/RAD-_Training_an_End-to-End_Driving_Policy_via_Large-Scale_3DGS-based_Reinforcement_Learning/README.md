# RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning

- **Authors:** Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, Xinggang Wang
- **Affiliations:** Huazhong University of Science & Technology, Horizon Robotics
- **Published:** NeurIPS 2025, arXiv:2502.13144, 21 Oct 2025
- **Keywords:** autonomous driving, reinforcement learning, 3D Gaussian Splatting, end-to-end, closed-loop, imitation learning, digital twin
- **Webpage:** https://hgao-cv.github.io/RAD/
- **GitHub:** https://github.com/hustvl/RAD

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|-----------|
| **Category** | System/method paper — first 3DGS-based closed-loop RL framework for training an end-to-end autonomous driving policy from raw sensor input |
| **Context** | Builds on 3D Gaussian Splatting (3DGS) for photorealistic scene reconstruction, BEV-based end-to-end AD (UniAD, VAD, VADv2), and prior IL+RL hybrid methods (CADRE, CIRL); directly extends StreetGaussian for dynamic urban driving scenes |
| **Correctness** | Assumptions are mostly sound; 3DGS-to-real consistency is validated quantitatively (trajectory overlay) and qualitatively; the use of log-replay for other agents is a known limitation explicitly acknowledged |
| **Contributions** | (1) First end-to-end AD policy trained via RL in a photorealistic 3DGS environment; (2) RL+IL hybrid training with a 4-component safety reward and dense auxiliary objectives; (3) A closed-loop 3DGS evaluation benchmark of 337 diverse, unseen scenes |
| **Clarity** | Well written; clear separation of concerns across sections; figures effectively illustrate training paradigms and qualitative comparisons |

RAD proposes training end-to-end autonomous driving policies via reinforcement learning inside photorealistic 3D Gaussian Splatting (3DGS) digital twins reconstructed from real-world data. By letting the AD policy explore freely in these environments, RL addresses the two core failures of imitation learning — causal confusion and the open-loop gap — while IL is retained as a regularizer to maintain human-like behavior. Evaluated on 337 previously unseen 3DGS scenes, RAD achieves a 3× lower collision rate than IL-only baselines while maintaining comparable trajectory fidelity.

---

## Pass 2 — Careful Read

### Core Idea in One Sentence

Train an end-to-end multi-camera AD policy by alternating PPO reinforcement learning in large-scale photorealistic 3DGS digital twins with imitation learning regularization, using a decoupled discrete action space and four safety-driven reward components.

### Method / Approach

- **3DGS Digital Twin**: 4305 real-world driving scenes (risky, dense-traffic clips) are reconstructed as 3DGS environments by extending StreetGaussian with mesh-constrained road surfaces, separate sky modeling, and depth/normal consistency supervision for foreground objects (vehicles, pedestrians), yielding photorealistic rendering suitable for sensor-input policies.
- **Three-Stage Training**: Stage 1 (perception pre-training) trains the BEV encoder, map head, and agent head on ground-truth annotations. Stage 2 (planning pre-training) trains the image encoder and planning head with IL on large-scale expert demonstrations to avoid cold-start instability. Stage 3 (reinforced post-training) alternates RL (PPO) and IL steps using 32 parallel workers each rolling out in randomly sampled 3DGS environments.
- **Decoupled Discrete Action Space**: Lateral displacement $a^x \in [-0.75, 0.75]$ m and longitudinal displacement $a^y \in [0, 15]$ m are represented as separate 61-bin discrete distributions over a 0.5-second horizon, enabling independent reward attribution and reducing exploration cost.
- **Safety Reward + Auxiliary Objectives**: A 4-component reward $R = \{r_{dc}, r_{sc}, r_{pd}, r_{hd}\}$ (dynamic collision, static collision, positional deviation, heading deviation) provides sparse episode-level signals; four directional auxiliary objectives supply dense per-step gradients by adjusting action probability mass in the correct direction relative to detected hazards.

### Key Results

Closed-loop quantitative comparison on 337 unseen 3DGS scenes (Table 4). All metrics are lower-is-better (↓).

| Method | CR↓ | DCR↓ | SCR↓ | DR↓ | PDR↓ | HDR↓ | ADD↓ |
|--------|-----|------|------|-----|------|------|------|
| TransFuser | 0.320 | 0.273 | 0.047 | 0.235 | 0.188 | 0.047 | 0.263 |
| VAD | 0.335 | 0.273 | 0.062 | 0.304 | 0.255 | 0.059 | 0.304 |
| GenAD | 0.341 | 0.299 | 0.042 | 0.291 | 0.160 | 0.131 | 0.265 |
| VADv2 | 0.270 | 0.240 | 0.030 | 0.243 | 0.139 | 0.104 | 0.273 |
| **RAD** | **0.089** | **0.080** | **0.009** | **0.063** | **0.042** | **0.021** | **0.257** |

Metric glossary: CR = Collision Ratio, DCR = Dynamic Collision Ratio, SCR = Static Collision Ratio, DR = Deviation Ratio (PDR+HDR), PDR = Positional Deviation Ratio, HDR = Heading Deviation Ratio, ADD = Average Deviation Distance.

**Ablation findings (Table 1 — training strategy):**
- Pure IL: CR=0.229, ADD=0.238 — low trajectory deviation but high collision rate
- Pure RL: CR=0.143, ADD=0.345 — better safety but deviates significantly from expert
- RL+IL (RAD): CR=0.089, ADD=0.257 — best safety without sacrificing behavioral fidelity

**Ablation findings (Table 2 — reward components):**
- Omitting the dynamic collision reward raises CR to 0.238 (the single largest impact), demonstrating its centrality to collision avoidance
- Including all four reward terms achieves both the lowest CR (0.089) and stable ADD (0.257)

**Ablation findings (Table 3 — auxiliary objectives):**
- Removing all auxiliary objectives substantially increases CR; all four auxiliary losses contribute to safety
- Auxiliary objectives alone (without PPO) give CR=0.187 > 0.089, confirming they work best jointly with the PPO objective

### Strengths

- **Photorealistic RL environment**: 3DGS reconstruction of real-world data eliminates the sim-to-real gap in visual appearance, enabling policies trained purely on rendered frames to transfer directly to real-world sensor data.
- **Synergistic RL+IL design**: RL addresses causal confusion and the open-loop gap; IL maintains human-aligned, smooth behavior — the two are complementary rather than conflicting.
- **Dense auxiliary objectives solve sparse reward**: The directional auxiliary losses provide informative gradients at every step without requiring additional labels, significantly improving convergence.
- **Large-scale validation**: 4305 reconstructed scenes, 3968 for RL training and 337 for closed-loop evaluation on previously unseen environments, demonstrating genuine generalization.
- **Decoupled action space**: Separating lateral and longitudinal actions reduces effective exploration dimensionality and enables attribution of distinct reward signals to each axis.

### Weaknesses / Open Questions

1. **3DGS quality ceiling**: Non-rigid pedestrians, unobserved views from novel ego trajectories, and low-light conditions remain challenging for 3DGS rendering, potentially introducing visual artifacts during RL rollouts.
2. **Non-reactive traffic agents**: Other vehicles are log-replayed with real-world trajectories, ignoring the ego vehicle's changed behavior. This limits the realism of multi-agent interaction and may allow the policy to exploit replay artifacts.
3. **Proprietary data**: The 2000-hour driving dataset and the 4305 reconstructed 3DGS environments are not publicly released, making reproducibility dependent on institutional resources.
4. **Extreme compute requirements**: Stage 1–2 training uses 128 RTX4090 GPUs; Stage 3 uses 32 RTX4090 — far beyond academic lab capacity.
5. **No real-world closed-loop test**: The closed-loop evaluation is conducted entirely within 3DGS environments. There is no safety-driver or autonomous vehicle test on public roads to confirm the learned policy transfers at deployment.

### References to Follow Up

1. **[Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting](../../2024/Street_Gaussians-_Modeling_Dynamic_Urban_Scenes_with_Gaussian_Splatting/)** — Yunzhi Yan et al., ECCV 2024: The 3DGS dynamic scene representation that RAD extends with mesh-constrained road and improved foreground reconstruction.
2. **[HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving](../../2024/HUGSIM-_A_Real-Time,_Photo-Realistic_and_Closed-Loop_Simulator_for_Autonomous_Driving/)** — Hongyu Zhou et al., arXiv 2024: Concurrent 3DGS-based AD simulator; RAD goes further by incorporating RL training rather than just closed-loop evaluation.
3. **VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning** — Shaoyu Chen et al., arXiv 2024: The strongest IL baseline from the same lab that RAD post-trains from; understanding VADv2's architecture helps understand RAD's Stage 2.
4. **Proximal Policy Optimization Algorithms** — Schulman et al., arXiv 2017: The PPO algorithm used as RAD's RL backbone, including the clipping and GAE formulations.
5. **Imitation is Not Enough: Robustifying Imitation with RL for Challenging Driving Scenarios** — Lu et al., IROS 2023: Influential prior work on combining IL and RL for driving (in CARLA), which RAD generalizes to a photorealistic 3DGS environment.

---

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Scene Representation and 3DGS Environment**

RAD extends StreetGaussian to support closed-loop training. The key enhancements over vanilla StreetGaussian are: (1) a road mesh is used to constrain background Gaussian spheres to the road surface, ensuring accurate geometry from any viewpoint the ego vehicle might visit during RL exploration; (2) sky is modeled as a separate component to avoid confusion with foreground objects; (3) foreground object (vehicle, pedestrian) poses are jointly optimized during reconstruction, with depth and normal consistency losses added as supervision signals to improve surface detail fidelity. These improvements matter because RL rollouts take the ego vehicle off the expert trajectory — the 3DGS environment must render correctly from these novel viewpoints, not just the original camera path.

**Policy Architecture**

The AD policy is a four-module end-to-end network taking multi-view image sequences as input and outputting a probability distribution over decoupled lateral and longitudinal actions.

- **BEV Encoder** (BEVFormer): Projects perspective-view multi-camera features into a Bird's-Eye View feature map, yielding instance-level map tokens and agent tokens.
- **Map Head** (MapTRv2): A group of learnable map tokens attends to the BEV feature map to predict vectorized HD map elements (lane centerlines, dividers, road boundaries, traffic signals).
- **Agent Head** (PIP): A group of learnable agent tokens predicts surrounding agent motion including location, orientation, size, speed, and multi-mode future trajectories.
- **Image Encoder** (ViT/ResNet): Encodes raw image patches into dense image tokens, complementing the sparse instance-level tokens with rich scene texture.
- **Planning Head**: A cascaded Transformer decoder $\phi$ with planning embedding $E_{plan}$ as query and scene representation $E_{scene}$ (map tokens + agent tokens + image tokens) as key and value. Combined with navigation information $E_{navi}$ and ego state $E_{state}$ , the decoder output passes through separate MLP heads to produce the lateral and longitudinal action distributions:

```math
\pi(a^x | s) = \text{softmax}(\text{MLP}(\phi(E_{plan}, E_{scene}) + E_{navi} + E_{state}))
```

```math
\pi(a^y | s) = \text{softmax}(\text{MLP}(\phi(E_{plan}, E_{scene}) + E_{navi} + E_{state}))
```

The same decoder output is used to produce two scalar value functions $V_x(s)$ and $V_y(s)$ that estimate expected cumulative lateral and longitudinal rewards respectively, used during RL training.

**Decoupled Discrete Action Space**

Each action represents a target displacement over a 0.5-second planning horizon, assuming constant linear and angular velocities within that window. Lateral displacement $a^x$ and longitudinal displacement $a^y$ are separately discretized:

```math
a^x \in \{d^x_{min}, \ldots, 0, \ldots, d^x_{max}\}, \quad N_x = 61, \; d^x_{min} = -0.75 \text{ m}, \; d^x_{max} = 0.75 \text{ m}
```

```math
a^y \in \{0, \ldots, d^y_{max}\}, \quad N_y = 61, \; d^y_{max} = 15 \text{ m}
```

From predicted $(a^x_t, a^y_t)$ , the linear velocity $v_t$ and steering angle $\delta_t$ are computed, and the ego vehicle pose is updated via a kinematic bicycle model:

```math
x^w_{t+1} = x^w_t + v_t \cos(\psi^w_t) \Delta t, \quad y^w_{t+1} = y^w_t + v_t \sin(\psi^w_t) \Delta t, \quad \psi^w_{t+1} = \psi^w_t + \frac{v_t}{L} \tan(\delta_t) \Delta t
```

where $\psi^w_t$ is heading angle in world coordinates and $L$ is the wheelbase.

**Three-Stage Training Paradigm**

- *Stage 1 — Perception Pre-Training*: BEV encoder, map head, and agent head trained with ground-truth map/agent labels. Image encoder and planning head are randomly initialized and frozen. This ensures the instance-level scene tokens carry high-quality semantic information before planning begins.
- *Stage 2 — Planning Pre-Training*: Image encoder and planning head trained with IL on 2000 hours of expert driving demonstrations. Predefined anchor positions $A = \{(a^x_i, a^y_j)\}_{i=1,j=1}^{N_x, N_y}$ are used; the ground-truth vehicle position at $t = 0.5$ s is matched to the nearest anchor $(\hat{i}, \hat{j})$ via normalized nearest-neighbor:

```math
\hat{i} = \arg\min_i \left\| \frac{a^x_i - d^x_{min}}{d^x_{max} - d^x_{min}} - \frac{p^x_{gt} - d^x_{min}}{d^x_{max} - d^x_{min}} \right\|_2
```

The IL objective is a dual focal loss: $L_{IL} = L_{focal}(\pi(a^x|s), \hat{i}) + L_{focal}(\pi(a^y|s), \hat{j})$ . BEV encoder, map head, and agent head are frozen to avoid conflicting gradients between perception and planning objectives.

- *Stage 3 — Reinforced Post-Training*: $N = 32$ parallel workers each maintain a 3DGS environment. Each worker randomly samples a scene, runs a rollout where the AD policy controls the ego vehicle, and stores transitions $(s_t, a_t, r_{t+1}, s_{t+1}, \ldots)$ in a shared replay buffer. A sliding window holds 4 clips (each 8 seconds at 10 Hz = 80 frames). Training cycles alternate: 4 rounds of RL (320 iterations each = 4 clips of data) followed by 1 round of IL. After each fixed number of steps, the policy is synchronized to all workers. Only image encoder and planning head parameters are updated; BEV encoder, map head, and agent head remain frozen.

**Reward Design**

The reward $R = \{r_{dc}, r_{sc}, r_{pd}, r_{hd}\}$ consists of four safety-driven components:

- $r_{dc}$ (dynamic collision): Negative reward when the ego vehicle's bounding box overlaps with annotated bounding boxes of dynamic obstacles (pedestrians, vehicles).
- $r_{sc}$ (static collision): Negative reward when the ego vehicle's bounding box overlaps with the Gaussians of static roadside obstacles.
- $r_{pd}$ (positional deviation): Negative reward when the Euclidean distance from ego to the nearest point on the expert trajectory exceeds $d_{max} = 2.0$ m.
- $r_{hd}$ (heading deviation): Negative reward when the angular difference between ego heading $\psi_t$ and the expert's matched heading exceeds $\psi_{max} = 40°$ .

Any of the four events also immediately terminates the episode, since subsequent frames typically produce corrupted sensor data in the 3DGS environment.

**PPO with Decoupled GAE**

Rewards are split into lateral and longitudinal components:

```math
r^x_t = r^{sc}_t + r^{pd}_t + r^{hd}_t, \qquad r^y_t = r^{dc}_t
```

Separate temporal-difference errors and advantage estimates are computed for each axis:

```math
\delta^x_t = r^x_t + \gamma V_x(s_{t+1}) - V_x(s_t), \qquad A^x_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta^x_{t+l}
```

```math
A^x_t = A^{sc}_t + A^{pd}_t + A^{hd}_t, \qquad A^y_t = A^{dc}_t
```

The full PPO objective with independent clipping per axis:

```math
L^{PPO}_x(\theta) = E_t \left[ \min\left( \rho^x_t A^x_t, \; \text{clip}(\rho^x_t, 1-\epsilon_x, 1+\epsilon_x) A^x_t \right) \right]
```

```math
L^{PPO}(\theta) = L^{PPO}_x(\theta) + L^{PPO}_y(\theta)
```

where $\rho^x_t = \pi_\theta(a^x_t | s_t) / \pi_{\theta_{old}}(a^x_t | s_t)$ is the importance sampling ratio. Clipping thresholds: $\epsilon_x = 0.1$ , $\epsilon_y = 0.2$ . GAE parameters: $\gamma = 0.9$ , $\lambda = 0.95$ .

**Auxiliary Objectives**

To address sparse rewards, four directional auxiliary losses provide dense per-step gradients. The action probability distribution for each axis is decomposed into cumulative probability mass in each direction relative to the old policy's action:

```math
\Delta\pi^{dec}_y = \sum_{a^y_t < a^{y,old}_t} \pi_\theta(a^y_t | s_t), \quad \Delta\pi^{acc}_y = \sum_{a^y_t > a^{y,old}_t} \pi_\theta(a^y_t | s_t)
```

```math
\Delta\pi^{left}_x = \sum_{a^x_t < a^{x,old}_t} \pi_\theta(a^x_t | s_t), \quad \Delta\pi^{right}_x = \sum_{a^x_t > a^{x,old}_t} \pi_\theta(a^x_t | s_t)
```

**Dynamic Collision Auxiliary Loss** — encourages deceleration if a dynamic collision is detected ahead ($f_{dc} = 1$) or acceleration if behind ($f_{dc} = -1$):

```math
L_{dc}(\theta) = E_t \left[ A^{dc}_t \cdot f_{dc} \cdot (\Delta\pi^{dec}_y - \Delta\pi^{acc}_y) \right]
```

**Static Collision Auxiliary Loss** — steers away from static obstacles on the left ($f_{sc} = 1$) or right ($f_{sc} = -1$):

```math
L_{sc}(\theta) = E_t \left[ A^{sc}_t \cdot f_{sc} \cdot (\Delta\pi^{right}_x - \Delta\pi^{left}_x) \right]
```

**Positional Deviation Auxiliary Loss** — corrects lateral position toward the expert trajectory when the vehicle has drifted left ($f_{pd} = 1$) or right ($f_{pd} = -1$):

```math
L_{pd}(\theta) = E_t \left[ A^{pd}_t \cdot f_{pd} \cdot (\Delta\pi^{right}_x - \Delta\pi^{left}_x) \right]
```

**Heading Deviation Auxiliary Loss** — corrects angular drift when heading is clockwise ($f_{hd} = 1$) or counterclockwise ($f_{hd} = -1$):

```math
L_{hd}(\theta) = E_t \left[ A^{hd}_t \cdot f_{hd} \cdot (\Delta\pi^{right}_x - \Delta\pi^{left}_x) \right]
```

Full composite objective:

```math
L(\theta) = L^{PPO}(\theta) + \lambda_1 L_{dc}(\theta) + \lambda_2 L_{sc}(\theta) + \lambda_3 L_{pd}(\theta) + \lambda_4 L_{hd}(\theta)
```

### Hidden Assumptions

1. The 3DGS rendering is visually close enough to the real sensor feed that a policy trained on 3DGS frames generalizes without explicit domain adaptation. The consistency analysis (Fig. 5) supports this but does not formally bound the gap.
2. Log-replaying other traffic participants preserves realistic interaction density even when the ego vehicle deviates significantly from its original trajectory. In practice, other agents do not react to the ego, which can make some collision scenarios (e.g., rear-ending a stopped car) easier to exploit or avoid than in reality.
3. The expert trajectory used for $r_{pd}$ and $r_{hd}$ supervision is always the optimal path. There is no mechanism to allow the policy to discover genuinely better routes than the human demonstrated.
4. The kinematic bicycle model accurately represents ego vehicle dynamics at the inference time step of $\Delta t = 0.5$ s. High-curvature maneuvers or slippery surfaces may violate this assumption.
5. Selecting risky, dense-traffic scenes for 3DGS reconstruction yields a training distribution that covers the tail of the safety-critical scenario space. Unrepresented scenarios (e.g., highway merges, construction zones) may not be learned.

### Reproducibility Notes

- **Code**: Available at https://github.com/hustvl/RAD
- **Data**: 2000 hours of proprietary driving data collected by Horizon Robotics; not publicly released. Scene selection criterion ("risky, dense-traffic clips") is qualitative and not precisely specified.
- **Compute — Stage 1–2**: 128 RTX 4090 GPUs, 30k training steps, batch size 512, lr 1e-4, AdamW with cosine decay
- **Compute — Stage 3**: 32 RTX 4090 GPUs, 32 workers, RL batch 32, IL batch 128, lr 5e-6, AdamW with cosine decay
- **Missing details**: Exact weighting coefficients $\lambda_1$–$\lambda_4$ for auxiliary losses are not reported in the main paper or appendix; the precise negative reward magnitudes for $r_{dc}$, $r_{sc}$, $r_{pd}$, $r_{hd}$ are also not listed; scene selection algorithm for the 4305 clips is not formalized.
- **Provided**: GAE $\gamma = 0.9$, $\lambda = 0.95$; clip $\epsilon_x = 0.1$, $\epsilon_y = 0.2$; $d_{max} = 2.0$ m, $\psi_{max} = 40°$; planning head dim 256; value function dim 256; lateral bins $N_x = 61$, $d^x \in [-0.75, 0.75]$ m; longitudinal bins $N_y = 61$, $d^y \in [0, 15]$ m.

### Ideas for Future Work

1. **Reactive traffic agent modeling**: Replace log-replay of surrounding agents with IDM-based or learned reactive agents that respond to the ego vehicle's behavior, enabling training on genuine multi-agent interaction scenarios.
2. **Improved non-rigid rendering**: Develop better 3DGS modeling for deformable objects (pedestrians, cyclists) — e.g., with articulated Gaussian primitives — to eliminate rendering artifacts that produce noisy training signals in pedestrian-heavy scenes.
3. **Scaling RL exploration**: The current 32 GPU setup processes 4 clips per worker per cycle. Scaling to hundreds of workers with larger scene diversity could further push down collision rates.
4. **Real-world closed-loop validation**: Deploy the learned policy in a safety-driver-assisted real vehicle to quantify sim-to-real transfer and identify any remaining domain gaps.
5. **Reward learning**: Replace hand-designed rewards with learned reward models from human preference feedback or inverse RL, allowing the reward signal to capture nuanced human judgment beyond binary collision and deviation thresholds.

---

## Pass 4 — Modern Perspective Review (as of July 2026)

### What Has Changed Since Publication

- **3DGS scene quality** has continued to improve rapidly (DrivingGaussian, HUGSIM, OmniRe); non-rigid rendering of pedestrians and cyclists remains an open problem but has seen significant progress since early 2025.
- **RL in AD** has gained further traction following the success of reasoning-model post-training (DeepSeek-R1, OpenAI o1), with multiple groups applying GRPO and similar group-based RL methods to planning and trajectory prediction.
- **Open-loop vs. closed-loop debate** in AD has been settled in favor of closed-loop evaluation; RAD's 3DGS-based benchmark aligns with this consensus.
- **Foundation model AD** (large-scale pre-trained world models) has emerged as a competing paradigm, raising the question of whether RL fine-tuning of IL models is the right lever versus training larger models on more diverse data.
- **Sim-to-real gap** remains underexplored — follow-on work is expected to test whether 3DGS-trained policies actually improve on-road deployment rather than just 3DGS-evaluation metrics.

### Has the Community Accepted the Claims?

RAD is a NeurIPS 2025 paper, so full community assessment is still in its early stages as of mid-2026. The core claim — that 3DGS-based RL significantly outperforms IL in closed-loop safety metrics — is well supported by the paper's own ablations and comparisons. The 3× reduction in collision rate over the strongest IL baseline (VADv2, from the same lab) is striking, though it is measured exclusively on the authors' proprietary 3DGS benchmark, which makes independent reproduction difficult. The combination of IL and RL is a principled and increasingly mainstream direction, and the architectural choices (PPO, decoupled action space, GAE) are standard. The primary open question is whether the 3DGS fidelity improvements translate to real-world safety gains, which the paper does not demonstrate. The GitHub release of code should facilitate partial reproduction by groups with sufficient compute.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl et al. | 2023 | Core scene representation RAD builds upon |
| [Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting](../../2024/Street_Gaussians-_Modeling_Dynamic_Urban_Scenes_with_Gaussian_Splatting/) | Yan et al. | 2024 | 3DGS extension for dynamic driving scenes; RAD's direct base for environment construction |
| VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning | Chen et al. | 2024 | IL-based predecessor from same lab; RAD Stage 2 initializes from this paradigm |
| [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Videos](../../2022/BEVFormer-_Learning_Bird's-Eye-View_Representation_from_Multi-Camera_Images_via_Spatiotemporal_Transformers/) | Li et al. | 2022 | BEV encoder used in RAD's policy architecture |
| Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | RL algorithm (PPO) used for policy optimization in Stage 3 |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | Schulman et al. | 2015 | GAE formulation used for advantage estimation in RAD's decoupled PPO |
| Imitation is Not Enough: Robustifying Imitation with RL for Challenging Scenarios | Lu et al. | 2023 | Prior IL+RL hybrid for CARLA driving; RAD generalizes to 3DGS photorealistic environment |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for AD | Zhou et al. | 2024 | Concurrent 3DGS AD simulator; focuses on evaluation rather than RL training |
| DiffusionDrive: Truncated Diffusion Model for End-to-End AD | Liao et al. | 2025 | Concurrent end-to-end AD with diffusion planning; uses IL only |
| SparseDrive: End-to-End AD via Sparse Scene Representation | Sun et al. | 2025 | Concurrent end-to-end IL baseline RAD compares against indirectly |
| NeuRAD: Neural Rendering for Autonomous Driving | Tonderski et al. | 2024 | NeRF-based AD simulator for sensor simulation; 3DGS-based RAD offers faster rendering |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| (none identified as of July 2026) | — | — | Paper too recent for confirmed follow-ons |

---

### Bottom Line

RAD is a timely and technically solid contribution that makes a convincing case for 3DGS-based closed-loop RL as a training paradigm for end-to-end AD. The collision rate improvements over IL baselines are large enough to be practically meaningful. As the first paper to jointly integrate RL and IL training within a photorealistic 3DGS digital twin for sensor-input policies, it is a foundational reference for this line of work. Its main limitation — that results are validated only on a proprietary 3DGS benchmark with log-replayed traffic agents and no real-world closed-loop test — means the community should treat the numbers as strong preliminary evidence rather than a definitive proof of deployment-ready safety. Anyone working on closed-loop AD training, photorealistic simulation, or RL fine-tuning of driving policies should read this paper.
