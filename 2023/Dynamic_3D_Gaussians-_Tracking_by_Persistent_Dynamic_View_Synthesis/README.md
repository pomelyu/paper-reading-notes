# Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

- **Authors:** Jonathon Luiten, Georgios Kopanas, Bastian Leibe, Deva Ramanan
- **Affiliations:** Carnegie Mellon University, RWTH Aachen University, Inria & Université Côte d'Azur
- **Published:** arXiv:2308.09295, 2023 (presented at 3DV 2024)
- **Keywords:** 3D Gaussian splatting, dynamic scene reconstruction, dense tracking, novel-view synthesis, non-rigid reconstruction, 6-DOF tracking
- **Website:** https://dynamic3dgaussians.github.io/
- **Github:** https://github.com/JonathonLuiten/Dynamic3DGaussians

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|-----------|
| **Category** | Method paper combining dynamic scene reconstruction, novel-view synthesis, and dense 6-DOF point tracking in a single unified framework |
| **Context** | Extends 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) to dynamic scenes; draws on non-rigid physical modelling (local rigidity priors from SLAM/reconstruction), and the emerging literature on long-term point tracking (PIPs, TAP-Vid, OmniMotion) |
| **Correctness** | Assumptions are physically motivated and well-grounded; local rigidity is a reasonable prior for most solid-object scenes; the multi-camera requirement is acknowledged as a limitation; minor concern over reliance on background frame-differencing for segmentation |
| **Contributions** | (1) First method to use 3D Gaussians for dynamic scene reconstruction; (2) Dense full 6-DOF scene tracking emerging purely from persistent view synthesis, without any optical flow or correspondence input; (3) New PanopticSports benchmark; (4) Multiple downstream applications (first-person view, scene editing, 4D composition) |
| **Clarity** | Well-written and logically structured; motivation is compelling; figures and tables effectively support the narrative |

Dynamic 3D Gaussians (D3DGS) represents a dynamic scene as 200–300k 3D Gaussians whose color, opacity, and size are fixed across time while their position and orientation are free to vary. Three physically-motivated regularization losses — local rigidity, local rotation similarity, and long-term isometry — constrain the Gaussians to move as locally rigid bodies. Trained online one timestep at a time via differentiable rendering on 27 synchronized cameras, the method achieves 28.7 PSNR on novel-view synthesis and 2.21 cm 3D median tracking error, a 10× improvement over the prior 2D tracking state-of-the-art, all without ever consuming optical flow as input.

![teaser](resources/fig_01_teaser.png)
---

## Pass 2 — Careful Read

### Core Idea in One Sentence

Freeze every appearance attribute of 3D Gaussians while letting their positions and rotations drift through time under local-rigidity regularization, so that physically correct dense 6-DOF tracking emerges as a by-product of fitting rendered images to multi-camera observations.

### Method / Approach

- **Persistent Gaussian Representation:** Each Gaussian $i$ carries $7t + 8$ parameters: time-varying center $\mu_{i,t}$ and quaternion rotation $q_{i,t}$, plus time-invariant scale $s_i$, color $c_i$, opacity logit $o_i$, and background logit $bg_i$. Fixing appearance forces each Gaussian to persistently represent the same physical patch of matter, making its trajectory synonymous with dense scene tracking.

- **Differentiable Rendering:** Inherits the 3DGS tile-rasterizer: each Gaussian is projected to a 2D Gaussian via the Jacobian of perspective projection, then composited front-to-back using the Max (1995) volume-rendering formula. This enables gradient flow from per-pixel RGB loss back to all Gaussian parameters at 850 FPS rendering speed.

- **Three Physical Regularizers:** (1) *Local rigidity* $\mathcal{L}^{\mathrm{rigid}}$ forces each Gaussian's $k=20$ nearest neighbours to undergo the same rigid-body transform as the Gaussian itself between consecutive timesteps — the dominant and most impactful loss. (2) *Local rotation similarity* $\mathcal{L}^{\mathrm{rot}}$ explicitly enforces neighbouring quaternion updates to be equal, aiding convergence. (3) *Long-term isometry* $\mathcal{L}^{\mathrm{iso}}$ preserves pairwise distances among neighbours over the full sequence to prevent drift. All three use the same distance-weighted Gaussian kernel ($\sigma \approx 2.2$ cm at first timestep).

- **Online Temporal Optimization:** The first frame is fully optimized (10k iterations, ~4 min) including Gaussian densification. Subsequent frames fix appearance and only optimize $\mu$ and $q$ (2k iterations, ~50 s each), initialized by forward-extrapolating velocity from the previous timestep. Adam momentum is reset per timestep. A background segmentation loss against frame-differencing masks prevents foreground–background confusion on low-contrast clothing.

### Key Results

| Task | Metric | 3GS-O [17] | PIPs [12] | **Ours** |
|------|--------|-----------|----------|---------|
| View synthesis | PSNR ↑ | 28.21 | — | **28.7** |
| View synthesis | SSIM ↑ | 0.90 | — | **0.91** |
| View synthesis | LPIPS ↓ | **0.16** | — | 0.17 |
| 3D tracking | MTE (cm) ↓ | 55.9 | — | **2.21** |
| 3D tracking | $\delta$ ↑ | 6.8 | — | **71.4** |
| 3D tracking | Survival ↑ | 43.8 | — | **100** |
| 2D tracking | MTE (px) ↓ | 43.8 | 15.7 | **1.57** |
| 2D tracking | $\delta$ ↑ | 10.9 | 39.6 | **78.4** |
| 2D tracking | Survival ↑ | 54.7 | 79.0 | **100** |

Ablation highlights (Juggle scene):
- Removing $\mathcal{L}^{\mathrm{rigid}}$ is by far the most damaging: 3D MTE degrades from 1.90 → 4.32 cm.
- Removing the background segmentation loss $\mathcal{L}^{Bg}$ collapses performance: MTE → 8.46 cm, PSNR → 24.14.
- Removing parameter fixing (allowing color/size to change) causes catastrophic drift: MTE → 30.7 cm.
- Forward velocity propagation for initialization is essential: without it, MTE → 6.32 cm.
- $\mathcal{L}^{\mathrm{rot}}$ and $\mathcal{L}^{\mathrm{iso}}$ individually produce only marginal metric gains, but the authors note qualitatively cleaner reconstructions when both are on.

### Strengths

- **No correspondence input required:** Tracking emerges entirely from fitting persistent Gaussians to raw multi-view images — no optical flow, no pose skeletons, no synthetic pretraining.
- **Full 6-DOF tracking:** Quaternion representation tracks rotation of every 3D point, enabling first-person view cameras, hat-on-handstander effects, and edit propagation.
- **Speed:** 850 FPS rendering; 2 hours training for 150 timesteps on a single RTX 3090 — orders of magnitude faster than comparable NeRF-based approaches.
- **100% track survival:** The persistent representation never loses a point, unlike any prior tracker.
- **Downstream versatility:** Scene composition, time-propagated edits, and first-person view are direct, nearly zero-cost applications of the representation.

### Weaknesses / Open Questions

1. **Multi-camera requirement:** 27 synchronized, calibrated cameras in a dome are required; the method does not generalize to monocular video out of the box, limiting real-world applicability.
2. **Closed-world assumption:** Objects entering the scene after frame 1 are completely unrepresented; the approach cannot handle occlusion-then-reveal of new content.
3. **Depth camera for initialization:** Although depth is only used to seed the initial point cloud and not in optimization, this is an extra sensor not always available.
4. **Foreground segmentation dependency:** The background loss relies on frame-differencing, which breaks when foreground and background share similar appearance — explicitly acknowledged for grey shirts vs. grey backgrounds.
5. **LPIPS slightly worse than baseline:** The per-frame 3GS-O achieves 0.16 vs. 0.17 for D3DGS, suggesting the temporal consistency constraint very slightly trades perceptual sharpness for coherence.

### References to Follow Up

1. **3D Gaussian Splatting for Real-Time Radiance Field Rendering** — Kerbl et al., SIGGRAPH 2023: the direct base renderer and initialization procedure this work builds on.
2. **Tracking Everything Everywhere All at Once (OmniMotion)** — Wang et al., arXiv 2023: the closest competing method; uses test-time NeRF fitting for tracking but requires dense optical flow as input — a key differentiator.
3. **Particle Video Revisited (PIPs)** — Harley, Fang & Fragkiadaki, ECCV 2022: the primary 2D tracking baseline; trained on 13k videos with ground-truth tracks, yet beaten 10× by D3DGS with no track supervision.
4. **PointOdyssey** — Zheng et al., ICCV 2023: the benchmark protocol this work adapts to 3D; important for understanding evaluation choices.
5. **Total-Recon** — Song et al., ICCV 2023: deformable scene reconstruction with linear blend skinning, representing the strongest canonical-deformation-field competitor for tracking applications.

---

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Representation.** The scene $S$ at timestep $t$ is a set of $N$ Dynamic 3D Gaussians ($N \approx 200$–$300\text{k}$). Gaussian $i$ stores:
- Time-varying: center $\mu_{i,t} \in \mathbb{R}^3$, rotation quaternion $q_{i,t} \in \mathbb{R}^4$ (normalized)
- Time-invariant: scale $s_i = [sx,\, sy,\, sz] \in \mathbb{R}^3$, color $c_i \in \mathbb{R}^3$, opacity logit $o_i \in \mathbb{R}$, background logit $bg_i \in \mathbb{R}$

The 3D covariance is $\Sigma_{i,t} = R_{i,t} S_i S_i^T R_{i,t}^T$ where $S_i = \mathrm{diag}([sx_i,\, sy_i,\, sz_i])$ and $R_{i,t} = \mathrm{q2R}(q_{i,t})$. The influence of Gaussian $i$ on world point $\boldsymbol{p}$ at time $t$ is:

$$f_{i,t}(\boldsymbol{p}) = \mathrm{\sigm}(o_i)\exp\left(-\frac{1}{2}(\boldsymbol{p} - \boldsymbol{\mu}_{i,t})^T \Sigma_{i,t}^{-1}(\boldsymbol{p} - \boldsymbol{\mu}_{i,t})\right)$$

Soft infinite extent is crucial: Gaussians far from their target location still receive nonzero gradients, which is necessary for gradient-based tracking when a Gaussian starts in the wrong position.

**Rendering.** The standard 3DGS pipeline is used unchanged. Each Gaussian's center is projected:

$$\mu^{\mathrm{2D}} = K\!\left((E\mu)/(E\mu)_z\right)$$

The 3D covariance is Jacobian-approximated into 2D: $\Sigma^{\mathrm{2D}} = J E \Sigma E^T J^T$, where $J = \partial\mu^{\mathrm{2D}}/\partial\mu$. Gaussians are sorted by depth per tile, then alpha-composited:

$$C_{\mathrm{pix}} = \sum_{i \in \mathcal{S}} c_i f_{i,\mathrm{pix}}^{\mathrm{2D}} \prod_{j=1}^{i-1}\!\left(1 - f_{j,\mathrm{pix}}^{\mathrm{2D}}\right)$$

The same formula with Gaussian depths replacing colors produces differentiable depth maps used for unprojection and tracking.

**Optimization schedule.** Timestep 0: all parameters optimized for 10,000 iterations including the 3DGS adaptive densification/pruning scheme; initial point cloud from 10 depth cameras (subsampled 2×). Timesteps $1 \ldots T$: only $\mu_{i,t}$ and $q_{i,t}$ optimized for 2,000 iterations; size/color/opacity/bg frozen from $t=0$. Initialization for each new timestep:

$$\mu_{i,t}^{\text{init}} = \mu_{i,t-1} + (\mu_{i,t-1} - \mu_{i,t-2})$$

(velocity extrapolation), same for quaternions with re-normalization. Adam first/second-order momentum is reset at each new timestep (critical to avoid stale momentum from the previous geometry warping gradients).

**Regularization losses.** All three losses share a weighting kernel:

$$w_{i,j} = \exp\!\left(-\lambda_w \|\mu_{j,0} - \mu_{i,0}\|^2\right)$$

with $\lambda_w = 2000$ ($\sigma \approx 2.2$ cm), computed once from $t=0$ positions and fixed. The $k=20$ nearest neighbours in $t=0$ are used for all subsequent timesteps.

*Local rigidity* $\mathcal{L}^{\mathrm{rigid}}$: for each pair $(i, j)$, the relative position of $j$ in $i$'s local frame should not change between $t-1$ and $t$:

$$\mathcal{L}_{i,j}^{\mathrm{rigid}} = w_{i,j} \left\|(\mu_{j,t-1} - \mu_{i,t-1}) - R_{i,t-1} R_{i,t}^{-1}(\mu_{j,t} - \mu_{i,t})\right\|_2$$

$$\mathcal{L}^{\mathrm{rigid}} = \frac{1}{k|S|} \sum_{i \in S} \sum_{j \in \mathrm{knn}_{i;k}} \mathcal{L}_{i,j}^{\mathrm{rigid}}$$

The term $R_{i,t-1} R_{i,t}^{-1}$ rotates the $t-1$ relative position into the $t$ coordinate frame of $i$. This simultaneously constrains translation ($j$'s position follows $i$'s rigid motion) and rotation ($i$'s rotation must be consistent with $j$'s induced displacement), enabling 6-DOF tracking even though the image loss only supervises color.

*Local rotation similarity* $\mathcal{L}^{\mathrm{rot}}$: neighbouring quaternion deltas should match:

$$\mathcal{L}^{\mathrm{rot}} = \frac{1}{k|\mathcal{S}|} \sum_{i \in \mathcal{S}} \sum_{j \in \mathrm{knn}_{i;k}} w_{i,j} \left\|\hat{q}_{j,t}\hat{q}_{j,t-1}^{-1} - \hat{q}_{i,t}\hat{q}_{i,t-1}^{-1}\right\|_2$$

where $\hat{q}$ is the normalized quaternion. Applied only between $t$ and $t-1$ (short-term). Provides explicit rotation gradient that $\mathcal{L}^{\mathrm{rigid}}$ only implicitly encodes.

*Long-term isometry* $\mathcal{L}^{\mathrm{iso}}$: pairwise distances between neighbours should equal their $t=0$ distances:

$$\mathcal{L}^{\mathrm{iso}} = \frac{1}{k|S|} \sum_{i \in S} \sum_{j \in \mathrm{knn}_{i;k}} w_{i,j} \left|\left\|\mu_{j,0} - \mu_{i,0}\right\|_2 - \left\|\mu_{j,t} - \mu_{i,t}\right\|_2\right|$$

Weaker than rigidity (only preserves distances, not directions), but acts over long horizons to counter drift accumulation.

**Background handling.** A foreground/background logit $bg_i$ is per-Gaussian. A pseudo-GT mask is obtained by frame-differencing against a reference frame with no foreground; a binary cross-entropy loss $\mathcal{L}^{Bg}$ supervises the rendered mask. Additionally, background Gaussians have their rigidity/rotation/isometry losses zeroed out (not applied between foreground and background), and a direct loss penalizes any movement or rotation of background Gaussians.

**Per-camera color calibration.** A scale $a_c$ and offset $b_c$ per color channel per camera is optimized in timestep 0 and frozen: $\tilde{c} = a_c \cdot c + b_c$. This corrects for white balance, exposure, and sensor differences across 27 cameras without affecting the learned Gaussian colors.

**Tracking inference.** To track a 3D point $p$ from time $t_1$ to $t_2$, find the most-influential Gaussian $i^* = \arg\max_i f_{i,t_1}(p)$. Express $p$ in $i^*$ 's local frame at $t_1$, then transform by $i^*$ 's motion to get its location at $t_2.$ If no Gaussian has $f > 0.5,$ the point is classified as static background. For 2D tracking: unproject pixel→3D using rendered depth, track in 3D, re-project into target camera.

### Hidden Assumptions

1. All foreground scene elements visible in frame 0 undergo locally rigid motion throughout the sequence — fails for fluids, smoke, cloth with complex draping, or highly articulated joints with very different local motions.
2. The scene background is static and can be separated from foreground via simple frame-differencing — breaks in dynamic lighting, camera auto-exposure, or scenes without a reference background frame.
3. 27 synchronized, geometrically calibrated cameras with accurate intrinsics/extrinsics are available — not a realistic assumption for casual video capture.
4. The initial sparse point cloud from depth cameras adequately covers the scene — works in a controlled dome but not general outdoor settings.
5. Scene objects are present from the very first frame — the representation has no mechanism for adding new Gaussians after $t=0$.
6. Foreground objects move much more than the camera (the frame-differencing mask assumption) — fails if the rig itself is moving.
7. $k=20$ nearest neighbours and $\sigma \approx 2.2$ cm are appropriate for all scene scales — these hyperparameters are tuned for human-scale motion in a studio.

### Reproducibility Notes

- **Dataset:** CMU Panoptic Studio (publicly available); PanopticSports is a prepared subset of 6 sports sequences; 21 ground-truth 3D trajectories from facial/hand keypoint annotations.
- **Code:** Not mentioned as released in the paper itself (code was later made available on GitHub).
- **Hardware:** Single NVIDIA RTX 3090; 2 hours total for 150 timesteps × 27 cameras at 640×360.
- **Optimizer:** Adam; exact learning rates not stated in the paper text.
- **Key hyperparameters stated:** $k=20$ neighbours, $\lambda_w=2000$, $t=0$ iterations $=10000$, $t>0$ iterations $=2000$, rendering size 640×360, 200–300k Gaussians total.
- **Missing/underspecified:** Adam learning rate schedule, weight balancing among $\mathcal{L}^{\mathrm{rigid}}$ / $\mathcal{L}^{\mathrm{rot}}$ / $\mathcal{L}^{\mathrm{iso}}$ / $\mathcal{L}^{Bg}$, exact densification thresholds (deferred to [17]), how the foreground/background logit threshold of 0.5 was chosen.
- **Depth camera dependency:** Only 10 depth cameras used for initialization sparse point cloud (not during training), but this is an additional sensor requirement beyond the 27 HD cameras.

### Ideas for Future Work

1. **Monocular extension:** Use learned monocular depth and/or video diffusion priors as pseudo-multi-view supervision to remove the multi-camera requirement.
2. **Open-world tracking:** Detect and spawn new Gaussians when unseen objects enter the frame, perhaps leveraging SAM-style segmentation to identify new regions.
3. **Learned rigidity priors:** Replace hand-crafted $k$-NN rigidity with a learned graph network that infers object part structure and applies part-level rigidity.
4. **Longer sequences without drift:** The isometry loss partially addresses drift but short-term rigidity can accumulate error; explicit loop-closure or global bundle adjustment over the Gaussian trajectories could help.
5. **Semantic editing:** Incorporate CLIP or DINO features into the Gaussian representation to enable language-guided editing and segmentation of dynamic components.
6. **Compression and streaming:** The $7t+8$ parameters per Gaussian scale linearly with timesteps; trajectory compression (e.g., B-splines) could make long videos tractable.

---

## Pass 4 — Modern Perspective Review (as of June 2026)

### What Has Changed Since Publication

- **Monocular dynamic Gaussians are now the norm:** Methods like Deformable 3DGS (Yang et al., 2024), SC-GS (Huang et al., 2024), and Shape of Motion (Wang et al., ECCV 2024) all handle monocular video — the key limitation of D3DGS — by leveraging optical flow, depth priors, or motion basis decompositions.
- **4D Gaussian representations proliferated:** 4D-GS (Wu et al., 2024) and related methods parameterize motion explicitly as polynomial trajectories or neural deformation fields, achieving comparable or better PSNR with simpler priors.
- **Foundation tracking models surpassed the benchmark:** TAPIR (Doersch et al., 2024), CoTracker2, and SpatialTracker now achieve strong 2D tracking without scene-specific optimization; the 10× advantage over PIPs has been substantially narrowed.
- **Video generation and world models:** Sora-class video diffusion models and world models (UniSim, DreamerV3-style) now handle dynamic scene synthesis generatively — a fundamentally different paradigm that implicitly models scene dynamics without explicit Gaussian tracking.
- **Evaluation moved to in-the-wild:** TAP-Vid-DAVIS, PointOdyssey, and CVO became standard; PanopticSports (controlled dome studio) is now seen as too constrained and not reflective of deployment conditions.
- **Real-time 4D editing became practical:** Gaussian grouping, GaussianEditor, and related methods make the downstream editing applications demonstrated in Fig. 7 routine rather than novel.

### Has the Community Accepted the Claims?

D3DGS is widely regarded as a landmark paper that established the paradigm of tracking-by-rendering with explicit 3D primitives. Its central claim — that persistent view synthesis naturally induces dense, accurate tracking without optical flow — has been validated by a large body of follow-on work that adopted and extended the framework. The 100% survival rate and 2.21 cm 3D MTE results held up as strong baselines for over a year before being surpassed. However, the community quickly moved to address its acknowledged limitations: monocular methods proved feasible (contrary to the paper's implied difficulty), and newer trackers trained on synthetic data substantially closed the 2D tracking gap. The physical priors (local rigidity) remain influential but are now often replaced with or augmented by learned motion priors from video foundation models. The paper's downstream applications have been generalized and productized in tools like Gaussian-based 4D video editors.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl, Kopanas, Leimkuehler, Drettakis | 2023 | Direct base renderer and scene representation extended to dynamics |
| NeRF: Representing Scenes as Neural Radiance Fields | Mildenhall et al. | 2020 | Volume rendering formulation (Max equation) and analysis-by-synthesis philosophy |
| Particle Video Revisited (PIPs) | Harley, Fang, Fragkiadaki | 2022 | Primary 2D long-term tracking baseline that D3DGS outperforms 10× |
| DynamicFusion | Newcombe, Fox, Seitz | 2015 | Classic depth-based non-rigid reconstruction with TSDF; motivates the local rigidity prior |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| Tracking Everything Everywhere All at Once (OmniMotion) | Wang et al. | 2023 | Most similar in spirit; NeRF-based test-time tracking but requires dense optical flow input |
| Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction | Yang et al. | 2023 | Concurrent; adds deformation field to 3DGS but focuses on monocular video, not tracking |
| 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering | Wu et al. | 2023 | Concurrent; uses polynomial trajectory Gaussians; targets rendering quality over tracking |
| SC-GS: Sparse-Controlled Gaussian Splatting | Huang et al. | 2023 | Concurrent; uses sparse control points to drive dense Gaussian motion |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| Shape of Motion: 4D Reconstruction from a Single Video | Wang et al. | 2024 | Extends tracking-by-rendering to monocular video using motion basis decomposition |
| Gaussian Grouping: Segment and Edit Anything in 3D Scenes | Ye et al. | 2024 | Extends D3DGS-style editing by adding identity-aware grouping to Gaussians |
| SplatFlow: Learning Multi-Frame Optical Flow via Splatting | Fischer et al. | 2024 | Uses Gaussian splatting to compute flow, reversing the paper's direction |
| PhysGaussian: Physically Integrated 3D Gaussians for Generative Dynamics | Xie et al. | 2024 | Integrates continuum mechanics physics (MPM) directly into Gaussian dynamics |

---

### Bottom Line

Dynamic 3D Gaussians is a foundational paper that every researcher working on dynamic scene reconstruction or 4D content creation should read. Its conceptual contribution — that freezing appearance and letting position drift under physical priors is sufficient to produce accurate dense 6-DOF tracking as a side effect of rendering — is elegant, practically effective, and has seeded an entire subfield. The specific limitations (multi-camera, no new objects) are now mostly addressed in successor work, but the core idea remains highly influential. The paper's writing is clear, the experiments are thorough for their time, and the downstream applications (Fig. 7) still convey an inspiring vision. It is best read as a foundational classic of the 4D Gaussian era, not as a deployable system for real-world video.
