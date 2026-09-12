                                           N O RD: A Data-Efficient Vision-Language-Action Model that Drives without
                                                                           Reasoning

                                                                     Ishaan Rawal1,2 * Shubh Gupta1             Yihan Hu1            Wei Zhan1,3†
                                                                 1                            2                                  3
                                                                     Applied Intuition            Texas A&M University               UC Berkeley


                                                                 Abstract




arXiv:2602.21172v3 [cs.AI] 5 Jun 2026
                                                                                                                                     Teacher LLM
                                                                                                                                     (e.g. GPT4o)
                                        Vision-Language-Action (VLA) models are advancing au-
                                        tonomous driving by replacing modular pipelines with uni-                                         Reasoning
                                                                                                                                                                          RL
                                                                                                                    Large                 Dataset        SFT with
                                        fied end-to-end architectures. However, current VLAs face                   Driving                              Reasoning        Fine-
                                                                                                                    Dataset                              Annotations      tuning
                                        two expensive requirements: (1) massive dataset collec-
                                        tion, and (2) dense reasoning annotations. In this work,
                                        we address both challenges with N O RD (No Reasoning                 a) Existing VLA training pipeline with large datasets and
                                                                                                             reasoning annotations
                                        for Driving). Compared to existing VLAs, N O RD achieves
                                        competitive performance while being fine-tuned on <60%
                                                                                                                    Small                              SFT with          RL
                                        of the data and no reasoning annotations, resulting in 3×                   Driving                            Trajectory        Fine-
                                        fewer tokens. We identify that standard Group Relative                      Dataset                            Data              tuning

                                        Policy Optimization (GRPO) fails to yield significant im-
                                        provements when applied to policies trained on such small,           b) NoRD: Our efficient training pipeline with less data and
                                        reasoning-free datasets. We show that this limitation stems          no reasoning annotations
                                        from difficulty bias, which disproportionately penalizes re-         Figure 1. Comparison of VLA training pipelines. (a) Exist-
                                        ward signals from scenarios that produce high-variance               ing approaches depend on large-scale reasoning data generation,
                                        rollouts within GRPO. N O RD overcomes this by incorpo-              followed by extensive SFT and RL fine-tuning. (b) In contrast,
                                        rating Dr. GRPO, a recent algorithm designed to mitigate             N O RD directly utilizes a small-scale driving dataset for SFT, and
                                        difficulty bias in LLMs. As a result, N O RD achieves com-           performs RL fine-tuning tailored for weak SFT policy, enabling
                                        petitive performance on Waymo and NAVSIM with a frac-                data-efficient learning without reasoning supervision.
                                        tion of the training data and no reasoning overhead, en-
                                        abling more efficient autonomous systems.
                                                                                                             on both massive data and dense reasoning introduces three
                                                                                                             non-scalable costs:
                                        1. Introduction                                                      1. Data cost of collecting and curating vast quantities of
                                                                                                                specialized driving scenarios
                                        The prevailing paradigm for end-to-end autonomous driv-              2. Annotation cost from generating high-quality reasoning
                                        ing is increasingly shifting toward Vision-Language-Action              traces for this data
                                        (VLA) models. The dominant training methodology for                  3. Training and inference cost from resulting reasoning to-
                                        these models is a two-stage training pipeline: (1) Super-               kens, increasing training time and creating inference la-
                                        vised Fine-Tuning (SFT) on large-scale datasets with de-                tency that is impractical for real-world deployment
                                        tailed, natural language Chain-of-Thought (CoT) reasoning            This motivates a natural hypothesis: Can we achieve com-
                                        annotations [41, 45, 51], followed by (2) a Reinforcement            petitive performance on driving benchmarks while be-
                                        Learning (RL) stage to align outputs with driving metrics,           ing both reasoning-free and data-efficient?
                                        for which Group Relative Policy Optimization (GRPO) [15]                This investigation is supported by two distinct lines of
                                        has been widely adopted [36, 51].                                    work. First, recent studies provide a theoretical motiva-
                                            While this paradigm has achieved state-of-the-art per-           tion by questioning the necessity of explicit reasoning, sug-
                                        formance on complex driving benchmarks [36], its reliance            gesting it may be a byproduct of planning rather than a
                                          * Work done during an internship at Applied Intuition.             causal determinant [38]. Second, existing work on end-to-
                                          † Corresponding author. Email: wei.zhan@applied.co                 end models like EMMA [18] and SimLingo [35], provides


                                                                                                         1
an empirical precedent by achieving strong performance                   tribution that deprives GRPO of a learning signal.
on nominal benchmarks without reasoning. We therefore                 3. We propose using Dr. GRPO as a drop-in replacement to
investigate if this data-efficient, reasoning-free approach              train N O RD, a data-efficient, reasoning-free VLA, and
can be extended to the more challenging benchmarks (e.g.                 are the first to validate this policy optimization method
NAVSIM [11], WaymoE2E [30]) that are currently domi-                     in the autonomous driving domain (see Fig. 1).
nated by their reasoning-centric counterparts.                        4. We demonstrate performance competitive with the state-
    We initially trained a reasoning-free N O RD- BASE VLA               of-the-art on the NAVSIM and WaymoE2E benchmarks
(based on Qwen-2.5VL-3B-Instruct [2]) using only SFT                     without using any reasoning annotations and at least
on 80,000 NAVSIM training samples; a greater than 60%                    60% less data than reasoning VLAs, while improving
reduction in data compared to state-of-the-art reasoning-                on inference time, proving the viability of our approach.
based models [51]. This model was then post-trained with
GRPO to optimize the PDM score [11].
                                                                      2. Related Works
    However, this data-efficient, reasoning-free model
achieves scores significantly lower than reasoning-based              Reasoning-based VLAs. Several works have incorporated
baselines (>12-point difference), and post-training with              high-level reasoning into the control loop, including hy-
GRPO only results in a meager improvement (+0.67%).                   brid architectures like ORION [14], unified transformers
This initial failure creates the illusion that reasoning data         like AutoVLA [51], and a wide variety of reasoning strate-
is a necessary component for achieving high performance.              gies, such as retrieval-augmented CoT [9], spatio-temporal
    In this work, we argue that this conclusion is prema-             reasoning [47], multi-agent reasoning [34, 49], and models
ture. We posit that the failure lies not in the reasoning-            combining memory and tool use [23, 28]. While this ap-
free SFT policy, but rather in the interaction between the            proach has achieved state-of-the-art performance on com-
policy optimization method (GRPO) and the reward land-                plex driving benchmarks [36], its reliance on large-scale,
scape. We find that the complex, sparse reward signals                specialized reasoning datasets [6, 41, 45] and the high in-
from driving benchmarks (like PDM score from NAVSIM                   ference latency of CoT generation [26, 31] motivate the ex-
or the RFS from WaymoE2E) induce a highly polarized                   ploration of alternatives.
distribution of intra-group rewards. A significant mass of
                                                                      Reasoning-Planning Disconnect. The high cost of
these mean rewards is clustered at the extremes (i.e., near 0
                                                                      reasoning-centric models has spurred an inquiry into
or 1), and correspond to rollouts with low variance. Con-
                                                                      their necessity, with recent work questioning whether the
versely, the remaining scenarios, which yield intermediate
                                                                      model’s reasoning improves its planning output. One
mean rewards, are characterized by high-variance rollouts.
                                                                      study [38], proposing a “Reasoning-Planning Decoupling
When GRPO is applied in this landscape to a weaker, data-
                                                                      Hypothesis” demonstrated that textual priors alone can
efficient SFT policy like N O RD- BASE, the resulting learn-
                                                                      match the performance of full multimodal reasoning mod-
ing signal disproportionately penalizes the intermediate-
                                                                      els. This skepticism extends to RL post-training, as other
mean (high-variance) scenarios, impeding effective opti-
                                                                      works argue that RL does not instill new reasoning capacity
mization.
                                                                      but instead optimizes within the SFT model’s existing latent
    We are the first to identify that the failure to optimize         distribution [46]. These findings motivate our reasoning-
weak SFT mode is caused by polarized intra-group reward               free approach and frame our central question: does the fail-
landscape, and that it stems from difficulty bias, which has          ure to align weak SFT models stem from an inherent limi-
also been observed in LLM reasoning domain [22, 29].                  tation of these models, or from an optimization failure?
Based on our analysis, we propose to mitigate this bias
by using Dr. GRPO [29], an existing policy optimization               Reasoning-Free VLAs. A separate line of VLA models
algorithm specifically designed to address this flaw. We              operate without explicit reasoning traces. This includes
demonstrate that by applying Dr. GRPO as a drop-in re-                models that map raw sensor data directly to trajectories like
placement, our reasoning free VLA, N O RD, can be suc-                EMMA [18], SimLingo [35], and S4-Driver [43], as well as
cessfully trained.                                                    generative approaches like ADriver-I [19], DrivingGPT [5],
    Our key contributions are as follows:                             and DiffVLA [20]. While these methods have demon-
1. We are the first to identify that the failure of reasoning-        strated strong performance on nominal driving benchmarks
    free and data-efficient VLA training for autonomous               like nuScenes [3], they have not yet proven competitive on
    driving is an instance of difficulty bias, triggered by the       the complex, long-tail benchmarks where reasoning-centric
    combination of a weak SFT policy and complex driving              models currently excel.
    metrics.                                                          Data Efficient VLAs. Our work aims to close the per-
2. We empirically characterize this failure, showing that the         formance gap between reasoning-free and reasoning-based
    data-efficient SFT policy induces a polarized reward dis-         methods while maintaining data efficiency. Many data-


                                                                  2
Table 1. Comparison of RL fine-tuning (RLFT) on N O RD- BASE
with GRPO and Dr. GRPO on NAVSIM test set. While GRPO fails
to improve N O RD- BASE, we get significant gains with Dr. GRPO.

      Model                           PDMS ↑
      N O RD- BASE                    76.66
      N O RD- BASE + GRPO             77.18 (+0.67%)
      N O RD- BASE + Dr. GRPO         85.62 (+11.68%)


efficient VLAs in broader domains [12, 13, 42, 44] mitigate
data scarcity by leveraging massive external out-of-domain
datasets. We instead focus on the distinct and more chal-
lenging problem of training a competitive model using only
small-sized specialized, in-domain driving data.
Mitigating Difficulty Bias. The literature to mitigate the
difficulty bias in GRPO, largely from the LLM reasoning
domain, is divided into two main strategies. Data-level in-
terventions attempt to manage data before the optimization,
using methods like online filtering of saturated or degen-
erate samples [10, 27], curriculum learning [32], or ad-
vanced sampling [1]. These approaches are designed for                 Figure 2. Reward distribution in the weak SFT model. The
binary rewards and are generally computationally infeasi-              group-mean PDM score is shown with band representing the mean
ble for expensive driving simulations, as they often require           of the corresponding group standard deviation for N O RD- BASE.
multiple rollouts to estimate sample difficulty. In contrast,          GRPO struggles to optimize high-variance regions (the majority)
                                                                       and is effective only in low-variance regions (the trajectories in
algorithmic-level interventions modify the optimization al-
                                                                       green and red are for ground truth and N O RD- BASE prediction).
gorithm itself. This includes reweighting schemes [48, 50],
alternative objectives [8, 22], or difficulty-based priors [4].
Our work incorporates this second strategy. We select
Dr. GRPO [29], a lightweight method that directly corrects             tral question: can VLAs fine-tuned on small-scale driv-
the bias by identifying and adjusting the specific normal-             ing data without reasoning supervision achieve competitive
ization term in the advantage estimation responsible for it.           performance, or is RL post-training inherently limited in
Dr. GRPO is thus a prime candidate for our setting, as it              optimizing weaker, data-efficient VLAs?
avoids the infeasible overhead of data-level methods.                      To investigate this, we train N O RD- BASE, a VLA built
                                                                       on Qwen-2.5VL-3B-Instruct [2], using supervised fine-
3. Limitations of GRPO for Data-Efficient                              tuning on only 80,000 NAVSIM training samples and with-
                                                                       out reasoning annotations. N O RD- BASE predicts physical
   Training                                                            trajectory tokens from current images and historical vehicle
VLAs for autonomous driving have achieved competitive                  states, followed by GRPO optimization on the PDM score
performance through a two-stage training pipeline - first              (details in Sec. 4.2). The PDM score evaluates predicted tra-
SFT followed by RL post-training. This paradigm relies                 jectories in simulation across metrics such as safety, com-
on large-scale domain-specific datasets that are additionally          fort, collision avoidance, and adherence to driving areas,
annotated with reasoning data. During RL post-training,                with higher scores indicating better performance.
GRPO optimizes the SFT model (i.e., the policy) for high-                  As shown in Tab. 1, GRPO post-training leads to only
level objectives such as preference alignment or safety by             a 0.67% improvement, resulting in negligible overall gains.
maximizing the group-relative advantage.                               This outcome is inconsistent with our goal of shifting the
   However, this existing SFT-heavy approach is costly and             primary learning burden from SFT to RL post-training. This
inefficient. First, collecting and labeling thousands or even          minimal improvement is in stark contrast to prior works
millions of driving scenarios is resource-intensive. Second,           like AutoVLA [51], which have demonstrated a 9% per-
generating reasoning traces from a teacher model increases             formance boost by post-training an SFT model, but one that
token load, training time, and compute requirements. Fi-               was trained on 212,000+ samples and with reasoning data.
nally, reasoning tokens during inference add latency, limit-           This discrepancy between GRPO’s effectiveness on strong
ing real-time deployment. These challenges raise our cen-              versus weak SFT policies motivates a deeper investigation


                                                                   3
                      (a) GRPO                                                                         (b) Dr. GRPO

Figure 3. Evolution of group-mean PDM score during RL fine-tuning. (a) GRPO struggles to optimize samples with high group
variance during training, particularly in the range [0.2–0.65]. (b) Dr. GRPO effectively optimizes high-variance samples during training,
resulting in significant overall performance gains.


                                                                          model performs reliably on simple behaviors, such as
                                                                          maintaining a straight trajectory at constant speed, re-
                                                                          sulting in high mean reward and low intra-group vari-
                                                                          ance. Conversely, in extremely difficult scenarios, e.g.,
                                                                          out-of-distribution driving, the model predicts trajecto-
                                                                          ries with trivially low PDM scores, yielding low mean
                                                                          and low variance within the rollout group. Notably, the
                                                                          proportion of such samples is very small, suggesting that
                                                                          N O RD is sufficiently expressive.
                                                                       2. High variance occurs in samples with intermediate
                                                                          group-mean values in the range [0.2, 0.65]. The PDM
                                                                          score penalizes collisions, off-road behavior, and other
                                                                          safety violations. Given the weakness of the SFT model,
                                                                          complex maneuvers such as sharp turns fail more often
                                                                          than they succeed, producing low mean rewards and high
                                                                          variance within the rollout group.
                                                                       With these observations, we analyze the evolution of group-
                                                                       mean reward distributions across GRPO training steps (see
Figure 4. Qualitative comparison of RL fine-tuning (RLFT)              Fig. 3a). We find that the density in the high-variance region
on the weak SFT model using GRPO and Dr. GRPO. With                    ([0.2, 0.65]) remains largely unchanged throughout training,
Dr. GRPO, N O RD successfully learns complex maneuvers such            whereas the density in the lowest-variance region (close to
as sharp turns and lane changes without collisions, whereas GRPO
                                                                       1) steadily increases. This pattern explains the marginal im-
fails to optimize the weak SFT model (N O RD- BASE) and collides
(in red).
                                                                       provement in the final PDM score after GRPO post-training:
                                                                       GRPO primarily optimizes the small subset of samples with
                                                                       low intra-group reward variance while failing to improve
                                                                       the majority of samples with high intra-group variance.
into the underlying cause.
                                                                           Our findings suggest that GRPO is fundamentally inef-
   To understand this discrepancy, we analyze the reward
                                                                       fective at learning from samples with high intra-group vari-
characteristics of the training set for N O RD- BASE, shown
                                                                       ance, which dominate the training dataset for our weak SFT
in Fig. 2. For each training example, we perform 8 rollouts
                                                                       model, N O RD- BASE, and therefore provides limited bene-
and plot the distribution of group-mean PDM scores, with
                                                                       fit for RL post-training. We interpret this failure as a form
bands indicating the corresponding group standard devia-
                                                                       of difficulty bias in GRPO. Originally, difficulty bias was
tion, averaged across groups. Our key observations are:
                                                                       proposed for binary reward settings, measured as the mean
1. Low variance occurs in samples with high or very                    group reward [22], and used to post-train LLMs for mathe-
   low group mean (≥ 0.8 or ≤ 0.15). The weak SFT                      matical reasoning [29]. Building on this, our analysis shows


                                                                   4
                                                                          multivariate normal distribution parameterized by the mean
                                                                          and covariance of the existing token embeddings [16]. The
                                                                          model is trained in two stages: (1) Supervised Fine-Tuning
                                                                          with limited data, followed by (2) RL Post-Training using
                                                                          Dr. GRPO for effective policy optimization starting from a
                                                                          weak SFT model, as illustrated in Fig. 1.

                                                                          4.1. Supervised Fine-Tuning with Limited Data
                                                                          N O RD- BASE is intentionally trained on a limited dataset
                                                                          during supervised fine-tuning to offload the majority of
                                                                          learning to the subsequent RL post-training phase. We
                                                                          model trajectory prediction as a next-token prediction prob-
                                                                          lem, where the model outputs trajectory tokens conditioned
Figure 5. Model architecture of N O RD. N O RD directly predicts
                                                                          on the inputs. As expected, the reduced training data results
action tokens without requiring reasoning traces, enabling a sig-
nificantly more efficient training and inference pipeline.
                                                                          in lower initial performance for N O RD- BASE (see Tab. 1).
                                                                          In the following section, we describe how to effectively op-
                                                                          timize this weak SFT policy using Dr. GRPO by explicitly
that the limitation of optimizing weak SFT policies with                  accounting for intra-group reward variance.
GRPO for data-efficient VLA training stems from this in-
                                                                          4.2. RL Post-Training for Weak SFT Policy
herent difficulty bias. To address this, we post-train N O RD-
BASE using Dr. GRPO, a GRPO variant originally designed                   As discussed in Sec. 3, weak SFT policies cannot be effec-
to mitigate difficulty bias in LLM reasoning. Dr. GRPO en-                tively optimized using standard GRPO, which can be un-
ables training with significantly less data and without any               derstood as an instance of the difficulty bias problem. To
reasoning annotations, as explained in the next section.                  address this, we employ Dr. GRPO, a recently proposed
                                                                          RL fine-tuning algorithm, for post-training our weak SFT
4. N O RD: No Reasoning for Driving                                       model, N O RD- BASE. In the original GRPO formulation,
                                                                          the group relative advantage is computed as
N O RD (No Reasoning for Driving) is our Vision-                                                               PG
                                                                                                           1
Language-Action (VLA) model for autonomous driving,                                            r(oi | x) − G    j=1 r(oj | x)
built upon Qwen-2.5VL-3B-Instruct. N O RD achieves both                            ÂGRPO
                                                                                     i,t  :=                                    .
                                                                                                  stdj=1,...,G (r(oj | x))
token and data efficiency by omitting reasoning annotations
entirely from the training and inference stages, by empha-                Here, r(oi | x) is the reward for sample i given input x,
sizing learning during the RL post-training phase rather                  G is the group size, and std denotes the standard devia-
than during supervised fine-tuning. However, as discussed                 tion across the group. Recent studies have shown that this
in Sec. 3, naively reducing the amount of SFT data signifi-               formulation unintentionally favors groups with low reward
cantly degrades performance because GRPO is ineffective                   variance [29]. When the standard deviation of the reward
in learning from samples with high intra-group variance.                  within the group is small (i.e., << 1), the group-relative
This section presents the design of N O RD and approach                   advantage is disproportionately large, whereas it is heavily
for training it effectively with limited data.                            attenuated for groups with high reward variance. This poses
    The inputs to N O RD are the past ego-trajectory, current             a major problem for us since N O RD- BASE, being a weak
speed, acceleration, and RGB images from the front, front-                SFT model, produces groups with high intra-group vari-
left, and front-right cameras, as shown in Fig. 5. The model              ance during the GRPO rollout for the majority of samples
predicts the future ego-trajectory at 10 Hz. To improve                   ( Fig. 2). Dr. GRPO mitigates difficulty bias by removing
token efficiency, we represent trajectories using k-disc to-              the standard deviation term from the group relative advan-
kenization [33] with a vocabulary size of 2048. Specifi-                  tage, enabling more effective optimization of weak policies.
cally, all future trajectories in the training set are first inter-       Notably, while other variants like VD-GRPO [39] preserve
polated to 10 Hz and segmented into 0.5 second intervals.                 absolute reward magnitudes to balance objective priorities
These segments are then clustered into 2048 clusters based                (e.g., safety vs. comfort), Dr. GRPO ensures that ’hard’ sce-
on the contour distance between trajectory segments. The                  narios contribute a sufficient gradient signal. Additionally,
resulting cluster centers form a discrete trajectory codebook             we employ DAPO-style asymmetric clipping to prevent en-
that can reconstruct any trajectory using vocabulary tokens.              tropy collapse during RL training and follow Liu et al. [29]
These trajectory tokens are appended to the original vocab-               by not using KL-divergence regularization. The resulting
ulary of the base Qwen model initialized by sample from a                 Dr. GRPO post-training objective is given by:


                                                                      5
                                                                              Table 2. Test results on the Waymo Vision-based End-to-End
                                                                              Driving Benchmark. N O RD achieves competitive performance,
                               G                                              without reasoning or ensembling.
                           1 X
ÂDrGRPO
  i,t    = r(oi | x) −           r(oi | x),                     (1)
                           G j=1                                                                 w/o         w/o
                                                                              Model                                     RFS↑     ADE@3↓
            |oi |                                                                               Reason     Ensemble
            X              πθ (oi,t |q, oi,<t ) DrGRPO
LDrGRPO =           min                          Â    ,                      Poutine [36]         ✗           ✓        7.986      1.2055
                          πθold (oi,t |q, oi,<t ) i,t
            t=1
                                                                      !       HMVLM [40]           ✗           ✓        7.736      1.3269
                 π (o |q, o )
                   θ i,t         i,<t
                                                                             DiffusionLTF         ✓           ✗        7.717      1.3561
           clip                          , 1 − ϵl , 1 + ϵh ÂDrGRPO
                                                             i,t              UniPlan              ✓           ✗        7.692      1.3083
                 πθold (oi,t |q, oi,<t )
                                                                              AutoVLA [51]         ✗           ✓        7.556      1.3507
                                                                (2)
                                                                              N O RD               ✓           ✓        7.709      1.2504
    This formulation enables N O RD, i.e., N O RD- BASE
trained with Dr. GRPO, to achieve improved performance
                                                                              5.2. Implementation Details
during RL post-training by mitigating difficulty bias and
stabilizing policy optimization for data- and token-efficient                 We use Qwen-2.5VL-3B-Instruct as our base model, as it
VLAs. We find that with Dr. GRPO finetuning, N O RD-                          offers a good trade-off between model capacity and compu-
BASE learns from mid-variance samples, leading to an over-                    tational efficiency. The model is fine-tuned on the NAVSIM
all improvement of 11.68% from the base model (as com-                        and WaymoE2E datasets separately using 16 A100 GPUs
pared to 0.67% with GRPO). We notice that Dr. GRPO is                         with a batch size of 128. We employ the AdamW optimizer
able to optimize even on the samples with high intra-group                    with a learning rate of 5 × 10−5 and a cosine decay sched-
variance, as shown in Fig. 3b. This enables N O RD even                       ule, fine-tuning all layers of Qwen-2.5VL-3B. This stage
learn complex maneuvers, as compared to the GRPO coun-                        yields the N O RD- BASE model. Subsequently, we apply
terpart (see Fig. 4).                                                         Dr. GRPO for RL post-training. For NAVSIM, we use 30
                                                                              A100 GPUs to optimize the base model for 160 steps with
                                                                              a constant learning rate of 5 × 10−6 . For WaymoE2E, we
5. Experiments                                                                post-train the model for 150 steps on 32 GPUs with a learn-
                                                                              ing rate of 1 × 10−6 .
5.1. Datasets
                                                                                  The RL post-training pipeline is implemented in
NAVSIM [11]: NAVSIM is a curated redistribution of the                        verl [37] with Fully Sharded Data Parallel (FSDP) for
OpenScenes dataset, comprising real-world urban driving                       memory-efficient training and vLLM [21] for rollout gen-
scenarios. The dataset comprises 120 hours of driving data                    eration. During rollouts, we set the group size to 8 and fix
from OpenScene. The dataset features diverse and challeng-                    the sampling temperature to 1.0. For validation, we em-
ing traffic situations, providing synchronized 360◦ camera                    ploy deterministic sampling with a temperature of 0.01. For
imagery, LiDAR scans, HD map data, and bounding-box                           NAVSIM, we use the PDM score as the primary reward
annotations of dynamic agents, along with historical con-                     function, while for WaymoE2E, we employ the normalized
trol signals. The task is to predict the ego-vehicle trajectory               RFS score. In both cases, we include additional rewards for
for the next 4 seconds at 2 Hz, with performance evaluated                    trajectory length and output format, each weighted by 0.25,
by executing the predicted trajectory within a simulation                     and then normalize to [0, 1].
environment, scored using PDM Score in terms of driving
safety, progress and comfort metrics.                                         5.3. Results
    Waymo Vision-Based End-to-End Dataset (Way-                               WaymoE2E performance: The WaymoE2E dataset
moE2E) [30]: WaymoE2E is a challenging long-tail dataset                      poses a challenging evaluation setting that emphasizes
providing 360◦ camera views and ego-vehicle trajectories.                     robustness under out-of-distribution driving scenarios.
Validation and test scenarios include three alternative tra-                  Consequently, most existing approaches rely on large-scale
jectories labeled with human per scene, representing vary-                    training datasets and explicit reasoning annotations to
ing driving preferences and scored in the range [4, 10].                      achieve competitive performance (Tab. 2). In contrast,
Predicted trajectories are evaluated using the Rated Feed-                    N O RD attains a RFS of 7.709, ranking as the third best-
back Score (RFS), which measures weighted similarity with                     performing VLA on the benchmark, while being the only
the reference preference trajectories. WaymoE2E contains                      top model trained without reasoning traces or ensembling.
4,021 challenging driving segments, partitioned into 2,037                    Remarkably, N O RD achieves this with merely 12,000
training, 479 validation, and 1,505 test segments.                            samples for supervised training and 450 samples for RLFT,


                                                                          6
 Table 3. Test results on NAVSIM benchmark (navtest subset). N O RD achieves competitive performance (w/o R: Without reasoning data,
 w/o L: without LiDAR data, and C: Number of RGB frames; * BoN refers to the average over best score per sample out of 6 outputs with
 different random seeds).

                                              w/o       w/o
 Method                                                                 C           PDMS↑          Collision↑   DAC↑                                 Direction↑          Progress↑              TTC↑              Comfort↑
                                               R         L
 BEV-based Methods
 UniAD [17]                                    ✓          ✓            32               83.4         97.7       91.9                                        -                 78.8               92.9                  100
 Transfuser [7]                                ✓          ✗             3               84.0         97.7       92.8                                       97.9               79.2               92.8                  100
 Hydra-MDP [24]                                ✓          ✗             3               86.5         98.2       96.2                                       95.8               78.7               94.6                  100
 DiffusionDrive [25]                           ✓          ✗             3               88.1         98.2       96.2                                        -                 82.2               94.7                  88.1
 VLA-based Methods
 AutoVLA [51]                                   ✗         ✓            12               89.1         98.4       95.6                                       95.4               81.9               98.0                  99.9
 AutoVLA-BoN* [51]                              ✗         ✓            12               92.1         99.1       97.1                                       95.5               87.6               97.1                  100
 RecogDrive [23]                                ✗         ✓            12               89.6         98.2       97.9                                        -                 83.5               95.2                  99.8
  N O RD                                       ✓          ✓               3             85.6         97.6       94.9                                       95.9               79.3               93.5                  100
  N O RD-BoN*                                  ✓          ✓               3             92.4         99.2       98.3                                       95.9               86.4               97.8                  99.9


            93                                                                                                                                 8.0                                                               Poutine

            91                                                                                                                                 7.9




                                                                                                                  Rated Feedback Score (RFS)
                                                               r              RecogDrive
                                                     y Frontie
            89                              Efficienc                                                                                          7.8                                            HMVLM
                                                            AutoVLA                                                                                   NoRD              Efficiency Frontier                      DiffusionLTF
                         Hydra-MDP                                                                                                                           UniPlan
            87                          DiffusionDrive                                                                                         7.7
                  NoRD
                                 High Efficiency High Performance
            85                                                                                                                                 7.6                AutoVLA




PDM Score
                  Transfuser     Low Efficiency Low Performance
                      UniAD                                                                                                                    7.5
            83
                                                       Our Method (Vision-only)
            81                                         VLA Baseline (Vision-only)                                                              7.4                                                Our Method
                                                                                                                                                                                                  VLA Baseline
                                                       Non-VLA Baseline
            79                                         Non-VLA Baseline (Vision-only)
                                                                                                                                               7.3                                                Non-VLA Baseline


            77                                                                                                                                 7.2

            75                                                                                                                                 7.1


                 80k      100k      120k     140k     220k       240k          3100k       3120k                                                     20k     130k      240k    350k      460k     570k         680k        790k

                         Dataset Size (Number of Samples)                                                                                                    Dataset Size (Number of Samples)
                                        (a) NAVSIM                                                                                                                      (b) WaymoE2E

 Figure 6. Pareto-optimal curves on two driving benchmarks. (a) N O RD is the only VLA in NAVSIM operating in the high-performance,
 high–data-efficiency region using only RGB inputs. (b) N O RD achieves competitive RFS on WaymoE2E with a fraction of the training
 data, without ensembling or reasoning supervision. Shaded regions provide a qualitative categorization of model efficiency and perfor-
 mance for ease of visualization.


 whereas Poutine and HMVLM require 17× and 12× larger                                                       solely on 3 camera frames, and uses no additional features
 datasets for only marginal RFS gains. Furthermore, N O RD                                                  like, LiDAR and HD Map. While other VLA models, such
 surpasses all other competitive models on the ADE metric,                                                  as AutoVLA and RecogDrive, require 1.6× and 34× more
 despite a ≥ 6× reduction in training data, underscoring its                                                training data, N O RD achieves competitive performance
 strong generalization ability, as shown in Fig. 8.                                                         with fewer than 90,000 samples. We also evaluate the
                                                                                                            best-of-N performance, where the oracle selects the best
 NAVSIM performance:           The NAVSIM benchmark                                                         trajectory out of 6 predictions based on the PDM score. In
 rigorously evaluates trajectory prediction by executing                                                    this configuration, N O RD-BoN surpasses reasoning-based
 models in a simulator and scoring them using the PDM                                                       AutoVLA-BoN, achieving a PDM score of 92.4, highlight-
 metric, a weighted measure of high-level driving factors                                                   ing its capabilities and data-efficiency.
 such as comfort, time-to-collision, and ego-progress. As
 shown in Tab. 3 (with qualitative results in Fig. 7), N O RD                                               Efficiency and Scalability: A central contribution of
 is the only model that requires no reasoning traces, relies                                                our work is the remarkable data efficiency of N O RD,


                                                                                                        7
Figure 7. Qualitative Results on NAVSIM (navtest subset). N O RD safely executes sharp turns, respects traffic lights, and avoids
collisions, demonstrating robust driving behavior. The predicted trajectory is shown in red.




                                                                            Figure 9. Comparison of token and runtime efficiency. N O RD
                                                                            is the most (a) token and (b) runtime efficient VLA.



                                                                            VLAs as shown in Fig. 9. Our findings strongly suggest
                                                                            that high-performance autonomous driving VLAs do not
                                                                            necessarily require large datasets, paving the way for more
                                                                            accessible and scalable yet efficient models.
Figure 8. Qualitative Results on WaymoE2E test set. N O RD
drives safely in challenging out-of-distribution scenarios like un-
safe pedestrian crossing and construction site. The predicted tra-          6. Conclusion
jectory is shown in red (stitched, center-cropped for visualization).
                                                                            We proposed N O RD, a reasoning-free, data-efficient VLA
                                                                            for autonomous driving. N O RD achieves strong perfor-
as highlighted in the Pareto-front analyses (Fig. 6a and                    mance while eliminating language reasoning and signifi-
Fig. 6b). On both benchmarks, N O RD establishes a                          cantly reducing training data requirements. By analyzing
competitive performance baseline while operating in                         rewards and modifying training pipelines, we demonstrate
the high-efficiency (i.e., low data) regime. While some                     that VLAs can be trained with substantially fewer sam-
VLA-based methods eventually achieve marginally higher                      ples while improving token efficiency and inference speed.
absolute scores, they do so at a prohibitive data cost of at                While Dr. GRPO mitigates difficulty bias better than GRPO,
least 3× more data. N O RD, in contrast, firmly establishes                 it remains imperfect [22], leaving room for future work. Im-
itself on the efficiency frontier, presenting an optimal and                portantly, N O RD does not suggest that VLAs cannot bene-
practical trade-off. While maintaining data efficiency, since               fit from language-based reasoning; rather, it shows that ef-
it directly predicts the trajectory tokens, it is extremely                 ficient, high-performing VLAs can be trained without rea-
lightweight and this enables it to achieve significantly                    soning and large-scale datasets, pushing the boundaries of
lower inference time and token count, as compared to other                  data and inference efficiency.


                                                                        8
References                                                           [11] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xin-
                                                                          shuo Weng, Zhiyu Huang, Zetong Yang, Hongyang
 [1] Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun                      Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone,
     Zhang, Shansan Gong, Ming Zhong, Jingjing Xu,                        Andreas Geiger, and Kashyap Chitta. Navsim: Data-
     Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong.
                                                                          driven non-reactive autonomous vehicle simulation
     Polaris: A post-training recipe for scaling reinforce-
                                                                          and benchmarking. In Advances in Neural Informa-
     ment learning on advanced reasoning models, 2025.
                                                                          tion Processing Systems (NeurIPS), 2024. 2, 6
     3
 [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang,                [12] Shengliang Deng, Mi Yan, Songlin Wei, Haixin
     Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie                    Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu
     Wang, Jun Tang, et al. Qwen2. 5-vl technical report.                 Yang, Xuheng Zhang, Wenhao Zhang, et al.
     arXiv preprint arXiv:2502.13923, 2025. 2, 3                          Graspvla: a grasping foundation model pre-trained
 [3] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh                   on billion-scale synthetic action data. arXiv preprint
     Vora, Venice Erin Liong, Qiang Xu, Anush Krish-                      arXiv:2505.03233, 2025. 3
     nan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom.               [13] Shichao Fan, Quantao Yang, Yajie Liu, Kun Wu,
     nuscenes: A multimodal dataset for autonomous driv-                  Zhengping Che, Qingjie Liu, and Min Wan. Diffusion
     ing. In Proceedings of the IEEE/CVF conference                       trajectory-guided policy for long-horizon robot ma-
     on computer vision and pattern recognition, pages                    nipulation. arXiv preprint arXiv:2502.10040, 2025.
     11621–11631, 2020. 2                                                 3
 [4] Mingrui Chen, Haogeng Liu, Hao Liang, Huaibo                    [14] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jian-
     Huang, Wentao Zhang, and Ran He. Unlocking the                       feng Cui, Dingkang Liang, Chong Zhang, Dingyuan
     potential of difficulty prior in rl-based multimodal rea-            Zhang, Hongwei Xie, Bing Wang, and Xiang Bai.
     soning, 2025. 3                                                      Orion: A holistic end-to-end autonomous driving
 [5] Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang.                         framework by vision-language instructed action gen-
     Drivinggpt: Unifying driving world modeling and                      eration. In Proceedings of the IEEE/CVF Interna-
     planning with multi-modal autoregressive transform-                  tional Conference on Computer Vision, 2025. 2
     ers.    In Proceedings of the IEEE/CVF Interna-                 [15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
     tional Conference on Computer Vision (ICCV), pages                   Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu
     26890–26900, 2025. 2                                                 Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incen-
 [6] Haohan Chi, Huan-ang Gao, Ziming Liu, Jianing Liu,                   tivizes reasoning in llms through reinforcement learn-
     Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu,                    ing. Nature, 645(8081):633–638, 2025. 1
     Zeda Wang, Wenyi Li, et al. Impromptu vla: Open                 [16] John Hewitt. Initializing new word embeddings for
     weights and open data for driving vision-language-                   pretrained language models, 2021. 5
     action models. arXiv preprint arXiv:2505.23757,
                                                                     [17] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chong-
     2025. 2
                                                                          hao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tian-
 [7] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Ze-
                                                                          wei Lin, Wenhai Wang, et al. Planning-oriented au-
     hao Yu, Katrin Renz, and Andreas Geiger. Transfuser:
                                                                          tonomous driving. In Proceedings of the IEEE/CVF
     Imitation with transformer-based sensor fusion for au-
                                                                          conference on computer vision and pattern recogni-
     tonomous driving. IEEE transactions on pattern anal-
                                                                          tion, pages 17853–17862, 2023. 7
     ysis and machine intelligence, 45(11):12878–12895,
     2022. 7                                                         [18] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-
 [8] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei                       Chih Hung, Jingwei Ji, Kristy Choi, Di Huang,
     Wei, and Yong Wang. Gpg: A simple and strong                         Tong He, Paul Covington, Benjamin Sapp, Yin
     reinforcement learning baseline for model reasoning.                 Zhou, James Guo, Dragomir Anguelov, and Mingxing
     arXiv preprint arXiv:2504.02546, 2025. 3                             Tan. EMMA: End-to-end multimodal model for au-
 [9] Charles Corbière, Simon Roburin, Syrielle Montariol,                tonomous driving. Transactions on Machine Learning
     Antoine Bosselut, and Alexandre Alahi. Retrieval-                    Research, 2025. 1, 2
     based interleaved visual chain-of-thought in real-              [19] Fan Jia, Weixin Mao, Yingfei Liu, Yucheng Zhao,
     world driving scenarios, 2025. 2                                     Yuqing Wen, Chi Zhang, Xiangyu Zhang, and Tian-
[10] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang,                      cai Wang. Adriver-i: A general world model for au-
     Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu,                       tonomous driving, 2023. 2
     Qixin Xu, Weize Chen, et al. Process reinforce-                 [20] Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun
     ment through implicit rewards.            arXiv preprint             Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao
     arXiv:2502.01456, 2025. 3                                            Jiang, Yunda Dong, et al. Diffvla: Vision-language


                                                                 9
     guided diffusion planning for autonomous driving.                  Understanding r1-zero-like training: A critical per-
     arXiv preprint arXiv:2505.19381, 2025. 2                           spective. In COLM, 2025. 2, 3, 4, 5
[21] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying                  [30] Waymo LLC. Vision-based end-to-end driving - 2025:
     Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gon-                  Waymo open dataset. https://waymo.com/
     zalez, Hao Zhang, and Ion Stoica. Efficient memory                 open / challenges / 2025 / e2e - driving/,
     management for large language model serving with                   2025. 2, 6
     pagedattention. In Proceedings of the ACM SIGOPS              [31] Yuechen Luo, Fang Li, Shaoqing Xu, Zhiyi Lai, Lei
     29th Symposium on Operating Systems Principles,                    Yang, Qimao Chen, Ziang Luo, Zixun Xie, Shengyin
     2023. 6                                                            Jiang, Jiaxin Liu, et al. Adathinkdrive: Adaptive
[22] Gang Li, Ming Lin, Tomer Galanti, Zhengzhong Tu,                   thinking via reinforcement learning for autonomous
     and Tianbao Yang. DisCO: Reinforcing large reason-                 driving. arXiv preprint arXiv:2509.13769, 2025. 2
     ing models with discriminative constrained optimiza-          [32] Shubham Parashar, Shurui Gui, Xiner Li, Hongyi
     tion. In The Thirty-ninth Annual Conference on Neu-                Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang,
     ral Information Processing Systems, 2025. 2, 3, 4, 8               James Caverlee, Dileep Kalathil, et al. Curriculum re-
[23] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li,                   inforcement learning from easy to hard tasks improves
     Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen,                       llm reasoning. arXiv preprint arXiv:2506.06632,
     Haiyang Sun, Bing Wang, et al. Recogdrive: A                       2025. 3
     reinforced cognitive framework for end-to-end au-             [33] Jonah Philion, Xue Bin Peng, and Sanja Fidler. Tra-
     tonomous driving. arXiv preprint arXiv:2506.08052,                 jeglish: Traffic modeling as next-token prediction.
     2025. 2, 7                                                         In The Twelfth International Conference on Learning
[24] Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhid-               Representations. 5
     ing Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz,            [34] Kangan Qian, Sicong Jiang, Yang Zhong, Ziang Luo,
     Zuxuan Wu, et al. Hydra-mdp: End-to-end mul-                       Zilin Huang, Tianze Zhu, Kun Jiang, Mengmeng
     timodal planning with multi-target hydra-distillation.             Yang, Zheng Fu, Jinyu Miao, et al. Agentthink: A uni-
     arXiv preprint arXiv:2406.06978, 2024. 7                           fied framework for tool-augmented chain-of-thought
[25] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang,                  reasoning in vision-language models for autonomous
     Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu                       driving. arXiv preprint arXiv:2505.15298, 2025. 2
     Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive:            [35] Katrin Renz, Long Chen, Elahe Arani, and Oleg
     Truncated diffusion model for end-to-end autonomous                Sinavski. Simlingo: Vision-only closed-loop au-
     driving. In Proceedings of the Computer Vision and                 tonomous driving with language-action alignment. In
     Pattern Recognition Conference, pages 12037–12047,                 Proceedings of the IEEE/CVF Conference on Com-
     2025. 7                                                            puter Vision and Pattern Recognition (CVPR), pages
[26] Haicheng Liao, Hanlin Kong, Bonan Wang, Chengyue                   11993–12003, 2025. 1, 2
     Wang, Wang Ye, Zhengbing He, Chengzhong Xu, and               [36] Luke Rowe, Rodrigue de Schaetzen, Roger Girgis,
     Zhenning Li. Cot-drive: Efficient motion forecast-                 Christopher Pal, and Liam Paull. Poutine: Vision-
     ing for autonomous driving with llms and chain-of-                 language-trajectory pre-training and reinforcement
     thought prompting. IEEE Transactions on Artificial                 learning post-training enable robust end-to-end au-
     Intelligence, pages 1–15, 2025. 2                                  tonomous driving. arXiv preprint arXiv:2506.11234,
[27] Jiacai Liu, Chaojie Wang, Chris Yuhao Liu, Liang                   2025. 1, 2, 6
     Zeng, Rui Yan, Yiwen Sun, and Yang Liu. DAPO :                [37] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
     Improving multi-step reasoning abilities of large lan-             Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
     guage models with direct advantage-based policy op-                Lin, and Chuan Wu. Hybridflow: A flexible and
     timization. In The Thirty-ninth Annual Conference on               efficient rlhf framework.     arXiv preprint arXiv:
     Neural Information Processing Systems, 2025. 3                     2409.19256, 2024. 6
[28] Xueyi Liu, Zuodong Zhong, Qichao Zhang, Yuxin                 [38] Xurui Song, Shuo Huai, JingJing Jiang, Jiayi Kong,
     Guo, Yupeng Zheng, Junli Wang, Dongbin Zhao, Yun-                  and Jun Luo. More than meets the eye? un-
     Fu Liu, Zhiguo Su, Yinfeng Gao, Qiao Lin, and Chen                 covering the reasoning-planning disconnect in train-
     Huiyong. Reasonplan: Unified scene prediction and                  ing vision-language driving models. arXiv preprint
     decision reasoning for closed-loop autonomous driv-                arXiv:2510.04532, 2025. 1, 2
     ing. In 9th Annual Conference on Robot Learning,              [39] Xiaolong Tang, Meina Kan, Shiguang Shan, and Xilin
     2025. 2                                                            Chen. Plan-R1: Safe and feasible trajectory planning
[29] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi,                   as language modeling. In International Conference on
     Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.                    Learning Representations (ICLR), 2026. 5


                                                              10
[40] Daming Wang, Yuhao Song, Zijian He, Kan-                           cal Methods in Natural Language Processing, pages
     gliang Chen, Xing Pan, Lu Deng, and Weihao                         5642–5665, Suzhou, China, 2025. Association for
     Gu. Hmvlm: Multistage reasoning-enhanced vision-                   Computational Linguistics. 3
     language model for long-tailed driving scenarios.             [49] Weicheng Zheng, Xiaofei Mao, Nanfei Ye, Pengxi-
     arXiv preprint arXiv:2506.05883, 2025. 6                           ang Li, Kun Zhan, Xianpeng Lang, and Hang Zhao.
[41] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong                 Driveagent-r1: Advancing vlm-based autonomous
     Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yi-                      driving with active perception and hybrid thinking.
     fan Ding, Wenhao Ding, et al. Alpamayo-r1: Bridg-                  arXiv preprint arXiv:2507.20879, 2025. 2
     ing reasoning and action prediction for generalizable         [50] Jingyu Zhou, Lu Ma, Hao Liang, Chengyu Shen,
     autonomous driving in the long tail. arXiv preprint                Bin Cui, and Wentao Zhang. Daro: Difficulty-
     arXiv:2511.00088, 2025. 1, 2                                       aware reweighting policy optimization. arXiv preprint
[42] Junjie Wen, Yichen Zhu, Minjie Zhu, Zhibin Tang,                   arXiv:2510.09001, 2025. 3
     Jinming Li, Zhongyi Zhou, Xiaoyu Liu, Chaomin                 [51] Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang,
     Shen, Yaxin Peng, and Feifei Feng. DiffusionVLA:                   Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. AutoVLA:
     Scaling robot foundation models via unified diffu-                 A vision-language-action model for end-to-end au-
     sion and autoregression. In Forty-second International             tonomous driving with adaptive reasoning and rein-
     Conference on Machine Learning, 2025. 3                            forcement fine-tuning. In The Thirty-ninth Annual
[43] Yichen Xie, Runsheng Xu, Tong He, Jyh-Jing Hwang,                  Conference on Neural Information Processing Sys-
     Katie Z Luo, Jingwei Ji, Hubert Lin, Letian Chen,                  tems, 2025. 1, 2, 3, 6, 7
     Yiren Lu, Zhaoqi Leng, Dragomir Anguelov, and
     Mingxing Tan. S4-driver: Scalable self-supervised
     driving multimodal large language model with spatio-
     temporal visual representation. In IEEE/CVF Con-
     ference on Computer Vision and Pattern Recognition
     (CVPR), 2025. 2
[44] Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui
     Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang,
     Xuxin Cheng, Ri-Zhao Qiu, et al. Egovla: Learning
     vision-language-action models from egocentric hu-
     man videos. arXiv preprint arXiv:2507.12440, 2025.
     3
[45] Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen,
     Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun
     Cai, Dapeng Zhang, and Shuo Li. Autodrive-r2 : In-
     centivizing reasoning and self-reflection capacity for
     vla model in autonomous driving. arXiv preprint
     arXiv:2509.01944, 2025. 1, 2
[46] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao,
     Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang.
     Does reinforcement learning really incentivize reason-
     ing capacity in LLMs beyond the base model? In The
     Thirty-ninth Annual Conference on Neural Informa-
     tion Processing Systems, 2025. 2
[47] Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran
     Liu, Yifan Bai, Zheng Pan, Mu Xu, and Xing Wei. Fu-
     turesightdrive: Thinking visually with spatio-temporal
     cot for autonomous driving. In The Thirty-ninth An-
     nual Conference on Neural Information Processing
     Systems, 2025. 2
[48] Jixiao Zhang and Chunsheng Zuo. GRPO-LEAD: A
     difficulty-aware reinforcement learning approach for
     concise mathematical reasoning in language models.
     In Proceedings of the 2025 Conference on Empiri-


                                                              11
   N O RD: A Data-Efficient Vision-Language-Action Model that Drives without
                                   Reasoning
                                             Supplementary Material
7. Comparison between GRPO and Dr. GRPO                             reward, format reward and dataset-specific reward (PDM
                                                                    score for NAVSIM and Normalized RFS for WaymoE2E).
We present a component-wise breakdown of Tab. 1 in                  The output of the model is a string of action tokens like
Tab. 4. Except for Ego Progress, Dr. GRPO significantly             TRAJ 0242 TRAJ 150 TRAJ 172 that are decoded to
outperforms GRPO. As shown in the training and validation           a list of waypoints of tuples [x,y,yaw] at 10 Hz.
curves in Fig. 11, both GRPO and Dr. GRPO improve over
time; however, GRPO consistently lags behind Dr. GRPO.
                                                                    Format Reward (rf ): A binary reward taking values
To further illustrate this, we visualize the change in mean         in {0, 0.25}. A reward of 0.25 is assigned if the prediction
PDM scores of the group, relative to the SFT model (step            consists of valid space-separated trajectory tokens of the
0), across different variance groups in Fig. 10. The vari-          form TRAJ i, where i is a zero-padded 4-digit integer in
ance groups are defined based on intra-group tertiles. Our          [0, 2047]; otherwise the reward is 0.
analysis reveals that:
1. Low-variance samples (Fig. 10 (a)): GRPO exhibits
                                                                    Length Reward (rl ): A binary reward taking values
   higher density above the y = x line, particularly for ini-       in {0, 0.25}. The model receives a reward of 0.25 if the
   tial scores in [0.8, 1.0].                                       prediction contains the correct number of trajectory tokens
2. Medium- and high-variance samples (Fig. 10 (b,c)):               (8 for NAVSIM and 10 for WaymoE2E); otherwise, the
   Dr. GRPO outperforms GRPO, with a denser concentra-              reward is 0.
   tion above the y = x line. The performance gap widens
   for high-variance samples, consistent with our observa-          Dataset Specific Reward (rd ):
   tion that GRPO attenuates policy updates for such sam-
   ples.                                                            1. PDM Score for NAVSIM: The PDM score (range:
                                                                       [0, 1]) comprehensively measures the driving quality and
8. Detailed Results                                                    safety. Is it given by:

8.1. Prompt Example
                                                                                                        5 · TTC + 2 · C + 5 · EP
We show an illustrative example in Fig. 12. N O RD main-                PDM Score = NC × DAC ×
                                                                                                                  12
tains token and inference efficiency by directly predicting
the trajectory tokens.                                                 where, No at-fault Collision (NC), Drivable Area Com-
8.2. Waymo E2E Scores                                                  pliance (DAC), Ego Progress (EP), Comfort (C), and
                                                                       Time-to-Collision (TTC) are all within [0, 1].
We present the detailed results of the performance of
                                                                    2. Normalized RFS for WaymoE2E: The RFS quantifies
N O RD on WaymoE2E test set in Tab. 6. As is evi-
                                                                       the alignment of the model’s predicted trajectory T̂ with
dent, N O RD is capable of performing complex multi-lane
                                                                       a set of three pre-rated human trajectories Tr . A score
switching maneuvers, while also performing well in less-
                                                                       sr ∈ [3, 10] is assigned to each rater trajectory based
represented scenes, such as intersections and construction
                                                                       on whether T̂ falls within a trust region defined by dy-
sites.
                                                                       namic longitudinal τ̄lng and lateral τ̄lat thresholds (scaled
8.3. Effect of Vocabulary Size                                         by current velocity). The final score is maxr (sr ), aver-
                                                                       aged over t ∈ {3, 5} seconds, and clipped to min(·, 4).
We experimented with a smaller k-disc vocabulary, consist-
                                                                       The Normlized RFS, with range [0, 1] is then given by:
ing of 512 trajectory tokens (as compared to 2048 trajec-
tory tokens in N O RD) and found that the performance on
NAVSIM degrades (Tab. 5). This is perhaps because the                                               max(maxr (sr ), 4) − 4
                                                                             Normalized RFS =
smaller vocabulary size cannot represent complex maneu-                                                     6
vers like sharp turn faithfully.
                                                                    The overall reward r for the predicted trajectory is therefore
9. Reward Functions                                                 given as:
In this section, we elaborate on the reward functions                                          rf + rl + rd
used for RL post-training. The reward consists of length                                  r=
                                                                                                   1.5

                                                                1
Figure 10. Training improvement patterns for GRPO (top, red) and Dr. GRPO (bottom, blue) across intra-group variance levels.
The y = x line indicates no change in PDM score. GRPO shows strong improvements for low-variance samples with initial scores
in [0.8, 1.0] (panel (a)), while Dr. GRPO outperforms GRPO for medium- and high-variance samples (panels (b) and (c)), with denser
concentration above y = x.

Table 4. Detailed comparison of RL-fine-tuning of N O RD- BASE with GRPO and Dr. GRPO. Dr. GRPO based RL fine-tuning is almost
always better than GRPO.

      Method                        PDMS↑       Collision↑        DAC↑     Direction↑    Progress↑     TTC↑      Comfort↑
      N O RD- BASE                    76.66        96.45          86.37       94.62         71.58       90.37      99.97
      N O RD- BASE+GRPO               77.18        91.89          90.12       91.84         80.06       80.13      99.96
      N O RD- BASE+Dr. GRPO           85.62        97.56          94.92       95.94         79.30       93.53       100


Table 5. Effect of k-disc vocabulary size on the performance of        10. Dataset Details
N O RD on navtest.
                                                                       10.1. WaymoE2E
                Vocabulary Size      PDMS ↑
                                                                       Supervised Finetuning: We curated the SFT dataset
                512                  83.07                             from the official WaymoE2E training set. Frames were
                2048                 85.62                             first strictly filtered, retaining only those that guaranteed
                                                                       four preceding time steps were available for consistent
                                                                       extraction of the ego-vehicle’s historical states. The final
                                                                       subset was then created by uniformly sampling 20% of
                                                                       these valid frame sequences from all contexts. This dataset
                                                                       was then randomly split into training and validation sets
                                                                       using an 85/15 ratio. The input images were resized to


                                                                   2
              0.95

              0.90


Mean Reward
              0.85

              0.80
                                                                NoRD+GRPO
                                                                NoRD+Dr. GRPO
              0.75
                     0       20   40      60     80    100      120   140   160
                                               Steps
                                         (a) Training Reward
              0.94
              0.92



Mean Reward
              0.90
              0.88
                                                                                      Figure 12. Example of N O RD inference. Given multi-view im-
              0.86                                                                    ages, past trajectory, and the current velocity, acceleration, and
                                                                NoRD+GRPO             driving command, N O RD directly predicts the trajectory tokens
              0.84                                              NoRD+Dr. GRPO         without explicit reasoning.
                     0       20   40      60     80    100      120   140   160
                                               Steps
                                       (b) Validation Reward
                                                                                      RL Finetuning: We use the official WaymoE2E vali-
  Figure 11. Training and validation curves for RL fine-tuning                        dation set, for which preference annotations are provided
  with GRPO and Dr.GRPO. Dr.GRPO (in red) consistently out-                           for a single frame per scenario. Consequently, we extract
  performs GRPO (in blue) on the (a) training and (b) validation sets                 one sample per scenario and randomly split the resulting
  by a significant margin.                                                            set into training and validation sets using an 85/15 ratio.

                     Table 6. Detailed results on WaymoE2E Test Set.                  10.2. NAVSIM
                                                                                      Supervised Finetuning: We use the official NAVSIM’s
                         Metric Name                           Value↑                 training set (navtrain) and split it into training and
                         Construction Score                   8.072616                validation sets for SFT using an 80/20 ratio. The input
                         Intersection Score                  7.9252014                images were resized to ensure the total number of pixels
                         Pedestrian Score                    7.7775736                lies between 784 and 401,408, following the Qwen vision
                         Cyclist Score                       7.8055406                encoder’s constraints.
                         Multi Lane Maneuver Score           7.8262477
                         Single Lane Maneuver Score           8.308635                RL Finetuning: We construct a RLFT dataset from
                         Cut In Score                         7.734755                the NAVSIM validation split originally used for supervised
                         Foreign Object Debris Score         7.6988134                fine-tuning. To remove trivial driving behaviors, we filter
                         Special Vehicle Score               7.7961473                trajectories using a constant-velocity baseline and discard
                         Spotlight Score                     6.5309787                samples with a final-point displacement error below 0.2
                         Others Score                         7.322814                m. For turning maneuvers, we additionally enforce a mini-
                         ADE at 3 seconds                     1.250462                mum average heading change of 0.01 rad per timestep to
                         ADE at 5 seconds                    2.8928785                eliminate mild curvature and drift. Straight trajectories are
                                                                                      exempt from the heading filter and are filtered solely using
                         Average Score                         7.709029               the displacement criterion. After filtering, the remaining
                                                                                      samples are balanced across three driving intents—straight,
                                                                                      left, and right—by uniformly subsampling each class. The
  ensure the total number of pixels lies between 784 and                              resulting dataset contains only non-trivial and dynamically
  401,408, following the Qwen vision encoder’s constraints.                           diverse trajectories, providing a more rigorous training sig-
                                                                                      nal for reinforcement learning-based trajectory prediction


                                                                                  3
and decision-making models.

11. Implementation Details
11.1. Supervised Finetuning
We perform supervised fine-tuning of N O RD on the
NAVSIM and WaymoE2E datasets using the Qwen2.5-VL-
3B-Instruct backbone, adapted to predict discretized trajec-
tory tokens from multi-view images, past trajectories, and
the ego-vehicle’s current kinematic states. For NAVSIM,
inputs consist of three camera frames (Front-Left, Front,
Front-Right), three past trajectory tokens covering the pre-
vious 1.5 seconds, current velocity and acceleration, and a
high-level driving command, with the model predicting 8
future trajectory tokens over a 4-second horizon at 10Hz.
For WaymoE2E, inputs include six past trajectory tokens
spanning 3 seconds, and the model predicts 10 future tokens
over a 5-second horizon. In both cases, trajectory tokens are
incorporated into the model vocabulary. All components of
the model, including the vision encoder, multimodal MLP,
and language model, are fine-tuned using mixed-precision
training with bf16 and gradient checkpointing to reduce
memory footprint. We train the model across 16 A100
GPUs, applying DeepSpeed ZeRO Stage 3 optimization for               Figure 13. Failure cases of N O RD. The predicted trajectory is
WaymoE2E and standard distributed training for NAVSIM.               shown in red and the violations marked in red circle.
We use consistent hyperparameters across datasets, includ-
ing a learning rate of 5 × 10−5 , a batch size of 8 per device
with 4 gradient accumulation steps, a cosine learning rate           frequently employ complex multi-dataset mixtures or uti-
scheduler, a warmup ratio of 0.03, and gradient clipping at          lize varying fractions of the available data. To standardize
1. We evaluate the model every 50 steps on the validation            these counts, we explicitly aggregated the reported dataset
sets and select the best model based on minimum evaluation           percentages and official splits detailed in the respective pa-
loss.                                                                pers’ training sections. For example, on WaymoE2E, we
                                                                     calculate HMVLM and DiffusionLTF at approx. 500k and
11.2. RL Finetuning                                                  730k samples based on the train and val splits in the Waymo
                                                                     Open Dataset for end-to-end driving and perception. Sim-
We perform RL fine-tuning of N O RD using Dr. GRPO to
                                                                     ilarly, for Poutine and AutoVLA, we aggregate their re-
optimize task-specific rewards. We generate 8 rollouts per
                                                                     ported multi-dataset percentages to approx. 700k and 210k
input to estimate group-relative advantages and update the
                                                                     samples, respectively. These standardizations ensure a fair
policy accordingly. We use a batch size of 128 trajectories
                                                                     relative comparison of data efficiency on the x-axis.
for NAVSIM and 256 trajectories for WaymoE2E, applying
asymmetric clipping with a high clip of 0.1 and a low clip
                                                                     13. Failure Cases
of -0.2 to stabilize policy updates. We train across 32 A100
GPUs for WaymoE2E and 30 A100 GPUs for NAVSIM,                       While N O RD achieves strong performance, it remains sus-
leveraging mixed-precision and gradient checkpointing for            ceptible to failure in certain scenarios. We present represen-
memory efficiency. We periodically evaluate the policy on            tative examples in Fig. 13. These cases can be attributed, in
validation sets and retain the checkpoint achieving the high-        part, to the fact that Dr. GRPO remains susceptible to dif-
est reward.                                                          ficulty bias, which still affects the policy optimization dy-
                                                                     namics. We therefore believe that targeted interventions to
12. Dataset Scale Estimation                                         better account for task difficulty could further push the per-
                                                                     formance frontier.
To visualize the performance-efficiency frontier in Fig. 6,
we estimated the total number of training samples for all
evaluated models based on their reported configurations.
Across both NAVSIM and WaymoE2E, baseline methods


                                                                 4
