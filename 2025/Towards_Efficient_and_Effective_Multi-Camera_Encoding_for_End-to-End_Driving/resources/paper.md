                                                    Towards Efficient and Effective Multi-Camera Encoding for
                                                                        End-to-End Driving
                                                  Jiawei Yangπ,∗ , Ziyu Chenρ,∗ , Yurong You∗ , Yan Wang∗ , Yiming Li∗ , Yuxiao Chen∗ , Boyi Li∗ ,
                                                                         Boris Ivanovic∗ , Marco Pavoneρ,∗ , Yue Wangπ,∗
                                                     π USC Physical Superintelligence (PSI) Lab              ρ Stanford University                       ∗ NVIDIA Research

                                                                                                Project Page: Flex

                                                                                                             Problem: Excessive image tokens across views and time burden policy models.
                                            Abstract— We present Flex, an efficient and effective scene




arXiv:2512.10947v2 [cs.CV] 12 Dec 2025
                                         encoder that addresses the computational bottleneck of process-     T-2                                T-1                               T
                                         ing high-volume multi-camera data in end-to-end autonomous
                                         driving. Flex employs a small set of learnable scene tokens
                                         to jointly encode information from all image tokens across
                                         different cameras and timesteps. By design, our approach is
                                         geometry-agnostic, learning a compact scene representation          Solution: Compress to obtain scene representations.
                                         directly from data without relying on the explicit 3D inductive     Multi-cam & timestep   Image Patchifier     Scene Encoder        Scene Repre.            Policy Model
                                         biases, such as Bird-Eye-View (BEV), occupancy or tri-plane
                                                                                                                                                                            2D representation
                                         representations, which are common in prior work. This holistic
                                                                                                                                        Image
                                         encoding strategy aggressively compresses the visual input                                                      Downsample                                        LLM
                                                                                                                                       Encoder
                                         for the downstream Large Language Model (LLM) based                        …                                                             …
                                         policy model. Evaluated on a large-scale proprietary dataset of                                                                      3D representation
                                         20,000 driving hours, our Flex achieves 2.2× greater inference                                                                             xT   Voxels

                                         throughput while improving driving performance by a large                                      Image             2D-to-3D                  x T BEV                LLM
                                         margin compared to state-of-the-art methods. Furthermore, we                                  Encoder          Scene Encoder
                                         show that these compact scene tokens develop an emergent                   …                                                               x T Tri-plane

                                         capability for scene decomposition without any explicit super-
                                                                                                                                                                                 Learned
                                         vision. Our findings challenge the prevailing assumption that                                                       Our            1D representation
                                         3D priors are necessary, demonstrating that a data-driven, joint                               Image
                                                                                                                                                             Flex                                          LLM
                                                                                                                                       Encoder
                                                                                                                                                        Scene Encoder
                                         encoding strategy offers a more scalable, efficient and effective
                                                                                                                    …
                                                                                                                                                                                Data-driven
                                                                                                             Ours                                                             No 3D/4D-prior
                                         path for future autonomous driving systems.
                                                                                                             Results: Faster and Better.
                                                             I. INTRODUCTION                                 Baseline                18.60                     Baseline                                       0.80
                                                                                                                   Ours                                41.08       Ours                             0.76
                                            Vision-Language-Action (VLA) models [1], [2], [3], [4],          Inference Throughput (clips / second, ↑)                    Driving Performance (minADE, ↓)
                                         which use Large Language Models (LLMs) as powerful pol-
                                         icy heads, have been revolutionizing end-to-end autonomous          Fig. 1: Problem, solution, and results. Multi-camera inputs
                                         driving. A key challenge in this new paradigm is to encode          across time generate excessive tokens that overload policy
                                         high-bandwidth visual data [5], efficiently and effectively.        models. Prior methods compress tokens into 2D or 3D scene
                                         A self-driving vehicle’s multi-camera rig usually collects          representations with limited efficiency and fixed granularity.
                                         tens of images per second, resulting in thousands or tens           Our Flex scene encoder learns compact scene representations
                                         of thousands of visual tokens per forward. Processing this          directly from data, without 3D/4D priors or camera poses. It
                                         massive data stream imposes a significant computational load        achieves both higher efficiency (41.08 vs. 18.60 clips/s) and
                                         on LLM-based policy models, leading to a critical bottleneck.       better driving accuracy (0.76 vs. 0.80 minADE6 ).
                                            Redundancy lies at the core of this problem. In a typical
                                         driving scenario, significant spatial overlap exists between
                                         adjacent wide field-of-view cameras, while temporal continu-        into a quasi-3D representation, presuming these geometric
                                         ity introduces further repetition between consecutive frames        priors are essential for coherent scene understanding. While
                                         (see Fig. 1 top). Naive per-image encoding increases latency        these approaches can improve efficiency by factorizing the
                                         and memory and, more importantly, fails to exploit the              scene, they also impose a rigid structure that may cap the
                                         spatiotemporal redundancy. Designing a scene encoder that           model’s performance. The development of deep learning has
                                         can leverage and compress this redundancy is therefore              repeatedly shown that replacing inductive biases with scal-
                                         essential for both efficiency and effectiveness.                    able, data-driven solutions often leads to better performance
                                            Prior efforts often manage this complexity by imposing           and generalization [11]. This trend motivates us to explore a
                                         strong 3D inductive biases. Methods based on bird-eye-              simpler, more scalable alternative, shown in Fig. 1.
                                         view (BEV) planes [6], [7], voxels [8], triplanes [9], [5],            In this paper, we present Flex, a flexible and geometry-
                                         or hexplanes [10] explicitly structure the visual information       agnostic Transformer scene encoder that jointly encodes
images from all views and timesteps into a compact set            grounded representations for compression. Perception mod-
of scene tokens. Concretely, Flex initializes a small set of      els convert images to Bird’s-Eye-View (BEV) or volumetric
learnable scene tokens, concatenates them with all image          occupancy as an intermediate representation. For example,
tokens across cameras and time, and runs a lightweight            PETR [16] and PETRv2 [17] inject 3D position encodings
Transformer [12] encoder that jointly processes all tokens        for camera-only 3D detection. BEVFormer [6] uses grid-
using self-attention layers. After encoding, we discard the       shaped BEV queries with spatiotemporal Transformers to
image tokens and keep only the updated scene tokens as the        aggregate multi-view, multi-timestep perceptions. BEVFu-
representation passed to the LLM-based policy model. This         sion [18] unifies multi-modal features in BEV efficiently;
creates an information-seeking bottleneck that forces cross-      SurroundOcc [19] predicts 3D occupancy from cameras via
view and cross-time compression, allowing the model to            2D-to-3D attention.
learn the optimal representation directly from data, avoiding        Beyond BEV and occupancy, neural field methods propose
any explicit 3D priors.                                           compact factorized 3D/4D representations. Tri-planes factor-
   Our design is intentionally made simple, targeting both        ize features into three axis-aligned planes for efficient 3D
efficiency and effectiveness. Efficiency comes from reducing      representation [5], [9], which are further extended to dynamic
the token budget passed to the policy model by about 3× to        scenes by HexPlane and K-Planes [20], [10], [21]. Instant-
20×, which is typically the dominant training and inference       NGP [22] and its variants use multi-resolution hash encod-
cost. Effectiveness comes from the joint encoding: the scene      ings to accelerate field learning [23]. These representations
tokens attend to all images across views and timesteps, so        improve speed and structure, but they predefine the basis and
redundancy suppression occurs at the scene level rather than      routing of information. Explicit 3D/4D representations also
image level. In contrast, per-image queries, like those used in   demand accurate camera poses and synchronized sensors,
Q-former [13], cannot enforce global coordination and tend        raising data requirements. Moreover, the pre-set granularity
to preserve global redundancy.                                    (e.g., grid or voxel size) constrains the information density,
   Our approach is surprisingly effective. We evaluate Flex       and camera perspective further skews this density: near
on a large-scale proprietary dataset of 20,000 driving hours.     objects occupy many pixels, whereas distant objects collapse
Compared to state-of-the-art image encoding strategies, our       into very few. Thus, fixed grid/voxel designs cannot adapt to
Flex achieves 2.2× the inference throughput of baseline           the highly non-uniform information density in perspective
methods while improving driving performance by a large            images. In contrast, we minimize geometry-specific priors
margin. These results suggest that a simple, data-driven          and compress via data-driven joint attention over the full
scene encoder can be more effective. Importantly, we find         image set. Our encoder uses a small set of learned scene
that jointly encoding the entire scene with self-attention is     queries to discover structure directly from data, without
critical to performance, and surprisingly, our compressed         committing to any fixed 3D decomposition, and naturally
token representation shows an emerging ability to focus on        adapts to the non-uniform density of perspective images.
destination, lane markers and safety-critical areas. We hope
our work will provide a new perspective on scene encoding
for end-to-end driving.
                                                                  C. Learned representations
                    II. R ELATED WORK
A. End-to-end driving and camera visual encoding                     A separate thread in computer vision learns a fixed-size
   End-to-end autonomous driving has evolved from modular         query set to summarize large inputs. Perceiver [24] and
stacks to architectures that fuse perception and planning         Perceiver IO [25] introduce asymmetric cross-attention from
into a single differentiable system. Recent planning-oriented     a small latent array to high-dimensional inputs, decoupling
designs unify tasks with shared queries and BEV features          compute from input length. Flamingo uses a Perceiver Re-
(e.g., UniAD [7]), or learn spatial–temporal features for         sampler to map image/video features to a small, learned set
perception–prediction–planning in one network (e.g., ST-          of visual tokens for VLMs [26]. TokenLearner adaptively ex-
P3 [14]), yet they still commit to explicit 3D structures, thus   tracts few informative tokens from images [27]. The encoder-
requiring precise camera pose information. Vectorized end-        decoder architecture in TiTok [28] and LVSM [29] leverage
to-end approaches (e.g., VADv2 [15]) consume multi-view           learnable queries to enable compact novel view synthesis and
image sequences directly and output actions, but do not ad-       auto-encoding tasks. STORM [30] use learnable tokens to
dress aggressive token-level compression across all cameras       capture the underlying motion biases of the scene. Our scene
and timesteps. Our work complements these systems: we             tokens follow this latent-query paradigm but differ from them
focus on the camera encoding stage, targeting a compact           in two ways critical for driving: (i) we jointly attend over
set of scene tokens learned jointly across views and time,        all cameras and timesteps at once with self-attention (not
without imposing a particular geometric scaffold.                 per-image or cross-attention-based), and (ii) we show that
                                                                  planning-centric compression naturally produces tokens that
B. Scene representations for autonomous driving                   are not only compact but also action-relevant. This design
   The massive volume of camera inputs creates a compu-           yields stronger efficiency–accuracy trade-offs and facilitates
tation bottleneck, leading recent works to adopt geometry-        end-to-end training.
Input Image                 Image tokens                    Policy Model           Input Image                Image tokens                Discarded                  Policy Model

                                           Resize+
                                            MLP
 Image 1      Patchifier                             …                              Image 1      Patchifier
                                                                                                                                …              …
                                           Resize+
                                            MLP
 Image 2      Patchifier                             …                              Image 2      Patchifier
                                                                                                                                …              …
                                                                                                                                      Flex
                                           Resize+         LLM-Based                                                                                                LLM-Based
                                                                                                                                     Scene
 Image 3      Patchifier
                                            MLP           Policy Model              Image 3      Patchifier                                                        Policy Model
                                                     …                                                                          …   Encoder    …
                                                     …                                                                                                      …
    …             …             …                                                      …            …             …             …              …

    Special Tokens

    Trajectory History                                                                                           Scene Tokens
                                                                                                                                …              …
    Trajectory Prediction

 (a) Baseline                        Number of Scene Tokens to Policy Model: ~3K    (b) Ours                                                  Number of Scene Tokens to Policy Model: <1K


Fig. 2: Method overview. In (a) baseline (left), thousands of image tokens from all cameras and timesteps are generated,
overloading the policy model. In (b) Flex (right), a Flex Scene Encoder compresses image tokens into a compact set of
scene tokens as scene representations. Only this small set of tokens is used as the scene representations. Special tokens
encode the start and end of each modality, camera type and timestamp information.

                                    III. M ETHOD                                              C camera views and T timesteps:
   Our work introduces Flex, a scene encoder designed                                            Sbaseline = Concat ({φ pro j (Resize(patchifier(Ic,t )))}c,t ) (1)
for Vision-Language-Action (VLA) models in autonomous
driving. The overall objective is to learn a policy π that                                    This results in a long sequence of C × T × 160 tokens, which
predicts a future ego-vehicle trajectory Y given a sequence                                   imposes a significant computational load on the policy head.
of observations O. Our primary contribution is a new scene                                    Trajectory Encoding. We adopt two complementary to-
encoder that produces a compact, yet highly effective, scene                                  kenization methods. (i) Continuous history. The ego-
representation from the massive amount of visual inputs,                                      vehicle’s recent trajectory sT −h:T is encoded by an MLP
which improves efficiency during training and inference, and                                  φhist into a single embedding hhist , compressing history into
enhances driving performance of the policy model.                                             one token. (ii) Discrete future. The future trajectory Y is
                                                                                              discretized into a vocabulary of waypoint tokens, added to
A. Vision-Language-Action Model                                                               the LLM embeddings. The policy model then predicts the
   We first establish a VLA architecture as our baseline.                                     future as an auto-regressive next-token generation task. This
This model processes a sequence of RGB images and past                                        encoding proved most effective in our early experiments.
trajectory data to auto-regressively predict a future trajectory.                             Policy Head. We initialize our policy model from a pre-
Fig 2-(a) shows the overview.                                                                 trained LLM backbone. This allows us to leverage the rich
Setup and Notation. Let C = {1, . . . ,C} be camera indices                                   world knowledge and reasoning capabilities acquired during
and T = {1, . . . , T } be past timesteps within a fixed window                               its pretraining on internet-scale data. The policy model is an
(covering around 2s). The observations O consist of multi-                                    auto-regressive Transformer that takes the scene representa-
camera (C), multi-timestep (T) images {Ic,t ∈ RH×W ×3 | c ∈                                   tion and history tokens as context to predict the sequence of
C , t ∈ T } and the most recent ego-state history sT −h:T (e.g.,                              future. The model is trained with a standard cross-entropy
pose/velocity), where h is the time horizon. The target is a                                  loss on the predicted future trajectory tokens. Given the large
                             T +h                                                             number of scene tokens, this policy model is the dominant
future trajectory Y = {yt }t=T    +1 in ego coordinates, where h
is the prediction horizon.                                                                    component in both training and inference costs.
Scene Representations. For the baseline model, scene rep-
resentations are derived directly from image tokens. Each                                     B. Flex Scene Encoder
input image Ic,t ∈ RH×W ×3 is first divided into patches and                                  Overview. Our Flex encoder is designed to reduce the ex-
processed by an image encoder (e.g., DINOv2 [31]), which                                      cessive number of image tokens in the scene representation.
we term “patchifier” following VGGT [32], to produce a set                                    Our core insight is that the concatenated image tokens,
of N tokens, typically N = 640 in our setting. These tokens                                   Sbaseline , are highly redundant due to significant spatial over-
are then bilinearly downsampled to a more manageable                                          lap between camera views and temporal continuity between
number, such as 160 tokens per image. After being resized,                                    frames. To address this, Flex jointly encode all views and
they are passed through a 2-layer MLP projection layer φ pro j                                timesteps into a small set of scene tokens, forcing the model
to match the hidden dimension Dllm of the LLM policy head.                                    to leverage cross-spacetime correspondences. This design
The final scene representation for the baseline, Sbaseline , is a                             promotes both efficiency and the learning of a more effective
direct concatenation of all processed image tokens from all                                   scene representation. Fig 2-(b) shows the overview.
                                                                           Loss from GT at t0
(a) Non-interleave: loss applied on one timestep t0                                             adjusting the attention mask, without splitting the sequence.
                                Policy Model                                                    This interleaved design forces the model to predict the future
   …            …              …       …                      …        …                        with incomplete context, greatly increasing the diversity of
                                                                                                training signals and leading to more efficient learning. We
         Loss from GT at t0 − T + 1           Loss from GT at t0 − 1       Loss from GT at t0
                                      …                            …                     …      modify the attention mask so that the past history and future
                                                                                                trajectory tokens in the context (e.g., from t0 − T + 1 to t0 − 1
                                  Policy Model
                                                                                                in Fig. 3-b) do not contribute to the final prediction.
   …                               …                                   …                 …
                                                                                                Adopting Holistic Scene Representations. Our Flex repre-
(b) Interleave: loss applied on T timesteps {t0 − T + 1, . . . , t0}
                                                                                                sentation SFlex is a single, jointly-encoded representation so
   Image / Scene Tokens                   History Trajectory           Future Trajectory
                                                                                                that it no longer has an explicit mapping to a specific camera
Fig. 3: Comparison of non-interleaved training strategies. (a)                                  or time. To resolve this, we introduce a remarkably effective
Loss applied only at the final timestep t0 . (b) Our proposed                                   heuristic. We evenly partition the K scene tokens into T
non-interleave setting: Loss applied across T timesteps {t0 −                                   sequential, non-overlapping chunks, {S1Flex , S2Flex , . . . , STFlex },
T + 1, . . . ,t0 }. Special tokens are abbreviated for simplicity.                              where each chunk SkFlex ∈ R(K/T )×Dllm .
GT denotes ground-truth.                                                                           During interleaved training at step k, the policy model
                                                                                                is conditioned on the concatenation of the first k chunks
Joint Scene Encoder. Our scene encoder is a lightweight                                         of scene tokens, i.e., Concat(S1Flex , . . . , SkFlex ), along with
Transformer that operates on all image tokens holistically. As                                  the most recent history token hhist . While this partitioning
in the baseline, input images are first patchified by a DINOv2                                  provides no explicit guidance to the encoder, the end-to-end
encoder [31]. We then add two sets of positional embeddings                                     training objective forces the model to learn a meaningful
to these image tokens: a timestep embedding PEttime and                                         allocation. This process forces an emergent specialization
a camera embedding PEccam . The timestep embeddings are                                         within the scene tokens, where different chunks implicitly
similar to those in DiT [33], while the camera embeddings                                       learn to capture information needed for the decision-making
are learnable vectors indexed by camera types.                                                  at every step, as shown in Fig. 5.
   We initialize a set of K learnable scene tokens S(0) ∈ RK×d ,
which act as queries. These scene tokens are prepended to the                                                           IV. E XPERIMENTS
sequence of all image tokens from all cameras and timesteps.
                                                                                                A. Setup
A L-layer Transformer encoder Eθ (we use L = 8) computes
full self-attention updates over this combined sequence:                                        Dataset. We train and evaluate on a large-scale, internal
     h           i                                                                          dataset comprising 20,000 hours of driving logs. The data
      S(L) ; X(L) = Eθ S(0) ; X ,           S(L) ∈ RK×d .   (2)                                 was collected from a fleet of ego-vehicles operating in over
                                                                                                1,700 cities across 25 countries, ensuring high diversity. It
After the final layer, we discard X(L) and use only the                                         encompasses a wide range of driving scenarios, including
updated scene tokens S(L) as the scene representation, further                                  urban and highway driving, various weather patterns, dif-
denoted as SFlex . After a linear projection to Dllm , these K                                  ferent times of day, and diverse traffic densities. We use a
tokens feed the policy model.                                                                   geographically-separate holdout for evaluation, with a 90%
                                                                                                training/validation and 10% test split. While ego-vehicles are
C. Interleaved Prediction                                                                       equipped with seven cameras, we primarily utilize a two-
   Next, we introduce an important design choice for VLA                                        camera setup (front-wide and front-telephoto) for efficiency.
models in autonomous driving, named interleaved prediction.                                     All videos are resized to a resolution of 320 × 512. We also
Diversifying Supervision. At inference time, the policy                                         present ablation studies on the number of camera inputs to
model conditions on C cameras over T timesteps together                                         demonstrate the wide applicability of our approach.
with a single history token, and generates the predicted                                        Training and Evaluation Details. We adopt a two-stage
trajectory. A straightforward training setup would mirror                                       training strategy. In the first stage, we freeze the image
this setting: the model takes the same number of timesteps                                      patchifier and train the scene encoder and policy model for
context plus one history token and minimizes loss only on the                                   300k iterations (100k for ablation studies) using AdamW
subsequent future tokens, as shown in Fig. 3-(a). However,                                      optimizer with a linear warmup over the first 1000 iterations
this turns out to be ineffective in practice (Tab. 4f). The                                     to a peak of 4 × 10−4 , followed by a cosine decay schedule.
model receives limited supervision.                                                             We train with a global batch size of 256 (4 clips/GPU on
   We instead propose an interleaved training strategy, as                                      64 GPUs). The second stage consists of end-to-end fine-
shown in Fig. 3-(b). For a sequence of length T , we apply                                      tuning, where all model parameters are unfrozen. For this
supervision on input contexts of varying lengths: at each                                       stage, we reduce the learning rate to 1 × 10−5 and train
timestep k, the model is given all image/scene tokens up                                        for 50k iterations. We sample observation windows of 2 s
to k and the corresponding trajectory history at tk , and it is                                 during training and 1 s during evaluation while keeping a
required to predict the next H future steps. This produces                                      fixed temporal sequence length of 9 timesteps. We train with
T supervision signals per sequence, one from every possible                                     standard mixed precision and enable activation checkpointing
prefix. In practice, this can be implemented efficiently by                                     in the policy model. By default, we use a total of 900 scene
TABLE I: System-level comparison. We compare minADE6 (lower is better), policy input length (#Scene Tokens), training
cost (A100 GPU hours), and inference throughput (clips/s). Flex achieves the best effectiveness–efficiency trade-off, with
lower error, fewer tokens, faster training, and higher throughput. Patch sizes are denoted as (px − py − pz ) for triplanes [5].
                                                                                                                       Training Time↓     Throughput↑
 Method                      LLM               Stage           #Camera×#Time       minADE6 ↓      #Scene tokens ↓
                                                                                                                      (A100 GPU hrs)        (clips/s)
 ∗ Triplane (8-8-8) [5]    Llama2-1B            Stage 1                4×6                1.046         1080 (1.2×)        4984 (2.2×)     69.62 (3.7×)
 ∗ Triplane (4-6-6) [5]    Llama2-1B            Stage 1                4×6                0.974         2496 (2.8×)        7960 (3.5×)     64.53 (3.5×)
 In-house VLA             Qwen2-0.5B            Stage 1                2×9                0.818         2880 (3.2×)        4134 (1.8×)     18.60 (1.0×)
 Flex (Ours)              Qwen2-0.5B            Stage 1                2×9                0.794          900 (1.0×)        2260 (1.0×)     41.08 (2.2×)
 In-house VLA             Qwen2-0.5B       Stage 1 + Stage 2           2×9                0.798         2880 (3.2×)        5750 (1.7×)     18.60 (1.0×)
 Flex (Ours)              Qwen2-0.5B Stage 1 + Stage 2                 2×9                0.761          900 (1.0×)        3318 (1.0×)     41.08 (2.2×)
 ∗ We include results from [5] mainly for reference as there are many differences in experimental setup, including LLM training (from scratch in [5] vs
                  pretrained), non-interleaved [5] vs interleaved prediction, and different choices of cameras × timesteps (4 × 6 in [5]).

tokens for 18 input images (2 cameras × 9 timesteps) and                     joint self-attention, and 2 cameras (front-wide and front-
an 8-layer scene encoder with full self-attention.                           telephoto). We analyze the effect of each component in the
Metrics. To provide a comprehensive comparison, we eval-                     following sections. Our baseline follows the identical setting
uate both the efficiency and effectiveness of our method                     (same DINOv2-base backbone, LLM-policy head, and 100k
against baselines. Efficiency is measured in terms of clips                  training iterations).
processed per second during inference and effectiveness                      Impact of patchifier size. We evaluate the impact of
is measured by driving performance using the standard                        the image patchifier by experimenting with three sizes of
minADEk metric, which is the minimum Average Displace-                       DINOv2 [31]: Small, Base, and Large while keeping the
ment Error over k predicted trajectories. We use k = 6:                      scene encoder fixed. As shown in Fig. 4a, scaling from
                                                                             Small to Base yields a significant improvement in driving
                                           1 H (k)
          minADE6 (Y, Ŷ ) =       min       ∑ ŷt − yt 2             (3)    performance with a moderate increase in encoding latency.
                                k∈{1,..,6} H t=1                             Moving from Base to Large yields more accuracy gains with
where yt is the ground truth ego-vehicle position at future                  more computational cost. We therefore adopt DINOv2-Base
              (k)                                                            for all other experiments, as it offers the best balance of
timestep t, ŷt is the k-th predicted position, and H is the
number of timesteps in the prediction horizon. We further                    performance and efficiency.
average minADE6 across time by 0.5s, 1.0s, 3.0s, and 5.0s.                   Number of scene tokens. We study how performance and
Baselines. We compare our method against our in-house                        efficiency scale with the number of scene tokens (K), varying
VLA model described in Section III-A, which directly feeds                   it from 144 (8 tokens/image) to 1152 (128 tokens/image).
the full, flattened sequence of all multi-view and multi-                    Fig. 4b shows a clear trade-off. Inference throughput is
timestep image tokens into the LLM policy model. This                        inversely proportional to K, as the cost of the policy model
comparison allows us to directly measure the efficiency gains                scales with the number of input tokens. Conversely, driving
and performance changes from introducing a compression                       performance (minADE6 ) improves nearly monotonically as
bottleneck. Both our method and the baseline use the same                    more scene tokens are used, saturating around K ≈ 900.
policy head (Qwen2-0.5B [34]), trajectory tokenizers, and                    This yields a smooth Pareto frontier, enabling practitioners
training schedule unless noted.                                              to select K for a target throughput—an advantage over rigid
                                                                             image-based encoding. K = 900 provides a good trade-off.
B. Comparisons with Prior Systems                                            Number of scene encoder layers. We ablate the depth of
  We compare Flex against prior state-of-the-art systems                     the Flex scene encoder, testing configurations with 1, 2,
on our held-out test set. The results, presented in Table 1,                 4, 8 and 16 Transformer layers. Fig. 4c shows the results.
demonstrate that our approach achieves a superior trade-off                  Performance improves notably when moving from 1 to 8
between efficiency and effectiveness.                                        layers, suggesting that sufficient depth is needed for the scene
  Compared to our state-of-the-art in-house VLA model,                       tokens to effectively extract information from the thousands
Flex achieves a 2.2× increase in inference throughput                        of image tokens, but excessively deep encoders provide little
while simultaneously improving the driving performance                       additional benefit. We therefore adopt 8 layers by default.
(minADE6 ) from 0.798 to 0.761. Flex requires only about                     Pareto frontier across design choices. Fig. 4d summarizes
60% of the total training time compared to the baseline.                     the ablation results by plotting the Pareto frontier across
This highlights the severe inefficiency of passing thousands                 different design choices. Each curve traces the trade-off be-
of redundant tokens to the policy head.                                      tween accuracy and throughput when varying a single factor.
                                                                             The frontier clearly shows that our default configuration (red
C. Ablation Study                                                            star) lies near the optimal trade-off region, offering strong
  For ablation study, we train all models following stage-                   driving performance without sacrificing efficiency.
1 strategy and reduce the training iterations from 300k to                   Scene encoder design. We compare four designs with
100k for manageable computation resources. By default, we                    the same output token budget K: (1) Independent per-
use DINOv2-base as the image patchifier, 50 scene tokens                     image tokens with cross- or self-attention. Each camera
per image (900 per scene), 8 scene encoder layers with                       image is independently compressed using its own set of
                                                                                                                                                                                                               1                              Baseline
                                                                                                                                                                 0.88         baseline                                             0.88
                                                                                                                                                                                                                                              Patchifier size
                                                                                                   160                                                                        default setting                                                 #Tokens/image
                                                                                         0.860                          baseline                                                                                                              #Encoder layers
                          base
                0.860                          baseline
                                                                                                                        default setting                          0.87                                                              0.87       Default setting

                                               default setting
                                                                                         0.855




                                                                                                                                                                                                                   minADE6 ( ↓ )
                                                                                                                                                                                                               2


                                                                                                                                                 minADE6 ( ↓ )
                0.855
                                                                                                                                                                 0.86                                                              0.86
                                                                 small



                                                                         minADE6 ( ↓ )
                                                                                         0.850
                0.850                                                                                                                                                                                      4


minADE6 ( ↓ )
                                                                                                                                                                 0.85                                                              0.85
                                                                                         0.845
                0.845
                                                                                                                                          8

                                                                                         0.840                                   16                              0.84                                                              0.84
                0.840

                                                                                                                                                                                                16    8
                                                                                                                            32
                0.835                                                                    0.835      128                                                          0.83                                                              0.83
                                                    base                                                         64 50
                                    large                                                                                                                                 20          30          40                                        20     30      40       50      60
                                                                                                                                                                        Inference Throughput (clips/second, ↑ )                           Inference Throughput (clips/second, ↑ )
                          20         30          40         50                                     20     30      40       50      60
                        Inference Throughput (clips/second, ↑ )                                  Inference Throughput (clips/second, ↑ )
                                                                                                                                   (c) Ablation on scene enc.           (d) Pareto frontier across
(a) Ablation on patchifier size. (b) Ablation on token number.                                                                                layers.                            settings.
                                                                                                                                                             (g) Ablation on camera count.
                        (e) Ablation on attention type.                                                       (f) Ablation on interleave settings.      (camera × time)        minADE (↓) Throughput ↑                                                    6
           Attention type                    minADE6 (↓)             Throughput ↑                          setting                            minADE6 (↓)                    Throughput ↑
                                                                                                                                                                                                     baseline (2×9)                                  0.860            18.60
           Per-Img. Cross-Attn                     0.966                 47.96                             baseline, non-interleave              0.894                           18.47               baseline† (4×9)                                 0.886             7.58
           Per-Img. Joint-Attn                     0.844                 47.07                             baseline, interleave                  0.860                           18.60               baseline† (7×9)                                 0.925             3.34
           Cross-Attn                              1.032                 47.27                             Ours, non-interleave                  0.991                           41.06               Ours (2×9), 50 tokens/img                       0.833            41.08
           Joint-Attn                              0.833                 41.08                             Ours, interleave                      0.833                           41.08               Ours (4×9), 50 tokens/img                       0.831            19.06
                                                                                                                                                                                                     Ours (7×9), 32 tokens/img                       0.830            11.40

Fig. 4: Ablation study on Flex design. We evaluate key components of Flex, including patchifier size, token count, scene
encoder depth, attention type, interleave setting, and camera count. The default configuration (red star or gray row) achieves
a strong accuracy–efficiency trade-off. Across all settings, Flex consistently improves both accuracy and throughput (2×–3×)
compared to the baseline. Each data point for our approach requires roughly 650 to 1,000 A100 GPU hours for training
while baselines take around 1,300 to 1,800 GPU hours. † : Trained with half the batch size due to out-of-memory constraints.

learnable tokens before being passed to the policy model.                                                                                                the usage of intermediate ground truth. Overall, interleaving
This is efficient but ignores cross-spacetime redundancy. We                                                                                             is key to enabling sample-efficient learning and unlocking
evaluate both cross-attention layers, as used in Q-former [13],                                                                                          the full benefits of our scene-token encoder.
and self-attention layers in this setting. (2) Joint scene                                                                                               The sensitivity to camera count. We further examine the
tokens with cross- or self-attention. A shared set of scene                                                                                              impact of increasing the number of input cameras (Table 4g).
tokens is introduced to summarize information from all                                                                                                   Our method continues to provide consistent gains over
camera views and timesteps. In the cross-attention setting,                                                                                              baselines in both efficiency and effectiveness. In particular,
scene tokens attend to the image tokens, but the image                                                                                                   the throughput advantage grows from 2.2× with 2 cameras
tokens themselves remain fixed (i.e., no self-attention among                                                                                            to 3.4× with 7 cameras, while maintaining strong driving
image tokens or between image and scene tokens). We                                                                                                      performance. Notably, as the number of cameras increases,
study both cross- and self-attention mechanisms to under-                                                                                                baseline performance degrades significantly, whereas our
stand the importance of updating scene tokens and image                                                                                                  method remains stable. We attribute this to two factors.
tokens. Table 4e summarizes the results. First, the widely                                                                                               First, learning effective policies from tens of thousands
employed per-image token strategy performs worse than joint                                                                                              of raw image tokens is increasingly challenging. Second,
encoding. This is expected, as it cannot leverage mutual                                                                                                 the front-wide and front-telephoto cameras already provide
information exist across different camera views and timesteps                                                                                            sufficient coverage for most driving scenarios in our test split,
to obtain a holistic scene representation. Second, within both                                                                                           which does not specifically emphasize rare cases requiring
families, the cross-attention variants underperform their self-                                                                                          side or rear cameras. Exploring these scenarios remains an
attention counterparts. These results suggest that enabling                                                                                              interesting direction for future work.
image tokens to interact with one another, mediated through                                                                                              D. Emerging Scene Decomposition and Intention Tokens
scene tokens, is critical for learning effective representations.
Overall, a holistic joint encoding method proves essential to                                                                                               To probe what the scene tokens capture, we analyze their
the success of Flex.                                                                                                                                     attention toward image tokens by inspecting the attention
                                                                                                                                                         weights produced by the scene encoder. Specifically, we
Interleaved prediction. Table 4f shows that interleaving is                                                                                              extract the attention from scene tokens to image tokens at the
crucial for both baseline and Flex. For the baseline model,                                                                                              final layer of the scene encoder, average these values across
interleaving reduces minADE6 from 0.894 to 0.860, with                                                                                                   attention heads, and then record the maximal response from
no loss in efficiency. For Flex, the effect is even stronger:                                                                                            each scene token to any image token. This procedure yields
interleaving improves minADE6 from 0.991 to 0.833 while                                                                                                  a per-scene-token measure of its strongest attentional link,
preserving high throughput (41 clips/s). This training strategy                                                                                          which we use for quantitative analysis and visualization of
forces the model to predict trajectories from partial contexts                                                                                           representative examples in Fig. 5.
rather than always conditioning on the full history. It also                                                                                                The results show clear specialization. The top-ranked
provides richer supervision and greater robustness because of                                                                                            tokens (Rank 1-3) consistently highlight the destination of
               Input Camera                                Token Rank 1                  Token Rank 2         Token Rank 3




     Input Camera        Token Rank 512         Input Camera         Token Rank 521      Input Camera        Token Rank 521



                                                                           3




    Input Camera         Token Rank 822         Input Camera         Token Rank 822      Input Camera        Token Rank 900



                                                                           3




Fig. 5: Emergent scene decomposition from learned scene tokens. Bottom: We show the sorted average token response
for all scene tokens. Larger value indicates that these tokens have stronger attention to image tokens. Top. Visualization of
representative scene tokens. Top three rows: the highest-ranked tokens (Ranks 1–3) consistently focus on the ego vehicle’s
destination. Middle two rows: mid-ranked tokens (e.g., Rank 512) exhibit interesting look-ahead scanning patterns across
the roadway, resembling human’s visual scanning, guiding fixation or look-ahead fixation strategies. Next two rows: a lower-
ranked token (Rank 822) clearly highlights road lane markings. Interestingly, the least responsive token (Rank 900) appears
to capture positional bias patterns. Notably, all these behaviors emerge naturally without explicit supervision.

the ego vehicle, suggesting that the model dedicates capacity     but similar phenomena have been observed in ViT [35],
to goal-related reasoning. Mid-ranked tokens (Rank 521)           where certain tokens serve structural roles (e.g., encoding
exhibit scanning behaviors (two-spot attention), shifting their   positions) rather than explicit content [23], [36], [11].
attention forward along the road in a manner resembling
look-ahead planning. Lower-ranked tokens (Rank 822) cap-            Overall, this analysis demonstrates that the scene encoder
ture static yet crucial structures such as lane markings. We      does more than compress visual inputs: it learns to allocate
also note that there are more tokens related to lane markings,    tokens to semantically meaningful components, including
meaning that these scene tokens work collectively. At the tail    destination, motion anticipation, road layout, and positional
end, the least responsive token (Rank 900) appears to encode      context. The emergence of intention-like tokens, often with
positional or bias-related patterns. Its semantics are unclear,   benign redundancy in critical regions such as the destination
                                                                  and landmarks, while devoting fewer resources to less infor-
mative areas like the sky or surrounding trees. Importantly,                    [9] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “Tri-perspective
these behaviors arise end-to-end without handcrafted priors                         view for vision-based 3d semantic occupancy prediction,” in CVPR,
                                                                                    2023.
or explicit supervision, showing that joint attention between                  [10] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic
scene and image tokens is sufficient to induce meaning-                             scenes,” in CVPR, 2023.
ful specialization. This emergent decomposition partially                      [11] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, “Vision transform-
                                                                                    ers need registers,” in ICLR, 2024.
demonstrates why the representation remains compact yet                        [12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
effective, enabling Flex to achieve high performance with                           Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
significantly fewer tokens.                                                         NeurIPS, 2017.
                                                                               [13] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-
                                                                                    image pre-training with frozen image encoders and large language
                          V. C ONCLUSION                                            models,” in ICML, 2023.
                                                                               [14] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao, “St-p3: End-
   In this work, we introduced Flex, a simple, efficient,                           to-end vision-based autonomous driving via spatial-temporal feature
and data-driven scene encoder for end-to-end autonomous                             learning,” in ECCV, 2022.
driving. Our core contribution is a lightweight, geometry-                     [15] S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang,
                                                                                    W. Liu, and X. Wang, “Vadv2: End-to-end vectorized autonomous
agnostic Transformer that jointly encodes multi-view, multi-                        driving via probabilistic planning,” arXiv preprint arXiv:2402.13243,
timestep images into a compact set of learnable scene tokens.                       2024.
Our extensive studies confirm that the joint, holistic nature                  [16] Y. Liu, T. Wang, X. Zhang, and J. Sun, “Petr: Position embedding
                                                                                    transformation for multi-view 3d object detection,” in ECCV, 2022.
of our encoder is critical to its success and that it is                       [17] Y. Liu, J. Yan, F. Jia, S. Li, A. Gao, T. Wang, and X. Zhang, “Petrv2:
flexible with respect to model size and sensor configuration                        A unified framework for 3d perception from multi-camera images,” in
and that our compressed scene tokens learn an emergent,                             ICCV, 2023.
                                                                               [18] T. Liang, H. Xie, K. Yu, Z. Xia, Z. Lin, Y. Wang, T. Tang, B. Wang,
semantically meaningful decomposition of the driving scene.                         and Z. Tang, “Bevfusion: A simple and robust lidar-camera fusion
Exploring the interpretability and the specialization of these                      framework,” in NeurIPS, 2022.
implicit scene tokens would be an intriguing future direction.                 [19] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, “Surroundocc:
Limitations. While Flex achieves strong efficiency–accuracy                         Multi-camera 3d occupancy prediction for autonomous driving,” in
                                                                                    ICCV, 2023.
trade-offs, we note that it inherits the biases and coverage                   [20] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. De Mello,
limitations of the proprietary driving dataset, which may                           O. Gallo, L. J. Guibas, J. Tremblay, S. Khamis et al., “Efficient
underrepresent rare long-tail scenarios. Broader evaluation                         geometry-aware 3d generative adversarial networks,” in CVPR, 2022.
                                                                               [21] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and
on more diverse and careful consideration of downstream                             A. Kanazawa, “K-planes: Explicit radiance fields in space, time, and
social impact, including safety and fairness across diverse                         appearance,” in CVPR, 2023.
geographies, remain important directions for future work.                      [22] T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
                                                                                    primitives with a multiresolution hash encoding,” ACM transactions
                             R EFERENCES                                            on graphics (TOG), 2022.
                                                                               [23] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che,
 [1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn,              D. Xu, S. Fidler, M. Pavone, and Y. Wang, “Emernerf: Emergent
     K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter,         spatial-temporal scene decomposition via self-supervision,” in ICLR,
     A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalash-          2024.
     nikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla,          [24] A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, and
     D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez,         J. Carreira, “Perceiver: General perception with iterative attention,”
     K. Pertsch, J. Quiambao, K. Rao, M. S. Ryoo, G. Salazar, P. R.                 in ICML, 2021.
     Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran,      [25] A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu,
     V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu,         D. Ding, S. Koppula, D. Zoran, A. Brock, E. Shelhamer, O. Hénaff,
     and B. Zitkovich, “Rt-1: Robotics transformer for real-world control           M. M. Botvinick, A. Zisserman, O. Vinyals, and J. Carreira, “Perceiver
     at scale,” in RSS, 2023.                                                       io: A general architecture for structured inputs & outputs,” in ICLR,
 [2] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna,                 2022.
     S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar,   [26] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson,
     B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn,         K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo:
     “Openvla: An open-source vision-language-action model,” in CoRL,               a visual language model for few-shot learning,” in NeurIPS, 2022.
     2024.                                                                     [27] M. S. Ryoo, A. Piergiovanni, A. Arnab, M. Dehghani, and A. An-
 [3] H. Gao, Z. Wang, Y. Li, K. Long, M. Yang, and Y. Shen, “A                      gelova, “Tokenlearner: What can 8 learned tokens do for images and
     survey for foundation models in autonomous driving,” arXiv preprint            videos?” in NeurIPS, 2021.
     arXiv:2402.01105, 2024.                                                   [28] Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.-C. Chen,
 [4] X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia,              “An image is worth 32 tokens for reconstruction and generation,” in
     X. Lang, and H. Zhao, “Drivevlm: The convergence of autonomous                 NeurIPS, 2024.
     driving and large vision-language models,” in CoRL, 2024.                 [29] H. Jin, H. Jiang, H. Tan, K. Zhang, S. Bi, T. Zhang, F. Luan,
 [5] B. Ivanovic, C. Saltori, Y. You, Y. Wang, W. Luo, and M. Pavone,               N. Snavely, and Z. Xu, “Lvsm: A large view synthesis model with
     “Efficient multi-camera tokenization with triplanes for end-to-end             minimal 3d inductive bias,” in ICLR, 2025.
     driving,” IEEE Robotics and Automation Letters, vol. 10, no. 11, pp.      [30] J. Yang, J. Huang, Y. Chen, Y. Wang, B. Li, Y. You, M. Igl, A. Sharma,
     11 713–11 720, 2025.                                                           P. Karkus, D. Xu, B. Ivanovic, Y. Wang, and M. Pavone, “Storm:
 [6] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai, “Bev-        Spatio-temporal reconstruction model for large-scale outdoor scenes,”
     former: learning bird’s-eye-view representation from lidar-camera via          in ICLR, 2025.
     spatiotemporal transformers,” IEEE Transactions on Pattern Analysis       [31] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec,
     and Machine Intelligence, 2024.                                                V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al.,
 [7] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du,               “Dinov2: Learning robust visual features without supervision,” Trans-
     T. Lin, W. Wang et al., “Planning-oriented autonomous driving,” in             actions on Machine Learning Research, 2023.
     CVPR, 2023.                                                               [32] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and
 [8] W. Tong, C. Sima, T. Wang, L. Chen, S. Wu, H. Deng, Y. Gu, L. Lu,              D. Novotny, “Vggt: Visual geometry grounded transformer,” in CVPR,
     P. Luo, D. Lin et al., “Scene as occupancy,” in ICCV, 2023.                    2025.
[33] W. Peebles and S. Xie, “Scalable diffusion models with transformers,”
     in ICCV, 2023.
[34] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li,
     D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang,
     J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J. Bai, J. He, J. Lin,
     K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang,
     P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan,
     T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang,
     X. Wei, X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu,
     Z. Cui, Z. Zhang, Z. Guo, and Z. Fan, “Qwen2 technical report,” arXiv
     preprint arXiv:2407.10671, 2024.
[35] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
     T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,
     J. Uszkoreit, and N. Houlsby, “An image is worth 16×16 words:
     Transformers for image recognition at scale,” in ICLR, 2021.
[36] J. Yang, K. Z. Luo, J. Li, C. Deng, L. Guibas, D. Krishnan, K. Q.
     Weinberger, Y. Tian, and Y. Wang, “Denoising vision transformers,”
     in ECCV, 2024.
