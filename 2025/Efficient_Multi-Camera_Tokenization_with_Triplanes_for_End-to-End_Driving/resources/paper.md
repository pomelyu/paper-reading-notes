                                            Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving

                                          Boris Ivanovic1        Cristiano Saltori1 Yurong You1 Yan Wang1 Wenjie Luo1                                    Marco Pavone1,2
                                                                           1                   2
                                                                             NVIDIA Research     Stanford University
                                               {bivanovic,csaltori,yurongy,yanwan,wenjiel,mpavone}@nvidia.com, pavone@stanford.edu



                                                                  Abstract                                       …




arXiv:2506.12251v2 [cs.CV] 21 Jul 2025
                                             Autoregressive Transformers are increasingly being de-
                                         ployed as end-to-end robot and autonomous vehicle (AV)
                                         policy architectures, owing to their scalability and poten-
                                         tial to leverage internet-scale pretraining for generaliza-
                                         tion. Accordingly, tokenizing sensor data efficiently is
                                         paramount to ensuring the real-time feasibility of such ar-
                                         chitectures on embedded hardware. To this end, we present
                                         an efficient triplane-based multi-camera tokenization strat-                                                …

                                         egy that leverages recent advances in 3D neural reconstruc-
                                         tion and rendering to produce sensor tokens that are ag-             Figure 1. Many Transformer-based AV policies employ patch-
                                         nostic to the number of input cameras and their resolu-              based image tokenization, whose token counts scale linearly with
                                         tion, while explicitly accounting for their geometry around          the number of cameras, complicating real-world AV deployment.
                                         an AV. Experiments on a large-scale AV dataset and state-            In contrast, our proposed triplane-based approach produces a fixed
                                         of-the-art neural simulator demonstrate that our approach            number of tokens, irrespective of the number of cameras or their
                                         yields significant savings over current image patch-based            resolution, significantly reducing online inference time.
                                         tokenization strategies, producing up to 72% fewer tokens,
                                         resulting in up to 50% faster policy inference while achiev-
                                         ing the same open-loop motion planning accuracy and im-              employ autoencoding architectures [13, 37, 44] or directly
                                         proved offroad rates in closed-loop driving simulations.             encode patches of pixels [11]. VLMs primarily employ Vi-
                                                                                                              sion Transformers (ViTs) [11] and adopt the latter, partition-
                                                                                                              ing images into patches that are encoded to form a 1D to-
                                         1. Introduction                                                      ken sequence. While this approach is practical, it produces
                                         Generalizing to novel environments remains a grand chal-             token counts that scale linearly with image resolution and
                                         lenge for robots and autonomous vehicles (AVs), requir-              the number of cameras [46]. To obtain a 360-degree view
                                         ing advanced reasoning capabilities to robustly handle un-           of their surroundings, AVs often use 6 to 10 cameras, the
                                         seen scenarios. Accordingly, coinciding with the rapid ad-           patch-based tokenization of which would yield thousands
                                         vancement of token-based autoregressive (AR) Transform-              of tokens per timestep, precluding real-time inference.
                                         ers [45] such as LLMs and VLMs, there has been a signif-                 Contributions. To address this, we present a novel
                                         icant growth in research aiming to endow robots and AVs              multi-camera image tokenization scheme (Fig. 1) based on
                                         with general reasoning capabilities by deploying internet-           triplanes, a recent advancement in 3D neural reconstruc-
                                         pretrained foundation models on-vehicle. However, such               tion and rendering, that produces tokens in a resolution-
                                         architectures oft contain billions of parameters, complicat-         agnostic, camera-number-agnostic, and geometrically-
                                         ing their real-time deployment in embedded systems [16].             aware manner (Sec. 3). Experiments on a large-scale dataset
                                            Many approaches are currently being investigated to re-           consisting of 20,000 hours of driving data from 25 coun-
                                         duce model size and inference time, including model com-             tries demonstrate the efficacy of our approach over patch-
                                         pression [62] and efficient sensor tokenization strategies. In       based tokenization, producing up to 72% fewer tokens, re-
                                         particular, image tokenizers aim to represent images with as         sulting in up to 50% faster inference while achieving the
                                         few latent embeddings (i.e., “tokens”) as possible. Current          same open-loop motion planning accuracy and improved
                                         methods largely focus on representing single images and              offroad rates in closed-loop driving simulations (Sec. 4).


                                                                                                          1
2. Related Work                                                       and VQ-GANs into a unified framework that further pares
                                                                      down the number of tokens needed to as low as 32. More
End-to-End Driving with Tokens. The first approaches                  broadly, while autoencoders can represent images parsimo-
applying token-based Transformer [45] architectures to                niously, they require a fixed input resolution and focus on
end-to-end (E2E) driving primarily focused on connect-                single images (neglecting camera geometries and scaling
ing previously-existing modules (e.g., detection, prediction,         linearly with camera count and resolution).
planning) along a differentiable pathway, treating their out-
                                                                          More recent works explore alternative image represen-
puts as queries, keys, and values into subsequent mod-
                                                                      tations (e.g., tokenizing the frequency spectrum [14]) and
ules. Initial works UniAD [23] and VAD [27] encode
                                                                      adaptive methods that allocate varying token amounts to
multi-camera observations into a birds-eye view (BEV) fea-
                                                                      images [12, 55]. Such advancements are exciting, and or-
ture grid, where each cell is treated as a token and pro-
                                                                      thogonal to our approach since it can be paired with any
cessed through a cascade of task-specific Transformer de-
                                                                      backbone image encoder or online token reducer.
coders (e.g., for object tracking, map segmentation, and mo-
                                                                          Volumetric Latent Representations. Accounting for
tion prediction). PARA-Drive [51] further tokenizes BEV
                                                                      multi-camera geometry is a critical aspect of the AV do-
feature grid cells into parallel multi-task decoding heads
                                                                      main, due to both the number of cameras and their orienta-
through a shared Transformer backbone. Each of these
                                                                      tion (pointing outwards). To this end, there have been many
works are trained from scratch on AV-specific data and all
                                                                      volumetric latent representations proposed for the neural re-
model outputs are supervised with ground-truth (GT) labels.
                                                                      construction and rendering of 3D environments that can also
    Inspired by progress in language modeling, a parallel
                                                                      be used for tokenization.
line of work aims to develop generalist embodied agents
that directly use an AR Transformer [45] to process in-                   Explicit representations like voxels and 3D Gaussian
put text, actions, and sensor data into downstream task-              Splatting [29] can be produced in a feedforward, differ-
specific tokens. RT-1 [2], Gato [35], RT-2 [3], RT-X [9], and         entiable manner from input images, but they produce too
VIMA [28] broadly introduce, extend, and deploy vision-               many elements (ie., voxels and Gaussians) for direct tok-
language-action models (VLAMs) onto multiple robotic                  enization. These can be alleviated with sparse voxel hier-
tasks and embodiments. Initial works in the AV domain                 archies [47] or Gaussian merging and pruning strategies,
focused on AR video generation [22] and predicting text               but they still result in thousands of tokens per scene. On
explanations alongside driving actions [49, 50, 54] from              the other hand, fully-implicit approaches such as NeRF [31]
a front-facing camera, tokenizing images with the patch-              and SDF [34, 39] require little memory, but lack the ability
based embedding scheme from ViT [11].                                 to be produced from input images in a feedforward manner.
    Multiple cameras have recently been incorporated in                   Hybrid explicit-implicit approaches such as K-planes [5,
VLM-based E2E AV models such as DriveVLM [41], Om-                    15] lie in the middle, factorizing time and space into K fea-
niDrive [48], and EMMA [26]. DriveVLM adopts the ViT-                 ture grids, with K = 3 (triplanes) being the most common
based SigLIP [60] to tokenize images, OmniDrive develops              choice [6]. Their main advantage is efficiency; the bulk of
a sparse 3D query architecture to encode multiple cameras,            modeling capacity is shifted to explicit features, allowing
and EMMA feeds camera images into Gemini Nano-1 [17].                 for a smaller feature decoder at rendering time without los-
However, these approaches require a separate token reduc-             ing expressiveness [6]. Accordingly, triplanes have been
tion model (e.g., LDPNetv2 [8]) to reduce inference times,            widely used to represent faces [6, 42, 43], objects [21, 36],
auxiliary training signals that are costly to obtain, or pro-         indoor environments [18], and, more recently, urban driving
duce a large number of tokens (500+) per image. In con-               scenes [24, 25, 53]. While these works mainly use triplanes
trast, our approach is fully self-supervised (it can optionally       for neural rendering or occupancy modeling, we focus on
use auxiliary signals if provided) and produces much fewer            their tokenization for use by an AR Transformer [45].
tokens (Sec. 4).
    Latent Image Representations. While current ap-                   3. Multi-Camera Tokenization with Triplanes
proaches mostly employ ViTs [11], another common to-
                                                                      3.1. Background: Triplanes
kenization strategy is to use the latent space of a pre-
trained image autoencoder. VAE [37], VQ-VAE [44], and                 Triplanes are a volumetric latent representation com-
VQ-GAN [13] have demonstrated that encoder-decoder ar-                prised of three axis-aligned orthogonal feature planes
chitectures can successfully compresses high-dimensional              Pxy , Pxz , Pyz ∈ RSi ×Sj ×Df (Fig. 2), where Si × Sj de-
images into a low-dimensional (optionally discrete) latent            notes each plane’s spatial dimensions, with each grid cell
space, which can be used as tokens for E2E driving mod-               corresponding to a si × sj m2 region of space, and Df is
els. MoST [32] adopts this approach, tokenizing each im-              the feature dimension. In this work, triplanes serve as a
age into 256 tokens using a VQ-GAN [13] encoder (along-               resolution-agnostic, camera-count-agnostic, and geometry-
side other ViT-based models). TiTok [59] combines ViTs                aware representation of multi-camera images, enabling ef-

                                                                  2
                                                                                 Offline Training (Section 3.3)



                                              A                                          fyz   f
                                              T                                                          fxz
                                                                                                                    Volumetric            Reconstruction
                                              T                                                    fxy
                                              N                                                                     Rendering                Losses
                                                                                                                                      ^
                                                                                                                                      I                         I
                                          A
                                          T
                     ImageEnc             T                                                                    px
                                          N                                                                          py
                                                                                                                                      DAR
                                 fc           A
             Ic                               T                                                                           xy
                                              T
                                              N                                                                pz
                                                                                                                    px    xz                                    …

                    Triplane Creation (Section 3.2)                                                                            Patchify         Tokens
                                                                                                                                + MLP                      t1,t2 ,…,tL
                                                                                                               pz
                                                                                                                     py   yz
                                                                                 Online Tokenization (Section 3.4)


Figure 2. Our triplane-based multi-camera tokenization strategy. Images from multiple cameras are first encoded through ImageEnc,
producing features which form triplanes through a set of geometric per-image and cross-image attention operations. The model is trained
in a self-supervised manner, and yields triplanes that can be effectively tokenized for downstream use by an autoregressive Transformer.

ficient tokenization for use by AR Transformers. Our ap-               the inner and outer cells of the triplane, respectively, and
proach for doing so is illustrated in Fig. 2.                          Sin,x ∈ N is the number of inner grid cells.
3.2. Encoding Multi-Camera Images into Triplanes                       3.3. Scalable Training via Volumetric Rendering
At time t, we assume an AV obtains an image                            In contrast to prior works which leverage multiple complex
Ic ∈ RH×W ×DCh from each of its N cameras c ∈                          (e.g., Hessian-based [25]) self-supervised losses, our work
{C1 , . . . , CN }. To encode these N images into a triplane           is trained to minimize only two pixel-wise reconstruction
{Pxy , Pxz , Pyz }, we first featurize images with a backbone          losses. Specifically,
encoder network
                                                                                    ˆ = λLPIPS LPIPS(I, I)
                                                                               L(I, I)                  ˆ + λ1 ∥I − I∥
                                                                                                                    ˆ1                                              (2)
         fc = ImageEnc(Ic ) ∀ c ∈ {C1 , . . . , CN },        (1)
                                                                       where LPIPS is the learned perceptual image patch similar-
where fc ∈ RHf ×Wf ×Df and Hf × Wf denotes the fea-                    ity metric [61], λ are scalar weighting factors, ∥ · ∥1 is the ℓ1
tures’ spatial size. Then, similar to TPVFormer [24], a grid           norm, Iˆ is the predicted (rendered) image from the triplane,
of 3D query points q ∈ RSx ×Sy ×Sz ×Df (where Sx , Sy , Sz             and I denotes the ground truth (GT) image from data.
are the number of triplane grid cells per spatial dimension)               Images Iˆ are rendered from triplanes via ray sampling
attend to the image features through a series of per-image             and aggregation. Triplanes are queried for 3D positions
and cross-image deformable attention operations, leverag-              x ∈ R3 by first projecting x onto each of the three fea-
ing each camera’s intrinsic and extrinsic parameters for 3D-           ture planes, retrieving the corresponding feature vectors
to-2D projections. Internally, a sinusoidal positional encod-          fxy , fxz , fyz via bilinear interpolation, and aggregating the
ing [45] is used to represent triplane grid locations. Finally,        three feature vectors into f via elementwise product. A
the updated queries are averaged along each spatial dimen-             lightweight MLP then decodes the 3D features f into color
sion to produce the desired triplane {Pxy , Pxz , Pyz }.               and density, which are rendered into RGB images using vol-
   Crucially, since the triplane sizes Sx , Sy , Sz are fixed,         umetric rendering [31].
this process decouples the number of cameras N and their                   Notably, in contrast to current autoencoder-based meth-
resolution H × W from the resulting number of tokens.                  ods [7, 13, 58], our approach does not use any GAN losses
   Finally, as driving scenes are unbounded, we employ a               (whose training process is particularly sensitive). As we
nonlinear scene parametrization as in Mip-NeRF 360 [1]                 will show in Sec. 4, even without these losses, our triplane-
to model far-away objects without increasing triplane sizes.           based driving model matches/exceeds the driving perfor-
Specifically, using the x axis as an example, a grid co-               mance of autoencoder-based approaches.
ordinate pg,x ∈ R is mapped to ego-relative coordinates
pego,x ∈ R with a symmetric bilinear grid resolution:                  3.4. Tokenizing Triplanes for Downstream Use
                                                                      With a trained model in hand, multi-camera input images
          Ro,x (pg,x + Sin,x ) − Ri,x Sin,x , if pg,x < −Sin,x
                                                                      can be converted into triplanes in a feedforward manner.
pego,x = Ri,x pg,x ,                 if − Sin,x ≤ pg,x ≤ Sin,x         Once a triplane is obtained, it can then be tokenized and
                                                                       ingested by a downstream AR Transformer.
          
            Ro,x (pg,x − Sin,x ) + Ri,x Sin,x , if pg,x > Sin,x
          
                                                                          Continuous Tokens. Triplanes can be converted into
where Ri,x , Ro,x ∈ R denote the resolutions (m / cell) of             a 1D token sequence in multiple ways. In this work, we


                                                                   3
                                                                          Choice of ImageEnc in Eq. (1)           PSNR ↑     SSIM ↑

                                                             45m
                                                                          DINOv2-small [33] (4 cameras)           29.15        0.85
                                                                          ResNet50 [19] (4 cameras)               29.58        0.86
                                                             15m
                                                                          DINOv2-small [33] (7 cameras)           28.94        0.84
                                                Sin,z
                                                                          ResNet50 [19] (7 cameras)               29.34        0.85
                                    Sin,x = Sin,y
                                                             0m
                                                                        Table 1. Irrespective of the choice of image encoder or number of
-180m
        -80m                                                            cameras, triplanes can accurately model outdoor driving scenes,
                   0m                                        -3m
                               +80m +180m                               matching state-of-the-art neural reconstruction performance.

Figure 3. We employ a bilinear triplane resolution, representing        Sec. 3.2, we employ a symmetric bilinear grid resolution
regions outside of Sin,x , Sin,y , Sin,z with fewer grid cells.         with Sin,x = Sin,y = 36 cells and, since we cannot see un-
                                                                        der ground, the z-axis asymmetrically represents [−3, 15]m
patchify them as in ViT [11], splitting triplanes into patches          with 36 cells and (15, 45]m with the remaining 12 cells.
of features and encoding them into a set of feature vectors                 We use the 22M-parameter DINOv2-small [33] model
with the desired number of dimensions for downstream use.               for ImageEnc in Eq. (1) and two sets of per-image and
   Formally, each feature plane Pij ∈ RSi ×Sj ×Df is con-               cross-image attention layers (4 total attention layers, total-
verted to a token sequence {tℓ } by first reshaping it into a           ing 6.5M parameters) to convert image features to triplanes.
set of patches Pij′ ∈ RSi /pi ×Sj /pj ×Df pi pj . Then, a single-           AR Transformer. An LLM-like backbone serves as our
layer MLP converts the resulting feature dimension Df pi pj             representative AR Transformer model [52]. It takes as in-
into DAR , producing a set of tokens {tℓ } ∈ RLij ×DAR where            put the above sensor tokens as well as past ego trajectory
          S S
Lij = pii pjj is the number of tokens needed to represent               information to produce future ego-vehicle positions. Future
plane Pij and DAR is the feature dimension of the down-                 trajectories are represented as discrete tokens and decoded
stream AR Transformer. Performing this for each plane re-               to positions at 10 Hz.
sults in an overall token sequence {tℓ } ∈ RL×DAR where                     Baselines. Since image autoencoders and ViTs are the
L = Lxy + Lxz + Lyz .                                                   most commonly-used image tokenizers (Sec. 2), we com-
   Discrete Tokens. While the above produces continuous                 pare our work to VQGAN [13] and DINOv2-small [33] as
tokens, discrete tokens can also be output. In particular, a            representative models.
Finite Scalar Quantization (FSQ) [30] layer can be added                    Training. We employ a multi-stage training pipeline.
after triplane creation to learn a discrete codebook and pro-           Tokenizers are first pre-trained to model driving data on our
duce discrete tokens. However, in initial experiments, we               internal dataset (except for DINOv2 as it is already internet-
found that discrete tokens underperformed continuous to-                pretrained), followed by training the randomly-initialized
kens. Thus, only continuous tokens are used in this work.               backbone on the resulting tokens. The overall tokenizer-
                                                                        LLM combination is trained to minimize a next-trajectory-
4. Experiments                                                          token prediction loss. We set λLPIPS = λ1 = 0.5 in Eq. (2).
                                                                        Additional training details can be found in [52].
Dataset. To train and evaluate driving performance, we                      Metrics. To evaluate triplane quality, we compare the
leverage a large internal dataset consisting of 20,000 hours            reconstructed images to GT images using peak signal-to-
of driving data from multiple ego-vehicles in 1700+ cities              noise ratio (PSNR) and structural similarity index measure
and 25 countries. Accordingly, it contains a variety of driv-           (SSIM). To evaluate the combined tokenizer-LLM open-
ing scenarios including highway and urban driving, mul-                 loop driving performance, we leverage the commonly-used
tiple weather conditions, day and night times, and vary-                minADE6 metric (denoting the minimum average displace-
ing amounts of traffic, with a geographically-separate 90%              ment error among 6 sampled trajectories), formulated as
trainval and 10% test split. Each ego-vehicle has 7 cam-
eras which we downsample to a per-camera resolution of                                                        T
                                                                                                           1 X (t)
H × W = 320 × 512 with 10 Hz observation frequency.                         minADEk (ô, o) = min                ∥ôk − o(t) ∥2       (3)
                                                                                                 k=1,...,6 T
Fig. 4 visualizes an example from the dataset.                                                               t=1
    Triplane Model. In all experiments, we set Sx = Sy =
96, Sz = 48, and Df = 192. To account for driving in
                                                                        4.1. Triplane Representation Quality
both high-speed freeways and slower urban streets, each tri-
plane metrically represents 180m ahead, behind, left, and               To evaluate whether triplanes can represent driving scenar-
right of the ego-vehicle, as well as 45m above and 3m be-               ios collected from outward-facing cameras, we first evalu-
low the ego-vehicle, visualized in Fig. 3. As described in              ate their reonstruction performance after the first stage of


                                                                    4
                                                                                                                                   yz


                                                   GT Images




                                                                                                                           xy           xz
                                              Reconstructed Images

Figure 4. Triplanes can accurately represent environments observed by outward-facing cameras (top), producing accurate reconstructions
(below) and semantically-meaningful features after only being trained with self-supervised pixel reconstruction losses (right, a PCA
visualization of triplane features).

                        Tokens per      minADE6 (m) ↓                      nuScenes E2E Planning              Traj L2 (m) ↓
  Image Tokenizer        Image ↓       @1s @3s @5s                                                       @1s @2s @3s Ave1,2,3s

  VQGANEnc                  160        0.08    0.33    0.74                UniAD [23]                    0.48   0.89   1.47      0.95
  DINOv2-small              160        0.08    0.32    0.69                VAD-Base [27]                 0.41   0.86   1.46      0.91
                                                                           PARA-Drive [51]               0.26   0.59   1.12      0.66
  Ours (8-8-8)               45        0.08    0.32    0.72                TOKEN 7B [40]                 0.26   0.70   1.46      0.81
  Ours (4-6-6)              104        0.08    0.31    0.67                DiMA 7B (VAD-Tiny) [20]       0.20   0.53   1.10      0.61
                                                                           DiMA 7B (VAD-Base) [20]       0.18   0.50   1.02      0.57
Table 2. When paired with a 1B AR backbone, our triplane-
based tokenizer achieves similar open-loop motion planning per-            DINOv2-small + 1B             0.34   0.69   1.17      0.73
formance as baseline tokenizers, while producing significantly             Ours (8-8-8) + 1B             0.31   0.63   1.10      0.68
fewer sensor tokens. Patch sizes are denoted as (px − py − pz ).           Ours (4-6-6) + 1B             0.29   0.62   1.08      0.66
                                                                         Table 3. Our approach is competitive with state-of-the-art ap-
training. Fig. 4 visualizes triplane-rendered images, GT im-             proaches, even with a much smaller backbone (1B) and much
ages, and a 3-color triplane feature visualization (obtained             fewer input tokens (35% to 72% less). Triplane patchification
with PCA). As can be seen in Tab. 1, triplanes are able to               sizes are denoted with (px − py − pz ).
represent both nearby and far-away elements of the scene,
achieving 28.94 dB PSNR and 0.84 SSIM when modeling                      first, (px , py , pz ) = (4, 6, 6), yields 416 tokens per timestep
all 7 cameras, comparable to state-of-the-art neural render-             (104 tokens per image, 35% fewer than the baselines) and
ing performance [56]. If modeling fewer cameras (e.g., only              matches the inference runtime of the baseline approaches.
the front 4 cameras), reconstruction performance increases               The second employs more aggressive triplane patchifica-
to 29.15 dB PSNR and 0.85 SSIM, since more modeling                      tion, with (px , py , pz ) = (8, 8, 8), resulting in 180 tokens
capacity can be used per-image.                                          per timestep (45 tokens per image, 72% fewer than the base-
    Alternative Choices for ImageEnc. Our work can op-                   lines) and a significantly faster runtime. As can be seen in
erate with any image backbone that produces 2D features.                 Tab. 2, the backbone trained with our approach performs
To demonstrate this, Tab. 1 also shows triplane reconstruc-              similarly to the baselines across all timescales while requir-
tion performance with a CNN-based image encoder (an                      ing significantly fewer tokens per image.
ImageNet-pretrained ResNet50 [19]). As can be seen, this
                                                                             In Tab. 3, we provide a comparison of our approach and
choice yields even better PSNR and SSIM values, although
                                                                         the DINOv2 baseline to a variety of other state-of-the-art
differences in model optimization result in an increase in in-
                                                                         (SotA) approaches for E2E planning on the nuScenes [4]
ference runtime compared to DINOv2-small. Accordingly,
                                                                         validation set. We follow the standardized evaluation setup
we use DINOv2-small for ImageEnc in Eq. (1) in all subse-
                                                                         from PARA-Drive [51]: predicting 3s of future ego-motion
quent experiments.
                                                                         from 2s of past image frames, training only on the nuScenes
                                                                         train set, and using the same displacement metric defini-
4.2. Open-Loop Driving Performance
                                                                         tions. As can be seen, our work is competitive with SotA
To evaluate driving performance with model sizes that are                approaches. Further, our approach performs similarly to
suitable for deployment, we train a 1B-parameter LLM-                    works like DiMA [20] and TOKEN [40] (which use much
like model (DAR = 2048) to drive from the 4 front-facing                 larger 7B LLM backbones compared to our 1B), while run-
cameras, with their past 6 frames as scene context (total-               ning much faster due to our reduced input token counts.
ing 24 frames of input). In addition to the VQGAN and                    Importantly, both the (4-6-6) and (8-8-8) versions of our
DINOv2-small tokenizer baselines, we evaluate two patch                  approach outperform the baseline DINOv2-based tokenizer.
size px , py , pz ablations for our triplane tokenizer. The              Further open-loop evaluations can be found in the appendix.


                                                                     5
                                  (a) Inference time scaling results when taking the 4 front-facing cameras as input.




                                        (b) Inference time scaling results when taking all 7 cameras as input.
Figure 5. Our triplane-based multi-camera tokenization approach scales much more favorably than baseline approaches as the number of
cameras and number of context frames per camera increases. This trend holds across multiple autoregressive Transformer sizes.

                                                                               4.3. Inference Time Scaling
                                         yz
                                                                               Our main hypothesis for using a triplane-based multi-
                                                                               camera tokenization strategy is that, while adding any layers
                                                                               to create an intermediate representation will increase tok-
                                                                               enization runtime, the corresponding decrease in sensor to-
                                                                               kens will lead to greater savings in backbone runtime, re-
                                                                               sulting in an overall inference time decrease since billion-
                                                                               parameter Transformers are virtually always the main infer-
                                                                               ence bottleneck in such AV stacks.
                          xy                 xz                                    To confirm this hypothesis, we profile the runtimes of
Figure 6. If modeling cameras that face a certain direction (e.g., all         combined tokenizer-backbone models with a variety of
front-facing cameras, as in this figure), the number of tokens can             backbone model sizes, number of historical frames, and
be further reduced by removing unused triplane regions, indicated              numbers of cameras. In addition to the 1B-parameter ver-
with an “x”. Depending on the chosen patch sizes px , py , pz , this           sion above, we also profile the runtime of 3B-parameter
can further reduce token counts by up to 40%.                                  and 7B-parameter backbones. Such model sizes were pre-
                                                                               viously considered for AV deployment, and even success-
                                                                               fully demonstrated on real AVs in works such as [41]. We
    Halfplane Token Reduction. Note that, in this partic-                      report the average inference times of different parts of the
ular setup (modeling only the front-facing cameras), one                       combined model over 100 forward passes on a single A100
can further reduce the number of tokens produced by our                        80GB GPU in Fig. 5. We omit error bars as there is very
triplane-based approach by removing half of the xy and xz                      little variation across runs (95% confidence intervals are
planes (they model space behind the car, and are thus un-                      within 0.5ms of means).
used, as visualized in Fig. 6). The results in Tab. 2 use                          As can be seen in Fig. 5, the majority of overall inference
this token-reduction technique, as do all subsequent exper-                    time is spent in the backbone (blue and green bars) and we
iments modeling only the 4 front-facing cameras. Depend-                       observe the expected linear scaling of inference time as the
ing on the chosen patch sizes, this can further reduce token                   number of cameras increases (from the top row to the bot-
counts by 30-40% (specifically, 37.5% for (px , py , pz ) =                    tom row) and as the number of context frames per camera
(4, 6, 6) and 40.9% for (px , py , pz ) = (8, 8, 8)).                          increases (within each plot).


                                                                          6
                                                                                                                                            yz


                                                   GT Images




                                                                                                                                       xy        xz

                                             Reconstructed Images

Figure 7. Triplanes can faithfully represent higher resolution data without requiring any changes to triplane sizes. Since these cameras are
all front-facing, the rear half of the xy and xz planes are unused.


    Fig. 5 confirms that our method demonstrates the trade-                                       7-Camera Open-Loop Driving Performance
off described above: an increase in tokenization time (larger                                           Relative to 4-Camera Setup
                                                                                            20%
purple bar) yields a much faster AR prefill stage (much                                               DINOv2 Baseline
smaller blue bar), with additional gains if using a more ag-                                          Triplane (Ours)
gressive patchification scheme (increasing px , py , pz ). This                             15%


                                                                         Relative minADE6
results in our method achieving significant inference time
reductions compared to baselines, often halving model run-                                  10%
time with larger context sizes and number of cameras.
    If targeting a 3 Hz inference frequency (e.g., as in [41]),
it is difficult to incorporate more than 2-4 frames of context                              5%
with model sizes above 1B parameters when using baseline
tokenization approaches. In contrast, our triplane-based to-                                0%
kenizer enables running a 7B-parameter model with 7 cam-                                          1                     3                        5
eras and 4 frames of context at 3 Hz!                                                                         Prediction Horizon (s)
    An additional benefit of triplane-based tokenization is              Figure 8. Our triplane model performs similarly with 7 cameras
that patchification can be modified after training, since it             and 4 cameras, whereas the DINOv2-small baseline struggles.
is executed after triplane creation, enabling easier tuning of
overall model performance without retraining the triplane.               strated in Tab. 1, we measure the motion planning perfor-
                                                                         mance of the 1B backbone with 7 cameras and compare it to
4.4. Resolution and Camera Count Agnosticism                             the 4-camera equivalent from above. Fig. 8 shows an inter-
Higher Resolutions. To validate that our approach can                    esting trend: When paired with our 7-camera triplane-based
seamlessly handle higher image resolutions, we resample                  approach, the 1B backbone is able to achieve performance
images from our dataset with a 3.6× higher resolution,                   within 2.5% of its 4-camera performance. However, when
H ′ × W ′ = 576 × 1024, and retrain our triplane-based                   paired with the DINOv2-small baseline, it yields 18% worse
approach with them. The resulting model achieves 28.00                   open-loop motion planning performance. We also observed
dB PSNR and 0.82 SSIM, which is lower than the lower-                    training instabilities with the 7-camera DINOv2-small base-
resolution results in Tab. 1. This is to be expected, how-               line, potentially due to the long sequence length brought by
ever, as we do not alter the triplane size, meaning 3.6×                 modeling 7 cameras each with 6 context frames (totaling 42
more pixels are being modeled by the same number of fea-                 frames of input).
tures. Fig. 7 confirms qualitatively that triplanes can faith-
fully represent higher resolution data, without requiring any            4.5. Performance in Closed-Loop Simulation
changes to triplane sizes.                                               Finally, it is well-known that open-loop driving perfor-
   To fully verify that the resulting tokens are still useful            mance does not necessarily correlate to closed-loop driving
for driving, we retrain the 1B backbone on the resulting to-             performance [10]. Accordingly, we additionally validate the
kens (using (px , py , pz ) = (4, 6, 6)) and observe identical           driving performance of the combined tokenizer-LLM mod-
performance to our previous (4, 6, 6) model in Tab. 2.                   els in a closed-loop simulation environment.
   Camera Count Agnosticism. To confirm our method’s                        We conduct closed-loop evaluation using a state-of-
agnosticism to the number of cameras, as initially demon-                the-art in-house neural reconstruction (NR)-based simula-


                                                                     7
Figure 9. In a challenging closed-loop simulation of a construction scene, the baseline DINOv2-based model (left) stops in the middle of
the road and does not start driving again, whereas our triplane-based model (right) has no difficulty nudging around the parked excavator
and driving behind the cement mixer. On the front camera views are the predicted ego-vehicle trajectories (selected driving trajectory in
orange, with the other 5 predicted trajectories in blue).


                       Tokens per        Offroad Rate ↓                  tokens, yielding significant inference time reductions while
 Image Tokenizer        Image ↓        @6s @12s @20s                     achieving comparable open-loop motion planning perfor-
                                                                         mance and even slightly improved offroad rates in closed-
 DINOv2-small              160         2.7%     2.7%      4.0%           loop simulations.
 Ours (8-8-8)               45         1.4%     1.4%      2.7%              A core limitation of the proposed approach is that it fo-
                                                                         cuses on per-timestep triplane modeling, leaving across-
Table 4. When paired with a 1B AR backbone, our triplane-
                                                                         timestep modeling strategies to future work. Other excit-
based tokenizer achieves slightly better closed-loop driving per-
formance as the DINOv2-small baseline, while producing signifi-
                                                                         ing areas of future work include further triplane size re-
cantly fewer sensor tokens. Patch sizes are denoted (px −py −pz ).       duction and feature dimension compression (to limit mem-
                                                                         ory requirements), and further reductions in tokenizer la-
tor, capable of reconstructing logged driving events and                 tency (potentially leveraging model optimization tools such
rendering novel views if the ego-vehicle departs from its                as pruning and optimized runtime environments, e.g., Ten-
originally-logged trajectory [52]. We place the combined                 sorRT), with the ultimate goal of deploying triplane-based
tokenizer-LLM models within 75 challenging scenarios,                    multi-camera tokenization on-vehicle.
mined specifically due to the amount of ego-agent and
agent-agent interactions within them, each of which are                  Acknowledgments. We thank Jiawei Yang, Yue
20s-long. Fig. 9 shows one such rollout with the currently-              Wang, Seung Wook Kim, Sérgio Agostinho, Am-
predicted trajectories superimposed over the image. Within               rita Mazumdar, Omer Shapira, and Shalini De Mello
                                                                         for many helpful technical discussions and guidance,
these scenarios, we have access to high-quality GT map la-
                                                                         as well as Michael Watson, Maximilian Igl, Michal
bels, and so we compare models by their offroad rates (in                Tyszkiewicz, Aaron Smith, Peter Karkus, and Zan Goj-
what proportion of closed-loop rollouts did the ego-vehicle              cic for their support in executing closed-loop simulations.
leave the road). Owing to the significant speedups achieved
by the (px , py , pz ) = (8, 8, 8) version of our triplane-LLM
combined model, we evaluate it against the DINOv2-LLM                    References
combined baseline. Tab. 4 shows that our approach achieves
                                                                          [1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
a slightly lower offroad rate compared to the baseline.
                                                                              Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded
                                                                              anti-aliased neural radiance fields. IEEE Conf. on Computer
5. Limitations and Conclusion                                                 Vision and Pattern Recognition, 2022. 3
                                                                          [2] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen
In conclusion, we present a novel multi-camera image tok-
                                                                              Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakr-
enization scheme based on triplanes that can produce sensor                   ishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Ju-
tokens in a resolution-agnostic, camera-number-agnostic,                      lian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally
and geometrically-aware manner. In comparison to autoen-                      Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalash-
coder and ViT-based baselines, our triplane-based approach                    nikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey
is able to represent multi-camera images with much fewer                      Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor


                                                                     8
     Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta,            [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
     Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka                    Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
     Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin                 Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
     Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clay-               vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
     ton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan                 worth 16x16 words: Transformers for image recognition at
     Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu,                 scale. In Int. Conf. on Learning Representations, 2020. 1, 2,
     and Brianna Zitkovich. RT-1: Robotics transformer for real-              4
     world control at scale. arXiv preprint arXiv:2212.06817,            [12] Shivam Duggal, Phillip Isola, Antonio Torralba, and
     2022. 2                                                                  William T. Freeman. How many tokens is an image worth?
 [3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen                     In Int. Conf. on Learning Representations, 2025. 2
     Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding,              [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming
     Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence,                transformers for high-resolution image synthesis. In IEEE
     Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakr-                  Conf. on Computer Vision and Pattern Recognition, 2021. 1,
     ishnan, Kehang Han, Karol Hausman, Alexander Herzog,                     2, 3, 4
     Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan           [14] Carlos Esteves, Mohammed Suhail, and Ameesh Makadia.
     Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal,                   Spectral image tokenizer. arXiv preprint arXiv:2412.09607,
     Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu,                   2024. 2
     Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kan-               [15] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
     ishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar,                 Warburg, Benjamin Recht, and Angjoo Kanazawa. K-
     Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait                  Planes: Explicit radiance fields in space, time, and appear-
     Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan                 ance. In IEEE Conf. on Computer Vision and Pattern Recog-
     Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin                nition, 2023. 2
     Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu,               [16] Haoxiang Gao, Zhongruo Wang, Yaqian Li, Kaiwen Long,
     and Brianna Zitkovich. RT-2: Vision-language-action mod-                 Ming Yang, and Yiqing Shen. A survey for foundation mod-
     els transfer web knowledge to robotic control. arXiv preprint            els in autonomous driving. arXiv preprint arXiv:2402.01105,
     arXiv:2307.15818, 2023. 2                                                2024. 1
 [4] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora,           [17] Gemini Team, Google. Gemini: A family of highly capable
     Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Gi-                 multimodal models. arXiv preprint arXiv:2312.11805, 2023.
     ancarlo Baldan, and Oscar Beijbom. nuScenes: A multi-                    2
     modal dataset for autonomous driving. In IEEE Conf. on              [18] Yan Han, Li Yang, Wu Zhennan, Chen Shenzhou, Sun Weix-
     Computer Vision and Pattern Recognition, 2020. 5                         uan, Shang Taizhang, Liu Weizhe, Chen Tian, Dai Xiaqiang,
 [5] Ang Cao and Justin Johnson. HexPlane: A fast representa-                 Ma Chao, Li Hongdong, and Ji Pan. Frankenstein: Gener-
     tion for dynamic scenes. In IEEE Conf. on Computer Vision                ating semantic-compositional 3d scenes in one tri-plane. In
     and Pattern Recognition, 2023. 2                                         Proc. of SIGGRAPH Asia, 2024. 2
 [6] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki                  [19] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
     Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo,                      for image recognition. In IEEE Conf. on Computer Vision
     Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero                   and Pattern Recognition, 2016. 4, 5
     Karras, and Gordon Wetzstein. Efficient geometry-aware 3D           [20] Deepti Hegde, Rajeev Yasarla, Hong Cai, Shizhong Han,
     generative adversarial networks. In IEEE Conf. on Computer               Apratim Bhattacharyya, Shweta Mahajan, Litian Liu,
     Vision and Pattern Recognition, 2022. 2                                  Risheek Garrepalli, Vishal M. Patel, and Fatih Porikli. Dis-
 [7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and                           tilling multi-modal large language models for autonomous
     William T. Freeman. MaskGIT: Masked generative image                     driving. In IEEE Conf. on Computer Vision and Pattern
     transformer. In IEEE Conf. on Computer Vision and Pattern                Recognition, 2025. 5
     Recognition, 2022. 3                                                [21] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
 [8] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu,                     Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao
     Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang,                    Tan. LRM: Large reconstruction model for single image to
     Xiaolin Wei, et al. MobileVLM: A fast, reproducible and                  3d. In Int. Conf. on Learning Representations, 2024. 2
     strong vision language assistant for mobile devices. arXiv          [22] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez,
     preprint arXiv:2312.16886, 2023. 2                                       George Fedoseev, Alex Kendall, Jamie Shotton, and Gian-
 [9] Embodiment Collaboration. Open X-Embodiment: Robotic                     luca Corrado. GAIA-1: A generative world model for au-
     learning datasets and RT-X models.           arXiv preprint              tonomous driving. arXiv preprint arXiv:2309.17080, 2023.
     arXiv:2310.08864, 2023. 2                                                2
[10] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo                [23] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima,
     Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor                        Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai
     Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas                     Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu
     Geiger, and Kashyap Chitta. NAVSIM: Data-driven non-                     Qiao, and Hongyang Li. Planning-oriented autonomous driv-
     reactive autonomous vehicle simulation and benchmarking.                 ing. In IEEE Conf. on Computer Vision and Pattern Recog-
     In Conf. on Neural Information Processing Systems, 2024. 7               nition, 2023. 2, 5


                                                                     9
[24] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou,              [35] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gómez
     and Jiwen Lu. Tri-perspective view for vision-based 3d se-               Colmenarejo, Alexander Novikov, Gabriel Barth-maron,
     mantic occupancy prediction. In IEEE Conf. on Computer                   Mai Giménez, Yury Sulsky, Jackie Kay, Jost Tobias Sprin-
     Vision and Pattern Recognition, 2023. 2, 3                               genberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Ed-
[25] Yuanhui Huang, Wenzhao Zheng, Borui Zhang, Jie Zhou,                     wards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol
     and Jiwen Lu. SelfOcc: Self-supervised vision-based 3d oc-               Vinyals, Mahyar Bordbar, and Nando de Freitas. A gener-
     cupancy prediction. In IEEE Conf. on Computer Vision and                 alist agent. Transactions on Machine Learning Research,
     Pattern Recognition, 2024. 2, 3                                          2022. 2
[26] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung,             [36] J. Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner,
     Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington,              Jiajun Wu, and Gordon Wetzstein. 3D neural field generation
     Benjamin Sapp, Yin Zhou, James Guo, Dragomir Anguelov,                   using triplane diffusion. In IEEE Conf. on Computer Vision
     and Mingxing Tan. EMMA: End-to-end multimodal model                      and Pattern Recognition, 2023. 2
     for autonomous driving. arXiv preprint arXiv:2410.23262,            [37] K. Sohn, H. Lee, and X Yan. Learning structured output
     2024. 2                                                                  representation using deep conditional generative models. In
[27] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie                    Conf. on Neural Information Processing Systems, 2015. 1, 2
     Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang,              [38] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
     and Xinggang Wang. VAD: Vectorized scene representation                  Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
     for efficient autonomous driving. In IEEE Int. Conf. on Com-             Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han,
     puter Vision, 2023. 2, 5                                                 Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Et-
                                                                              tinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang,
[28] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang,
                                                                              Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov.
     Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandku-
                                                                              Scalability in perception for autonomous driving: Waymo
     mar, Yuke Zhu, and Linxi Fan. VIMA: General robot manip-
                                                                              open dataset. In IEEE Conf. on Computer Vision and Pattern
     ulation with multimodal prompts. In Int. Conf. on Machine
                                                                              Recognition, 2020. 1
     Learning, 2023. 2
                                                                         [39] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten
[29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler,
                                                                              Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacob-
     and George Drettakis. 3d gaussian splatting for real-time
                                                                              son, Morgan McGuire, and Sanja Fidler. Neural geometric
     radiance field rendering. Proc. of SIGGRAPH, 42(4), 2023.
                                                                              level of detail: Real-time rendering with implicit 3D shapes.
     2
                                                                              In IEEE Conf. on Computer Vision and Pattern Recognition,
[30] Fabian Mentzer, David Minnen, Eirikur Agustsson, and                     2021. 2
     Michael Tschannen. Finite scalar quantization: VQ-VAE               [40] Ran Tian, Boyi Li, Xinshuo Weng, Yuxiao Chen, Edward
     made simple. In Int. Conf. on Learning Representations,                  Schmerling, Yue Wang, Boris Ivanovic, and Marco Pavone.
     2024. 4                                                                  Tokenize the world into object-level knowledge to address
[31] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,                    long-tail events in autonomous driving. In Conf. on Robot
     Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:                  Learning, 2024. 5
     Representing scenes as neural radiance fields for view syn-         [41] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang,
     thesis. In European Conf. on Computer Vision, 2020. 2, 3                 Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and
[32] Norman Mu, Jingwei Ji, Zhenpei Yang, Nate Harada, Hao-                   Hang Zhao. DriveVLM: The convergence of autonomous
     tian Tang, Kan Chen, Charles R. Qi, Runzhou Ge, Kratarth                 driving and large vision-language models. In Conf. on Robot
     Goel, Zoey Yang, Scott Ettinger, Rami Al-Rfou, Dragomir                  Learning, 2024. 2, 6, 7
     Anguelov, and Yin Zhou. MoST: Multi-modality scene tok-             [42] Alex Trevithick, Matthew Chan, Michael Stengel, Eric R.
     enization for motion prediction. In IEEE Conf. on Computer               Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan
     Vision and Pattern Recognition, 2024. 2                                  Chandraker, Ravi Ramamoorthi, and Koki Nagano. Real-
[33] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V.                 time radiance fields for single-image portrait view synthesis.
     Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,                   In Proc. of SIGGRAPH, 2023. 2
     Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby,                 [43] Alex Trevithick, Matthew Chan, Towaki Takikawa, Umar
     Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell                    Iqbal, Shalini De Mello, Manmohan Chandraker, Ravi Ra-
     Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael                  mamoorthi, and Koki Nagano. Rendering every pixel for
     Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Je-                  high-fidelity geometry in 3D gans. In IEEE Conf. on Com-
     gou, Julien Mairal, Patrick Labatut, Armand Joulin, and Pi-              puter Vision and Pattern Recognition, 2024. 2
     otr Bojanowski. DINOv2: Learning robust visual features             [44] Aaron van den Oord, Oriol Vinyals, and Koray
     without supervision. Transactions on Machine Learning Re-                Kavukcuoglu.       Neural discrete representation learning.
     search, 2024. Featured Certification. 4                                  In Conf. on Neural Information Processing Systems, 2017.
[34] Jeong Joon Park, Peter Florence, Julian Straub, Richard                  1, 2
     Newcombe, and Steven Lovegrove. DeepSDF: Learning                   [45] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
     continuous signed distance functions for shape representa-               reit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia
     tion. In IEEE Conf. on Computer Vision and Pattern Recog-                Polosukhin. Attention is all you need. In Conf. on Neural
     nition, 2019. 2                                                          Information Processing Systems, 2017. 1, 2, 3


                                                                    10
[46] Feng Wang, Yaodong Yu, Guoyizhe Wei, Wei Shao, Yuyin                [59] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen,
     Zhou, Alan Yuille, and Cihang Xie. Scaling laws in patchi-               Daniel Cremers, and Liang-Chieh Chen. An image is worth
     fication: An image is worth 50,176 tokens and more. arXiv                32 tokens for reconstruction and generation. In Conf. on Neu-
     preprint arXiv:2502.03738, 2025. 1                                       ral Information Processing Systems, 2024. 2
[47] Letian Wang, Seung Wook Kim, Jiawei Yang, Cunjun Yu,                [60] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and
     Boris Ivanovic, Steven Waslander, Yue Wang, Sanja Fidler,                Lucas Beyer. Sigmoid loss for language image pre-training.
     Marco Pavone, and Peter Karkus. DistillNeRF: Perceiv-                    In IEEE Int. Conf. on Computer Vision, 2023. 2
     ing 3d scenes from single-glance images by distilling neu-          [61] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
     ral fields and foundation model features. In Conf. on Neural             and Oliver Wang. The unreasonable effectiveness of deep
     Information Processing Systems, 2024. 2                                  features as a perceptual metric. In IEEE Conf. on Computer
[48] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min                   Vision and Pattern Recognition, 2018. 3
     Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez.          [62] Xunyu Zhu, Jian Li, Yong Liu, Can Ma, and Weiping Wang.
     OmniDrive: A holistic llm-agent framework for autonomous                 A survey on model compression for large language models.
     driving with 3d perception, reasoning and planning. arXiv                Transactions of the Association for Computational Linguis-
     preprint arXiv:2405.01533, 2024. 2                                       tics, 12:1556–1577, 2024. 1
[49] Wayve.       LINGO-1: Exploring natural language for
     autonomous driving, 2023.           Available at https :
     / / wayve . ai / thinking / lingo - natural -
     language-autonomous-driving/. 2
[50] Wayve.       LINGO-2: Driving with natural language,
     2024. Available at https://wayve.ai/thinking/
     lingo-2-driving-with-language/. 2
[51] Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and
     Marco Pavone. PARA-Drive: Parallelized architecture for
     real-time autonomous driving. In IEEE Conf. on Computer
     Vision and Pattern Recognition, 2024. 2, 5
[52] Xinzhou Wu. Accelerate the future of AI-defined vehicles
     and autonomous driving, 2025. Available at https://
     www.nvidia.com/en-us/on-demand/session/
     gtc25-dd40000/. 4, 8
[53] Huaiyuan Xu, Junliang Chen, Shiyu Meng, Yi Wang, and
     Lap-Pui Chau. A survey on occupancy perception for au-
     tonomous driving: The information fusion perspective. In-
     formation Fusion, 114:102671, 2025. 2
[54] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo,
     Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao.
     DriveGPT4: Interpretable end-to-end autonomous driving
     via large language model. IEEE Robotics and Automation
     Letters, 2024. 2
[55] Wilson Yan, Volodymyr Mnih, Aleksandra Faust, Matei Za-
     haria, Pieter Abbeel, and Hao Liu. ElasticTok: Adaptive
     tokenization for image and video. In Int. Conf. on Learning
     Representations, 2025. 2
[56] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi
     Li, Yurong You, Maximilian Igl, Apoorva Sharma, Peter
     Karkus, Danfei Xu, Boris Ivanovic, Yue Wang, and Marco
     Pavone. STORM: Spatio-temporal reconstruction model for
     large-scale outdoor scenes. In Int. Conf. on Learning Repre-
     sentations, 2025. 5
[57] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
     gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
     thing v2. In Conf. on Neural Information Processing Sys-
     tems, 2024. 1
[58] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang,
     James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge,
     and Yonghui Wu. Vector-quantized image modeling with im-
     proved VQGAN. In Int. Conf. on Learning Representations,
     2022. 3


                                                                    11
   Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving
                                                        Supplementary Material
A. Additional Open-Loop Evaluations                                           If additional sources of labeled data are available, e.g., as
                                                                          part of prior AV development efforts, they can easily be in-
                                                                          corporated into training as supplementary terms to the self-
   WOD E2E Planning                    Traj L2 (m) ↓                      supervised rendering loss (Eq. (2)). In Fig. 10, we show the
                             @1s      @3s @5s Ave1,3,5s                   results of adding a depth supervision loss to Eq. (2). Specif-
   DINOv2-small + 1B         0.21     1.00       2.31     1.17            ically, we use DepthAnythingV2-Small [57] to provide rel-
   Ours (8-8-8) + 1B         0.11     0.66       1.72     0.83            ative depth labels, and include them in the loss through an
   Ours (4-6-6) + 1B         0.11     0.69       1.79     0.86            additional depth term Ldepth (d, d)ˆ = λd ∥d − d∥  ˆ 1 where d
Table 5. Our approach significantly outperforms the DINOv2                is the provided depth label and dˆ is the triplane-estimated
baseline on the Waymo Open Dataset. Triplane patchification               depth. On the left-most image of Fig. 10, a car can be seen
sizes are denoted with (px − py − pz ).                                   two lanes over from the ego-vehicle. This car is also visible
                                                                          in the triplane-estimated depth, with more details (e.g., the
    Tab. 5 shows the performance of our work on the Waymo                 wheels and the shape of the driver-side windows are visi-
Open Dataset [38] validation split, predicting 5s of future               ble). While these preliminary results are encouraging, and
ego-motion from 1s of past observations (with the same                    validate the ability for multi-task training with triplanes, we
metrics as above at 1s, 3s, 5s horizons). Here, the DINOv2-               leave more accurate depth (and other auxiliary tasks) mod-
based tokenizer performs much worse than our work, indi-                  eling to future work.
cating that the non-geometry-aware ViT-based tokenization
struggles to capture the details needed to produce the more
complicated driving behaviors found in Waymo.
    Interestingly, we also see that our planning performance
is preserved even with aggressive patchification schemes
((8-8-8) slightly outperforms (4-6-6)!), indicating that our
triplane-based tokenizer features are robust to downstream
patchification choices. Unfortunately, there do not appear to
be many other works that utilize Waymo for E2E planning
benchmarking, so we offer these results primarily as further
evidence that our conclusions generalize to other datasets.

B. Optional Auxiliary Supervision




                             GT Images




                      Triplane Estimated Depth




                   Depth Labels (DepthAnythingV2)


Figure 10. Triplanes can optionally be trained with auxiliary la-
bels, if available. Here, DepthAnythingV2 [57] provides relative
depth supervision which the triplane is able to learn from and pre-
dict. As there are many high-frequency details in these images,
they are best viewed on a high-resolution screen.


                                                                      1
