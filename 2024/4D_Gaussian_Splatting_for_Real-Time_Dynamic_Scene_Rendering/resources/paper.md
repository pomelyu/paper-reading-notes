                                                          4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

                                                         Guanjun Wu1 ,* Taoran Yi2 ,* Jiemin Fang3†, Lingxi Xie3 , Xiaopeng Zhang3 ,
                                                                     Wei Wei1 , Wenyu Liu2 , Qi Tian3 , Xinggang Wang2†‡
                                                                  1
                                                                    School of CS, Huazhong University of Science and Technology
                                                         2
                                                           School of EIC, Huazhong University of Science and Technology 3 Huawei Inc.
                                                                              {guajuwu, taoranyi, weiw, liuwy, xgwang}@hust.edu.cn




arXiv:2310.08528v3 [cs.CV] 15 Jul 2024
                                                                  {jaminfong, 198808xc, zxphistory}@gmail.com                                         tian.qi1@huawei.com




                                         Figure 1. Our method achieves real-time rendering‡ for dynamic scenes at high image resolutions while maintaining high rendering quality.
                                         The right figure is tested on synthetic datasets [42], where the radius of the dot corresponds to the training time. “Res”: resolution.
                                         ‡
                                          The rendering speed not only depends on the image resolution but also the number of 3D Gaussians and the scale of deformation fields which are determined by the complexity
                                         of the scene.
                                                                          Abstract                                                1. Introduction
                                            Representing and rendering dynamic scenes has been an
                                                                                                                                  Novel view synthesis (NVS) stands as a critical task in the
                                         important but challenging task. Especially, to accurately
                                                                                                                                  domain of 3D vision and plays a vital role in many appli-
                                         model complex motions, high efficiency is usually hard to
                                                                                                                                  cations, e.g. VR, AR, and movie production. NVS aims
                                         guarantee. To achieve real-time dynamic scene rendering
                                                                                                                                  at rendering images from any desired viewpoint or times-
                                         while also enjoying high training and storage efficiency, we
                                                                                                                                  tamp of a scene, usually requiring modeling the scene ac-
                                         propose 4D Gaussian Splatting (4D-GS) as a holistic rep-
                                                                                                                                  curately from several 2D images. Dynamic scenes are quite
                                         resentation for dynamic scenes rather than applying 3D-GS
                                                                                                                                  common in real scenarios, rendering which is important but
                                         for each individual frame. In 4D-GS, a novel explicit rep-
                                                                                                                                  challenging as complex motions need to be modeled with
                                         resentation containing both 3D Gaussians and 4D neural
                                                                                                                                  both spatially and temporally sparse input.
                                         voxels is proposed. A decomposed neural voxel encoding
                                         algorithm inspired by HexPlane is proposed to efficiently                                   NeRF [35] has achieved great success in synthesizing
                                         build Gaussian features from 4D neural voxels and then a                                 novel view images by representing scenes with implicit
                                         lightweight MLP is applied to predict Gaussian deforma-                                  functions. The volume rendering techniques [8] are in-
                                         tions at novel timestamps. Our 4D-GS method achieves                                     troduced to connect 2D images and 3D scenes. However,
                                         real-time rendering under high resolutions, 82 FPS at an                                 the original NeRF method bears big training and rendering
                                         800×800 resolution on an RTX 3090 GPU while main-                                        costs. Though some NeRF variants [5, 9, 11, 12, 36, 48, 51]
                                         taining comparable or better quality than previous state-                                reduce the training time from days to minutes, the rendering
                                         of-the-art methods. More demos and code are available at                                 process still bears a non-negligible latency.
                                         https://guanjunwu.github.io/4dgs/.                                                          Recent 3D Gaussian Splatting (3D-GS) [22] signifi-
                                                                                                                                  cantly boosts the rendering speed to a real-time level by
                                                                                                                                  representing the scene as 3D Gaussians. The cumbersome
                                             * Equal contributions. † Project lead. ‡ Corresponding author.                       volume rendering in the original NeRF is replaced with ef-


                                                                                                                              1
                                                                            Original Sampled Points              Original Sampled Points
ficient differentiable splatting [63], which directly projects              Canonical Mapped Points              The Original Cast Ray
3D Gaussian onto the 2D image plane. 3D-GS not only en-                     The Original Cast Ray                Time Features of the Points
joys real-time rendering speed but also represents the scene                The Canonical Mapped Ray
more explicitly, making it easier to manipulate the scene
representation.
                                                                                                                      𝑡
    However, 3D-GS focuses on the static scenes. Extend-
                                                                                                                    𝑡
ing it to dynamic scenes as a 4D representation is a reason-                                                        𝑡
able, important but difficult topic. The key challenge lies in       (a) Canonical Mapping Volume Rendering (b) Time-aware Volume Rendering
modeling complicated point motions from sparse input. 3D-
                                                                                                             Original 3D Gaussians G
GS holds a natural geometry prior by representing scenes                                                     Deformed 3D Gaussians 𝐺
with point-like Gaussians. One direct and effective exten-                                                   Gaussian Deformation Field 𝐹 𝐺, 𝑡
sion approach is to construct 3D Gaussians at each times-                                                    Gaussian Rasterization Paths

tamp [33] but the storage/memory cost will multiply espe-
cially for long input sequences. Our goal is to construct
                                                                                               (c) 4D Gaussian Splatting
a compact representation while maintaining both training
and rendering efficiency, i.e. 4D Gaussian Splatting (4D-            Figure 2. Illustration of different dynamic scene rendering meth-
GS). To this end, we propose to represent Gaussian mo-               ods. (a) Points are sampled in the cast ray during volume ren-
tions and shape changes by an efficient Gaussian deforma-            dering. The point deformation fields proposed in [9, 42] map the
tion field network, containing a temporal-spatial structure          points into a canonical space. (b) Time-aware volume rendering
encoder and an extremely tiny multi-head Gaussian defor-             computes the features of each point directly and does not change
mation decoder. Only one set of canonical 3D Gaussians is            the rendering path. (c) The Gaussian deformation field converts
maintained. For each timestamp, the canonical 3D Gaus-               original 3D Gaussians into another group of 3D Gaussians with a
                                                                     certain timestamp.
sians will be transformed by the Gaussian deformation field
into new positions with new shapes. The transformation
process represents both the Gaussian motion and deforma-             2.1. Novel View Synthesis
tion. Note that different from modeling motions of each              Novel view synthesis is a important and challenging task in
Gaussian separately [33, 61], the spatial-temporal structure         3D reconstruction. Much approaches are proposed to rep-
encoder can connect different adjacent 3D Gaussians to pre-          resent a 3D object and render novel views. Efficient rep-
dict more accurate motions and shape deformation. Then               resentations such as light fields [4], mesh [7, 17, 27, 50],
the deformed 3D Gaussians can be directly splatted for ren-          voxels [18, 20, 26], multi-planes [10] can render high
dering the according-timestamp image. Our contributions              quality image with enough supervisions. NeRF-based ap-
can be summarized as follows.                                        proaches [3, 35, 65] demonstrate that implicit radiance
• An efficient 4D Gaussian splatting framework with an ef-           fields can effectively learn scene representations and syn-
  ficient Gaussian deformation field is proposed by mod-             thesize high-quality novel views. [38, 39, 42] have chal-
  eling both Gaussian motion and Gaussian shape changes              lenged the static hypothesis, expanding the boundary of
  across time.                                                       novel view synthesis for dynamic scenes. [9] proposes
• A multi-resolution encoding method is proposed to con-             to use an explicit voxel grid to model temporal informa-
  nect the nearby 3D Gaussians and build rich 3D Gaussian            tion, accelerating the learning time for dynamic scenes to
  features by an efficient spatial-temporal structure encoder.       half an hour and applied in [19, 32, 62]. The proposed
• 4D-GS achieves real-time rendering on dynamic scenes,              deformation-based neural rendering methods are shown in
  up to 82 FPS at a resolution of 800×800 for synthetic              Fig. 2 (a). Flow-based [14, 28, 32, 52, 67] methods adopt-
  datasets and 30 FPS at a resolution of 1352×1014 in                ing warping algorithm to synthesis novel views by blend-
  real datasets, while maintaining comparable or superior            ing nearby frames.       [5, 12, 13, 25, 48, 53] represent
  performance than previous state-of-the-art (SOTA) meth-            further advancements in faster dynamic scene learning by
  ods. It also shows potential for editing and tracking in 4D        adopting decomposed neural voxels. They treat sampled
  scenes.                                                            points in each timestamp individually as shown in Fig. 2
                                                                     (b). [16, 30, 41, 54, 56, 58] are efficient methods to han-
                                                                     dle multi-view setups. The aforementioned methods though
2. Related Works                                                     achieve fast training speed, real-time rendering for dynamic
                                                                     scenes is still challenging, especially for monocular input.
In this section, we simply review the difference of dynamic          Our method aims at constructing a highly efficient training
NeRFs in Sec. 2.1, then discuss the point clouds-based neu-          and rendering pipeline in Fig. 2 (c), while maintaining the
ral rendering algorithm in Sec. 2.2.                                 quality, even for sparse inputs.


                                                                 2
                                                                                      𝑥, 𝑦, 𝑧: Position of 3D Gaussians
                                                                                      𝑡: Timestamp
                                                                                      ∆𝑥, ∆𝑦, ∆𝑧: Deformation of Position Deformed 3D
                            𝑥𝑡           𝑦𝑡             𝑧𝑡                            ∆𝑟, ∆𝑠: Deformation of Covariance Gaussians 𝐺
             𝑥, 𝑦, 𝑧

                                                                                                  Position
                                                                                                  Head 𝜑 ∆𝑥, ∆𝑦, ∆𝑧
                                                                           MLP                                               Splatting
3D Gaussians 𝐺                                                                                   Rotation
                                                                                                 Head 𝜑 ∆𝑟

                          𝑥𝑦            𝑦𝑧             𝑥𝑧                                        Scaling
                                                                                                         ∆𝑠
                 𝑡                                                                               Head 𝜑

                                   Spatial-Temporal Structure Encoder              Multi-head Gaussian Deformation Decoder

Figure 3. The overall pipeline of our model. Given a group of 3D Gaussians G, we extract the center coordinate of each 3D Gaussian X
and timestamp t to compute the voxel feature by querying multi-resolution voxel planes. Then a tiny multi-head Gaussian deformation
decoder is used to decode the feature and get the deformed 3D Gaussians G ′ at timestamp t. The deformed Gaussians are then splatted to
get the rendered images.

2.2. Neural Rendering with Point Clouds                                 proach also models 3D Gaussian motions but with a com-
                                                                        pact network, resulting in highly efficient training and real-
Effectively representing 3D scenes remains a challenging                time rendering.
topic. The community has explored various neural repre-
sentations [35], e.g. meshes, point clouds [59], voxels [11],           3. Preliminary
and hybrid approaches [36, 51]. Point-cloud-based meth-
ods [31, 43, 44, 64] initially target at 3D segmentation and            In this section, we simply review the representation and ren-
classification. A representative approach for rendering pre-            dering process of 3D-GS [22] in Sec. 3.1 and the formula of
sented in [1, 59] combines point cloud representations with             dynamic NeRFs in Sec. 3.2.
volume rendering, achieving rapid convergence speed even                3.1. 3D Gaussian Splatting
for dynamic novel view synthesis [37, 67]. [23, 24, 45]
adopt differential point rendering technique for scene recon-           3D Gaussians [22] is an explicit 3D scene representation in
structions.                                                             the form of point clouds. Each 3D Gaussian is characterized
                                                                        by a covariance matrix Σ and a center point X , which is
   Recently, 3D-GS [6, 22] is notable for its pure explicit             referred to as the mean value of the Gaussian:
representation and differential point-based splatting meth-
                                                                                                        1   T   −1
ods, enabling real-time rendering of novel views. Dy-                                     G(X) = e− 2 X Σ            X
                                                                                                                         .               (1)
namic3DGS [33] models dynamic scenes by tracking the
position and variance of each 3D Gaussian at each times-                For differentiable optimization, the covariance matrix Σ can
tamp ti . An explicit table is utilized to store information            be decomposed into a scaling matrix S and a rotation matrix
about each 3D Gaussian at every timestamp, leading to a                 R:
linear memory consumption increase, denoted as O(tN ), in                                     Σ = RSST RT .                       (2)
which N is num of 3D Gaussians. For long-term scene re-                     When rendering novel views, differential splatting [63]
construction, the storage cost will become non-negligible.              is employed for the 3D Gaussians within the camera planes.
The memory complexity of our approach only depends                      As introduced by [68], using a viewing transform matrix W
on the number of 3D Gaussians and parameters of Gaus-                   and the Jacobian matrix J of the affine approximation of
sians deformation fields network F, which is denoted as                 the projective transformation, the covariance matrix Σ′ in
O(N + F). Another method to extend 3D Gaussians to                      camera coordinates can be computed as
4D [61] adds a marginal temporal Gaussian distribution into
the origin 3D Gaussians, which uplifts 3D Gaussians into                                    Σ′ = JW ΣW T J T .                           (3)
4D. However, it may cause each 3D Gaussian to only focus
on their local temporal space. Deformable-3DGS [60] is a                In summary, each 3D Gaussian is characterized by the fol-
concurrent work that introduces an MLP deformation net-                 lowing attributes: position X ∈ R3 , color defined by spher-
work to model the motion of dynamic scenes. Spacetime-                  ical harmonic (SH) coefficients C ∈ Rk (where k repre-
GS [29] tracks each 3D Gaussians individually. Our ap-                  sents nums of SH functions), opacity α ∈ R, rotation factor


                                                                   3
r ∈ R4 , and scaling factor s ∈ R3 . Specifically, for each                  Iter 0                    Iter 3000                      Iter 20000
pixel, the color and opacity of all the Gaussians are com-
puted using the Gaussian’s representation Eq. 1. The blend-
ing of N ordered points that overlap the pixel is given by
the formula:
                      X            i−1
                                   Y
                C=         ci αi         (1 − αi ).       (4)
                     i∈N           j=1

Here, ci , αi represents the density and color of this point
computed by a 3D Gaussian G with covariance Σ multi-
plied by an optimizable per-point opacity and SH color co-          Random Point Cloud Input   3D Gaussian Initialization   4D Gaussian Joint Optimization
efficients.
                                                                    Figure 4. Illustration of the optimization process. With static 3D
3.2. Dynamic NeRFs with Deformation Fields                          Gaussian initialization, our model can learn high-quality 3D Gaus-
                                                                    sians of the motion part.
All the dynamic NeRF algorithms can be formulated as:

                   c, σ = M(x, d, t, λ),                  (5)          Specifically, the deformation of 3D Gaussians ∆G is in-
where M is a mapping that maps 8D space (x, d, t, λ) to             troduced by the Gaussian deformation field network ∆G =
4D space (c, σ). x reveals to the spatial point, and λ is the       F(G, t), in which the spatial-temporal structure encoder H
optional input as used to build topological and appearance          can encode both the temporal and spatial features of 3D
changes in [39], and d stands for view-dependency.                  Gaussians fd = H(G, t). And the multi-head Gaussian de-
   As shown in Fig. 2 (a), all the deformation NeRF based           formation decoder D can decode the features and predict
methods estimate the world-to-canonical mapping by a                each 3D Gaussian’s deformation ∆G = D(f ), then the de-
deformation network ϕt : (x, t) → ∆x. Then a network is             formed 3D Gaussians G ′ can be introduced.
introduced to compute volume density and view-dependent                The rendering process of our 4D Gaussian Splatting is
RGB color from each ray. The formula for rendering can be           depicted in Fig. 2 (c). Our 4D Gaussian splatting converts
expressed as:                                                       the original 3D Gaussians G into another group of 3D Gaus-
                                                                    sians G ′ given a timestamp t, maintaining the effectiveness
               c, σ = NeRF(x + ∆x, d, λ),                 (6)       of the differential splatting as referred in [63].

where ‘NeRF’ stands for vanilla NeRF pipeline, λ is a               4.2. Gaussian Deformation Field Network
frame-dependent code to model the topological and appear-           The network to learn the Gaussian deformation field in-
ance changes [34, 39].                                              cludes an efficient spatial-temporal structure encoder H and
   However, our 4D Gaussian splatting framework presents            a Gaussian deformation decoder D for predicting the defor-
a novel rendering technique. We successfully compute                mation of each 3D Gaussian.
the canonical-to-world mapping directly at time t using
a Gaussian deformation field network F, and differential
splatting [22] follows. This enables the capability of com-         Spatial-Temporal Structure Encoder. Nearby 3D Gaus-
puting backward flow and tracking for 3D Gaussians.                 sians always share similar spatial and temporal information.
                                                                    To model 3D Gaussians’ features effectively, we introduce
4. Method                                                           an efficient spatial-temporal structure encoder H including
                                                                    a multi-resolution HexPlane R(i, j) and a tiny MLP ϕd in-
Sec. 4.1 introduces the overall 4D Gaussian Splatting               spired by [5, 9, 12, 48]. While the vanilla 4D neural voxel is
framework. Then, the Gaussian deformation field is pro-             memory-consuming, we adopt a 4D K-Planes [12] module
posed in Sec. 4.2. Finally, we describe the optimization            to decompose the 4D neural voxel into 6 multi-resolution
process in Sec. 4.3.                                                planes. All 3D Gaussians in a certain area can be contained
                                                                    in the bounding plane voxels and the deformation of Gaus-
4.1. 4D Gaussian Splatting Framework
                                                                    sians can also be encoded in nearby temporal voxels.
As shown in Fig. 3, given a view matrix M = [R, T ], times-             Specifically, the spatial-temporal structure encoder H
tamp t, our 4D Gaussian splatting framework includes 3D             contains 6 multi-resolution plane modules Rl (i, j) and
Gaussians G and Gaussian deformation field network F.               a tiny MLP ϕd , i.e. H(G, t) = {Rl (i, j), ϕd |(i, j) ∈
Then a novel-view image Iˆ is rendered by differential splat-       {(x, y), (x, z), (y, z), (x, t), (y, t), (z, t)}, l ∈ {1, 2}}. The
ting [63] S following Iˆ = S(M, G ′ ), where G ′ = ∆G + G.          position µ = (x, y, z) is the mean value of 3D Gaussians


                                                                4
       Ground Truth            K-Planes             HexPlane               TiNeuVox              3D-GS                Ours




Hook




Jack



Figure 5. Visualization of synthesized datasets compared with other models [5, 9, 12, 19, 22, 53]. The rendering results of [12] are
displayed with a default green background. We adopt their rendering settings.

G. Each voxel module is defined by R(i, j) ∈ Rh×lNi ×lNj ,               for warm-up and then render images with 3D Gaussians
where h stands for the hidden dim of features, and N de-                 Iˆ = S(M, G) instead of 4D Gaussians Iˆ = S(M, G ′ ). The
notes the basic resolution of voxel grid and l equals to the             illustration of the optimization process is shown in Fig. 4.
upsampling scale. This entails encoding information of the
3D Gaussians within the 6 2D voxel planes while consider-
ing temporal information. The formula for computing sep-                 Loss Function. Similar to other reconstruction meth-
arate voxel features is as follows:                                      ods [9, 22, 42], we use the L1 color loss to supervise the
            [Y                                                           training process. A grid-based total-variational loss [5, 9,
      fh =         interp(Rl (i, j)),                                    12, 51] Ltv is also applied.
             l                                                (7)
   (i, j) ∈ {(x, y), (x, z), (y, z), (x, t), (y, t), (z, t)}.                                L = |Iˆ − I| + Ltv .                 (9)

fh ∈ Rh∗l is the feature of neural voxels. ‘interp’ denotes              5. Experiment
the bilinear interpolation for querying the voxel features lo-
cated at 4 vertices of the grid. The discussion of the produc-           In this section, we mainly introduce the hyperparameters
tion process is similar to K-Planes [12]. Then a tiny MLP                and datasets of our settings in Sec. 5.1 and the results be-
ϕd merges all the features by fd = ϕd (fh ).                             tween different datasets are compared with [2, 5, 9, 12, 22,
                                                                         30, 49, 53, 54] in Sec. 5.2. Then, ablation studies are pro-
                                                                         posed to prove the effectiveness of our approach in Sec. 5.3
Multi-head Gaussian Deformation Decoder. When all
                                                                         and more discussion about 4D-GS in Sec. 5.4. Finally, we
the features of 3D Gaussians are encoded, we can com-
                                                                         discuss the limitation of our proposed 4D-GS in Sec. 5.5.
pute any desired variable with a multi-head Gaussian de-
formation decoder D = {ϕx , ϕr , ϕs }. Separate MLPs are
                                                                         5.1. Experimental Settings
employed to compute the deformation of position ∆X =
ϕx (fd ), rotation ∆r = ϕr (fd ), and scaling ∆s = ϕs (fd ).             Our implementation is primarily based on the PyTorch [40]
Then, the deformed feature (X ′ , r′ , s′ ) can be addressed as:         framework and tested on a single RTX 3090 GPU, and
                                                                         we’ve fine-tuned our optimization parameters by the config-
        (X ′ , r′ , s′ ) = (X + ∆X , r + ∆r, s + ∆s).          (8)       uration outlined in the 3D-GS [22]. More hyperparameters
                                                                         are shown in the appendix.
Finally, we obtain the deformed 3D Gaussians G ′ =
{X ′ , s′ , r′ , σ, C}.
                                                                         Synthetic Dataset. We primarily assess the performance
4.3. Optimization
                                                                         of our model using a synthetic dataset, as introduced by
3D Gaussian Initialization. 3D-GS [22] shows that 3D                     D-NeRF [42]. The dataset is designed for monocular set-
Gaussians can be well-trained with structure from motion                 tings, although it’s worth noting that the camera poses for
(SfM) [46] points initialization. Similarly, 4D Gaussians                each timestamp are close to randomly generated. Each
can also leverage the power of proper 3D Gaussian initial-               scene within these datasets contains dynamic frames, rang-
ization. We optimize 3D Gaussians at initial 3000 iterations             ing from 50 to 200 in number.


                                                                     5
         GT     HyperNeRF     TiNeuVox       3D-GS         Ours           GT       HyperNeRF TiNeuVox         3D-GS         Ours




Broom                                                                                                                                 Chicken




Banana
                                                                                                                                      3DPrinter




Figure 6. Visualization of the HyperNeRF [39] dataset compared with other methods [9, 19, 22, 39]. ‘GT’ stands for ground truth images.

Table 1. Quantitative results on the synthetic dataset. The best and the second best results are denoted by pink and yellow. The rendering
resolution is set to 800×800. “Time” in the table stands for training times.

                     Model                 PSNR (dB)↑      SSIM↑     LPIPS↓      Time↓      FPS ↑    Storage (MB)↓
                     TiNeuVox-B [9]           32.67         0.97        0.04     28 mins     1.5            48
                     KPlanes [12]             31.61         0.97          -      52 mins     0.97          418
                     HexPlane-Slim [5]        31.04         0.97        0.04    11m 30s      2.5            38
                     3D-GS [22]               23.19         0.93        0.08     10 mins     170            10
                     FFDNeRF [19]             32.68         0.97        0.04        -        <1            440
                     MSTH [53]                31.34         0.98        0.02     6 mins        -             -
                     V4D [13]                 33.72         0.98        0.02    6.9 hours    2.08          377
                     Ours                     34.05         0.98        0.02     8 mins       82            18


Real-world Datasets. We utilize datasets provided by                     the compared methods are from their papers, reproduced by
HyperNeRF [39] and Neu3D [25] as benchmark datasets                      their code or provided by the authors. The rendering speed
to evaluate the performance of our model in real-world sce-              and storage data for [5, 9, 12, 22] are estimated based on the
narios. The HyperNeRF [39] dataset is captured using one                 official implementations.
or two cameras, following straightforward camera motion,                    The results in synthetic dataset [42] are summarized
while the Neu3D dataset is captured using 15 to 20 static                in Tab. 1. While current dynamic hybrid representations
cameras, involving extended periods and intricate camera                 can produce high-quality results, they often come with the
motions. We use the points computed by SfM [46] from                     drawback of rendering speed. The lack of modeling dy-
the first frame of each video in the Neu3D dataset and 200               namic motion part makes 3D-GS [22] fail to reconstruct dy-
frames randomly selected in the HyperNeRF dataset.                       namic scenes. In contrast, our method enjoys both the high-
                                                                         est rendering quality within the synthetic dataset and excep-
5.2. Results
                                                                         tionally fast rendering speeds while keeping extremely low
We primarily assess our experimental results using various               storage consumption and convergence time.
metrics, encompassing peak-signal-to-noise ratio (PSNR),                    Additionally, the results obtained from real-world
perceptual quality measure LPIPS [66], structural similar-               datasets are presented in Tab. 2 and Tab. 3. It becomes
ity index (SSIM) [57] and its extensions including structural            apparent that some NeRFs [2, 5, 49] suffer from slow
dissimilarity index measure (DSSIM), multiscale structural               convergence speed, and the other grid-based NeRF meth-
similarity index (MS-SSIM), FPS, training times and stor-                ods [5, 9, 12, 53] encounter difficulties when attempting
age. To assess the quality of novel view synthesis, we con-              to capture intricate object details. In stark contrast, our
duct comparisons with several state-of-the-art methods in                methods research comparable rendering quality, fast con-
the field, including [2, 5, 9, 12, 13, 19, 22, 30, 38, 39, 49,           vergence, and excel in free-view rendering speed in in-
53]. The K-Planes results on the synthetic dataset originate             door cases. Though Im4D [30] addresses the high qual-
from the Deformable-3DGS [60] paper. The other results of                ity in comparison to ours, the need for multi-cam setups


                                                                    6
                 Table 2. Quantitative results on HyperNeRF [39] vrig dataset with the rendering resolution of 960×540.

                          Model                  PSNR (dB)↑       MS-SSIM↑             Times↓     FPS↑     Storage (MB)↓
                          Nerfies [38]                 22.2            0.803        ∼ hours       <1                  -
                          HyperNeRF [39]               22.4            0.814        32 hours      <1                  -
                          TiNeuVox-B [9]               24.3            0.836         30 mins       1                48
                          3D-GS [22]                   19.7            0.680         40 mins       55               52
                          FFDNeRF [19]                 24.2            0.842            -         0.05              440
                          V4D [13]                     24.8            0.832        5.5 hours     0.29              377
                          Ours                         25.2            0.845         30 mins       34                61

                   Table 3. Quantitative results on the Neu3D [25] dataset with the rendering resolution of 1352×1014.

                  Model                    PSNR (dB)↑         D-SSIM↓          LPIPS↓      Time ↓        FPS↑         Storage (MB)↓
                  NeRFPlayer [49]              30.69           0.034           0.111       6 hours        0.045             -
                  HyperReel [2]                31.10           0.036           0.096       9 hours         2.0            360
                  HexPlane-all* [5]            31.70           0.014           0.075      12 hours         0.2            250
                  KPlanes [12]                 31.63             -               -        1.8 hours        0.3            309
                  Im4D [30]                    32.58             -             0.208       28 mins         ∼5              93
                  MSTH [53]                    32.37           0.015           0.056       20 mins       2 (15‡ )         135
                  Ours                         31.15           0.016           0.049       40 mins          30             90
   *: The metrics of the models are tested without “coffee martini” and resolution is set to 1024×768.
   ‡ : The FPS is tested with fixed-view rendering.




makes it hard to model monocular scenes and other meth-                         3D Gaussian Initialization. In some cases without
ods [2, 5, 12, 49, 53] also limit free-view rendering speed                     SfM [46] points initialization, training 4D-GS directly may
and storage.                                                                    cause difficulty in convergence. Optimizing 3D Gaussians
                                                                                for warm-up enjoys: (a) making some 3D Gaussians stay in
5.3. Ablation Study                                                             the dynamic part, which releases the pressure of large de-
Spatial-Temporal Structure Encoder. The explicit Hex-                           formation learning by 4D Gaussians as shown in Fig. 4. (b)
Plane encoder Rl (i, j) possesses the capacity to retain 3D                     learning proper 3D Gaussians G and suggesting deforma-
Gaussians’ spatial and temporal information, which can re-                      tion fields paying more attention to the dynamic part. (c)
duce storage consumption in comparison with purely ex-                          avoiding numeric errors in optimizing the Gaussian defor-
plicit method [33]. Discarding this module, we observe that                     mation network F and keeping the training process stable.
using only a shallow MLP ϕd falls short in modeling com-                        Tab. 4 also shows that if we train our model without the
plex deformations across various settings. Tab. 4 demon-                        warm-up coarse stage, the rendering quality will suffer.
strates that, while the model incurs minimal memory costs,
it does come at the expense of rendering quality.                               5.4. Discussions
                                                                                Tracking with 3D Gaussians. Tracking in 3D is also
Gaussian Deformation Decoder. Our proposed Gaus-                                a important task. FFDNeRF [19] also shows the results
sian deformation decoder D decodes the features from the                        of tracking objects’ motion in 3D. Different from dy-
spatial-temporal structure encoder H. All the changes in 3D                     namic3DGS [33], our methods even can present tracking
Gaussians can be explained by separate MLPs {ϕx , ϕr , ϕs }.                    objects in monocular settings with pretty low storage i.e.
As shown in Tab. 4, 4D Gaussians cannot fit dynamic scenes                      10MB in 3D Gaussians G and 8 MB in Gaussian deforma-
well without modeling 3D Gaussian motion. Meanwhile,                            tion field network F. Fig. 7 shows the 3D Gaussian’s de-
the movement of human body joints is typically manifested                       formation at certain timestamps.
as stretching and twisting of surface details in a macro-                       Composition with 4D Gaussians. Similar to Dy-
scopic view. If one aims to accurately model these move-                        namic3DGS [33], our proposed methods can also perform
ments, the size and shape of 3D Gaussians should also be                        editing in 4D Gaussians, as shown in Fig. 8. Thanks to
adjusted accordingly. Otherwise, there may be underfitting                      the explicit representation of 3D Gaussians, all the trained
of details during excessive stretching, or an inability to cor-                 models can predict deformed 3D Gaussians in the same
rectly simulate the movement of objects at a microscopic                        space following G ′ = {G1′ , G2′ , ..., Gn′ } and differential ren-
level.                                                                          dering [63] can project all the point clouds into viewpoints


                                                                          7
                               Table 4. Ablation studies on synthetic datasets using our proposed methods.

         Model                              PSNR(dB)↑        SSIM↑            LPIPS↓                   Time↓            FPS↑          Storage (MB)↓
         Ours w/o HexPlane Rl (i, j)             27.05         0.95                      0.05          4 mins           140                    12
         Ours w/o initialization                 31.91         0.97                      0.03         7.5 mins           79                    18
         Ours w/o ϕx                             26.67         0.95                      0.07          8 mins            82                    17
         Ours w/o ϕr                             33.08         0.98                      0.03          8 mins            83                    17
         Ours w/o ϕs                             33.02         0.98                      0.03          8 mins            82                    17
         Ours                                    34.05         0.98                      0.02          8 mins            82                    18


                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                           3 R L Q W V
                                                                                              
                                                                                              
                                                                                              
                                                                                              
           (a) Cook Spinach            (b) Coffee Martini
                                                                                                                                                      
Figure 7. Visualization of tracking with 3D Gaussians. Lines in                                                                        ) 3 6
the figures of the second row stand for the trajectory of 3D Gaus-
                                                                             Figure 9. Visualization of the relationship between rendering
sians.
                                                                             speed and numbers of 3D Gaussians. All the tests are finished
                                                                             in the synthetic dataset.


                                                                             straints.
                                                                             5.5. Limitations
                                                                             Though 4D-GS can indeed attain rapid convergence and
                                                                             yield real-time rendering outcomes in many scenarios, there
                                                                             are a few key challenges to address. First, large motions,
   Figure 8. Visualization of composition with 4D Gaussians.                 the absence of background points, and the unprecise camera
                                                                             pose cause the struggle of optimizing 4D Gaussians. Mean-
                                                                             while, it is still challenging for 4D-GS to split the joint mo-
by Iˆ = S(M, G ′ ) as referred in Sec. 4.1.                                  tion of static and dynamic Gaussians under the monocular
                                                                             settings without any additional supervision. Finally, a more
                                                                             compact algorithm needs to be designed to handle urban-
Analysis of Rendering Speed. As shown in Fig. 9, we
                                                                             scale reconstruction due to the heavy querying of Gaussian
also test the relationship between the number of 3D Gaus-
                                                                             deformation fields by huge numbers of 3D Gaussians.
sians and rendering speed at the resolution of 800×800.
We observe that if the rendered Gaussians are fewer than                     6. Conclusion
30,000, the rendering speed can be up to 90 FPS on a single
RTX 3090 GPU. The configuration of Gaussian deforma-                         This paper proposes 4D Gaussian splatting to achieve real-
tion fields is discussed in the appendix. To achieve real-                   time dynamic scene rendering. An efficient deformation
time rendering speed, we should strike a balance among all                   field network is constructed to accurately model Gaussian
the rendering resolutions, 4D Gaussians representation in-                   motions and shape deformations, where adjacent Gaus-
cluding numbers of Gaussians, the capacity of the Gaussian                   sians are connected via a spatial-temporal structure encoder.
deformation field network, and any other hardware con-                       Connections between Gaussians lead to more complete de-


                                                                      8
formed geometry, effectively avoiding avulsion. Our 4D                        on Computer Vision and Pattern Recognition, pages 2367–
Gaussians can not only model dynamic scenes but also have                     2376, 2019. 2
the potential for 4D objective tracking and editing.                     [11] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
                                                                              Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Acknowledgments                                                               Radiance fields without neural networks. In Proceedings of
                                                                              the IEEE/CVF Conference on Computer Vision and Pattern
This work was supported by the National Natural Science                       Recognition, pages 5501–5510, 2022. 1, 3
Foundation of China (No. 62376102). The authors would                    [12] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
like to thank Haotong Lin for providing the quantitative re-                  Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
sults of Im4D [30].                                                           Explicit radiance fields in space, time, and appearance. In
                                                                              Proceedings of the IEEE/CVF Conference on Computer Vi-
                                                                              sion and Pattern Recognition, pages 12479–12488, 2023. 1,
References                                                                    2, 4, 5, 6, 7, 13, 14
 [1] Jad Abou-Chakra, Feras Dayoub, and Niko Sünderhauf. Par-           [13] Wanshui Gan, Hongbin Xu, Yi Huang, Shifeng Chen, and
     ticlenerf: Particle based encoding for online neural radiance            Naoto Yokoya. V4d: Voxel for 4d novel view synthesis.
     fields in dynamic scenes. arXiv preprint arXiv:2211.04041,               IEEE Transactions on Visualization and Computer Graph-
     2022. 3, 15                                                              ics, 2023. 2, 6, 7
 [2] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael          [14] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang.
     Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil                  Dynamic view synthesis from dynamic monocular video. In
     Kim. Hyperreel: High-fidelity 6-dof video with ray-                      Proceedings of the IEEE/CVF International Conference on
     conditioned sampling. In Proceedings of the IEEE/CVF                     Computer Vision, pages 5712–5721, 2021. 2
     Conference on Computer Vision and Pattern Recognition,              [15] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell,
     pages 16610–16620, 2023. 5, 6, 7                                         and Angjoo Kanazawa. Monocular dynamic view synthesis:
                                                                              A reality check. Advances in Neural Information Processing
 [3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
                                                                              Systems, 35:33768–33780, 2022. 14
     Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
     Mip-nerf: A multiscale representation for anti-aliasing neu-        [16] Xiangjun Gao, Jiaolong Yang, Jongyoo Kim, Sida Peng,
     ral radiance fields. In Proceedings of the IEEE/CVF Inter-               Zicheng Liu, and Xin Tong. Mps-nerf: Generalizable 3d hu-
     national Conference on Computer Vision, pages 5855–5864,                 man rendering from multiview images. IEEE Transactions
     2021. 2                                                                  on Pattern Analysis and Machine Intelligence, 2022. 2
                                                                         [17] Kaiwen Guo, Feng Xu, Yangang Wang, Yebin Liu, and
 [4] Michael Broxton, John Flynn, Ryan Overbeck, Daniel Erick-
                                                                              Qionghai Dai. Robust non-rigid motion tracking and sur-
     son, Peter Hedman, Matthew Duvall, Jason Dourgarian, Jay
                                                                              face reconstruction using l0 regularization. In Proceedings
     Busch, Matt Whalen, and Paul Debevec. Immersive light
                                                                              of the IEEE International Conference on Computer Vision,
     field video with a layered mesh representation. ACM Trans-
                                                                              pages 3083–3091, 2015. 2
     actions on Graphics (TOG), 39(4):86–1, 2020. 2
                                                                         [18] Kaiwen Guo, Peter Lincoln, Philip Davidson, Jay Busch,
 [5] Ang Cao and Justin Johnson. Hexplane: A fast representa-
                                                                              Xueming Yu, Matt Whalen, Geoff Harvey, Sergio Orts-
     tion for dynamic scenes. In Proceedings of the IEEE/CVF
                                                                              Escolano, Rohit Pandey, Jason Dourgarian, et al. The re-
     Conference on Computer Vision and Pattern Recognition,
                                                                              lightables: Volumetric performance capture of humans with
     pages 130–141, 2023. 1, 2, 4, 5, 6, 7, 13, 14
                                                                              realistic relighting. ACM Transactions on Graphics (ToG),
 [6] Guikun Chen and Wenguan Wang. A survey on 3d gaussian                    38(6):1–19, 2019. 2
     splatting. arXiv preprint arXiv:2401.03890, 2024. 3                 [19] Xiang Guo, Jiadai Sun, Yuchao Dai, Guanying Chen, Xiao-
 [7] Alvaro Collet, Ming Chuang, Pat Sweeney, Don Gillett, Den-               qing Ye, Xiao Tan, Errui Ding, Yumeng Zhang, and Jingdong
     nis Evseev, David Calabrese, Hugues Hoppe, Adam Kirk,                    Wang. Forward flow for novel view synthesis of dynamic
     and Steve Sullivan. High-quality streamable free-viewpoint               scenes. In Proceedings of the IEEE/CVF International Con-
     video. ACM Transactions on Graphics (ToG), 34(4):1–13,                   ference on Computer Vision, pages 16022–16033, 2023. 2,
     2015. 2                                                                  5, 6, 7, 12
 [8] Robert A Drebin, Loren Carpenter, and Pat Hanrahan. Vol-            [20] Tao Hu, Tao Yu, Zerong Zheng, He Zhang, Yebin Liu, and
     ume rendering. ACM Siggraph Computer Graphics, 22(4):                    Matthias Zwicker. Hvtr: Hybrid volumetric-textural render-
     65–74, 1988. 1                                                           ing for human avatars. In 2022 International Conference on
 [9] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-                   3D Vision (3DV), pages 197–208. IEEE, 2022. 2
     aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.             [21] Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe,
     Fast dynamic radiance fields with time-aware neural vox-                 Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser
     els. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9,                 Sheikh. Panoptic studio: A massively multiview system for
     2022. 1, 2, 4, 5, 6, 7, 12, 13, 14, 15                                   social motion capture. In Proceedings of the IEEE Inter-
[10] John Flynn, Michael Broxton, Paul Debevec, Matthew Du-                   national Conference on Computer Vision, pages 3334–3342,
     Vall, Graham Fyffe, Ryan Overbeck, Noah Snavely, and                     2015. 14, 15
     Richard Tucker. Deepview: View synthesis with learned gra-          [22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler,
     dient descent. In Proceedings of the IEEE/CVF Conference                 and George Drettakis. 3d gaussian splatting for real-time


                                                                     9
     radiance field rendering. ACM Transactions on Graphics                     Representing scenes as neural radiance fields for view syn-
     (ToG), 42(4):1–14, 2023. 1, 3, 4, 5, 6, 7, 11, 12, 14                      thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
[23] Leonid Keselman and Martial Hebert. Approximate differ-                    2, 3
     entiable rendering with algebraic surfaces. In European Con-          [36] Thomas Müller, Alex Evans, Christoph Schied, and Alexan-
     ference on Computer Vision, pages 596–614. Springer, 2022.                 der Keller. Instant neural graphics primitives with a mul-
     3                                                                          tiresolution hash encoding. ACM Transactions on Graphics
[24] Leonid Keselman and Martial Hebert. Flexible techniques                    (ToG), 41(4):1–15, 2022. 1, 3
     for differentiable rendering with 3d gaussians. arXiv preprint        [37] Byeongjun Park and Changick Kim. Point-dynrf: Point-
     arXiv:2308.14737, 2023. 3                                                  based dynamic radiance fields from a monocular video. In
[25] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon                       Proceedings of the IEEE/CVF Winter Conference on Appli-
     Green, Christoph Lassner, Changil Kim, Tanner Schmidt,                     cations of Computer Vision, pages 3171–3181, 2024. 3
     Steven Lovegrove, Michael Goesele, Richard Newcombe,                  [38] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
     et al. Neural 3d video synthesis from multi-view video. In                 Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
     Proceedings of the IEEE/CVF Conference on Computer Vi-                     Martin-Brualla. Nerfies: Deformable neural radiance fields.
     sion and Pattern Recognition, pages 5521–5531, 2022. 2, 6,                 In Proceedings of the IEEE/CVF International Conference
     7, 12, 13, 15                                                              on Computer Vision, pages 5865–5874, 2021. 2, 6, 7, 12
[26] Zhong Li, Yu Ji, Wei Yang, Jinwei Ye, and Jingyi Yu. Ro-              [39] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
     bust 3d human motion reconstruction via dynamic template                   Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
     construction. In 2017 International Conference on 3D Vision                Brualla, and Steven M Seitz.        Hypernerf: A higher-
     (3DV), pages 496–505. IEEE, 2017. 2                                        dimensional representation for topologically varying neural
[27] Zhong Li, Minye Wu, Wangyiteng Zhou, and Jingyi Yu. 4d                     radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2,
     human body correspondences from panoramic depth maps.                      4, 6, 7, 12, 13, 14, 15
     In Proceedings of the IEEE Conference on Computer Vision              [40] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
     and Pattern Recognition, pages 2877–2886, 2018. 2                          James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
                                                                                Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-
[28] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang.
                                                                                perative style, high-performance deep learning library. Ad-
     Neural scene flow fields for space-time view synthesis of dy-
                                                                                vances in neural information processing systems, 32, 2019.
     namic scenes. In Proceedings of the IEEE/CVF Conference
                                                                                5
     on Computer Vision and Pattern Recognition, pages 6498–
                                                                           [41] Sida Peng, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xi-
     6508, 2021. 2
                                                                                aowei Zhou. Representing volumetric videos as dynamic
[29] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
                                                                                mlp maps. In Proceedings of the IEEE/CVF Conference
     sian feature splatting for real-time dynamic view synthesis.
                                                                                on Computer Vision and Pattern Recognition, pages 4252–
     arXiv preprint arXiv:2312.16812, 2023. 3
                                                                                4262, 2023. 2
[30] Haotong Lin, Sida Peng, Zhen Xu, Tao Xie, Xingyi He, Hu-              [42] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
     jun Bao, and Xiaowei Zhou. High-fidelity and real-time                     Francesc Moreno-Noguer. D-nerf: Neural radiance fields
     novel view synthesis for dynamic scenes. In SIGGRAPH                       for dynamic scenes. In Proceedings of the IEEE/CVF Con-
     Asia Conference Proceedings, 2023. 2, 5, 6, 7, 9, 15                       ference on Computer Vision and Pattern Recognition, pages
[31] Xingyu Liu, Mengyuan Yan, and Jeannette Bohg. Meteor-                      10318–10327, 2021. 1, 2, 5, 6, 12, 15
     net: Deep learning on dynamic 3d point cloud sequences. In            [43] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas.
     Proceedings of the IEEE/CVF International Conference on                    Pointnet: Deep learning on point sets for 3d classification
     Computer Vision, pages 9246–9255, 2019. 3                                  and segmentation. In Proceedings of the IEEE conference
[32] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu                            on computer vision and pattern recognition, pages 652–660,
     Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Jo-                       2017. 3
     hannes Kopf, and Jia-Bin Huang. Robust dynamic radiance               [44] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J
     fields. In Proceedings of the IEEE/CVF Conference on Com-                  Guibas. Pointnet++: Deep hierarchical feature learning on
     puter Vision and Pattern Recognition, pages 13–23, 2023. 2                 point sets in a metric space. Advances in neural information
[33] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and                      processing systems, 30, 2017. 3
     Deva Ramanan. Dynamic 3d gaussians: Tracking by per-                  [45] Darius Rückert, Linus Franke, and Marc Stamminger. Adop:
     sistent dynamic view synthesis. In 3DV, 2024. 2, 3, 7, 14,                 Approximate differentiable one-pixel point rendering. ACM
     15                                                                         Transactions on Graphics (ToG), 41(4):1–14, 2022. 3
[34] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi,                [46] Johannes L Schonberger and Jan-Michael Frahm. Structure-
     Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duck-                    from-motion revisited. In Proceedings of the IEEE con-
     worth. Nerf in the wild: Neural radiance fields for uncon-                 ference on computer vision and pattern recognition, pages
     strained photo collections. In Proceedings of the IEEE/CVF                 4104–4113, 2016. 5, 6, 7
     Conference on Computer Vision and Pattern Recognition,                [47] Johannes L Schonberger and Jan-Michael Frahm. Structure-
     pages 7210–7219, 2021. 4                                                   from-motion revisited. In Proceedings of the IEEE con-
[35] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,                       ference on computer vision and pattern recognition, pages
     Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:                     4104–4113, 2016. 12


                                                                      10
[48] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,                  [59] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin
     Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural                  Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf:
     4d decomposition for high-fidelity dynamic reconstruction                 Point-based neural radiance fields. In Proceedings of the
     and rendering. In Proceedings of the IEEE/CVF Conference                  IEEE/CVF Conference on Computer Vision and Pattern
     on Computer Vision and Pattern Recognition, pages 16632–                  Recognition, pages 5438–5448, 2022. 3
     16642, 2023. 1, 2, 4                                                 [60] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
[49] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele                    Zhang, and Xiaogang Jin. Deformable 3d gaussians for
     Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerf-                      high-fidelity monocular dynamic scene reconstruction. arXiv
     player: A streamable dynamic scene representation with de-                preprint arXiv:2309.13101, 2023. 3, 6
     composed neural radiance fields. IEEE Transactions on Visu-          [61] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li
     alization and Computer Graphics, 29(5):2732–2742, 2023.                   Zhang. Real-time photorealistic dynamic scene representa-
     5, 6, 7, 13                                                               tion and rendering with 4d gaussian splatting. arXiv preprint
[50] Zhuo Su, Lan Xu, Zerong Zheng, Tao Yu, Yebin Liu, and Lu                  arXiv:2310.10642, 2023. 2, 3
     Fang. Robustfusion: Human volumetric capture with data-              [62] Taoran Yi, Jiemin Fang, Xinggang Wang, and Wenyu Liu.
     driven visual cues using a rgbd camera. In Computer Vision–               Generalizable neural voxels for fast human radiance fields.
     ECCV 2020: 16th European Conference, Glasgow, UK, Au-                     arXiv preprint arXiv:2303.15387, 2023. 2
     gust 23–28, 2020, Proceedings, Part IV 16, pages 246–264.            [63] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Öztireli,
     Springer, 2020. 2                                                         and Olga Sorkine-Hornung. Differentiable surface splatting
[51] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel                    for point-based geometry processing. ACM Transactions on
     grid optimization: Super-fast convergence for radiance fields             Graphics (TOG), 38(6):1–14, 2019. 2, 3, 4, 7
     reconstruction. In Proceedings of the IEEE/CVF Conference            [64] Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and
     on Computer Vision and Pattern Recognition, pages 5459–                   Pheng-Ann Heng. Pu-net: Point cloud upsampling network.
     5469, 2022. 1, 3, 5                                                       In Proceedings of the IEEE conference on computer vision
[52] Fengrui Tian, Shaoyi Du, and Yueqi Duan. Monon-                           and pattern recognition, pages 2790–2799, 2018. 3
     erf: Learning a generalizable dynamic radiance field from            [65] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen
     monocular videos. In Proceedings of the IEEE/CVF Interna-                 Koltun. Nerf++: Analyzing and improving neural radiance
     tional Conference on Computer Vision, pages 17903–17913,                  fields. arXiv preprint arXiv:2010.07492, 2020. 2
     2023. 2                                                              [66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
[53] Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, and                     man, and Oliver Wang. The unreasonable effectiveness of
     Huaping Liu. Masked space-time hash encoding for efficient                deep features as a perceptual metric. In Proceedings of the
     dynamic scene reconstruction. Advances in neural informa-                 IEEE conference on computer vision and pattern recogni-
     tion processing systems, 2023. 2, 5, 6, 7                                 tion, pages 586–595, 2018. 6
[54] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei                 [67] Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu,
     Song, and Huaping Liu. Mixed neural voxels for fast multi-                Yiyuan Yang, Andrew Markham, and Niki Trigoni. Dyn-
     view video synthesis. In Proceedings of the IEEE/CVF In-                  point: Dynamic neural point for view synthesis. Advances in
     ternational Conference on Computer Vision, pages 19706–                   Neural Information Processing Systems, 36, 2024. 2, 3
     19716, 2023. 2, 5, 13                                                [68] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
[55] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P                       Markus Gross. Surface splatting. In Proceedings of the
     Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo                       28th annual conference on Computer graphics and interac-
     Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibr-                 tive techniques, pages 371–378, 2001. 3
     net: Learning multi-view image-based rendering. In Pro-
     ceedings of the IEEE/CVF Conference on Computer Vision               A. Appendix
     and Pattern Recognition, pages 4690–4699, 2021. 15
[56] Yiming Wang, Qin Han, Marc Habermann, Kostas Dani-                   In the supplementary material, we mainly introduce our hy-
     ilidis, Christian Theobalt, and Lingjie Liu. Neus2: Fast             perparameter settings of experiments in Sec. A.1. Then
     learning of neural implicit surfaces for multi-view recon-           more ablation studies are conducted in Sec. A.2. Finally,
     struction. In Proceedings of the IEEE/CVF International              we delve into the limitations of our proposed 4D-GS in
     Conference on Computer Vision, pages 3295–3306, 2023. 2              Sec. A.3.
[57] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
     moncelli. Image quality assessment: from error visibility to         A.1. Hyperparameter Settings
     structural similarity. IEEE transactions on image processing,
     13(4):600–612, 2004. 6                                               Our hyperparameters mainly follow the settings of 3D-
[58] Qingshan Xu, Weihang Kong, Wenbing Tao, and Marc Polle-              GS [22]. The basic resolution of our multi-resolution Hex-
     feys. Multi-scale geometric consistency guided and planar            Plane module R(i, j) is set to 64, which is upsampled by 2
     prior assisted multi-view stereo. IEEE Transactions on Pat-          and 4. The learning rate is set as 1.6 × 10−3 , decayed to
     tern Analysis and Machine Intelligence, 45(4):4945–4963,             1.6 × 10−4 at the end of training. The Gaussian deforma-
     2022. 2                                                              tion decoder is a tiny MLP with a learning rate of 1.6×10−4


                                                                     11
       (a) Cook Spinach                      (b) Cut Beef                        (c) Flame Salmon                      (d) Coffee Martini

Figure 10. More visualization of composition in 4D Gaussians. (a) Composition with Punch and Standup. (b) Composition with Lego and
Trex. (c) Composition with Hellwarrior and Mutant. (d) Composition with Bouncingballs and Jumpingjacks.

                           Table 5. Perscene results on the HyperNeRF vrig dataset [39] of different models.

                                    3D Printer                     Chicken                  Broom                      Banana
         Method
                               PSNR      MS-SSIM           PSNR       MS-SSIM       PSNR      MS-SSIM        PSNR          MS-SSIM
         Nerfies [38]           20.6        0.83            26.7        0.94         19.2           0.56        22.4          0.87
         HyperNeRF [39]         20.0        0.59            26.9        0.94         19.3           0.59        23.3          0.90
         TiNeuVox-B [9]         22.8        0.84            28.3        0.95         21.5           0.69        24.4          0.87
         FFDNeRF [19]           22.8        0.84            28.0        0.94         21.9           0.71        24.3          0.86
         3D-GS [22]             18.3        0.60            19.7        0.70         20.6           0.63        20.4          0.80
         Ours                   22.1        0.81            28.7        0.93         22.0           0.70        28.0          0.94




      （a) Ours w/o dx                           （b) Ours

      Figure 11. Visualization of ablation study about ϕx .


                                                                                (a) Ours         (b) Ours w 𝜑    ,𝜑          (c) TiNeuVox
which decreases to 1.6 × 10−5 . The batch size in training
is set to 1. The opacity reset operation in 3D-GS [22] is not              Figure 12. Visualization of ablation study in ϕC and ϕα comparing
used as it does not bring evident benefit in most of our tested            with TiNeuVox [9].
scenes. Besides, we find that expanding the batch size will
indeed contribute to rendering quality but the training cost
increases accordingly.                                                     from growing at the iteration of 15000.
    Different datasets are constructed under different captur-                The Neu3D dataset [25] includes 15 – 20 fixed camera
ing settings. D-NeRF [42] is a synthetic dataset in which                  setups, so it’s easy to get the SfM [47] point in the first
each timestamp has only one single captured image follow-                  frame. We utilize the dense point-cloud reconstruction and
ing the monocular setting. This dataset has no background                  downsample it lower than 100k to avoid out of memory er-
which is easy to train, and can reveal the upper bound of                  ror. Thanks to the efficient design of our 4D Gaussian splat-
our proposed framework. We change the pruning interval                     ting framework and the tiny movement of all the scenes,
to 8000 and only set a single upsampling rate of the multi-                only 14000 iterations are needed and we can get the high
resolution HexPlane Module R(i, j) as 2 because the struc-                 rendering quality images.
ture information is relatively simple in this dataset. The                    HyperNeRF dataset [39] is captured with fewer than 2
training iteration is set to 20000 and we stop 3D Gaussians                cameras in feed-forward settings. We change the upsam-


                                                                      12
          (a) 𝑅 (𝑥, 𝑦)                       (b) 𝑅 (𝑥, 𝑧)                        (c) 𝑅 (𝑦, 𝑧)                   (d) Training View 1




          (e) 𝑅 (𝑥, 𝑡)                       (f) 𝑅 (𝑦, 𝑡)                        (g) 𝑅 (𝑧, 𝑡)                   (h) Training View 2

Figure 13. More visualization of the HexPlane voxel grids R(i, j) in bouncing balls. (a)-(c), (e)-(f) stand for visualization of R1 (i, j),
where grids resolution equals to 64×64.

                                         Table 6. Per-scene results on the DyNeRF [25] dataset.

                                                       Cut Beef          Cook Spinach            Sear Steak
                             Method
                                                    PSNR      SSIM       PSNR      SSIM     PSNR        SSIM
                             NeRFPlayer [49]        31.83     0.928      32.06     0.930        32.31   0.940
                             HexPlane [5]           32.71     0.985      31.86     0.983        32.09   0.986
                             KPlanes [12]           31.82     0.966      32.60     0.966        32.52   0.974
                             MixVoxels [54]         31.30     0.965      31.65     0.965        31.43   0.971
                             Ours                   32.90     0.957      32.46     0.949        32.49   0.957
                                                     Flame Steak         Flame Salmon       Coffee Martini
                             Method
                                                    PSNR      SSIM       PSNR      SSIM     PSNR        SSIM
                             NeRFPlayer [49]        27.36     0.867      26.14     0.849        32.05   0.938
                             HexPlane [5]           31.92     0.988      29.26     0.980          -       -
                             KPlanes [12]           32.39     0.970      30.44     0.953        29.99   0.953
                             MixVoxels [54]         31.21     0.970      29.92     0.945        29.36   0.946
                             Ours                   32.51     0.954      29.20     0.917        27.34   0.905


pling resolution up to [2, 4] and the hidden dim of the de-              future works.
coder to 128. Similar to other works [9, 39], we found
that Gaussian deformation fields always fall into the local              A.2. More Ablation Studies
minima that link the correlation of motion between cameras
and objects even with static 3D Gaussian initialization. And             Editing with 4D Gaussians. We provide more visualiza-
we’re going to reserve the splitting of the relationship in the          tion in editing with 4D Gaussians in Fig. 10. This work
                                                                         only proposes a naive approach to transformation. It is


                                                                    13
                                                         Table 7. Per-scene results on synthetic datasets.

                             Bouncing Balls                         Hellwarrior                         Hook                      Jumpingjacks
   Method
                         PSNR     SSIM          LPIPS       PSNR       SSIM       LPIPS      PSNR       SSIM     LPIPS    PSNR       SSIM       LPIPS
   3D-GS [22]            23.20   0.9591         0.0600      24.53     0.9336      0.0580     21.71     0.8876    0.1034   23.20     0.9591      0.0600
   K-Planes[12]          40.05   0.9934         0.0322      24.58     0.9520      0.0824     28.12     0.9489    0.0662   31.11     0.9708      0.0468
   HexPlane[5]           39.86   0.9915         0.0323      24.55     0.9443      0.0732     28.63     0.9572    0.0505   31.31     0.9729      0.0398
   TiNeuVox[9]           40.23   0.9926         0.0416      27.10     0.9638      0.0768     28.63     0.9433    0.0636   33.49     0.9771      0.0408
   Ours                  40.62   0.9942         0.0155      28.71     0.9733      0.0369     32.73     0.9760    0.0272   35.42     0.9857      0.0128
                                   Lego                                Mutant                          Standup                       Trex
   Method
                         PSNR     SSIM          LPIPS       PSNR       SSIM       LPIPS      PSNR       SSIM     LPIPS    PSNR       SSIM       LPIPS
   3D-GS [22]            23.06   0.9290         0.0642      20.64     0.9297      0.0828     21.91     0.9301    0.0785   21.93     0.9539      0.0487
   K-Planes [12]         25.49   0.9483         0.0331      32.50     0.9713      0.0362     33.10     0.9793    0.0310   30.43     0.9737      0.0343
   HexPlane [5]          25.10   0.9388         0.0437      33.67     0.9802      0.0261     34.40     0.9839    0.0204   30.67     0.9749      0.0273
   TiNeuVox [9]          24.65   0.9063         0.0648      30.87     0.9607      0.0474     34.61     0.9797    0.0326   31.25     0.9666      0.0478
   Ours                  25.03   0.9376         0.0382      37.59     0.9880      0.0167     38.11     0.9898    0.0074   34.23     0.9850      0.0131




   (a) Training View         (b) Novel View 1            (c) Novel View 2

Figure 14. Novel view rendering results in the iPhone dataset [15].




              (a) Ours                           (b) Ground Truth                                 (a) Broom                        (b) Teapot

Figure 15. Rendering results on sports dataset [21], also used in                    Figure 16. Failure cases of modeling large motions and dramatic
Dynamic3DGS [33].                                                                    scene changes. (a) The sudden motion of the broom makes op-
                                                                                     timization harder. (b) Teapots have large motion and a hand is
                                                                                     entering/leaving the scene.
Table 8. Ablation Study on ϕC and ϕα , comparing with TiNeu-
Vox [9] in Americano of the HyperNeRF [39] dataset.

                                                                                     s need to be considered. Meanwhile, some interpolation
                                          Americano
           Method                                                                    methods should be applied to enlarge or reduce 4D Gaus-
                                    PSNR          MS-SSIM                            sians.
           TiNeuVox-B [9]            28.4             0.96
           Ours w/ ϕC ,ϕα            31.53            0.97                           Position Deformation. We find that removing the output
           Ours                      30.90            0.96                           of the position deformation head can also model the ob-
                                                                                     ject motion. It is mainly because leaving some 3D Gaus-
                                                                                     sians in the dynamic part, keeping them small in shape, and
worth noting that when applying the rotation of the scenes,                          then scaling them up at a certain timestamp can also model
3D Gaussian’s rotation quaternion r and scaling coefficient                          the dynamic part. However, this approach can only model


                                                                                14
coarse object motion and lost potential for tracking. The               3D Gaussians, which may fail in modeling large motions or
visualization is shown in Fig. 11.                                      dramatic scene changes. This phenomenon is also observed
                                                                        in previous NeRF-based methods [9, 25, 39, 42], produc-
Color and Opacity’s Deformation. When encountered                       ing blurring results. Fig. 16 shows some failed samples.
with fluid or non-rigid motion, we adopt another two out-               Exploring more useful priors could be a promising future
put MLP decoder ϕC , ϕα to compute the deformation of 3D                direction.
Gaussian’s color and opacity ∆C = ϕC (fd ), ∆α = ϕα (fd ).
Tab. 8 and Fig. 12 show the results in comparison with
TiNeuVox [9]. However, it is worth noting that model-
ing Gaussian color and opacity change may cause irrational
shape changes when rendering novel views. i.e. the Gaus-
sians on the surface should move with other Gaussians but
stay in the place and the color is changed, making the track-
ing difficult to achieve.

Spatial-temporal Structure Encoder. We explore why
4D-GS can achieve such a fast convergence speed and ren-
dering quality. As shown in Fig. 13, we visualize the full
features of R1 in bouncingballs. It’s explicit that in the
R1 (x, y) plane, the spatial structure of the scenes is en-
coded. Similarily, R1 (x, z) and R1 (y, z) also show dif-
ferent view structure features. Meanwhile, temporal voxel
grids R1 (x, t), R1 (y, t) and R1 (z, t) also show the inte-
grated motion of the scenes, where large motions always
stand for explicit features. So, it seems that the proposed
HexPlane module encodes the features of spatial and tem-
poral information.

A.3. More Discussions
Monocular Dynamic Scene Novel View Synthesis. In
monocular settings, the input is sparse in both camera pose
and timestamp dimensions. This may cause the local min-
ima of overfitting with training images in some complicated
scenes. As shown in Fig. 14, though 4D-GS can render rel-
atively high quality in the training set, the strong overfitting
effects of the proposed model cause the failure of render-
ing novel views. To solve the problem, more priors such as
depth supervision or optical flow may be needed.

Large Motion Modeling with Multi-Camera Settings.
In the Neu3D [25] dataset, all the motion parts of the scene
are not very large and the multi-view camera setup also pro-
vides a dense sampling of the scene. That is the reason
why 4D-GS can perform a relatively high rendering quality.
However, in large motion such as sports datasets [21] used
in Dynamic 3DGS [33], 4D-GS cannot fit well within short
times as shown in Fig. 15. Online training [1, 33] or using
information from other views like [30, 55] could be a better
approach to solve the problem with multi-camera input.

Large Motion Modeling with Monocular Settings. 4D-
GS uses a deformation field network to model the motion of


                                                                   15
