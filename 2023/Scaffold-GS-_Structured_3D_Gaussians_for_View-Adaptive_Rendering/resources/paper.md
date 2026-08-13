                                                         Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

                                                                         Tao Lu 1,3 * Mulin Yu1 * Linning Xu2 Yuanbo Xiangli4
                                                                                    Limin Wang1,3 Dahua Lin1,2 Bo Dai1
                                                             1
                                                               Shanghai Artificial Intelligence Laboratory, 2 The Chinese University of Hong Kong,
                                                                                    3
                                                                                      Nanjing University, 4 Cornell University




arXiv:2312.00109v1 [cs.CV] 30 Nov 2023
                                               Initial points                                        3D Gaussians               3D-GS RGB                                  Ours RGB                              GT      3D-GS Opacity                          Ours Opacity
                                                                                                     (3D-GS, Ours-anchor)




                                                                                Scaffold-GS




                                                                                                                                                         PSNR: 28.04dB                         PSNR: 28.57dB




                                             (3D-GS) 17.16dB / 242MB / 127FPS                 (Ours) 20.41 dB / 66MB / 110FPS         (3D-GS) 29.93 dB / 288MB / 109 FPS              (Ours) 31.13 dB / 133MB / 128FPS      (3D-GS) 34.60dB / 204MB / 113 FPS        (Ours) 35.41dB / 48MB / 88 FPS



                                         Figure 1. Scaffold-GS represents the scene using a set of 3D Gaussians structured in a dual-layered hierarchy. Anchored on a sparse
                                         grid of initial points, a modest set of neural Gaussians are spawned from each anchor to dynamically adapt to various viewing angles
                                         and distances. Our method achieves rendering quality and speed comparable to 3D-GS with a more compact model (last row metrics:
                                         PSNR/storage size/FPS). Across multiple datasets, Scaffold-GS demonstrates more robustness in large outdoor scenes and intricate indoor
                                         environments with challenging observing views e.g. transparency, specularity, reflection, texture-less regions and fine-scale details.

                                                                                        Abstract                                                                               servations, without sacrificing the rendering speed. Project
                                             Neural rendering methods have significantly advanced                                                                              page: https://city-super.github.io/scaffold-gs/.
                                         photo-realistic 3D scene rendering in various academic and
                                         industrial applications. The recent 3D Gaussian Splatting
                                         method has achieved the state-of-the-art rendering quality
                                         and speed combining the benefits of both primitive-based                                                                              1. Introduction
                                         representations and volumetric representations. However,                                                                              Photo-realistic and real-time rendering of 3D scenes has al-
                                         it often leads to heavily redundant Gaussians that try to                                                                             ways been a pivotal interest in both academic research and
                                         fit every training view, neglecting the underlying scene ge-                                                                          industrial domains, with applications spanning virtual real-
                                         ometry. Consequently, the resulting model becomes less                                                                                ity [51], media generation [36], and large-scale scene vi-
                                         robust to significant view changes, texture-less area and                                                                             sualization [43, 45, 49]. Traditional primitive-based repre-
                                         lighting effects. We introduce Scaffold-GS, which uses an-                                                                            sentations like meshes and points [6, 26, 32, 55] are faster
                                         chor points to distribute local 3D Gaussians, and predicts                                                                            due to the use of rasterization techniques tailored for mod-
                                         their attributes on-the-fly based on viewing direction and                                                                            ern GPUs. However, they often yield low-quality render-
                                         distance within the view frustum. Anchor growing and                                                                                  ings, exhibiting discontinuity and blurry artifacts. In con-
                                         pruning strategies are developed based on the importance                                                                              trast, volumetric representations and neural radiance fields
                                         of neural Gaussians to reliably improve the scene cover-                                                                              utilize learning-based parametric models [3, 5, 30], hence
                                         age. We show that our method effectively reduces redun-                                                                               can produce continuous rendering results with more details
                                         dant Gaussians while delivering high-quality rendering. We                                                                            preserved. Nevertheless, they come with the cost of time-
                                         also demonstrates an enhanced capability to accommodate                                                                               consuming stochastic sampling, leading to slower perfor-
                                         scenes with varying levels-of-detail and view-dependent ob-                                                                           mance and potential noise.
                                            * denotes equal contribution.                                                                                                          In recent times, 3D Gaussian Splatting (3D-GS) [22] has


                                                                                                                                                                       1
achieved state-of-the-art rendering quality and speed. Ini-             2. Related work
tialized from point clouds derived from Structure from Mo-
tion (SfM) [42], this method optimizes a set of 3D Gaus-                MLP-based Neural Fields and Rendering. Early neu-
sians to represent the scene. It preserves the inherent con-            ral fields typically adopt a multi-layer perceptron (MLP) as
tinuity found in volumetric representations, whilst facilitat-          the global approximator of 3D scene geometry and appear-
ing rapid rasterization by splatting 3D Gaussians onto 2D               ance. They directly use spatial coordinates (and viewing
image planes.                                                           direction) as input to the MLP and predict point-wise at-
                                                                        tribute, e.g. signed distance to scene surface (SDF) [33, 34,
    While this approach offers several advantages, it tends             46, 54], or density and color of that point [2, 30, 49]. Be-
to excessively expand Gaussian balls to accommodate ev-                 cause of its volumetric nature and inductive bias of MLPs,
ery training view, thereby neglecting scene structure. This             this stream of methods achieves the SOTA performance in
results in significant redundancy and limits its scalability,           novel view synthesis. The major challenge of this scene rep-
particularly in the context of complex large-scale scenes.              resentation is that the MLP need to be evaluated on a large
Furthermore, view-dependent effects are baked into indi-                number of sampled points along each camera ray. Con-
vidual Gaussian parameters with little interpolation capa-              sequently, rendering becomes extremely slow, with limited
bilities, making it less robust to substantial view changes             scalability towards complex and large-scale scenes. Despite
and lighting effects.                                                   several works have been proposed to accelerate or mitigate
                                                                        the intensive volumetric ray-marching, e.g. using proposal
    We present Scaffold-GS, a Gaussian-based approach that              network [4], baking technique [11, 19], and surface render-
utilizes anchor points to establish a hierarchical and region-          ing [41]. They either incorporated more MLPs or traded
aware 3D scene representation. We construct a sparse grid               rendering quality for speed.
of anchor points initiated from SfM points. Each of these
anchors tethers a set of neural Gaussians with learnable off-
sets, whose attributes (i.e. opacity, color, rotation, scale) are       Grid-based Neural Fields and Rendering. This type of
dynamically predicted based on the anchor feature and the               scene representations are usually based on a dense uniform
viewing position. Unlike the vanilla 3D-GS which allows                 grid of voxels. They have been greatly used in 3D shape
3D Gaussians to freely drift and split, our strategy exploits           and geometry modeling [12, 15, 21, 29, 35, 44, 57]. Some
scene structure to guide and constrain the distribution of 3D           recent methods have also focused on faster training and in-
Gaussians, whilst allowing them to locally adaptive to vary-            ference of radiance field by exploiting spatial data struc-
ing viewing angles and distances. We further develop the                ture to store scene features, which were interpolated and
corresponding growing and pruning operations for anchors                queried by sampled points during ray-marching. For in-
to enhance the scene coverage.                                          stance, Plenoxel [13] adopted a sparse voxel grid to inter-
                                                                        polate a continuous density field, and represented view-
   Through extensive experiments, we show that our                      dependent visual effects with Spherical Harmonics. The
method delivers rendering quality on par with or even sur-              idea of tensor factorization has been studied in multiple
passing the original 3D-GS. At inference time, we limit                 works [9, 10, 50, 52] to further reduce data redundancy and
the prediction of neural Gaussians to anchors within the                speed-up rendering. K-planes [14] used neural planes to
view frustum, and filter out trivial neural Gaussians based             parameterize a 3D scene, optionally with an additional tem-
on their opacity with a filtering step (i.e. learnable selec-           poral plane to accommodate dynamics. Several generative
tor). As a result, our approach can render at a similar speed           works [8, 40] also capitalized on triplane structure to model
(around 100 FPS at 1K resolution) as the original 3D-GS                 a 3D latent space for better geometry consistency. Instant-
with little computational overhead. Moreover, our storage               NGP [31] used a hash grid and achieved drastically faster
requirements are significantly reduced as we only need to               feature query, enabling real-time rendering of neural radi-
store anchor points and MLP predictors for each scene.                  ance field. Although these approaches can produce high-
                                                                        quality results and are more efficient than global MLP rep-
   In conclusion, our contributions are: 1) Leveraging scene
                                                                        resentation, they still need to query many samples to render
structure, we initiate anchor points from a sparse voxel
                                                                        a pixel, and struggle to represent empty space effectively.
grid to guide the distribution of local 3D Gaussians, form-
ing a hierarchical and region-aware scene representation; 2)
Within the view frustum, we predict neural Gaussians from               Point-based Neural Fields and Rendering. Point-based
each anchor on-the-fly to accommodate diverse viewing di-               representations utilize the geometric primitive (i.e. point
rections and distances, resulting in more robust novel view             clouds) for scene rendering. A typical procedure is to ras-
synthesis; 3) We develop a more reliable anchor growing                 terize an unstructured set of points using a fixed size, and
and pruning strategy utilizing the predicted neural Gaus-               exploits specialized modules on GPU and graphics APIs for
sians for better scene coverage.                                        rendering [7, 37, 38]. In spite of its fast speed and flexibil-

                                                                    2
ity to solve topological changes, they usually suffer from               of a 3D Gaussian:
holes and outliers that lead to artifacts in rendering. To alle-                                            1      T    −1
viate the discontinuity issue, differentiable point-based ren-                          G(x) = e− 2 (x−µ) Σ                  (x−µ)
                                                                                                                                     ,        (1)
dering has been extensively studied to model objects geom-
etry [16, 20, 27, 48, 55]. In particular, [48, 55] used dif-             where x is an arbitrary position within the 3D scene and
ferentiable surface splatting that treats point primitives as            Σ denotes the covariance matrix of the 3D Gaussian. Σ is
discs, ellipsoids or surfels that are larger than a pixel. [1, 24]       formulated using a scaling matrix S and rotation matrix R
augmented points with neural features and rendered using                 to maintain its positive semi-definite:
2D CNNs. As a comparison, Point-NeRF [53] achieved
high-quality novel view synthesis utilizing 3D volume ren-                                           Σ = RSS T RT ,                           (2)
dering, along with region growing and point pruning dur-
ing optimization. However, they resorted to volumetric ray-              In addition to color c modeled by Spherical harmonics, each
marching, hence hindered display rate. Notably, the recent               3D Gaussian is associated with an opacity α which is mul-
work 3D-GS [22] employed anisotropic 3D Gaussians ini-                   tiplied by G(x) during the blending process.
tialized from structure from motion (SfM) to represent 3D                    Distinct from conventional volumetric representations,
scenes, where a 3D Gaussian was optimized as a volume                    3D-GS efficiently renders the scene via tile-based rasteri-
and projected to 2D to be rasterized as a primitive. Since               zation instead of resource-intensive ray-marching. The 3D
it integrated pixel color using α-blender, 3D-GS produced                Gaussian G(x) are first transformed to 2D Gaussians G′ (x)
high-quality results with fine-scale detail, and rendered at             on the image plane following the projection process as de-
real-time frame rate.                                                    scribed in [58]. Then a tile-based rasterizer is designed to
                                                                         efficiently sort the 2D Gaussians and employ α-blending:
3. Methods                                                                             X             i−1
                                                                                                     Y
                                                                            C(x′ ) =         ci σi         (1 − σj ),    σi = αi G′i (x′ ),   (3)
The original 3D-GS [22] optimizes Gaussians to reconstruct                             i∈N           j=1
every training view, with heuristic splitting and pruning op-
erations but in general neglects the underlying scene struc-             where x′ is the queried pixel position and N denotes the
ture. This often leads to highly redundant Gaussians and                 number of sorted 2D Gaussians associated with the queried
makes the model less robust to novel viewing angles and                  pixel. Leveraging the differentiable rasterizer, all attributes
distances. To address this issue, we propose a hierarchical              of the 3D Gaussians are learnable and directly optimized
3D Gaussian scene representation that respects the scene                 end-to-end via training view reconstruction.
geometric structure, with anchor points initialized from
SfM to encode local scene information and spawn local neu-               3.2. Scaffold-GS
ral Gaussians. The physical properties of neural Gaussians
                                                                         3.2.1   Anchor Point Initialization
are decoded from the learned anchor features in a view-
dependent manner on-the-fly. Fig. 2 illustrates our frame-               Consistent with existing methods [22, 53], we use the sparse
work. We start with a brief background of 3D-GS then un-                 point cloud from COLMAP [39] as our initial input. We
fold our proposed method in details. Sec. 3.2.1 introduces               then voxelize the scene from this point cloud P ∈ RM ×3
how to initialize the scene with a regular sparse grid of an-            as:                        
chor points from the irregular SfM point clouds. Sec. 3.2.2                                           P
                                                                                              V=            · ϵ,                  (4)
explains how we predict neural Gaussians properties based                                              ϵ
on anchor points and view-dependent information. To make
our method more robust to the noisy initialization, Sec. 3.3             where V ∈ RN ×3 denotes voxel centers, and ϵ is the voxel
introduces a neural Gaussian based “growing” and “prun-                  size. We then remove duplicate entries, denoted by {·} to
ing” operations to refine the anchor points. Sec. 3.4 elabo-             reduce the redundancy and irregularity in P.
rates training details.                                                     The center of each voxel v ∈ V is treated as an anchor
                                                                         point, equipped with a local context feature fv ∈ R32 , a
3.1. Preliminaries                                                       scaling factor lv ∈ R3 , and k learnable offsets Ov ∈ Rk×3 .
                                                                         In a slight abuse of terminology, we will denote the anchor
3D-GS [22] represents the scene with a set of anisotropic                point as v in the following context. We further enhance fv
3D Gaussians that inherit the differential properties of vol-            to be multi-resolution and view-dependent. For each anchor
umetric representation while be efficiently rendered via a               v, we 1) create a features bank {fv , fv↓1 , fv↓2 }, where ↓n
tile-based rasterization.                                                denotes fv being down-sampled by 2n factors; 2) blend the
    Starting from a set of Structure-from-Motion (SfM)                   feature bank with view-dependent weights to form an inte-
points, each point is designated as the position (mean) µ                grated anchor feature fˆv . Specifically, Given a camera at


                                                                     3
   (a) Sparse Voxel from SfM Points                                                                                                              (c) Neural Gaussian Splatting
                                                                         (b) Neural Gaussian Derivation (k=4)
                                                                                                                                                         & 𝞪-blending
            Visible Voxels
                                               Position & Opacity                                                  Color, Scale & Quaternion
                                                                                    Each Voxel
                                                                                   𝛼 < 𝜏𝛼

                                                                                          O1
                                                                         F𝞪 -> opac.               O2
                                                                         Fc -> rgb
                                                                                                    O3
                                                                         Fs -> scale
                                                                         Fq -> quatrn.
                                                                                           O4                                                             L1, LSSIM, Lvol

                                                                                          S(fa)k


                                      anchor          neural Gaussian
                                      voxel           learnable offset                                                                              Rendered RGB        GT


Figure 2. Overview of Scaffold-GS. (a) We start by forming a sparse voxel grid from SfM-derived points. An anchor associated with
a learnable scale is placed at the center of each voxel, roughly sculpturing the scene occupancy. (b) Within a view frustum, k neural
Gaussians are spawned from each visible anchor with offsets {Ok }. Their attributes, i.e. opacity, color, scale and quaternion are then
decoded from the anchor feature, relative camera-anchor viewing direction and distance using Fα , Fc , Fs , Fq respectively. (c) Note that
to alleviate redundancy and improve efficiency, only non-trivial neural Gussians (i.e. α ≥ τα ) are rasterized following [22]. The rendered
image is supervised via reconstruction (L1 ), structural similarity (LSSIM ) and a volume regularization (Lvol ).


position xc and an anchor at position xv , we calculate their                                      denoted as Fα , Fc , Fq and Fs . Note that attributes are de-
relative distance and viewing direction with:                                                      coded in one-pass. For example, opacity values of neural
                                                                                                   Gaussians spawned from an anchor point are given by:
                                                     xv − xc
              δvc = ∥xv − xc ∥2 , ⃗dvc =                        ,                  (5)
                                                    ∥xv − xc ∥2                                                 {α0 , ..., αk−1 } = Fα (fˆv , δvc , ⃗dvc ),                  (9)

then weighted sum the feature bank with weights predicted                                          their colors {ci }, quaternions {qi } and scales {si } are simi-
from a tiny MLP Fw :                                                                               larly derived. Implementation details are in supplementary.
                                                                                                       Note that the prediction of neural Gaussian attributes
        {w, w1 , w2 } = Softmax(Fw (δvc , ⃗dvc )),                                 (6)             are on-the-fly, meaning that only anchors visible within
                  fˆv = w · fv + w1 · fv↓1 + w2 · fv↓2 ,                           (7)             the frustum are activated to spawn neural Gaussians. To
                                                                                                   make the rasterization more efficient, we only keep neural
                                                                                                   Gaussians whose opacity values are larger than a predefined
3.2.2     Neural Gaussian Derivation                                                               threshold τα . This substantially cuts down the computa-
In this section, we elaborate on how our approach derives                                          tional load and helps our method maintain a high rendering
neural Gaussians from anchor points. Unless specified oth-                                         speed on-par with the original 3D-GS.
erwise, F∗ represents a particular MLP throughout the sec-                                         3.3. Anchor Points Refinement
tion. Moreover, we introduce two efficient pre-filtering
strategies to reduce MLP overhead.                                                                 Growing Operation. Since neural Gaussians are closely
    We parameterize a neural Gaussian with its position                                            tied to their anchor points which are initialized from SfM
µ ∈ R3 , opacity α ∈ R, covariance-related quaternion                                              points, their modeling power is limited to a local region, as
q ∈ R4 and scaling s ∈ R3 , and color c ∈ R3 . As shown                                            has been pointed out in [22, 53]. This poses challenges to
in Fig. 2(b), for each visible anchor point within the view-                                       the initial placement of anchor points, especially in texture-
ing frustum, we spawn k neural Gaussians and predict their                                         less and less observed areas. We therefore propose an error-
attributes. Specifically, given an anchor point located at xv ,                                    based anchor growing policy that grows new anchors where
the positions of its neural Gaussians are calculated as:                                           neural Gaussians find significant. To determine a significant
                                                                                                   area, we first spatially quantize the neural Gaussians by con-
         {µ0 , ..., µk−1 } = xv + {O0 , . . . , Ok−1 } · lv ,                      (8)             structing voxels of size ϵg . For each voxel, we compute the
                                                                                                   averaged gradients of the included neural Gaussians over
where {O0 , O1 , ..., Ok−1 } ∈ Rk×3 are the learnable offsets                                      N training iterations, denoted as ∇g . Then, voxels with
and lv is the scaling factor associated with that anchor, as                                       ∇g > τg is deemed as significant, where τg is a pre-defined
described in Sec. 3.2.1. The attributes of k neural Gaussians                                      threshold; and a new anchor point is thereby deployed at
are directly decoded from the anchor feature fˆv , the relative                                    the center of that voxel if there was no anchor point estab-
viewing distance δvc and direction ⃗dvc between the cam-                                           lished. Fig. 3 illustrates this growing operation. In practice,
era and the anchor point (Eq. 5) through individual MLPs,                                          we quantize the space into multi-resolution voxel grid to al-


                                                                                          4
                                                                                     all available scenes tested in the 3D-GS [22], including
                                                                                     seven scenes from Mip-NeRF360 [4], two scenes from
                                                                                     Tanks&Temples [23], two scenes from DeepBlending [18]
                                                                                     and synthetic Blender dataset [30]. We additionally evalu-
                  Gradient (small to large)
                                                                                     ated on datasets with contents captured at multiple LODs
                     Multi-res voxel                                                 to demonstrate our advantages in view-adaptive rendering.
                   (colored for new anchor)
                                                                                     Six scenes from BungeeNeRF [49] and two scenes from
Figure 3. Growing operation. We develop an anchor growing                            VR-NeRF [51] are selected. The former provides multi-
policy guided by the gradients of the neural Gaussians. From                         scale outdoor observations and the latter captures intricate
left to right, we spatially quantize neural Gaussians into multi-                    indoor environments. Apart from the commonly used met-
                                             (m)
resolution voxels (m ∈ {1, 2, 3}) of size {ϵg }. New anchors                         rics (PSNR, SSIM [47], and LPIPS [56]), we additionally
                                                            (m)
are added to voxels with aggregated gradients larger than {τg }.                     report the storage size (MB) and the rendering speed (FPS)
                                                                                     for model compactness and performance efficiency. We
                                                                                     provide the averaged metrics over all scenes of each dataset
low new anchors to be added at different granularity, where                          in the main paper and leave the full quantitative results on
                                                                                     each scene in the supplementary.
          ϵ(m)
           g   = ϵg /4m−1 ,                         τg(m) = τg ∗ 2m−1 ,   (10)

where m denotes the level of quantization. To further regu-                          Baseline and Implementation. 3D-GS [22] is selected as
late the addition of new anchors, we apply a random elimi-                           our main baseline for its established SOTA performance in
nation to these candidates. This cautious approach to adding                         novel view synthesis. Both 3D-GS and our method were
points effectively curbs the rapid expansion of anchors.                             trained for 30k iterations. We also record the results of Mip-
                                                                                     NeRF360 [4], iNGP [31] and Plenoxels [13] as in [22].
Pruning Operation To eliminate trivial anchors, we ac-                                   For our method, we set k = 10 for all experiments. All
cumulate the opacity values of their associated neural Gaus-                         the MLPs employed in our approach are 2-layer MLPs with
sians over N training iterations. If an anchor fails to pro-                         ReLU activation; the dimensions of the hidden units are all
duce neural Gaussians with a satisfactory level of opacity,                          32. For anchor points refinement, we average gradients over
we then remove it from the scene.                                                    N = 100 iterations, and by default use τg = 64ϵ. On intri-
                                                                                     cate scenes and the ones with dominant texture-less regions,
3.4. Losses Design                                                                   we use τg = 16ϵ. An anchor is pruned if the accumulated
                                                                                     opacity of its neural Gaussians is less than 0.5 at each round
We optimize the learnable parameters and MLPs with re-
                                                                                     of refinement. The two loss weights λSSIM and λvol are set
spect to the L1 loss over rendered pixel colors, with SSIM
                                                                                     to 0.2 and 0.001 in our experiments. Please check the sup-
term [47] LSSIM and volume regularization [28] Lvol . The
                                                                                     plementary material for more details.
total supervision is given by:
                                                                                     4.2. Results Analysis
             L = L1 + λSSIM LSSIM + λvol Lvol ,                           (11)
                                                                                     Our evaluation was conducted on diverse datasets, ranging
where the volume regularization Lvol is:                                             from synthetic object-level scenes, indoor and outdoor envi-
                                                                                     ronments, to large-scale urban scenes and landscapes. A va-
                                              Nng
                                              X                                      riety of improvements can be observed especially on chal-
                       Lvol =                       Prod(si ).            (12)       lenging cases, such as texture-less area, insufficient obser-
                                              i=1
                                                                                     vations, fine-scale details and view-dependent light effects.
Here, Nng denotes the number of neural Gaussians in the                              See Fig. 1 and Fig. 4 for examples.
scene and Prod(·) is the product of the values of a vector,
e.g., in our case the scale si of each neural Gaussian. The                          Comparisons. In assessing the quality of our approach,
volume regularization term encourages the neural Gaus-                               we compared with 3D-GS [22], Mip-NeRF360 [4],
sians to be small with minimal overlapping.                                          iNGP [31] and Plenoxels [13] on real-world datasets. Qual-
                                                                                     itative results are presented in Tab. 1. The quality met-
4. Experiments                                                                       rics for Mip-NeRF360, iNGP and Plenoxels align with
                                                                                     those reported in the 3D-GS study. It can be noticed that
4.1. Experimental Setup
                                                                                     our approach achieves comparable results with the SOTA
Dataset and Metrics. We conducted a comprehen-                                       algorithms on Mip-NeRF360 dataset, and surpassed the
sive evaluation across 27 scenes from publicly avail-                                SOTA on Tanks&Temples and DeepBlending, which cap-
able datasets. Specifically, we tested our approach on                               tures more challenging environments with the presence


                                                                                 5
Table 1. Quantitative comparison to previous methods on real-world datasets. Competing metrics are extracted from respective papers.

                           Dataset                                   Mip-NeRF360                                                   Tanks&Temples                                                         Deep Blending
                       Method Metrics                          PSNR ↑ SSIM ↑ LPIPS ↓                                          PSNR ↑ SSIM ↑ LPIPS ↓                                                PSNR ↑ SSIM ↑ LPIPS ↓
                        3D-GS [22]                               28.69                 0.870               0.182                    23.14              0.841                    0.183                   29.41               0.903                 0.243
                      Mip-NeRF360 [4]                            29.23                 0.844               0.207                    22.22              0.759                    0.257                   29.40               0.901                 0.245
                         iNPG [31]                               26.43                 0.725               0.339                    21.72              0.723                    0.330                   23.62               0.797                 0.423
                       Plenoxels [13]                            23.62                 0.670               0.443                    21.08              0.719                    0.379                   23.06               0.795                 0.510
                           Ours                                  28.84                 0.848               0.220                    23.96              0.853                    0.177                   30.21               0.906                 0.254

             GT (scene name)                    Ours (frame PSNR / avg PSNR)                 3D-GS (frame PSNR / avg PSNR)                          GT (scene name)                           Ours (frame PSNR / avg PSNR)                    3D-GS (frame PSNR / avg PSNR)




Mip360-Room(a)                           29.88 / 31.93                                 27.76 / 31.52                                    Mip360-Counter(a)                               31.51 / 29.34                                    30.15 / 28.88




Mip360-Room(b)                           33.46 / 31.93                                 32.32 / 31.52                                    Mip360-Counter(b)                               30.12 / 29.34                                    29.10 / 28.88




                           TandT-Train                                23.52 / 22.15                                 22.56 / 21.90                                     DB-DrJohnson                                       30.17 / 29.8                                    28.48 / 28.95




                           TandT-Truck                                 26.76 / 25.77                                26.00 / 25.23                                         DB-Playroom                                    34.14 / 30.62                                   32.04 / 29.80


                                                                                                                                                            Closer view                                    Closer view                                     Closer view




VR-Kitchen                               26.97 / 29.61                                 21.80 / 29.40                                    VR-Apartment                                    27.04 / 28.87       30.88                        22.76 / 28.48      26.24


Figure 4. Qualitative comparison of Scaffold-GS and 3D-GS [22] across diverse datasets [4, 17, 23, 51]. Patches that highlight the
visual differences are emphasized with arrows and green & yellow insets for clearer visibility. Our approach constantly outperforms 3D-GS
on these scenes, with evident advantages in challenging scenarios, e.g. thin geometry and fine-scale details (M IP 360-ROOM(a), M IP 360-
C OUNTER(a)), texture-less regions (DB-D R J OHNSON, DB-P LAYROOM), light effects (M IP 360-C OUNTER(b), DB-D R J OHNSON), in-
sufficient observations (TAND T-T RAIN, VR-K ITCHEN). It can also be observed (e.g. VR-A PARTMENT) that our model is superior in
representing contents at varying scales and viewing distances.

Table 2. Performance comparison. Rendering FPS and storage                                                                                  Tab. 2. Our method achieved real-time rendering while us-
size are reported. Storage size reduction ratio is indicated by (↓).                                                                        ing less storage, indicating that our model is more com-
Rendering speed of both methods are measured on our machine.                                                                                pact than 3D-GS without sacrificing rendering quality and
                                                                                                                                            speed. Additionally, akin to prior grid-based methods, our
  Dataset           Mip-NeRF360                           Tanks&Temples                    Deep Blending
                  FPS   Mem (MB)                         FPS Mem (MB)                    FPS   Mem (MB)                                     approach converged faster than 3D-GS. See supplementary
  3D-GS             97             693                   123        411                   109               676                             material for more analysis.
   Ours            102         156 (4.4× ↓)              110    87 (4.7× ↓)               139          66 (10.2× ↓)                             We also examined our method on the synthetic Blender
                                                                                                                                            dataset, which provides an exhaustive set of views capturing
                                                                                                                                            objects at 360◦ . A good set of initial SfM points is not read-
of e.g. changing lighting, texture-less regions and reflec-                                                                                 ily available in this dataset, thus we start from 100k grid
tions. In terms of efficiency, we evaluated rendering speed                                                                                 points and learn to grow and prune points with our anchor
and storage size of our method and 3D-GS, as shown in                                                                                       refinement operations. After 30k iterations, we used the re-


                                                                                                                                    6
Table 3. Qualitative comparison. Our method is able to handle
large-scale scenes (e.g. B UNGEE N E RF) with light-weight repre-
sentation. Our method shows consistent compactness and effec-
tiveness in complex lighting conditions and synthetic scenes.

 Dataset      BungeeNeRF             VR-NeRF             Synthetic Blender
           PSNR   Mem (MB)        PSNR Mem (MB)         PSNR Mem (MB)
 3D-GS     24.89      1606        28.94       263       33.32        53
  Ours     27.01   203 (7.9× ↓)   29.24   69 (3.8× ↓)   33.68   14 (3.8× ↓)




                                                                                  Figure 6. Anchor feature clustering. We cluster anchor features
                                                                                  (DB-P LAYROOM) into 3 clusters using K-means [25] and visu-
                                                                                  alize the result. The clustered features show clues of scene con-
                                                                                  tents, e.g. the banister, stroller, desk and monitor can be clearly
                                                                                  identified. Anchors on the wall and floor are also respectively
                                                                                  grouped together. This shows that our approach improves the in-
                                                                                  terpretability of 3D-GS model, and has the potential to be scaled-
Figure 5. Comparison on multi-scale scenes (w/ zoom-in cases).
                                                                                  up on much larger scenes exploiting reusable features.
We showcase the rendering outcomes at an unsceen closer scale on
the A MSTERDAM scene from BungeeNeRF. Our method smoothly
extrapolates to new viewing distances using refined neural Gaus-
                                                                                  Feature Analysis. We further perform an analysis of the
sian properties, remedying the needle-like artifacts of original 3D-
GS caused by fixed Gaussian scaling values.                                       learnable anchor features and the selector mechanism. As
                                                                                  depicted in Fig. 6, the clustered pattern suggests that the
                                                                                  compact anchor feature spaces adeptly capture regions with
                                                                                  similar visual attributes and geometries, as evidenced by
mained points as initialized anchors and re-run our frame-                        their proximity in the encoded feature space.
work. The PSNR score and storage size compared with 3D-
GS are presented in Tab. 3. Fig. 1 also demonstrates that our
                                                                                  View Adaptability. To support that our neural Gaussians
method can achieve better visual quality with more reliable
                                                                                  are view-adaptive, we explore how the values of attributes
geometry and texture details.
                                                                                  change when the same Gaussian is observed from differ-
                                                                                  ent positions. Fig. 7 demonstrates a varying distribution
                                                                                  of attributes intensity at different viewing positions, while
Multi-scale Scene Contents. We examined our model’s                               maintaining a degree of local continuity. This accounts for
capability in handling multi-scale scene details on the                           the superior view adaptability of our method compared to
BungeeNeRF and VR-NeRF datasets. As shown in Tab. 3,                              the static attributes of 3D-GS, as well as its enhanced gen-
our method achieved superior quality whilst using fewer                           eralizability to novel views.
storage size to store the model compared to 3D-GS [22].
As illustrated in Fig. 4 and Fig. 5, our method was supe-                         Selection Process by Opacity. We examine the decoded
rior in accommodating varying levels of detail in the scene.                      opacity from neural Gaussians and our opacity-based selec-
In contrast, images rendered from 3D-GS often suffered                            tion process (Fig. 2(b)) from two aspects. First, without the
from noticeable blurry and needle-shaped artifacts. This                          anchor point refinement module, we filter neural Gaussian
is likely because that Gaussian attributes are optimized to                       using their decoded opacity values to extract geometry from
overfit multi-scale training views, creating excessive Gaus-                      a random point cloud. Fig. 8 demonstrates that the remained
sians that work for each observing distance. However, it can                      neural Gaussians effectively reconstruct the coarse structure
easily lead to ambiguity and uncertainty when synthesizing                        of the bulldozer model from random points, highlighting
novel views, since it lacks the ability to reason about view-                     its capability for implicit geometry modeling under mainly
ing angle and distance. On contrary, our method efficiently                       rendering-based supervision. We found this is conceptually
encoded local structures into compact neural features, en-                        similar to the proposal network utilized in [4], serving as
hancing both rendering quality and convergence speed. De-                         the geometry proxy estimator for efficient sampling.
tails are provided in the supplementary material.                                     Second, we apply different k values in our method.


                                                                              7
                                                                                                             Table 4. Effects of filtering. F ILTER 1 refers to selecting an-
                                                                                                             chors by view frustum and F ILTER 2 refers to the opacity-based
                                                                                                             selection process. The filtering method has no notable impact on
                                                                                                             fidelity, but greatly affects inference speed.

                                                                                                                      Scene        DB-P LAYROOM        DB-D R J OHNSON
                                                                                                                                   PSNR    FPS         PSNR       FPS
                                                                                                                  N O F ILTERS      30.4       84       29.7        79
                                                                                                                   F ILTER 1        30.3      118       29.6       100
                                                                                                                   F ILTER 2        30.6      109       29.7       104
                                                                                                                      F ULL        30.62      150       29.8       129
  Figure 7. View-adaptive neural Gaussian attributes. We visu-
  alize the decoded attributes of a single neural Gaussian observed
                                                                                                             Table 5. Anchor refinement. The growing operation is essential
  at different positions. Each point corresponds to a viewpoint in
                                                                                                             for fidelity since it improves the poor initialization. The pruning
  space. The color of the point denotes the intensity of attributes de-
                                                                                                             operation controls the increasing of storage size and optimizes the
  coded for this view (left: Fs → si ; right: Fα → αi ). This pattern
                                                                                                             quality of remained anchors.
  indicates that attributes of a neural Gaussian adapt to viewpoint
  changing, while exhibiting a certain degree of local continuity.
                                                                                                                   Scene          DB-P LAYROOM           DB-D R J OHNSON
                                                                                                                                 PSNR Mem (MB)          PSNR Mem (MB)
                             Anchor points              Selected Neural Gaussians in a test view
                                                                                                                   N ONE         28.45        24         28.81        12
                                                                                                               W / P RUNING      29.12        23         28.51        12
                                                                                                               W / G ROWING      30.54        71         29.75        76
                                                                                                                   F ULL         30.62        63         29.80        68
                                              𝞪>𝝉
                                             selector


                                                                                                             4.3. Ablation Studies
                                                                                                             Efficacy of Filtering Strategies. We evaluated our fil-
                                                                                                             tering strategies (Sec. 3.2.2), which we found crucial for
  Figure 8. Geometry culling via selector. (Left) Anchor points                                              speeding up our method. As Tab. 4 shows, while these
  from randomly initialized points; (Right) Activated neural Gaus-                                           strategies had no notable effect on fidelity, they significantly
  sians derived from each anchor under the current view. In synthetic                                        enhanced inference speed. However, there was a risk of
  Blender scenes, with all 3D Gaussians visible in the viewing frus-                                         masking pertinent neural Gaussians, which we aim to ad-
  tum, our opacity filtering functions similar to a geometry proxy                                           dress in future works.
  estimator, excluding unoccupied regions before rasterization.
                                                                                                             Efficacy of Anchor Points Refinement Policy. We eval-
                                                                                                             uated our growing and pruning operations described in



Activated neural Gaussians
                                                                                                             Sec. 3.3. Tab. 5 shows the results of disabling each opera-
                                                                                                             tion in isolation and maintaining the rest of the method. We
                                                                                                             found that the addition operation is crucial for accurately re-
                                                                                                             constructing details and texture-less areas, while the prun-
                                                                                                             ing operation plays an important role in eliminating trivial
                                                                                                             Gaussians and maintaining the efficiency of our approach.
                                                                                   Training iterations

  Figure 9. Learning with different k values. Despite varying
                                                                                                             4.4. Discussions and Limitations
  initial k values under different hyper-parameter settings, they con-                                       Through our experiments, we found that the initial points
  verge to activate a similar number of neural Gaussians with com-                                           play a crucial role for high-fidelity results. Initializing our
  parable rendering fidelity.                                                                                framework from SfM point clouds is a swift and viable
                                                                                                             solution, considering these point clouds usually arise as a
                                                                                                             byproduct of image calibration processes. However, this ap-
  Fig. 9 shows that regardless of the k value, the final number                                              proach may be suboptimal for scenarios dominated by large
  of activated neural Gaussians converges to a similar amount                                                texture-less regions. Despite our anchor point refinement
  through the training, indicating Scaffold-GS’s preference to                                               strategy can remedy this issue to some extent, it still suffers
  select a collection of non-redundant Gaussians that are suf-                                               from extremely sparse points. We expect that our algorithm
  ficient to represent the scene.                                                                            will progressively improve as the field advances, yielding


                                                                                                         8
more accurate results. Further details are discussed in the                    Guibas, Jonathan Tremblay, S. Khamis, Tero Karras, and
supplementary material.                                                        Gordon Wetzstein. Efficient geometry-aware 3d generative
                                                                               adversarial networks. 2022 IEEE/CVF Conference on Com-
5. Conclusion                                                                  puter Vision and Pattern Recognition (CVPR), pages 16102–
                                                                               16112, 2021. 2
In this work, we introduce Scaffold-GS, a novel 3D neural                  [9] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
scene representation for efficient view-adaptive rendering.                    Hao Su. Tensorf: Tensorial radiance fields. ArXiv,
The core of Scaffold-GS lies in its structural arrangement of                  abs/2203.09517, 2022. 2
3D Gaussians guided by anchor points from SfM, whose at-                  [10] Anpei Chen, Zexiang Xu, Xinyue Wei, Siyu Tang, Hao Su,
tributes are on-the-fly decoded from view-dependent MLPs.                      and Andreas Geiger. Factor fields: A unified framework for
We show that our approach leverages a much more com-                           neural fields and beyond. ArXiv, abs/2302.01226, 2023. 2
pact set of Gaussians to achieve comparable or even bet-                  [11] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and An-
                                                                               drea Tagliasacchi. Mobilenerf: Exploiting the polygon ras-
ter results than the SOTA algorithms. The advantage of
                                                                               terization pipeline for efficient neural field rendering on mo-
our view-adaptive neural Gaussians is particularly evident                     bile architectures. In The Conference on Computer Vision
in challenging cases where 3D-GS usually fails. We fur-                        and Pattern Recognition (CVPR), 2023. 2
ther show that our anchor points encode local features in                 [12] Christopher Bongsoo Choy, Danfei Xu, JunYoung Gwak,
a meaningful way that exhibits semantic patterns to some                       Kevin Chen, and Silvio Savarese. 3d-r2n2: A unified ap-
degree, suggesting its potential applicability in a range of                   proach for single and multi-view 3d object reconstruction.
versatile tasks such as large-scale modeling, manipulation                     ArXiv, abs/1604.00449, 2016. 2
and interpretation in the future.                                         [13] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
                                                                               Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
References                                                                     Radiance fields without neural networks. In CVPR, 2022. 2,
                                                                               5, 6, 3
 [1] Kara-Ali Aliev, Dmitry Ulyanov, and Victor S. Lempitsky.             [14] Sara Fridovich-Keil, Giacomo Meanti, Frederik Warburg,
     Neural point-based graphics. In European Conference on                    Benjamin Recht, and Angjoo Kanazawa. K-planes: Ex-
     Computer Vision, 2019. 3                                                  plicit radiance fields in space, time, and appearance. 2023
 [2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter                 IEEE/CVF Conference on Computer Vision and Pattern
     Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.                 Recognition (CVPR), pages 12479–12488, 2023. 2
     Mip-nerf: A multiscale representation for anti-aliasing neu-         [15] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna,
     ral radiance fields. 2021 IEEE/CVF International Confer-                  and Thomas A. Funkhouser. Local deep implicit functions
     ence on Computer Vision (ICCV), pages 5835–5844, 2021.                    for 3d shape. 2020 IEEE/CVF Conference on Computer
     2                                                                         Vision and Pattern Recognition (CVPR), pages 4856–4865,
 [3] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter                 2019. 2
     Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.            [16] Markus Gross and Hanspeter Pfister. Point-based graphics.
     Mip-nerf: A multiscale representation for anti-aliasing neu-              Elsevier, 2011. 3
     ral radiance fields. In Proceedings of the IEEE/CVF Interna-         [17] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
     tional Conference on Computer Vision (ICCV), pages 5855–                  George Drettakis, and Gabriel Brostow. Deep blending for
     5864, 2021. 1                                                             free-viewpoint image-based rendering. 37(6):257:1–257:15,
 [4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.                 2018. 6, 3
     Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded                [18] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
     anti-aliased neural radiance fields. CVPR, 2022. 2, 5, 6, 7, 3            George Drettakis, and Gabriel Brostow. Deep blending for
 [5] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.                 free-viewpoint image-based rendering. ACM Transactions
     Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased                      on Graphics (ToG), 37(6):1–15, 2018. 5, 2, 3
     grid-based neural radiance fields. In Proceedings of the             [19] Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall,
     IEEE/CVF International Conference on Computer Vision                      Jonathan T. Barron, and Paul E. Debevec. Baking neural
     (ICCV), pages 19697–19705, 2023. 1                                        radiance fields for real-time view synthesis. 2021 IEEE/CVF
 [6] Mario Botsch, Alexander Hornung, Matthias Zwicker, and                    International Conference on Computer Vision (ICCV), pages
     Leif Kobbelt. High-quality surface splatting on today’s                   5855–5864, 2021. 2
     gpus. In Proceedings Eurographics/IEEE VGTC Symposium                [20] Eldar Insafutdinov and Alexey Dosovitskiy. Unsupervised
     Point-Based Graphics, 2005., pages 17–141. IEEE, 2005. 1                  learning of shape and pose with differentiable point clouds.
 [7] Mario Botsch, Alexander Sorkine-Hornung, Matthias                         In Advances in Neural Information Processing Systems
     Zwicker, and Leif P. Kobbelt. High-quality surface splatting              (NeurIPS), 2018. 3
     on today’s gpus. Proceedings Eurographics/IEEE VGTC                  [21] Abhishek Kar, Christian Häne, and Jitendra Malik. Learning
     Symposium Point-Based Graphics, 2005., pages 17–141,                      a multi-view stereo machine. ArXiv, abs/1708.05375, 2017.
     2005. 2                                                                   2
 [8] Eric Chan, Connor Z. Lin, Matthew Chan, Koki Nagano,                 [22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler,
     Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J.                   and George Drettakis. 3d gaussian splatting for real-time


                                                                      9
     radiance field rendering. ACM Transactions on Graphics, 42           [36] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Milden-
     (4), 2023. 1, 3, 4, 5, 6, 7, 2                                            hall. Dreamfusion: Text-to-3d using 2d diffusion. In The
[23] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen                    Eleventh International Conference on Learning Representa-
     Koltun. Tanks and temples: Benchmarking large-scale scene                 tions, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, 2023. 1
     reconstruction. ACM Transactions on Graphics, 36(4), 2017.           [37] Liu Ren, Hanspeter Pfister, and Matthias Zwicker. Object
     5, 6, 2, 3                                                                space ewa surface splatting: A hardware accelerated ap-
[24] Georgios Kopanas, Julien Philip, Thomas Leimkühler, and                  proach to high quality point rendering. Computer Graphics
     George Drettakis. Point-based neural rendering with per-                  Forum, 21, 2002. 2
     view optimization. Computer Graphics Forum, 40, 2021.                [38] Miguel Sainz and Renato Pajarola. Point-based rendering
     3                                                                         techniques. Computers & Graphics, 28(6):869–879, 2004. 2
[25] K Krishna and M Narasimha Murty. Genetic k-means algo-               [39] Johannes Lutz Schönberger and Jan-Michael Frahm.
     rithm. IEEE Transactions on Systems, Man, and Cybernet-                   Structure-from-motion revisited. In Conference on Com-
     ics, Part B (Cybernetics), 29(3):433–439, 1999. 7                         puter Vision and Pattern Recognition (CVPR), 2016. 3, 2
[26] Christoph Lassner and Michael Zollhofer. Pulsar: Effi-               [40] Jessica Shue, Eric Chan, Ryan Po, Zachary Ankner, Jiajun
     cient sphere-based neural rendering. In Proceedings of                    Wu, and Gordon Wetzstein. 3d neural field generation us-
     the IEEE/CVF Conference on Computer Vision and Pattern                    ing triplane diffusion. 2023 IEEE/CVF Conference on Com-
     Recognition, pages 1440–1449, 2021. 1                                     puter Vision and Pattern Recognition (CVPR), pages 20875–
[27] Chen-Hsuan Lin, Chen Kong, and Simon Lucey. Learn-                        20886, 2022. 2
     ing efficient point cloud generation for dense 3d object re-         [41] Vincent Sitzmann, Semon Rezchikov, William T. Freeman,
     construction. In AAAI Conference on Artificial Intelligence               Joshua B. Tenenbaum, and Frédo Durand. Light field net-
     (AAAI), 2018. 3                                                           works: Neural scene representations with single-evaluation
[28] Stephen Lombardi, Tomas Simon, Gabriel Schwartz,                          rendering. In Neural Information Processing Systems, 2021.
     Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mix-                 2
     ture of volumetric primitives for efficient neural rendering.
                                                                          [42] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo
     ACM Transactions on Graphics (ToG), 40(4):1–13, 2021. 5
                                                                               Tourism: Exploring Photo Collections in 3D. Association
[29] Lars M. Mescheder, Michael Oechsle, Michael Niemeyer,
                                                                               for Computing Machinery, New York, NY, USA, 1 edition,
     Sebastian Nowozin, and Andreas Geiger. Occupancy net-
                                                                               2023. 2
     works: Learning 3d reconstruction in function space. 2019
                                                                          [43] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
     IEEE/CVF Conference on Computer Vision and Pattern
                                                                               han, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron,
     Recognition (CVPR), pages 4455–4465, 2018. 2
                                                                               and Henrik Kretzschmar. Block-nerf: Scalable large scene
[30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
                                                                               neural view synthesis. In Proceedings of the IEEE/CVF Con-
     Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
                                                                               ference on Computer Vision and Pattern Recognition, pages
     Representing scenes as neural radiance fields for view syn-
                                                                               8248–8258, 2022. 1
     thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
     2, 5, 3                                                              [44] Shubham Tulsiani, Tinghui Zhou, Alyosha A. Efros, and Ji-
                                                                               tendra Malik. Multi-view supervision for single-view re-
[31] Thomas Müller, Alex Evans, Christoph Schied, and Alexan-
                                                                               construction via differentiable ray consistency. 2017 IEEE
     der Keller. Instant neural graphics primitives with a mul-
                                                                               Conference on Computer Vision and Pattern Recognition
     tiresolution hash encoding. ACM Transactions on Graphics
                                                                               (CVPR), pages 209–217, 2017. 2
     (ToG), 41(4):1–15, 2022. 2, 5, 6, 3
[32] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao,             [45] Haithem Turki, Deva Ramanan, and Mahadev Satya-
     Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fi-                  narayanan. Mega-nerf: Scalable construction of large-
     dler. Extracting Triangular 3D Models, Materials, and Light-              scale nerfs for virtual fly-throughs. In Proceedings of
     ing From Images. In Proceedings of the IEEE/CVF Confer-                   the IEEE/CVF Conference on Computer Vision and Pattern
     ence on Computer Vision and Pattern Recognition (CVPR),                   Recognition (CVPR), pages 12922–12931, 2022. 1
     pages 8280–8290, 2022. 1                                             [46] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
[33] Michael Oechsle, Songyou Peng, and Andreas Geiger.                        Komura, and Wenping Wang. Neus: Learning neural implicit
     Unisurf: Unifying neural implicit surfaces and radiance                   surfaces by volume rendering for multi-view reconstruction.
     fields for multi-view reconstruction. 2021 IEEE/CVF In-                   arXiv preprint arXiv:2106.10689, 2021. 2
     ternational Conference on Computer Vision (ICCV), pages              [47] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
     5569–5579, 2021. 2                                                        Image quality assessment: from error visibility to structural
[34] Jeong Joon Park, Peter R. Florence, Julian Straub,                        similarity. IEEE Transactions on Image Processing, 13(4):
     Richard A. Newcombe, and S. Lovegrove. Deepsdf: Learn-                    600–612, 2004. 5
     ing continuous signed distance functions for shape represen-         [48] Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin
     tation. 2019 IEEE/CVF Conference on Computer Vision and                   Johnson. Synsin: End-to-end view synthesis from a single
     Pattern Recognition (CVPR), pages 165–174, 2019. 2                        image. 2020 IEEE/CVF Conference on Computer Vision and
[35] Songyou Peng, Michael Niemeyer, Lars M. Mescheder,                        Pattern Recognition (CVPR), pages 7465–7475, 2019. 3
     Marc Pollefeys, and Andreas Geiger. Convolutional occu-              [49] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao,
     pancy networks. ArXiv, abs/2003.04618, 2020. 2                            Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin.


                                                                     10
     Bungeenerf: Progressive neural radiance field for extreme
     multi-scale scene rendering. In Computer Vision - ECCV
     2022 - 17th European Conference, Tel Aviv, Israel, Octo-
     ber 23-27, 2022, Proceedings, Part XXXII, pages 106–122.
     Springer, 2022. 1, 2, 5, 3
[50] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao,
     Bo Dai, and Dahua Lin. Assetfield: Assets mining and re-
     configuration in ground feature plane representation. ArXiv,
     abs/2303.13953, 2023. 2
[51] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia,
     Aayush Bansal, Changil Kim, Samuel Rota Bulò, Lorenzo
     Porzi, Peter Kontschieder, Aljaž Božič, Dahua Lin, Michael
     Zollhöfer, and Christian Richardt. VR-NeRF: High-fidelity
     virtualized walkable spaces. In SIGGRAPH Asia Conference
     Proceedings, 2023. 1, 5, 6, 2, 3
[52] Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan,
     Nanxuan Zhao, Christian Theobalt, Bo Dai, and Dahua Lin.
     Grid-guided neural radiance fields for large urban scenes. In
     Proceedings of the IEEE/CVF Conference on Computer Vi-
     sion and Pattern Recognition, pages 8296–8306, 2023. 2
[53] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin
     Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf:
     Point-based neural radiance fields. In Proceedings of the
     IEEE/CVF Conference on Computer Vision and Pattern
     Recognition, pages 5438–5448, 2022. 3, 4
[54] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman.
     Volume rendering of neural implicit surfaces. In Thirty-
     Fifth Conference on Neural Information Processing Systems,
     2021. 2
[55] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Öztireli,
     and Olga Sorkine-Hornung. Differentiable surface splatting
     for point-based geometry processing. ACM Transactions on
     Graphics (TOG), 38(6):1–14, 2019. 1, 3
[56] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shecht-
     man, and Oliver Wang. The unreasonable effectiveness of
     deep features as a perceptual metric. In Proceedings of the
     IEEE Conference on Computer Vision and Pattern Recogni-
     tion (CVPR), 2018. 5
[57] Xi Zhao, Ruizhen Hu, Haisong Liu, Taku Komura, and
     Xinyu Yang. Localization and completion for 3d object inter-
     actions. IEEE Transactions on Visualization and Computer
     Graphics, 26(8):2634–2644, 2019. 2
[58] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
     Markus Gross. Ewa volume splatting. In Proceedings Visu-
     alization, 2001. VIS’01., pages 29–538. IEEE, 2001. 3




                                                                     11
                    Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
                                                                   Supplementary Material
6. Overview
This supplementary is organized as follows: (1) In the
first section, we elaborate implementation details of our
Scaffold-GS, including anchor point feature enhancement
(Sec.3.2.1), structure of MLPs (Sec.3.2.2) and anchor point
refinement strategies (Sec.3.3); (2) The second part de-                                                           (a)                        (b)                             (c)


scribes our dataset preparation steps. We then show addi-
                                                                                       Figure 11. View-based feature bank’s weight distribution.
tional experimental results and analysis based on our train-                           (a), (b) and (c) denote the predicted weights {w2 , w1 , w} for
ing observations.                                                                      {fv↓2 , fv↓1 , fv } from a group of uniformally distributed view-
                                                                                       points. Light color denotes larger weights. For this anchor, finer
7. Implementation details.                                                             features are more activated at center view positions. The patterns
                                                                                       exhibit the ability to capture different scene granularities based on
Feature Bank. To enhance the view-adaptability, we up-                                 view direction and distance.
date the anchor feature through a view-dependent encoding.
Following calculating the relative distance δvc and view-                                                                                                                Opacity: N×1×(𝑘)
ing direction ⃗dvc of a camera and an anchor, we predict                                                           F𝛼    36x32
                                                                                                                                    ReLU
                                                                                                                                            32×1×(𝑘)
                                                                                                                                                            Tanh
                                                                                                                                                                        1 1 1 1         …    1 1
a weight vector w ∈ R3 as follows:
                                                                                                           ^
                                                                                                           fv                                                                Color: N×3×(𝑘)
                    (w, w1 , w2 ) = Softmax(Fw (δvc , ⃗dvc )),

                                                                                       Input: N×(32+3+1)
                                                                            (13)                                                    ReLU                  Sigmoid
                                                                                                                                                                                        …
                                                                                                                    Fc   36x32               32×3×(𝑘)                    3     3              3

                                                                                                           →
where Fw is a tiny MLP that serves as a view encoding func-                                                dvc                                                               Scale: N×3×(𝑘)
tion. We then encode the view direction information to the                                                                                              Sigmoid*sv       3     3        …     3
                                                                                                           𝛿vc
anchor feature fv by compositing a feature bank containing                                                                          ReLU
                                                                                                                 Fs & Fq 36x32               32×7×(𝑘)                   Quaternion: N×4×(𝑘)
information with different resolutions as follows:
                                                                                                                                                        Normalization    4          4   …     4

                     fˆv = w · fv + w1 · fv↓1 + w2 · fv↓2 ,                 (14)
                                                                                       Figure 12. MLP Structures. For each anchor point, we use small
In practice, we implement the feature bank via slicing and                             MLPs (Fα , Fc , Fs , Fq ) to predict attributes (opacity, color, scale
repeating, as illustrated in Fig. 10. We found this slicing and                        and quaternion) of k neural Gaussians. The input to MLPs are
mixture operation improves Scaffold-GS’s ability to cap-                               anchor feature fˆv , relative viewing direction ⃗
                                                                                                                                       dvc and distance δvc
ture different scene granularity. The distribution of feature                          between the camera and anchor point.
bank’s weights is illustrated in Fig. 11.

      fv
                                                                                       • For opacity, the output is activated by Tanh, where value
                                                                                         0 serves as a natural threshold for selecting valid samples
       fv↓      1
                      repeat            repeat   repeat            repeat                and the final valid values can cover the full range of [0,1).
                                                                                       • For color, we activate the output with Sigmoid function:
      fv↓   2
                               repeat                     repeat

                                                                                                                                 {c0 , ..., ck−1 } = Sigmoid(Fc ),                          (15)
Figure 10. Generation of Feature Bank. We expand the anchor
feature f into a set of multi-resolution features {fv , fv↓1 , fv↓2 }                    which constrains the color into a range of (0,1).
via slicing and repeating. This operation improves Scaffold-GS’s                       • For rotation, we follow 3D-GS [22] and activate it with a
ability to capture different scene granularity.                                          normalization to obtain a valid quaternion.
                                                                                       • For scaling, we adjust the base scaling sv of each anchor
                                                                                         with the MLP output as follows:
MLPs as feature decoders. The core MLPs include the
opacity MLP Fα , the color MLP Fc and the covariance                                                                       {s0 , ..., sk−1 } = Sigmoid(Fs ) · sv ,                          (16)
MLP Fs and Fq . All of these F∗ are implemented in a L IN -
EAR → R E LU → L INEAR style with the hidden dimension                                 Voxel Size. The voxel size ϵ sets the finest anchor reso-
of 32, as illustrated in Fig. 12. Each branch’s output is acti-                        lution. We employ two strategies: 1) Use the median of
vated with a head layer.                                                               the nearest-neighbor distances among all initial points: ϵ


                                                                                   1
                             Example scene                                       Table 6. SSIM scores for Mip-NeRF360 [4] scenes.
                                                                             Method   Scenes    bicycle   garden    stump    room      counter   kitchen   bonsai
                                                                               3D-GS [22]       0.771      0.868    0.775     0.914     0.905     0.922    0.938
                                                                             Mip-NeRF360 [4]    0.685      0.813    0.744     0.913     0.894     0.920    0.941
                                                                                iNPG [31]       0.491      0.649    0.574     0.855     0.798     0.818    0.890
                                                                              Plenoxels [13]    0.496     0.6063    0.523    0.8417     0.759     0.648    0.814
           Initial anchors                   Reﬁned anchors                       Ours          0.705      0.842    0.784    0.925      0.914     0.928    0.946


                                                                                 Table 7. PSNR scores for Mip-NeRF360 [4] scenes.
                                                                             Method   Scenes    bicycle   garden    stump    room      counter   kitchen   bonsai
                                                                               3D-GS [22]        25.25     27.41    26.55    30.63     28.70     30.32     31.98
                                                                             Mip-NeRF360 [4]     24.37     26.98    26.40    31.63     29.55     32.23     33.46
                                                                                iNPG [31]        22.19     24.60    23.63    29.27     26.44     28.55     30.34
                                                                              Plenoxels [13]     21.91     23.49    20.66    27.59     23.62     23.42     24.67
                                                                                  Ours           24.50     27.17    26.27    31.93     29.34     31.30     32.70

                                 zoom
                                  out
                                                                                 Table 8. LPIPS scores for Mip-NeRF360 [4] scenes.
                                                                             Method   Scenes    bicycle   garden    stump    room      counter   kitchen   bonsai
                                                                               3D-GS [22]       0.205      0.103    0.210    0.220      0.204     0.129    0.205
                                                                             Mip-NeRF360 [4]    0.301      0.170    0.261     0.211     0.204     0.127    0.176
                                                                                iNPG [31]       0.487      0.312    0.450     0.301     0.342     0.254    0.227
                                                                              Plenoxels [13]    0.506     0.3864    0.503    0.4186     0.441     0.447    0.398
                                                                                  Ours          0.306      0.146    0.284    0.202      0.191     0.126    0.185



                                 zoom
                                                                              Table 9. Storage size (MB) for Mip-NeRF360 [4] scenes.
                                  out
                                                                             Method   Scenes    bicycle    garden   stump    room      counter   kitchen   bonsai
                                                                               3D-GS [22]        1291      1268      1034     327       261       414       281
                                                                                  Ours            248       271      493      133       194       173       258


                                                                           Table 10. SSIM scores for Tanks&Temples [23] and Deep
                                                                           Blending [18] scenes.
                                                                              Method        Scenes        Truck      Train      Dr Johnson          Playroom
                                                                               3D-GS [22]                 0.879     0.802             0.899             0.906
Figure 13. Anchor Refinement. We visualize the initial and re-               Mip-NeRF360 [4]              0.857     0.660             0.901             0.900
fined anchor points on the truck scene [23]. The truck is high-                 iNPG [31]                 0.779     0.666             0.839             0.754
lighted by the circle. Note that the refined points effectively cov-          Plenoxels [13]              0.774     0.663             0.787             0.802
ers surrounding regions and fine-scale structures, leaning to more                    Ours                0.883     0.822             0.907             0.904
complete and detailed scene renderings.

                                                                           8. Experiments and Results
is adapted to point cloud density, yielding denser anchors                 Additional       Data       Preprocessing. We           used
with enhanced rendering quality but might introduce more                   COLMAP [39] to estimate camera poses and generate
computational overhead; 2) Set ϵ manually to either 0.005                  SfM points for VR-NeRF [51] and BungeeNeRF [49]
or 0.01: this is effective in most scenarios but might lead to             datasets. Both two datasets are challenging in terms of
missing details in texture-less regions. We found these two                varying levels of details presented in the captures. The
strategies adequately accommodate various scene complex-                   VR-NeRF dataset was tested using its eye-level subset with
ities in our experiments.                                                  3 cameras. For all other datasets, we adhered to the original
                                                                           3D-GS [22] method, sourcing them from public resources.

                                                                           Per-scene Results. Here we list the error metrics used in
Anchor Refinement. As briefly discussed in the main pa-                    our evaluation in Sec.4 across all considered methods and
per, the voxelization process suggests that our method may                 scenes, as shown in Tab. 6-17.
behave sensitive to initial SfM results. We illustrate the ef-
fect of the anchor refinement process in Fig. 13, where new                Training Process Analysis. Figure 14 illustrates the vari-
anchors enhance scene details and fill gaps in large texture-              ations in PSNR during the training process for both train-
less regions and less observed areas.                                      ing and testing views. Our method demonstrates quicker


                                                                       2
                           Amsterdam (BungeeNeRF)                             Pompidou (BungeeNeRF)                          Train (Tanks&Temples)                                Counter (Mip-NeRF360)




Training views




Testing views




                                                                                                      3D-GS           Ours

Figure 14. PSNR curve of Scaffold-GS and 3D-GS [22] across diverse datasets [4, 17, 49]. We illustrate the variations in PSNR during
the training process for both training and testing views. The orange curve represents Scaffold-GS, while the blue curve corresponds to
3D-GS. Our method not only achieves rapid convergence but also exhibits superior performance, marked by a significant rise in training
PSNR and consistently higher testing PSNR, in contrast to 3D-GS.

Table 11. PSNR scores for Tanks&Temples [23] and Deep                                                             Table 15. Storage size (MB) for Synthetic Blender [30] scenes.
Blending [18] scenes.                                                                                               Method      Scenes     Mic    Chair     Ship      Materials    Lego     Drums      Ficus      Hotdog
                                                                                                                      3D-GS [22]            50     116       63          35          78       93        59          44
                 Method        Scenes       Truck         Train       Dr Johnson          Playroom
                                                                                                                            Ours            12       13      16          18          13       35        11          8
                   3D-GS [22]               25.19         21.10          28.77              30.04
                 Mip-NeRF360 [4]            24.91         19.52          29.14              29.66
                                                                                                                  Table 16. PSNR scores for BungeeNeRF [49] and VR-
                    iNPG [31]               23.26         20.17          27.75              19.48
                  Plenoxels [13]            23.22         18.93          23.14              22.98                 NeRF [51] scenes.
                                                                                                                   Method    Scenes   Amsterdam   Bilbao   Pompidou    Quebec     Rome    Hollywood   Apartment    Kitchen
                        Ours                25.77         22.15          29.80              30.62                    3D-GS [22]          25.74    26.35     21.20       28.79     23.54     23.25       28.48       29.40
                                                                                                                       Ours              27.10    27.66     25.34       30.51     26.50     24.97       28.87       29.61

Table 12. LPIPS scores for Tanks&Temples [23] and Deep
Blending [18] scenes.                                                                                             Table 17. Storage size (MB) for BungeeNeRF [49] and VR-
                                                                                                                  NeRF [51] scenes.
                 Method        Scenes       Truck         Train       Dr Johnson          Playroom                 Method    Scenes   Amsterdam   Bilbao   Pompidou    Quebec     Rome    Hollywood   Apartment    Kitchen

                   3D-GS [22]               0.148         0.218          0.244              0.241                    3D-GS [22]          1453     1337       2129       1438      1626      1642        202         323
                                                                                                                       Ours              243       197       230         166      200       182          48          90
                 Mip-NeRF360 [4]            0.159         0.354          0.237              0.252
                    iNPG [31]               0.274         0.386          0.381              0.465
                  Plenoxels [13]            0.335         0.422          0.521              0.499
                                                                                                                  the Amsterdam and Pompidou scenes in BungeeNeRF, we
                        Ours                0.147         0.206          0.250              0.258
                                                                                                                  trained them with images at three coarser scales and eval-
                                                                                                                  uated them at a novel finer scale. The fact that 3D-GS
Table 13. Storage size (MB) for Tanks&Temples [23] and Deep                                                       achieved higher training PSNR but lower testing PSNR in-
Blending [18] scenes.
                                                                                                                  dicates its tendency to overfit at training scales.
                 Method        Scenes       Truck         Train       Dr Johnson         Playroom
                    3D-GS [22]                578         240             715                515
                        Ours                  107          66             69                     63


                 Table 14. PSNR scores for Synthetic Blender [30] scenes.
           Method       Scenes    Mic     Chair   Ship    Materials    Lego     Drums    Ficus    Hotdog
                  3D-GS [22]      35.36   35.83   30.80     30.00     35.78      26.15   34.87    37.72
                    Ours          37.25   35.28   31.17     30.65     35.69      26.44   35.21    37.73




convergence, enhanced robustness, and better generaliza-
tion compared to 3D-GS, as evidenced by the rapid increase
in training PSNR and higher testing PSNR. Specifically, for


                                                                                                              3
