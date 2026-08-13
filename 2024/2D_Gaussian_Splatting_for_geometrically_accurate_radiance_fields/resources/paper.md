                                         2D Gaussian Splatting for Geometrically Accurate Radiance Fields
                                         BINBIN HUANG, ShanghaiTech University, China
                                         ZEHAO YU, University of Tübingen Tübingen AI Center, Germany
                                         ANPEI CHEN, University of Tübingen Tübingen AI Center, Germany
                                         ANDREAS GEIGER, University of Tübingen Tübingen AI Center, Germany
                                         SHENGHUA GAO, ShanghaiTech University, China
                                         https://surfsplatting.github.io

                                              Disk (color)                                        Radiance field                               Mesh




arXiv:2403.17888v3 [cs.CV] 22 Feb 2025
                                              Disk (normal)                                       Surface normal




                                               (a) 2D disks as surface elements                           (b) 2D Gaussian splatting                               (c) Meshing

                                         Fig. 1. Our method, 2DGS, (a) optimizes a set of 2D oriented disks to represent and reconstruct a complex real-world scene from multi-view RGB images. These
                                         optimized 2D disks are tightly aligned to the surfaces. (b) With 2D Gaussian splatting, we allow real-time rendering of high quality novel view images with
                                         view consistent normals and depth maps. (c) Finally, our method provides detailed and noise-free triangle mesh reconstruction from the optimized 2D disks.

                                         3D Gaussian Splatting (3DGS) has recently revolutionized radiance field                      optimization, we introduce a perspective-accurate 2D splatting process uti-
                                         reconstruction, achieving high quality novel view synthesis and fast render-                 lizing ray-splat intersection and rasterization. Additionally, we incorporate
                                         ing speed. However, 3DGS fails to accurately represent surfaces due to the                   depth distortion and normal consistency terms to further enhance the quality
                                         multi-view inconsistent nature of 3D Gaussians. We present 2D Gaussian                       of the reconstructions. We demonstrate that our differentiable renderer al-
                                         Splatting (2DGS), a novel approach to model and reconstruct geometrically                    lows for noise-free and detailed geometry reconstruction while maintaining
                                         accurate radiance fields from multi-view images. Our key idea is to collapse                 competitive appearance quality, fast training speed, and real-time rendering.
                                         the 3D volume into a set of 2D oriented planar Gaussian disks. Unlike 3D
                                         Gaussians, 2D Gaussians provide view-consistent geometry while modeling                      CCS Concepts: • Computing methodologies → Reconstruction; Render-
                                         surfaces intrinsically. To accurately recover thin surfaces and achieve stable               ing; Machine learning approaches.

                                                                                                                                     Additional Key Words and Phrases: Novel View Synthesis, Radiance Fields,
                                         Authors’ Contact Information: Binbin Huang, huangbb@shanghaitech.edu.cn, Shang-             Surface Splatting, Surface Reconstruction
                                         haiTech University, Shanghai, China; Zehao Yu, zehao.yu@uni-tuebingen.de, Uni-
                                         versity of Tübingen and Tübingen AI Center, Tübingen, Germany; Anpei Chen,                  ACM Reference Format:
                                         anpei.chen@uni-tuebingen.de, University of Tübingen and Tübingen AI Center, Tübin-          Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao.
                                         gen, Germany; Andreas Geiger, a.geiger@uni-tuebingen.de, University of Tübingen and
                                         Tübingen AI Center, Tübingen, Germany; Shenghua Gao, gaoshh@shanghaitech.edu.cn,            2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields.
                                         ShanghaiTech University, Shanghai, China.                                                   In Special Interest Group on Computer Graphics and Interactive Techniques
                                                                                                                                     Conference Conference Papers ’24 (SIGGRAPH Conference Papers ’24), July
                                                                                                                                     27-August 1, 2024, Denver, CO, USA. ACM, New York, NY, USA, 13 pages.
                                         Permission to make digital or hard copies of all or part of this work for personal or
                                         classroom use is granted without fee provided that copies are not made or distributed
                                                                                                                                     https://doi.org/10.1145/3641519.3657428
                                         for profit or commercial advantage and that copies bear this notice and the full citation
                                         on the first page. Copyrights for third-party components of this work must be honored.
                                         For all other uses, contact the owner/author(s).                                             1   INTRODUCTION
                                         SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA                      Photorealistic novel view synthesis (NVS) and accurate geometry
                                         © 2024 Copyright held by the owner/author(s).
                                         ACM ISBN 979-8-4007-0525-0/24/07                                                             reconstruction stand as pivotal long-term objectives in computer
                                         https://doi.org/10.1145/3641519.3657428                                                      graphics and vision. Recently, 3D Gaussian Splatting (3DGS) [Kerbl

                                                                                                                                             SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
2 •   Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao


et al. 2023] has emerged as an appealing alternative to implicit [Bar-
ron et al. 2022a; Mildenhall et al. 2020] and feature grid-based rep-
resentations [Barron et al. 2023; Müller et al. 2022] in NVS, due to
its real-time photorealistic NVS results at high resolutions. Rapidly
evolving, 3DGS has been quickly extended with respect to multiple
domains, including anti-aliasing rendering [Yu et al. 2024], material
modeling [Jiang et al. 2023; Shi et al. 2023], dynamic scene recon-
struction [Yan et al. 2023], and animatable avatar creation [Qian
et al. 2023; Zielonka et al. 2023]. Nevertheless, it falls short in cap-     Fig. 2. Comparison of 3DGS and 2DGS. 3DGS utilizes different intersec-
turing intricate geometry since the volumetric 3D Gaussian, which            tion planes for value evaluation when viewing from different viewpoints,
                                                                             resulting in inconsistency. Our 2DGS provides multi-view consistent value
models the complete angular radiance, conflicts with the thin nature
                                                                             evaluations.
of surfaces.
   On the other hand, earlier works [Pfister et al. 2000; Zwicker et al.       In summary, we make the following contributions:
2001a,b] have shown surfels (surface elements) to be an effective                • We present a highly efficient differentiable 2D Gaussian ren-
representation of complex geometry. Surfels approximate the object                 derer, enabling perspective-correct splatting by leveraging
surface locally with shape and shade attributes and can be derived                 2D surface modeling, ray-splat intersection, and volumetric
from known geometry. They are widely used in SLAM [Whelan et al.                   integration.
2016] and other robotics tasks [Schöps et al. 2019] as an efficient ge-          • We introduce two regularization losses for improved and
ometry representation. Subsequent advancements [Yifan et al. 2019]                 noise-free surface reconstruction.
have incorporated surfels into a differentiable framework. However,              • Our approach achieves state-of-the-art geometry reconstruc-
these methods typically require ground truth (GT) geometry, depth                  tion and NVS results compared to other explicit representa-
sensor data, or operate under constrained scenarios with known                     tions.
lighting.
   Inspired by these works, we propose 2D Gaussian Splatting for             2 RELATED WORK
3D scene reconstruction and novel view synthesis that combines
the benefits of both worlds, while overcoming their limitations. Un-         2.1 Novel view synthesis
like 3DGS, our approach represents a 3D scene with 2D Gaussian               Significant advancements have been achieved in NVS, particularly
primitives, each defining an oriented elliptical disk. The significant       since the introduction of Neural Radiance Fields (NeRF) [Milden-
advantage of 2D Gaussian over its 3D counterpart lies in the accu-           hall et al. 2021]. NeRF employs a multi-layer perceptron (MLP) to
rate geometry representation during rendering. Specifically, 3DGS            represent geometry and view-dependent appearance, optimized via
evaluates a Gaussian’s value at the intersection between a pixel ray         volume rendering to deliver exceptional rendering quality. Post-
and a 3D Gaussian [Keselman and Hebert 2022, 2023], which leads              NeRF developments have further enhanced its capabilities. For in-
to inconsistency depth when rendered from different viewpoints.              stance, Mip-NeRF [Barron et al. 2021] and subsequent works [Barron
In contrast, our method utilizes explicit ray-splat intersection, re-        et al. 2022a, 2023; Hu et al. 2023] tackle NeRF’s aliasing issues. Ad-
sulting in a perspective correct splatting, as illustrated in Figure 2,      ditionally, the rendering efficiency of NeRF has seen substantial
which in turn significantly improves reconstruction quality. Fur-            improvements through techniques such as distillation [Reiser et al.
thermore, the inherent surface normals in 2D Gaussian primitives             2021; Yu et al. 2021] and baking [Chen et al. 2023a; Hedman et al.
enable direct surface regularization through normal constraints. In          2021; Reiser et al. 2023; Yariv et al. 2023]. Moreover, the training
contrast with surfels-based models [Pfister et al. 2000; Yifan et al.        and representational power of NeRF have been enhanced using
2019; Zwicker et al. 2001a], our 2D Gaussians can be recovered from          feature-grid based scene representations [Chen et al. 2022, 2023c;
unknown geometry with gradient-based optimization.                           Fridovich-Keil et al. 2022; Liu et al. 2020; Müller et al. 2022; Sun et al.
   While our 2D Gaussian approach excels in geometric model-                 2022a].
ing, optimizing solely with photometric losses can lead to noisy                Recently, 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] has
reconstructions, due to the inherently unconstrained nature of 3D            emerged, demonstrating impressive real-time NVS results. This
reconstruction tasks, as noted in [Barron et al. 2022b; Yu et al. 2022b;     method has been quickly extended to multiple domains [Xie et al.
Zhang et al. 2020]. To enhance reconstructions and achieve smoother          2023; Yan et al. 2023; Yu et al. 2024; Zielonka et al. 2023]. In this work,
surfaces, we introduce two regularization terms: depth distortion            we propose to “flatten” 3D Gaussians to 2D Gaussian primitives to
and normal consistency. The depth distortion term concentrates 2D            better align their shape with the object surface. Combined with
primitives distributed within a tight range along the ray, address-          two novel regularization losses, our approach reconstructs surfaces
ing the rendering process’s limitation where the distance between            more accurately than 3DGS while preserving its high-quality and
Gaussians is ignored. The normal consistency term minimizes dis-             real-time rendering capabilities.
crepancies between the rendered normal map and the gradient of
the rendered depth, ensuring alignment between the geometries                2.2   3D reconstruction
defined by depth and normals. Employing these regularizations                3D Reconstruction from multi-view images has been a long-standing
in combination with our 2D Gaussian model enables us to extract              goal in computer vision. Multi-view stereo based methods [Schön-
highly accurate surface meshes, as demonstrated in Figure 1.                 berger et al. 2016; Yao et al. 2018; Yu and Gao 2020] rely on a modular

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
                                                                                    2D Gaussian Splatting for Geometrically Accurate Radiance Fields   • 3


pipeline that involves feature matching, depth prediction, and fu-         3D Gaussians, our method directly employs 2D Gaussians, simpli-
sion. In contrast, recent neural approaches [Niemeyer et al. 2020;         fying the process and enhancing the resulting geometry without
Yariv et al. 2020] represent surface implicitly via an MLP [Mescheder      additional mesh refinement. NeuSG optimizes 3D Gaussian primi-
et al. 2019; Park et al. 2019] , extracting surfaces post-training via     tives and an implicit SDF network jointly and extracts the surface
the Marching Cube algorithm. Further advancements [Oechsle et al.          from the SDF network, while our approach leverages 2D Gaussian
2021; Wang et al. 2021; Yariv et al. 2021] integrated implicit surfaces    primitives for surface approximation, offering a faster and concep-
with volume rendering, achieving detailed surface reconstructions          tually simpler solution.
from RGB images. These methods have been extended to large-scale
reconstructions via additional regularization [Li et al. 2023; Yu et al.   3   3D GAUSSIAN SPLATTING
2022a,b], and efficient reconstruction for objects [Wang et al. 2023].
                                                                           Kerbl et al. [Kerbl et al. 2023] propose to represent 3D scenes with
Despite these impressive developments, efficient large-scale scene
                                                                           3D Gaussian primitives and render images using differentiable vol-
reconstruction remains a challenge. For instance, Neuralangelo [Li
                                                                           ume splatting. Specifically, 3DGS explicitly parameterizes Gaussian
et al. 2023] requires 128 GPU hours for reconstructing a single scene
                                                                           primitives via 3D covariance matrix 𝚺 and their location p𝑘 :
from the Tanks and Temples Dataset [Knapitsch et al. 2017]. In this
work, we introduce 2D Gaussian splatting, a method that signifi-                                     1
cantly accelerates the reconstruction process. It achieves similar or                    G(p) = exp(− (p − p𝑘 ) ⊤ 𝚺 −1 (p − p𝑘 ))                      (1)
                                                                                                     2
slightly better results compared to previous implicit neural surface
representations, while being an order of magnitude faster.                 where the covariance matrix 𝚺 = RSS⊤ R⊤ is factorized into a scal-
                                                                           ing matrix S and a rotation matrix R. To render an image, the 3D
2.3   Differentiable Point-based Graphics                                  Gaussian is transformed into the camera coordinates with world-
                                                                           to-camera transform matrix W and projected to image plane via a
Differentiable point-based rendering [Aliev et al. 2020; Insafutdinov
                                                                           local affine transformation J [Zwicker et al. 2001a]:
and Dosovitskiy 2018; Rückert et al. 2022; Wiles et al. 2020; Yifan
et al. 2019] has been explored extensively due to its efficiency and                                     ′
                                                                                                        𝚺 = JW𝚺W⊤ J⊤                                   (2)
flexibility in representing intricate structures. Notably, NPBG [Aliev
et al. 2020] rasterizes point cloud features onto an image plane,                                                            ′
                                                                           By skipping the third row and column of 𝚺 , we obtain a 2D Gaussian
subsequently utilizing a convolutional neural network for RGB
image prediction. DSS [Yifan et al. 2019] focuses on optimizing ori-       G 2𝐷 with covariance matrix 𝚺2𝐷 . Next, 3DGS [Kerbl et al. 2023]
ented point clouds from multi-view images under known lighting             employs volumetric alpha blending to integrate alpha-weighted
conditions. Pulsar [Lassner and Zollhofer 2021] introduces a tile-         appearance from front to back:
based acceleration structure for more efficient rasterization. More
                                                                                                𝐾                      −1
                                                                                                                      𝑘Ö
recently, 3DGS [Kerbl et al. 2023] optimizes anisotropic 3D Gauss-                             ∑︁
                                                                                      c(x) =         c𝑘 𝛼𝑘 G𝑘2𝐷 (x)         (1 − 𝛼 𝑗 G𝑗2𝐷 (x))         (3)
ian primitives, demonstrating real-time photorealistic NVS results.                                                   𝑗=1
                                                                                               𝑘=1
Despite these advances, using point-based representations from un-
constrained multi-view images remains challenging. In this paper,          where 𝑘 is the index of the Gaussian primitives, 𝛼𝑘 denotes the alpha
we demonstrate detailed surface reconstruction using 2D Gaussian           values and c𝑘 is the view-dependent appearance. The attributes of
primitives. We also highlight the critical role of additional regular-     3D Gaussian primitives are optimized using a photometric loss.
ization losses in optimization, showcasing their significant impact
on the quality of the reconstruction.                                         Challenges in Surface Reconstruction. Reconstructing surfaces us-
                                                                           ing 3D Gaussian modeling and splatting faces several challenges.
2.4   Concurrent works                                                     First, the volumetric radiance representation of 3D Gaussians con-
                                                                           flicts with the thin nature of surfaces. Second, 3DGS does not na-
Since 3DGS [Kerbl et al. 2023] was introduced, it has been rapidly
                                                                           tively model surface normals, essential for high-quality surface
adapted across multiple domains. We now review the closest work
                                                                           reconstruction. Third, the rasterization process in 3DGS lacks multi-
in inverse rendering. These work [Gao et al. 2023; Jiang et al. 2023;
                                                                           view consistency, leading to varied 2D intersection planes for dif-
Liang et al. 2023; Shi et al. 2023] extend 3DGS by modeling normals
                                                                           ferent viewpoints [Keselman and Hebert 2023], as illustrated in
as additional attributes of 3D Gaussian primitives. Our approach,
                                                                           Figure 2 (a). Additionally, using an affine matrix for transforming a
in contrast, inherently defines normals by representing the tangent
                                                                           3D Gaussian into ray space only yields accurate projections near the
space of the 3D surface using 2D Gaussian primitives, aligning them
                                                                           center, compromising on perspective accuracy around surrounding
more closely with the underlying geometry. Additionally, the afore-
                                                                           regions [Zwicker et al. 2004]. Therefore, it often results in noisy
mentioned works predominantly focus on estimating the material
                                                                           reconstructions, as shown in Figure 5.
properties of the scene and evaluating their results for relighting
tasks. Notably, none of these works specifically target surface re-
construction, the primary focus of our work.                               4   2D GAUSSIAN SPLATTING
   We also highlight the distinctions between our method and con-          To accurately reconstruct geometry while maintaining high-quality
current works SuGaR [Guédon and Lepetit 2023] and NeuSG [Chen              novel view synthesis, we present differentiable 2D Gaussian splat-
et al. 2023b]. Unlike SuGaR, which approximates 2D Gaussians with          ting (2DGS).

                                                                                 SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
4   •   Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao


             Tangent frame (u,v)                Image frame (x,y)                   4.2   Splatting
                                                                                    One common strategy for rendering 2D Gaussians is to project the
                                                        𝑠! 𝐭 !
                                                                                    2D Gaussian primitives onto the image space using the affine ap-
                         𝑠" 𝐭 "                                  𝑠" 𝐭 "             proximation of the perspective projection [Zwicker et al. 2001a,b].
                              𝑠 𝐭                                                   However, as noted in [Zwicker et al. 2004], this projection is only
                         𝐩! # #

                                                                                    accurate at the center of the Gaussian and has increasing approxi-
                                                                                    mation error with increased distance to the center. To address this
                 2D Gaussian Splat               2D Gaussian Splat                  issue, Zwicker et al. proposed a formulation based on homogeneous
                   in object space                 in image space                   coordinates. Specifically, projecting the 2D splat onto an image plane
                                                                                    can be described by a general 2D-to-2D mapping in homogeneous
Fig. 3. Illustration of 2D Gaussian Splatting. 2D Gaussian Splats are ellip-        coordinates. Let W ∈ 4 × 4 be the combined transformation matrix
tical disks characterized by a center point p𝑘 , tangential vectors t𝑢 and          from world space to screen space. The screen space points are hence
t𝑣 , and two scaling factors (𝑠𝑢 and 𝑠 𝑣 ) control the variance. Their elliptical   obtained by
projections are sampled through the ray-splat intersection ( Section 4.2) and
                                                                                              x = (𝑥𝑧, 𝑦𝑧, 𝑧, 𝑧) T = W𝑃 (𝑢, 𝑣) = WH(𝑢, 𝑣, 1, 1) T        (7)
accumulated via alpha-blending in image space. 2DGS reconstructs surface
attributes such as colors, depths, and normals through gradient descent.            where x represents a homogeneous ray emitted from the camera and
                                                                                    passing through pixel (𝑥, 𝑦) and intersecting the splat at depth 𝑧. To
                                                                                    rasterize a 2D Gaussian, Zwicker et al. proposed to project its conic
4.1     Modeling                                                                    into the screen space with an implicit method using M = (WH) −1 .
Unlike 3DGS [Kerbl et al. 2023], which models the entire angular                    However, the inverse transformation introduces numerical insta-
radiance in a blob, we simplify the 3-dimensional modeling by adopt-                bility, especially when the splat degenerates into a line segment
ing “flat” 2D Gaussians embedded in 3D space. With 2D Gaussian                      (i.e., if it is viewed from the side). To address this issue, previous
modeling, the primitive distributes densities within a planar disk,                 surface splatting rendering methods discard such ill-conditioned
defining the normal as the direction of the steepest change of den-                 transformations using a predefined threshold [Zwicker et al. 2004].
sity. This feature enables better alignment with thin surfaces. While               However, such a scheme poses challenges within a differentiable
previous methods [Kopanas et al. 2021; Yifan et al. 2019] also utilize              rendering framework, as thresholding can lead to unstable opti-
2D Gaussians for geometry reconstruction, they require a dense                      mization. To address this problem, we utilize an explicit ray-splat
point cloud or ground-truth normals as input. By contrast, we si-                   intersection inspired by [Sigg et al. 2006].
multaneously reconstruct the appearance and geometry given only
a sparse calibration point cloud and photometric supervision.                          Ray-splat Intersection. We efficiently locate the ray-splat inter-
   As illustrated in Figure 3, our 2D splat is characterized by its                 sections by finding the intersection of three non-parallel planes, a
central point p𝑘 , two principal tangential vectors t𝑢 and t𝑣 , and                 method originally designed for specialized hardware [Weyrich et al.
a scaling vector S = (𝑠𝑢 , 𝑠 𝑣 ) that controls the variances of the 2D              2007]. Given an image coordinate x = (𝑥, 𝑦), we parameterize the ray
Gaussian. Notice that the primitive normal is defined by two orthog-                of a pixel in the projective space as the intersection of two orthogo-
onal tangential vectors t𝑤 = t𝑢 × t𝑣 . We can arrange the orientation               nal planes: the x-plane and the y-plane. Specifically, the x-plane is
into a 3 × 3 rotation matrix R = [t𝑢 , t𝑣 , t𝑤 ] and the scaling factors            defined by a normal vector (−1, 0, 0) and an offset 𝑥. The x-plane
into a 3 × 3 diagonal matrix S whose last entry is zero.                            can be represented as a 4D homogeneous plane h𝑥 = (−1, 0, 0, 𝑥) T .
   A 2D Gaussian is therefore defined in a local tangent plane in                   Similarly, the y-plane is h𝑦 = (0, −1, 0, 𝑦) T . Thus, the ray x = (𝑥, 𝑦)
world space, which is parameterized:                                                is determined by the intersection of the two planes.
                                                                                       Next, we transform both planes into the local coordinates of
                                                                                    the 2D Gaussian primitives, the 𝑢𝑣-coordinate system. Note that
              𝑃 (𝑢, 𝑣) = p𝑘 + 𝑠𝑢 t𝑢 𝑢 + 𝑠 𝑣 t𝑣 𝑣 = H(𝑢, 𝑣, 1, 1) T           (4)
                                                                                transforming points on a plane using a transformation matrix M
                          𝑠𝑢 t𝑢 𝑠 𝑣 t𝑣 0 p𝑘             RS p𝑘                       is equivalent to transforming homogeneous plane parameters us-
             where H =                              =                        (5)
                            0       0      0 1           0       1                  ing the inverse transpose M −T [Blinn 1977]. Therefore, applying
                                                                                    M = (WH) −1 is equivalent to (WH) T , eliminating explicit matrix
where H ∈ 4 × 4 is a homogeneous transformation matrix repre-                       inversion and yielding:
senting the geometry of the 2D Gaussian. For the point u = (𝑢, 𝑣) in
                                                                                                      h𝑢 = (WH) T h𝑥        h𝑣 = (WH) T h𝑦               (8)
𝑢𝑣 space, its 2D Gaussian value can then be evaluated by standard
Gaussian                                                                              As introduced in Section 4.1, points on the 2D Gaussian plane are
                                   2                                               represented as (𝑢, 𝑣, 1, 1). At the same time, the intersection point
                                    𝑢 + 𝑣2
                                           
                     G(u) = exp −                                (6)                should fall in the transformed 𝑥-plane and 𝑦-plane. Thus,
                                       2
                                                                                                    h𝑢 · (𝑢, 𝑣, 1, 1) T = h𝑣 · (𝑢, 𝑣, 1, 1) T = 0        (9)
The center p𝑘 , scaling (𝑠𝑢 , 𝑠 𝑣 ), and the rotation (t𝑢 , t𝑣 ) are learnable      This leads to an efficient solution for the intersection point u(x):
parameters. Following 3DGS [Kerbl et al. 2023], each 2D Gaussian
                                                                                                     h2 h4 − h𝑢4 h2𝑣                h4 h1 − h𝑢1 h4𝑣
primitive has opacity 𝛼 and view-dependent appearance 𝑐 parame-                               𝑢 (x) = 𝑢1 2𝑣                  𝑣 (x) = 𝑢1 2𝑣              (10)
terized with spherical harmonics.                                                                    h𝑢 h𝑣 − h𝑢2 h1𝑣                h𝑢 h𝑣 − h𝑢2 h1𝑣

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
                                                                                          2D Gaussian Splatting for Geometrically Accurate Radiance Fields   • 5


where h𝑢𝑖 , h𝑖𝑣 are the 𝑖-th parameter of the 4D plane. Note that h𝑢3              Normal Consistency. As our representation is based on 2D Gauss-
and h3𝑣 are always zero according to Eq. 5. Once we obtain the local            ian surface elements, we must ensure that all 2D splats are locally
coordinates (𝑢, 𝑣), we can calculate the depth 𝑧 of the intersected             aligned with the actual surfaces. In the context of volume rendering
points using Eq. 7 and evaluate the Gaussian value with Eq. 6.                  where multiple semi-transparent surfels may exist along the ray,
                                                                                we consider the actual surface at the median point of intersection
   Degenerate Solutions. When a 2D Gaussian is observed from a
                                                                                p𝑠 , where the accumulated opacity reaches 0.5. We then align the
slanted viewpoint, it degenerates to a line in screen space. Therefore,
                                                                                splats’ normal with the gradients of the depth maps as follows:
it might be missed during rasterization. To deal with these cases and                                        ∑︁
stabilize optimization, we employ the object-space low-pass filter                                    L𝑛 =       𝜔𝑖 (1 − n𝑖T N)                  (14)
introduced in [Botsch et al. 2005]:                                                                               𝑖
                               n              x−c o                             where 𝑖 indexes over intersected splats along the ray, 𝜔 denotes
                  Ĝ(x) = max G(u(x)), G(           )              (11)
                                                𝜎                               the blending weight of the intersection point, n𝑖 represents the
where u(x) is given by Eq. 10 and c is the projection of center p𝑘 .            normal of the splat that is oriented towards the camera, and N is
Intuitively, Ĝ(x) is lower-bounded by a fixed screen-space Gaussian            the normal estimated by the gradient of the depth map. Specifically,
                                                                                N is computed with finite differences from nearby depth points as
low-pass√ filter with center c and radius 𝜎. In our experiments, we
set 𝜎 = 2/2 to ensure sufficient pixels are used during rendering.              follows:
                                                                                                                 ∇𝑥 p𝑠 × ∇𝑦 p𝑠
   Rasterization. We follow a similar rasterization process as in                                    N(𝑥, 𝑦) =                                 (15)
                                                                                                                |∇𝑥 p𝑠 × ∇𝑦 p𝑠 |
3DGS [Kerbl et al. 2023]. First, a screen space bounding box is com-
puted for each Gaussian primitive. Then, 2D Gaussians are sorted                By aligning the splat normal with the estimated surface normal, we
based on the depth of their center and organized into tiles based on            ensure that 2D splats locally approximate the actual object surface.
their bounding boxes. Finally, volumetric alpha blending is used to                Final Loss. Finally, we optimize our model from an initial sparse
integrate alpha-weighted appearance from front to back:                         point cloud using a set of posed images. We minimize the following
                   ∑︁                        −1
                                            𝑖Ö                                  loss function:
          c(x) =         c𝑖 𝛼𝑖 Ĝ𝑖 (u(x))         (1 − 𝛼 𝑗 Ĝ𝑗 (u(x)))   (12)                           L = L𝑐 + 𝛼 L𝑑 + 𝛽L𝑛                     (16)
                   𝑖=1                      𝑗=1
                                                                                where L𝑐 is an RGB reconstruction loss combining L1 with the
The iterative process is terminated when the accumulated opacity                D-SSIM term from [Kerbl et al. 2023], while L𝑑 and L𝑛 are regu-
reaches saturation.                                                             larization terms. We set 𝛼 = 1000 for bounded scenes, 𝛼 = 100 for
                                                                                unbounded scenes, and 𝛽 = 0.05 for all scenes.
5   TRAINING
Our 2D Gaussian method, while effective in geometric modeling, can              6     EXPERIMENTS
result in noisy reconstructions when optimized only with photomet-              We now present evaluations of our 2D Gaussian Splatting recon-
ric losses, a challenge inherent to 3D reconstruction tasks [Barron             struction method, including appearance and geometry comparison
et al. 2022b; Yu et al. 2022b; Zhang et al. 2020]. To mitigate this             with previous state-of-the-art implicit and explicit approaches. We
issue and improve the geometry reconstruction, we introduce two                 then analyze the contribution of the proposed components.
regularization terms: depth distortion and normal consistency.
   Depth Distortion. Different from NeRF, 3DGS’s volume rendering
                                                                                6.1    Implementation
doesn’t consider the distance between intersected Gaussian primi-               We implement our 2D Gaussian Splatting with custom CUDA ker-
tives. Therefore, spreading out Gaussians might result in a similar             nels, building upon the framework of 3DGS [Kerbl et al. 2023]. We
color and depth rendering. This is different from surface rendering,            extend the renderer to output depth distortion maps, depth maps
where rays intersect the first visible surface exactly once. To miti-           and normal maps for regularizations (See detailed computations in
gate this issue, we take inspiration from Mip-NeRF360 [Barron et al.            Appendices A and B of the supplemental material). During train-
2022a] and propose a depth distortion loss to concentrate the weight            ing, we increase the number of 2D Gaussian primitives following
distribution along the rays by minimizing the distance between the              the adaptive control strategy in 3DGS. Since our method does not
ray-splat intersections:                                                        directly rely on the gradient of the projected 2D center, we hence
                                 ∑︁                                             project the gradient of 3D center p𝑘 onto the screen space as an
                          L𝑑 =         𝜔𝑖 𝜔 𝑗 |𝑧𝑖 − 𝑧 𝑗 |                (13)   approximation. Similarly, we employ a gradient threshold of 0.0002
                                 𝑖,𝑗                                            and remove splats with opacity lower than 0.05 every 3000 step. We
                           Î −1                                                 conduct all the experiments on a single GTX RTX3090 GPU.
where 𝜔𝑖 = 𝛼𝑖 Ĝ𝑖 (u(x)) 𝑖𝑗=1    (1 − 𝛼 𝑗 Ĝ𝑗 (u(x))) is the blending
weight of the 𝑖−th intersection and 𝑧𝑖 is the depth of the intersection            Mesh Extraction. To extract meshes from reconstructed 2D splats,
points. Unlike the distortion loss in Mip-NeRF360, where 𝑧𝑖 is the              we render depth maps of the training views using the depth value
distance between sampled points and is not optimized, our approach              of the splats projected to the pixels and utilize truncated signed
directly encourages the concentration of the splats by adjusting the            distance fusion (TSDF) to fuse the reconstruction depth maps, using
intersection depth 𝑧𝑖 . Note that we implement this regularization              Open3D [Zhou et al. 2018]. We set the voxel size to 0.004 and the
term efficiently with CUDA in a manner similar to [Sun et al. 2022b].           truncated threshold to 0.02 during TSDF fusion. We also extend the

                                                                                       SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
6 •                  Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao




                     Ground truth                 Ours (color)              Ours (normal)                      Ours                         3DGS                       SuGaR
Fig. 4. Visual comparisons (test-set view) between our method, 3DGS [Kerbl et al. 2023], and SuGaR [Guédon and Lepetit 2023] using scenes from an real-world
dataset [Barron et al. 2022b]. Our method excels at synthesizing geometrically accurate radiance fields and surface reconstruction, outperforming 3DGS and
SuGaR in capturing sharp edges and intricate details.

Table 1. Quantitative comparison on the DTU Dataset [Jensen et al. 2014]. Our 2DGS achieves the highest reconstruction accuracy among other methods and
provides 100× speed up compared to the SDF based baselines.

                                                             24      37     40     55     63     65     69     83     97    105    106    110     114    118    122    Mean     Time
                     NeRF [Mildenhall et al. 2021]          1.90    1.60   1.85   0.58   2.28   1.27   1.47   1.67   2.05   1.07   0.88   2.53    1.06   1.15   0.96   1.49    > 12h



 explicit implicit
                     VolSDF [Yariv et al. 2021]             1.14    1.26   0.81   0.49   1.25   0.70   0.72   1.29   1.18   0.70   0.66   1.08    0.42   0.61   0.55   0.86     >12h
                     NeuS [Wang et al. 2021]                1.00    1.37   0.93   0.43   1.10   0.65   0.57   1.48   1.09   0.83   0.52   1.20    0.35   0.49   0.54   0.84     >12h
                     3DGS [Kerbl et al. 2023]               2.14    1.53   2.08   1.68   3.49   2.21   1.43   2.07   2.22   1.75   1.79   2.55    1.53   1.52   1.50   1.96    11.2 m
                     SuGaR [Guédon and Lepetit 2023]        1.47    1.33   1.13   0.61   2.25   1.71   1.15   1.63   1.62   1.07   0.79   2.45    0.98   0.88   0.79   1.33     ∼ 1h
                     2DGS-15k (Ours)                        0.48    0.92   0.42   0.40   1.04   0.83   0.83   1.36   1.27   0.76   0.72   1.63    0.40   0.76   0.60   0.83     5.5 m
                     2DGS-30k (Ours)                        0.48    0.91   0.39   0.39   1.01   0.83   0.81   1.36   1.27   0.76   0.70   1.40    0.40   0.76   0.52   0.80    10.9 m


Table 2. Quantitative results on the Tanks and Temples Dataset [Knapitsch                               Table 3. Performance comparison between 2DGS (ours), 3DGS and SuGaR
et al. 2017]. We report the F1 score and training time.                                                 on the DTU dataset [Jensen et al. 2014]. We report the averaged chamfer
                                                                                                        distance, PSNR (training-set view), reconstruction time, and model size.
                               NeuS   Geo-Neus    Neurlangelo      SuGaR    3DGS      Ours
 Barn                          0.29     0.33         0.70           0.14     0.13     0.41                                                       CD ↓ PSNR ↑ Time ↓ MB (Storage) ↓
 Caterpillar                   0.29     0.26         0.36           0.16     0.08     0.23               3DGS [Kerbl et al. 2023]                1.96  35.76 11.2 m      113
 Courthouse                    0.17     0.12         0.28           0.08     0.09     0.16               SuGaR [Guédon and Lepetit 2023]         1.33  34.57  ∼1 h      1247
 Ignatius                      0.83     0.72         0.89           0.33     0.04     0.51               2DGS-15k (Ours)                         0.83  33.42 5.5 m       52
 Meetingroom                   0.24     0.20         0.32           0.15     0.01     0.17               2DGS-30k (Ours)                         0.80  34.52 10.9 m      52
 Truck                         0.45     0.45         0.48           0.26     0.19     0.45
 Mean                          0.38     0.35         0.50           0.19     0.09     0.32
                                                                                                        resolution 1600 × 1200. We use Colmap [Schönberger and Frahm
 Time                          >24h     >24h         >24h           >1h     14.3 m   15.5 m
                                                                                                        2016] to generate a sparse point cloud for each scene and down-
original 3DGS to render depth and employ the same technique for                                         sample the images into a resolution of 800 × 600 for efficiency. We
surface reconstruction for a fair comparison.                                                           use the same training process for 3DGS [Kerbl et al. 2023] and
                                                                                                        SuGaR [Guédon and Lepetit 2023] for a comparison.
6.2                  Comparison                                                                           Geometry Reconstruction. In Table 1 and Table 3, we compare
   Dataset. We evaluate the performance of our method on vari-                                          our geometry reconstruction to SOTA implicit (i.e., NeRF [Milden-
ous datasets, including DTU [Jensen et al. 2014], Tanks and Tem-                                        hall et al. 2020], VolSDF [Yariv et al. 2021], and NeuS [Wang et al.
ples [Knapitsch et al. 2017], and Mip-NeRF360 [Barron et al. 2022a].                                    2021]), explicit (i.e., 3DGS [Kerbl et al. 2023] and concurrent work
The DTU dataset comprises 15 scenes, each with 49 or 69 images of                                       SuGaR [Guédon and Lepetit 2023]) methods on Chamfer distance

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
                                                                                         2D Gaussian Splatting for Geometrically Accurate Radiance Fields    • 7


                                                                               Table 5. Quantitative studies for the regularization terms and mesh extrac-
                                                                               tion methods on the DTU dataset.

                                                                                                                Accuracy ↓ Completion ↓ Average ↓
                                                                                A. w/o normal consistency          1.35       1.13        1.24
                                                                                B. w/o depth distortion            0.89       0.87        0.88
                                                                                C. w / expected depth              0.88       1.01        0.94
                                                                                D. w / SPSR                        1.25       0.89        1.07
                                                                                E. Full Model                      0.79       0.86        0.83



Fig. 5. Qualitative comparison on the DTU benchmark [Jensen et al. 2014].
Our 2DGS produces detailed and noise-free surfaces.

Table 4. Quantitative results on Mip-NeRF 360 [Barron et al. 2022a] dataset.
All scores of the baseline methods are directly taken from their papers
whenever available. We report the performance of 3DGS, SuGaR and ours
using 30𝑘 iterations.

                                                                                       Input          (A) w/o. NC          (B) w/o. DD          Full Model
                        Outdoor Scene                Indoor scene
                 PSNR ↑ SSIM ↑ LIPPS ↓        PSNR ↑ SSIM ↑ LIPPS ↓
 NeRF             21.46    0.458    0.515      26.84    0.790     0.370        Fig. 6. Qualitative studies for the regularization effects. From left to right –
 Deep Blending    21.54    0.524    0.364      26.40    0.844     0.261        input image, surface normals without normal consistency, without depth
 Instant NGP      22.90    0.566    0.371      29.15    0.880     0.216        distortion, and our full model. Disabling the normal consistency loss leads
 MERF             23.19    0.616    0.343      27.80    0.855     0.271        to noisy surface orientations; conversely, omitting depth distortion regular-
 BakedSDF         22.47    0.585    0.349      27.06    0.836     0.258        ization results in blurred surface normals. The complete model, employing
 MipNeRF360       24.47    0.691    0.283      31.72    0.917     0.180        both regularizations, successfully captures sharp and flat features.
 Mobile-NeRF       21.95    0.470    0.470       -         -        -
 SuGaR             22.93    0.629    0.356     29.43     0.906    0.225
 3DGS              24.64    0.731    0.234     30.41     0.920    0.189        competitive NVS results across state-of-the-art techniques while
 2DGS (Ours)       24.34    0.717    0.246     30.40     0.916    0.195        providing geometrically accurate surface reconstruction. We include
                                                                               the appearance rendering results in Figure 11.
and training time using the DTU dataset. Our method outperforms                6.3    Ablations
all compared methods in terms of Chamfer distance. Moreover, as
                                                                               In this section, we isolate the design choices and measure their effect
shown in Table 2, 2DGS achieves competitive results with SDF mod-
                                                                               on reconstruction quality, including regularization terms and mesh
els (i.e., NeuS [Wang et al. 2021] and Geo-Neus [Fu et al. 2022])
                                                                               extraction. We conduct experiments on the DTU dataset [Jensen et al.
on the TnT dataset, and significantly better reconstruction than
                                                                               2014] with 15𝑘 iterations and report the reconstruction accuracy,
explicit reconstruction methods (i.e., 3DGS and SuGaR). Notably,
                                                                               completeness and average reconstruction quality. The quantitative
our model demonstrates exceptional efficiency, offering a recon-
                                                                               effect of the choices is reported in Table 5. Additional baseline com-
struction speed that is approximately 100 times faster compared
                                                                               parisons can be found in Appendix C of the supplemental material.
to implicit reconstruction methods and more than 3 times faster
than the concurrent work SuGaR. Our approach can also achieve                     Regularization. We first examine the effects of the proposed nor-
qualitatively better reconstructions with more appearance and ge-              mal consistency and depth distortion regularization terms. Our
ometry details and fewer outliers, as shown in Figure 5. Moreover,             model (Table 5 E) provides the best performance when applying
SDF-based reconstruction methods require predefining the spherical             both regularization terms. We observe that disabling the normal
size for initialization, which plays a critical role in the success of         consistency (Table 5 A) can lead to incorrect orientation, as shown
SDF reconstruction. By contrast, our method leverages radiance                 in Figure 6 A. Additionally, the absence of depth distortion (Table 5
field based geometry modeling and is less sensitive to initialization.         B) results in a noisy surface, as shown in Figure 6 B.
We include the full geometry reconstruction results for both DTU
                                                                                  Mesh Extraction. We now analyze our choice for mesh extraction.
and TnT in Figure 9 and Figure 10.
                                                                               Our full model (Table 5 E) utilizes TSDF fusion for mesh extraction
   Appearance Reconstruction. Our method represents 3D scenes                  with median depth. One alternative option is to use the expected
as radiance fields, providing high-quality novel view synthesis. In            depth instead of the median depth. However, it yields worse recon-
this section, we compare our novel view renderings using the Mip-              structions as it is more sensitive to outliers, as shown in Table 5
NeRF360 dataset against baseline approaches, as shown in Table 4               C. Further, our approach surpasses screened Poisson surface recon-
and Figure 4. Note that, since the ground truth geometry is not                struction (SPSR)(Table 5 D) [Kazhdan and Hoppe 2013] using 2D
available in the Mip-NeRF360 dataset and we hence focus on quan-               Gaussians’ center and normal as inputs, due to SPSR’s inability to
titative comparison. Remarkably, our method consistently achieves              incorporate the opacity and the size of 2D Gaussian primitives.

                                                                                      SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
8 •    Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao


7     CONCLUSION                                                                          Qiancheng Fu, Qingshan Xu, Yew-Soon Ong, and Wenbing Tao. 2022. Geo-Neus:
                                                                                             Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruc-
We presented 2D Gaussian splatting, a novel approach for geomet-                             tion. Advances in Neural Information Processing Systems (NeurIPS) (2022).
rically accurate radiance field reconstruction. We utilized 2D Gauss-                     Jian Gao, Chun Gu, Youtian Lin, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. 2023. Re-
                                                                                             lightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition
ian primitives for 3D scene representation, facilitating accurate and                        and Ray Tracing. arXiv:2311.16043 (2023).
view consistent geometry modeling and rendering. We proposed                              Antoine Guédon and Vincent Lepetit. 2023. SuGaR: Surface-Aligned Gaussian Splatting
two regularization techniques to further enhance the reconstructed                           for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. arXiv
                                                                                             preprint arXiv:2311.12775 (2023).
geometry. Extensive experiments on several challenging datasets                           Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul De-
verify the effectiveness and efficiency of our method.                                       bevec. 2021. Baking neural radiance fields for real-time view synthesis. In Proceedings
                                                                                             of the IEEE/CVF International Conference on Computer Vision. 5875–5884.
   Limitations. While our method successfully delivers accurate ap-                       Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen
                                                                                             Ma. 2023. Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural
pearance and geometry reconstruction for a wide range of objects                             Radiance Fields. In ICCV.
and scenes, we also discuss its limitations: First, we assume surfaces                    Eldar Insafutdinov and Alexey Dosovitskiy. 2018. Unsupervised learning of shape and
                                                                                             pose with differentiable point clouds. Advances in neural information processing
with full opacity and extract meshes from multi-view depth maps.                             systems 31 (2018).
This can pose challenges in accurately handling semi-transparent                          Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. 2014.
surfaces, such as glass, due to their complex light transmission prop-                       Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference
                                                                                             on computer vision and pattern recognition. 406–413.
erties, as shown in Figure 12. Secondly, our current densification                        Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and
strategy favors texture-rich over geometry-rich areas, occasionally                          Yuexin Ma. 2023. GaussianShader: 3D Gaussian Splatting with Shading Functions
leading to less accurate representations of fine geometric structures.                       for Reflective Surfaces. arXiv preprint arXiv:2311.17977 (2023).
                                                                                          Michael Kazhdan and Hugues Hoppe. 2013. Screened poisson surface reconstruction.
A more effective densification strategy could mitigate this issue.                           ACM Transactions on Graphics (ToG) 32, 3 (2013), 1–13.
Finally, our regularization often involves a trade-off between image                      Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
quality and geometry, and can potentially lead to over-smoothing                             3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on
                                                                                             Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
in certain regions.                                                                       Leonid Keselman and Martial Hebert. 2022. Approximate differentiable rendering with
                                                                                             algebraic surfaces. In European Conference on Computer Vision. Springer, 596–614.
                                                                                          Leonid Keselman and Martial Hebert. 2023. Flexible techniques for differentiable
ACKNOWLEDGMENTS                                                                              rendering with 3d gaussians. arXiv preprint arXiv:2308.14737 (2023).
BH and SG are supported by NSFC #62172279, #61932020, Program of                          Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and
                                                                                             Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions on
Shanghai Academic Research Leader. ZY, AC and AG are supported                               Graphics 36, 4 (2017).
by the ERC Starting Grant LEGO-3D (850533) and DFG EXC number                             Georgios Kopanas, Julien Philip, Thomas Leimkühler, and George Drettakis. 2021. Point-
                                                                                             Based Neural Rendering with Per-View Optimization. In Computer Graphics Forum,
2064/1 - project number 390727645.                                                           Vol. 40. Wiley Online Library, 29–43.
                                                                                          Christoph Lassner and Michael Zollhofer. 2021. Pulsar: Efficient sphere-based neural
REFERENCES                                                                                   rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
                                                                                             Recognition. 1440–1449.
Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lem-         Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-
   pitsky. 2020. Neural point-based graphics. In Computer Vision–ECCV 2020: 16th             Yu Liu, and Chen-Hsuan Lin. 2023. Neuralangelo: High-Fidelity Neural Surface
   European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16.          Reconstruction. In IEEE Conference on Computer Vision and Pattern Recognition
   Springer, 696–712.                                                                        (CVPR).
Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-         Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia. 2023. GS-IR: 3D Gaussian
   Brualla, and Pratul P. Srinivasan. 2021. Mip-NeRF: A Multiscale Representation for        Splatting for Inverse Rendering. arXiv preprint arXiv:2311.16473 (2023).
   Anti-Aliasing Neural Radiance Fields. ICCV (2021).                                     Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. 2020.
Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman.        Neural Sparse Voxel Fields. NeurIPS (2020).
   2022a. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings     Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. 2024. Dynamic
   of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5470–5479.         3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In 3DV.
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.   Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas
   2022b. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. CVPR (2022).          Geiger. 2019. Occupancy Networks: Learning 3D Reconstruction in Function Space.
Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.      In Conference on Computer Vision and Pattern Recognition (CVPR).
   2023. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. ICCV (2023).           Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ra-
James F Blinn. 1977. A homogeneous formulation for lines in 3 space. In Proceedings of       mamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields
   the 4th annual conference on Computer graphics and interactive techniques. 237–241.       for View Synthesis. In ECCV.
Mario Botsch, Alexander Hornung, Matthias Zwicker, and Leif Kobbelt. 2005. High-          Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-
   quality surface splatting on today’s GPUs. In Proceedings Eurographics/IEEE VGTC          mamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields
   Symposium Point-Based Graphics, 2005. IEEE, 17–141.                                       for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. 2022. TensoRF:             Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant
   Tensorial Radiance Fields. In European Conference on Computer Vision (ECCV).              Neural Graphics Primitives with a Multiresolution Hash Encoding. ACM Trans.
Hanlin Chen, Chen Li, and Gim Hee Lee. 2023b. NeuSG: Neural Implicit Surface Re-             Graph. 41, 4, Article 102 (July 2022), 15 pages.
   construction with 3D Gaussian Splatting Guidance. arXiv preprint arXiv:2312.00846      Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. 2020. Differ-
   (2023).                                                                                   entiable Volumetric Rendering: Learning Implicit 3D Representations without 3D
Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. 2023a.                Supervision. In Conference on Computer Vision and Pattern Recognition (CVPR).
   Mobilenerf: Exploiting the polygon rasterization pipeline for efficient neural field   Michael Oechsle, Songyou Peng, and Andreas Geiger. 2021. UNISURF: Unifying Neural
   rendering on mobile architectures. In Proceedings of the IEEE/CVF Conference on           Implicit Surfaces and Radiance Fields for Multi-View Reconstruction. In International
   Computer Vision and Pattern Recognition. 16569–16578.                                     Conference on Computer Vision (ICCV).
Zhang Chen, Zhong Li, Liangchen Song, Lele Chen, Jingyi Yu, Junsong Yuan, and Yi          Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Love-
   Xu. 2023c. NeuRBF: A Neural Fields Representation with Adaptive Radial Basis              grove. 2019. DeepSDF: Learning Continuous Signed Distance Functions for Shape
   Functions. In Proceedings of the IEEE/CVF International Conference on Computer            Representation. In The IEEE Conference on Computer Vision and Pattern Recognition
   Vision. 4182–4194.                                                                        (CVPR).
Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and           Hanspeter Pfister, Matthias Zwicker, Jeroen Van Baar, and Markus Gross. 2000. Surfels:
   Angjoo Kanazawa. 2022. Plenoxels: Radiance Fields without Neural Networks. In             Surface elements as rendering primitives. In Proceedings of the 27th annual conference
   CVPR.


SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
                                                                                                     2D Gaussian Splatting for Geometrically Accurate Radiance Fields      • 9


   on Computer graphics and interactive techniques. 335–342.                              Zehao Yu, Anpei Chen, Bozidar Antic, Songyou Peng, Apratim Bhattacharyya, Michael
Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain,          Niemeyer, Siyu Tang, Torsten Sattler, and Andreas Geiger. 2022a. SDFStudio: A Uni-
   and Matthias Nießner. 2023. GaussianAvatars: Photorealistic Head Avatars with             fied Framework for Surface Reconstruction. https://github.com/autonomousvision/
   Rigged 3D Gaussians. arXiv preprint arXiv:2312.02069 (2023).                              sdfstudio
Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. 2021. KiloNeRF: Speed-     Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. Mip-
   ing up Neural Radiance Fields with Thousands of Tiny MLPs. In International               Splatting: Alias-free 3D Gaussian Splatting. Conference on Computer Vision and
   Conference on Computer Vision (ICCV).                                                     Pattern Recognition (CVPR) (2024).
Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan, Ben Mildenhall, Andreas   Zehao Yu and Shenghua Gao. 2020. Fast-MVSNet: Sparse-to-Dense Multi-View Stereo
   Geiger, Jon Barron, and Peter Hedman. 2023. Merf: Memory-efficient radiance fields        With Learned Propagation and Gauss-Newton Refinement. In Conference on Com-
   for real-time view synthesis in unbounded scenes. ACM Transactions on Graphics            puter Vision and Pattern Recognition (CVPR).
   (TOG) 42, 4 (2023), 1–12.                                                              Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger.
Darius Rückert, Linus Franke, and Marc Stamminger. 2022. Adop: Approximate dif-              2022b. MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface
   ferentiable one-pixel point rendering. ACM Transactions on Graphics (ToG) 41, 4           Reconstruction. Advances in Neural Information Processing Systems (NeurIPS) (2022).
   (2022), 1–14.                                                                          Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. 2020. NeRF++: Analyzing
Johannes Lutz Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion                 and Improving Neural Radiance Fields. arXiv:2010.07492 (2020).
   Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR).            Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. 2018. Open3D: A Modern Library for
Johannes Lutz Schönberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm.             3D Data Processing. arXiv:1801.09847 (2018).
   2016. Pixelwise View Selection for Unstructured Multi-View Stereo. In European         Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollhöfer, Jus-
   Conference on Computer Vision (ECCV).                                                     tus Thies, and Javier Romero. 2023. Drivable 3D Gaussian Avatars. (2023).
Thomas Schöps, Torsten Sattler, and Marc Pollefeys. 2019. Surfelmeshing: Online              arXiv:2311.08581 [cs.CV]
   surfel-based mesh reconstruction. IEEE transactions on pattern analysis and machine    Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. 2001a. EWA
   intelligence 42, 10 (2019), 2494–2507.                                                    volume splatting. In Proceedings Visualization, 2001. VIS’01. IEEE, 29–538.
Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen Zhao, Haocheng Feng, Jingtuo            Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. 2001b. Surface
   Liu, Liangjun Zhang, Jian Zhang, Bin Zhou, Errui Ding, and Jingdong Wang. 2023.           splatting. In Proceedings of the 28th annual conference on Computer graphics and
   GIR: 3D Gaussian Inverse Rendering for Relightable Scene Factorization. Arxiv             interactive techniques. 371–378.
   (2023). arXiv:2312.05133                                                               Matthias Zwicker, Jussi Rasanen, Mario Botsch, Carsten Dachsbacher, and Mark Pauly.
Christian Sigg, Tim Weyrich, Mario Botsch, and Markus H Gross. 2006. GPU-based               2004. Perspective accurate splatting. In Proceedings-Graphics Interface. 247–254.
   ray-casting of quadratic surfaces.. In PBG@ SIGGRAPH. 59–65.
Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022a. Direct Voxel Grid Optimization:
   Super-fast Convergence for Radiance Fields Reconstruction. In CVPR.
Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022b. Improved Direct Voxel Grid               A    DETAILS OF DEPTH DISTORTION
   Optimization for Radiance Fields Reconstruction. arxiv cs.GR 2206.05085 (2022).
Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping            While Barron et al. [Barron et al. 2022b] calculates the distortion
   Wang. 2021. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for            loss with samples on the ray, we operate Gaussian primitives, where
   Multi-view Reconstruction. Advances in Neural Information Processing Systems 34
   (2021), 27171–27183.                                                                   the intersected depth may not be ordered. To this end, we adopt
Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and          an L2 loss and transform the intersected depth 𝑧 to NDC space to
   Lingjie Liu. 2023. NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view     down-weight distant Gaussian primitives, 𝑚 = NDC(𝑧), with near
   Reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer
   Vision (ICCV).                                                                         and far plane empirically set to 0.2 and 1000. We implemented our
Tim Weyrich, Simon Heinzle, Timo Aila, Daniel B Fasnacht, Stephan Oetiker, Mario          depth distortion loss based on [Sun et al. 2022b], also powered by
   Botsch, Cyril Flaig, Simon Mall, Kaspar Rohrer, Norbert Felber, et al. 2007. A
   hardware architecture for surface splatting. ACM Transactions on Graphics (TOG)
                                                                                          tile-based rendering. Here we show that the nested algorithm can
   26, 3 (2007), 90–es.                                                                   be implemented in a single forward pass:
Thomas Whelan, Renato F Salas-Moreno, Ben Glocker, Andrew J Davison, and Stefan
   Leutenegger. 2016. ElasticFusion: Real-time dense SLAM and light source estimation.
   The International Journal of Robotics Research 35, 14 (2016), 1697–1716.                               𝑁
                                                                                                          ∑︁−1 ∑︁
                                                                                                               𝑖 −1
Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. 2020. SynSin:
   End-to-end View Synthesis from a Single Image. In Proceedings of the IEEE/CVF
                                                                                                   L=                 𝜔𝑖 𝜔 𝑗 (𝑚𝑖 − 𝑚 𝑗 ) 2
   Conference on Computer Vision and Pattern Recognition (CVPR).                                          𝑖=0 𝑗=0
Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu                         𝑁 −1       𝑖 −1      𝑖 −1                𝑖 −1
   Jiang. 2023. PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynam-                       ∑︁    © ∑︁           ∑︁                  ∑︁
                                                                                                      =      𝜔𝑖 ­𝑚𝑖2      𝜔𝑗 +      𝜔 𝑗 𝑚 2𝑗 − 2𝑚𝑖                        (17)
                                                                                                                                                               ª
   ics. arXiv preprint arXiv:2311.12198 (2023).                                                                                                         𝜔 𝑗𝑚 𝑗 ®
Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xi-                           𝑖=0          𝑗=0       𝑗=0                 𝑗=0
   anpeng Lang, Xiaowei Zhou, and Sida Peng. 2023. Street Gaussians for Modeling
                                                                                                                «                                              ¬
   Dynamic Urban Scenes. (2023).                                                                        𝑁
                                                                                                        ∑︁−1                                    
Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. 2018. MVSNet: Depth                          =      𝜔𝑖 𝑚𝑖2𝐴𝑖 −1 + 𝐷𝑖2−1 − 2𝑚𝑖 𝐷𝑖 −1 ,
   Inference for Unstructured Multi-view Stereo. European Conference on Computer                          𝑖=0
   Vision (ECCV) (2018).
Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural
   implicit surfaces. Advances in Neural Information Processing Systems 34 (2021),                        Í𝑖            Í𝑖                 2 Í𝑖          2
   4805–4815.                                                                             where 𝐴𝑖 =     𝑗=0 𝜔 𝑗 , 𝐷𝑖 =  𝑗=0 𝜔 𝑗 𝑚 𝑗 and 𝐷𝑖 = 𝑗=0 𝜔 𝑗 𝑚 𝑗 .
Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P. Srinivasan, Richard
   Szeliski, Jonathan T. Barron, and Ben Mildenhall. 2023. BakedSDF: Meshing Neural          Specifically, we let 𝑒𝑖 = 𝑚𝑖2𝐴𝑖 −1 + 𝐷𝑖2−1 − 2𝑚𝑖 𝐷𝑖 −1 so that the
   SDFs for Real-Time View Synthesis. arXiv (2023).                                                                                     Í
Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and
                                                                                          distortion loss can be “rendered” as L𝑖 = 𝑖𝑗=0 𝜔 𝑗 𝑒 𝑗 . Here, L𝑖 mea-
   Yaron Lipman. 2020. Multiview Neural Surface Reconstruction by Disentangling           sures the depth distortion up to the 𝑖-th Gaussian. During marching
   Geometry and Appearance. Advances in Neural Information Processing Systems 33          Gaussian front-to-back, we simultaneously accumulate 𝐴𝑖 , 𝐷𝑖 and
   (2020).
Wang Yifan, Felice Serena, Shihao Wu, Cengiz Öztireli, and Olga Sorkine-Hornung.          𝐷𝑖2 , preparing for the next distortion computation L𝑖+1 . Similarly,
   2019. Differentiable surface splatting for point-based geometry processing. ACM        the gradient of the depth distortion can be back-propagated to the
   Transactions on Graphics (TOG) 38, 6 (2019), 1–14.
Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. 2021.
                                                                                          primitives back-to-front. Different from implicit methods where
   PlenOctrees for Real-time Rendering of Neural Radiance Fields. In ICCV.                𝑚 are the pre-defined sampled depth and non-differentiable, we
                                                                                          additionally back-propagate the gradient through the intersection
                                                                                          𝑚, encouraging the Gaussians to move tightly together directly.

                                                                                                  SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
10   •    Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao


                                          Color                                  Table 6. Additional baselines on DTU dataset. All the models are trained
                                                                                 with 30𝑘 iterations.
                                                                                                                                     Accuracy ↓ Completion ↓ Average ↓
                                                                                 SuGaR                                                  1.48       1.17        1.33
                                                                                 SuGaR + TSDF                                           2.47       1.90        2.18
                                                                                 3DGS + SPSR (center)                                   2.05       1.25        1.65
                                          Depth                                  3DGS + TSDF (mean)                                     1.93       1.99        1.96
                                                                                 2DGS + SPSR (center)                                   1.25       0.89        1.07
                                                                                 2DGS (affine approx) + TSDF (mean)                     0.96       1.20        1.08
                                                                                 2DGS (our rasterizer) + TSDF (mean)                    0.79       0.98        0.88
                                                                                 2DGS (our rasterizer) + TSDF (median)                  0.78       0.83        0.80


                Ground truth                      Ours           3DGS            improved overall completion metrics compared to results obtained
                                                                                 using TSDF.
Fig. 7. Visualization of a plane tiled by 2D Gaussians. Affine approxima-          Finally, we conduct ablation experiments on our 2DGS. Notably,
tion [Zwicker et al. 2001b] adopted in 3DGS [Kerbl et al. 2023] causes           2DGS demonstrates enhanced performance by iteratively integrat-
perspective distortion and inaccurate depth, violating normal consistency.       ing components such as TSDF, perspective-correct rasterization,
                                                                                 and median depth. For the affine approximation baseline, we utilize
B        DEPTH CALCULATIONS                                                      3DGS’s rasterization method by configuring one scale of the 3D
  Mean depth: There are two optional depth computations used for                 Gaussian to 1𝑒 −6 . While affine approximation already yields promis-
our meshing process. The mean (expected) depth is calculated by                  ing results, integrating the proposed ray-splat intersection scheme
weighting the intersected depth:                                                 results in more accurate depth map generation under perspective
                           ∑︁         ∑︁                                         projection, as depicted in Figure 7, thus enhancing depth fusion
                  𝑧 mean =    𝜔𝑖 𝑧𝑖 /( 𝜔𝑖 + 𝜖)              (18)                 performance.
                                   𝑖              𝑖

where 𝜔𝑖 = 𝑇𝑖 𝛼𝑖 Ĝ𝑖 (u(x) is the weight contribution of the 𝑖-th                D     ADDITIONAL RESULTS
                    Î −1                                                         Our 2D Gaussian Splatting method achieves comparable perfor-
Gaussian and 𝑇𝑖 = 𝑖𝑗=1   (1 − 𝛼 𝑗 Ĝ𝑗 (u(x))) measures its visibility.
It is important to normalize the depth with the accumulated alpha                mance even without the need for regularizations, as Table 7 shows.
      Í
𝐴 = 𝑖 𝜔𝑖 to ensure that a 2D Gaussian can be rendered as a planar                We have included a detailed breakdown of per-scene metrics for the
2D disk in the depth visualization.                                              MipNeRF360 dataset [Barron et al. 2022b] in Table 9. Additionally,
                                                                                 we have provided a comparison of our rendered depth maps with
   Median depth: We compute the median depth as the largest “vis-                those from 3DGS and MipNeRF360 in Figure 8.
ible” depth, considering 𝑇𝑖 = 0.5 as the pivot for surface and free
space:                                                                           Table 7. PSNR scores for Synthetic NeRF dataset. Our model achieve com-
                   𝑧 median = max{𝑧𝑖 |𝑇𝑖 > 0.5}.               (19)              parable performance without using regularizations.

We find our median depth computation is more robust to [Luiten                                   Mic     Chair   Ship     Materials     Lego    Drums     Ficus    Hotdog    Mean
et al. 2024]. When a ray’s accumulated alpha does not reach 0.5,                     Plenoxels   33.26   33.98   29.62     29.14        34.10    25.35    31.83     36.81    31.76
                                                                                     INGP-Base   36.22   35.00   31.10     29.78        36.39    26.02    33.51     37.40    33.18
while Luiten et al. sets a default value of 15, our computation selects              Mip-NeRF    36.51   35.14   30.41     30.71        35.70    25.48    33.29     37.48    33.09
the last Gaussian, which is more accurate and suitable for training.                 3DGS        35.36   35.83   30.80     30.00        35.78    26.15    34.87     37.72    33.32
                                                                                     Ours        35.09   35.05   30.60     29.74        35.10    26.05    35.57     37.36    33.07

C        ADDITIONAL BASELINES
In this section, we present additional baselines to ablate the impact                                    Table 8. PSNR scores for TnT dataset.
of our design choices, as summarized in Table 6. Furthermore, we                            Barn     Caterpillar   Courthouse Ignatius               Meetingroom     Truck    Mean
integrate our meshing approach into the comparison against these                  SuGaR     28.63      23.27         23.33      20.72                   25.47        24.40    24.16
                                                                                  3DGS      27.99      24.82         23.33      23.95                   26.89        25.01    25.33
baselines for a comprehensive analysis. SuGaR extracts a mesh from                Ours      28.79      24.23         23.51      23.82                   26.15        26.85    25.56
depth points utilizing SPSR (Screen Poisson Surface Reconstruction)
during the coarse stage, followed by refinement using a mesh ren-
derer. To assess the effect of this meshing strategy, we substituted                   Table 9. PSNR↑, SSIM↑, LIPPS↓ scores for MipNeRF360 dataset.
SPSR with TSDF using the depth maps, followed by an identical
refinement stage. However, we found that the depth map generated                          bicycle   flowers   garden     stump    treehill   room    counter   kitchen   bonsai   mean
                                                                                  SugaR    23.34      19.54    25.40      25.07    21.30     29.97    27.56     29.41     30.77   25.82
from their flat Gaussian intersection is sparse and discontinuous. As             3DGS     25.24      21.52    27.41      26.55    22.49     30.63    28.70     30.32     31.98   27.20
a result, the adaptation of TSDF with its discontinuous depth map                 Ours     24.87      21.15    26.95      26.47    22.27     31.06    28.55     30.50     31.52   27.03
                                                                                  SuGaR    0.634      0.499    0.762      0.705    0.546     0.904    0.885     0.902     0.933   0.752
yields inferior results. For 3DGS, we leverage SPSR for mesh gener-               3DGS     0.771      0.605    0.868      0.775    0.638     0.914    0.905     0.922     0.938   0.815
ation. Because the 3D Gaussian lacks a surface normal, we treat its               Ours     0.752      0.588    0.852      0.765    0.627     0.912    0.900     0.919     0.933   0.805
                                                                                  SuGaR    0.354      0.407    0.240      0.325    0.452     0.259    0.244     0.178     0.220   0.298
normal as a trainable parameter [Gao et al. 2023; Liang et al. 2023]              3DGS     0.205      0.336    0.103      0.210    0.317     0.220    0.204     0.129     0.205   0.214
distilled from the depth map, employing the normal consistency                    Ours     0.218      0.346    0.115      0.222    0.329     0.223    0.208     0.133     0.214   0.223

regularization. We then utilize all center points for SPSR, resulting in

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
                                                                                            2D Gaussian Splatting for Geometrically Accurate Radiance Fields   •   11



                  (a) Ground-truth                         (c) 3DGS, normals from depth gradient           (e) Our model (2DGS), normals from depth gradient




 (b) MipNeRF360 [Barron et al. 2022b], SSIM=0.813         (d) 3DGS [Kerbl et al. 2023], SSIM=0.834                 (f) Our model (2DGS), SSIM=0.845

Fig. 8. We visualize the depth maps generated by MipNeRF360 [Barron et al. 2022b], 3DGS [Kerbl et al. 2023], and our method. The depth maps for 3DGS (d)
and 2DGS (f) are rendered using Eq. 18 and visualized following MipNeRF360. To highlight the surface smoothness, we further visualize the normal estimated
from depth gradient using Eq. 15 for both 3DGS (c) and ours (e). While MipNeRF360 is capable of producing plausibly smooth depth maps, its sampling
process may result in the loss of detailed structures. Both 3DGS and 2DGS excel at modeling thin structures; however, as illustrated in (c) and (e), the depth
map of 3DGS exhibits significant noise. In contrast, our approach generates sampled depth points with normals consistent with the rendered normal map
(refer to Figure 1b), thereby enhancing depth fusion during the meshing process.




                                                                                          SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA.
