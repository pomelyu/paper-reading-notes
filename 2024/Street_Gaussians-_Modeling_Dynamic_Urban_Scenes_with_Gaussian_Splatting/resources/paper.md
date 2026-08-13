arXiv:2401.01339v3 [cs.CV] 18 Aug 2024
                                             Street Gaussians: Modeling Dynamic Urban
                                                    Scenes with Gaussian Splatting

                                                Yunzhi Yan1,2 Haotong Lin1 Chenxu Zhou1 Weijie Wang1
                                                         Haiyang Sun2 Kun Zhan2 Xianpeng Lang2
                                                                 Xiaowei Zhou1 Sida Peng1†

                                                                    Zhejiang University1   Li Auto2



                                               Abstract. This paper aims to tackle the problem of modeling dynamic
                                               urban streets for autonomous driving scenes. Recent methods extend
                                               NeRF by incorporating tracked vehicle poses to animate vehicles, en-
                                               abling photo-realistic view synthesis of dynamic urban street scenes.
                                               However, significant limitations are their slow training and rendering
                                               speed. We introduce Street Gaussians, a new explicit scene representa-
                                               tion that tackles these limitations. Specifically, the dynamic urban scene
                                               is represented as a set of point clouds equipped with semantic logits and
                                               3D Gaussians, each associated with either a foreground vehicle or the
                                               background. To model the dynamics of foreground object vehicles, each
                                               object point cloud is optimized with optimizable tracked poses, along
                                               with a 4D spherical harmonics model for the dynamic appearance. The
                                               explicit representation allows easy composition of object vehicles and
                                               background, which in turn allows for scene editing operations and render-
                                               ing at 135 FPS (1066 * 1600 resolution) within half an hour of training.
                                               The proposed method is evaluated on multiple challenging benchmarks,
                                               including KITTI and Waymo Open datasets. Experiments show that
                                               the proposed method consistently outperforms state-of-the-art methods
                                               across all datasets. The code will be released to ensure reproducibility.

                                               Keywords: 3D Gaussians · View Synthesis · Real-Time Rendering




                                 1           Introduction

                                 Modeling dynamic 3D streets from images has many important applications,
                                 such as city simulation, autonomous driving, and gaming. For instance, the digi-
                                 tal twin of city streets can be used as the simulation environment for self-driving
                                 vehicles, thereby reducing the training and test costs. These applications require
                                 us to efficiently reconstruct 3D street models from captured data and render
                                 high-quality novel views in real-time.
                                     With the development of neural scene representations, there have been some
                                 methods [32, 38, 43, 52, 78] that attempt to reconstruct street scenes with neural
                                         †
                                             Corresponding author
2       Y. Yan et al.




              Ours                      EmerNeRF                        MARS
       Train: 30min, FPS: 135       Train: 2.5h, FPS: 0.21       Train: >18h, FPS: 0.68



Fig. 1: Rendering results on the Waymo dataset [51]. Our method produces
high-quality rendering at 135 FPS (1066×1600) within half an hour of training. Current
SOTA methods [64, 69] suffer from high training and rendering cost.




radiance fields [35]. To improve the modeling capability, Block-NeRF [52] divides
the scene into several blocks and represents each one with a NeRF network. Al-
though this strategy enables photo-realistic rendering of large-scale street scenes,
Block-NeRF suffers from long training time due to the large amount of network
parameters. Moreover, it cannot handle dynamic vehicles on the street, which
are crucial aspects in autonomous driving environment simulation.
     Recently, some methods [23, 39, 64, 71] propose to represent dynamic driving
scenes as compositional neural representations that consist of foreground moving
cars and static background. To handle the dynamic car, they leverage tracked
vehicle poses to establish the mapping between the observation space and the
canonical space, where they use NeRF networks to model the car’s geometry
and appearance. Although these methods produce reasonable results, they are
still limited to the high training cost and low rendering speed.
    In this work, we propose a novel explicit scene representation for recon-
structing dynamic 3D street scenes from images. The basic idea is utilizing
point clouds to build dynamic scenes, which significantly increases the train-
ing and rendering efficiency. Specifically, we decompose urban street scenes into
the static background and moving vehicles, which are separately built based on
3D Gaussians [19]. To handle the dynamics of foreground vehicles, we model
their geometry as a set of points with optimizable tracked vehicle poses, where
each point stores learnable 3D Gaussian parameters. Furthermore, the time-
varying appearance is represented by a 4D spherical harmonics model that uses
a time series function to predict spherical harmonics coefficients at any time step.
Thanks to the dynamic 3D Gaussians representation, we can faithfully recon-
struct the target urban street within half an hour and achieve real-time rendering
(135FPS@1066x1600). Building upon the proposed scene representation, we de-
                      Street Gaussians for Modeling Dynamic Urban Scenes         3

velop several strategies to further improve the rendering performance, including
the tracked pose optimization,point cloud initialization, and sky modeling.
    We evaluate the proposed method on Waymo Open [51] (Waymo) and KITTI
[15] datasets, which present dynamic street scenes with complex vehicle motions
and various environment conditions. Across all datasets, our approach achieves
state-of-the-art performance in terms of rendering quality, while being rendered
over 100 times faster than previous methods [39, 64, 69]. Furthermore, detailed
ablations and scene editing applications are conducted to demonstrate the effec-
tiveness of proposed components and the flexibility of the proposed representa-
tion, respectively.
    Overall, this work makes the following contributions:
 – We propose Street Gaussians, a novel scene representation for modeling com-
   plex dynamic urban scenes, which efficiently reconstructs and renders high-
   fidelity urban street scenes in real-time.
 – We propose several strategies including 4D spherical harmonics appear-
   ance model, tracked pose optimization, and point cloud initialization, which
   largely improve the rendering performance of Street Gaussians.


2   Related work
Static scene modeling. Neural scene representation proposes to represent
3D scenes with neural networks, which can model complex scenes from images
through differentiable rendering. NeRF [3–5, 35, 36] represents continuous vol-
umetric scenes with MLP networks and achieves impressive rendering results.
Some works have been proposed to extend NeRF to urban scenes [9, 16, 18, 30,
32, 38, 43, 52, 55]. GridNeRF [66] proposes muti-resolution feature planes to help
NeRF generate photorealistic results on large-scale scenes. DNMP [32] models
the scene with deformable mesh primitives initialized by voxelizing point clouds.
NeuRas [30] takes scaffold mesh as input and optimizes the neural texture field
to perform fast rasterization.
    Point-based rendering works [1,10,22,27,45] define learned neural descriptors
on point clouds and perform differentiable rasterization with a neural renderer.
However, they require dense point clouds as input and generate blurry results un-
der regions with low point counts. A very recent work 3D Gaussian Splatting (3D
GS) [19] defines a set of anisotropic Gaussians in 3D world and performs adap-
tive density control to achieve high-quality rendering results with only sparse
point clouds input. However, 3D GS assumes the scene to be static and can not
model dynamic moving objects.
Dynamic scene modeling. Recent methods build 4D neural scene represen-
tation on single-object scenes by encoding time as additional input [2, 13, 26, 28,
29, 40, 41, 50]. Some works learn a scene decomposition of outdoor scenes under
the supervision of optical flow [56] or vision transformer feature [69]. However,
their scene representation is not instance aware, limiting the applications for au-
tonomous driving simulation. Another line of works model the scene as the com-
position of moving object models and a background model [23, 39, 54, 64, 65, 71]
4       Y. Yan et al.

with neural fields, which is most similar to us. However, they suffer from high
memory cost on large scale scene and can not perform real-time rendering.
    Extending point-based rendering to dynamic scene is also investigated re-
cently [68, 76]. Recent approaches extend 3D GS to small-scale dynamic scenes
by introducing deformation field [61, 75], physical priors [33] or 4D parametriza-
tion [73] to 3D Gaussian model. More recently, some concurrent works [8,79] also
explore 3D Gaussians in urban street scenes. DrivingGaussian [79] introduces
Incremental 3D Static Gaussians and Composite Dynamic Gaussian Graphs.
PVG [8] utilizes Periodic Vibration 3D Gaussians to model dynamic urban scene.
Simulation environments for autonomous driving. Existing self-driving
simulation engines such as CARLA [11] or AirSim [47] suffer from costly man-
ual effort to create virtual environments and the lack of realism in the generated
data. In recent years, a lot of effort has been put into building sensor simulations
from autonomous driving data captured in real scenes. Some works [12, 34, 74]
concentrate on LiDAR simulation by aggregating LiDAR and reconstructing tex-
tured primitives. However, they have difficulty handling high-resolution images
and usually produce noisy appearance. Other works [7, 57, 72] reconstruct ob-
jects from multi-view images and LiDAR input, which can be interacted with
other environments. However, these methods are restricted to existing images
and fail to render novel views. Some methods utilize neural fields to perform
multiply tasks including view synthesis [17, 39, 71], perception [14, 23, 78], gen-
eration [25, 37, 48, 67, 70] and inverse rendering [42, 58–60] on driving scenes.
However, they struggle with high training and rendering cost. In contrast, Our
method focuses on performing real-time rendering of dynamic urban scenes,
which is crucial for autonomous driving simulation.

3     Method
Given a sequence of images captured from a moving vehicle in an urban street
scene, our goal is to develop a model capable of generating photorealistic images
for view synthesis. Towards this objective, we propose a novel scene represen-
tation, named Street Gaussians, specifically designed for representing dynamic
street scenes. As shown in the Figure 2, we represent a dynamic urban scene
as a set of point clouds, each corresponding to either the static background or
a moving vehicle (Section 3.1). The explicit point-based representation allows
easy composition of separate models, enabling real-time rendering as well as
the decomposition of foreground objects for editing applications (Section 3.2).
The proposed scene representation can be effectively trained along with tracked
vehicle poses from an off-the-shelf tracker, enhanced by our pose optimization
strategy (Section 3.3).

3.1   Street Gaussians
In this section, we seek to find a dynamic scene representation that can be
quickly constructed and rendered in real-time. Previous methods [23, 64] typi-
cally face challenges with low training and rendering speed as well as accurate
                                               Street Gaussians for Modeling Dynamic Urban Scenes                                                                                                                    5

                                                Scene representation
                 Geometry model




                                                                                                                                                                            Point-based Rendering
  Position
  Opacity
  Rotation                                                                                                                                                                                          Rendering Images




                                                                                                  Composition
  Scale
                                                Background model
        Dynamic appearance model                                                                                                                                                                     Decomposition

        ......


  Fourier coeﬃcients   Time basis   SH basis
                                                                                    Optimizable
                                                  Object model                     Tracked boxes                                                                                                     Semantic maps



Fig. 2: Overview of Street Gaussians. The dynamic urban street scene is repre-
sented as a set of point-based background and foreground objects with optimizable
tracked vehicle poses. Each point is assigned with a 3D Gaussian [19] including posi-
tion, opacity, and covariance consisting of rotation and scale to represent the geometry.
To represent the appearance, we assign each background point with a spherical har-
monics model while the foreground points are associated with a dynamic spherical
harmonics model. The explicit point-based representation allows easy composition of
separate models, which enables real-time rendering of high-quality images and seman-
tic maps (optional if 2D semantic information is provided during training), as well as
the decomposition of foreground objects for editing applications.



tracked vehicle poses. To tackle this problem, we propose a novel explicit scene
representation, named Street Gaussians, which is built upon 3D Gaussians [19].
In Street Gaussians, we represent the static background and each moving vehicle
object with a separate neural point cloud.
    In the following, we will first focus on the background model, elaborating on
several common attributes that are shared with the object model. Subsequently,
we will delve into the dynamic aspects of the object model’s design.
Background model. The background model is represented as a set of points in
the world coordinate system. Each point is assigned with a 3D Gaussian to softly
represent the continuous scene geometry and color. The Gaussian parameters
consist of a covariance matrix Σb and a position vector µb ∈ R3 , which denotes
the mean value. To avoid invalid value during optimization, each covariance
matrix is further reduced to a scaling matrix Sb and a rotation matrix Rb ,
where Sb is characterized by its diagonal elements and Rb is converted into a
unit quaternion.The covariance matrix Σb can be recovered from Sb and Rb as:

                                                           \label {eq:covariance matrix} \bm {\Sigma }_b = \mathbf {R}_b\mathbf {S}_b{\mathbf {S}_b}^T{\mathbf {R}_b}^T.                                        (1)

Apart from the position and covariance matrix, each Gaussian is also assigned
with an opacity value αb ∈ R and a set of spherical harmonics coefficients zb =
(zm,l )m:−ℓ≤m≤ℓ
       l:0≤ℓ≤ℓmax to represent scene geometry and appearance. To obtain the view-
dependent color, the spherical harmonics coefficients are further multiplied by
the spherical harmonics basis functions projected from the view direction. To
represent 3D semantic information, each point is added with a semantic logit
βb ∈ RM , where M is the number of semantic classes.
6       Y. Yan et al.

Object model. Consider a scene containing N moving foreground object ve-
hicles. Each object is represented with a set of optimizable tracked vehicle poses
and a point cloud, where each point is assigned a 3D Gaussian, semantic logits,
and a dynamic appearance model.
    The Gaussian properties of both the object and the background are similar,
sharing the same meaning for opacity αo and scale matrix So . However, their
position, rotation, and appearance models differ from those of the background
model. The position µo and rotation Ro are defined in the object local coordinate
system. To transform them into the world coordinate system (the background’s
coordinate system), we introduce the definition of tracked poses for objects.
Specifically, the tracked poses of vehicles are defined as a set of rotation matrices
{Rt }N                                    Nt
      t=1 and translation vectors {Tt }t=1 , where Nt represents the number of
        t


frames. The transformation can be defined as:

                                                \label {eq:object gaussian transform} \begin {aligned} \bm {\mu }_{w} &= \mathbf {R}_t \bm {\mu }_o + \mathbf {T}_t, \\ \mathbf {R}_{w} &= \mathbf {R}_t \mathbf {R}_o, \\ \end {aligned} 
                                                                                                                                                                                                                                             (2)

where µw and Rw are the position and rotation of the corresponding object
Gaussian in the world coordinate system, respectively. After transformation, the
object’s covariance matrix Σw can be obtained by Eq. 1 with Rw and So . Note
that we also found the tracked vehicle poses from the off-the-shelf tracker to
be noisy. To address this issue, we treat the tracked vehicle poses as learnable
parameters. We detail it in Section 3.3.
    Simply representing object appearance with the spherical harmonics coeffi-
cients is insufficient for modeling the appearance of moving vehicles, as shown in
Figure 3, because the appearance of a moving vehicle is influenced by its position
in the global scene. One straightforward solution is to use separate spherical har-
monics to represent the object for each timestep. However, this representation
will significantly increase the storage cost. Instead, we introduce the 4D spher-
ical harmonics model by replacing each SH coefficient zm,l with a set of fourier
transform coefficients f ∈ Rk where k is the number of fourier coefficient. Given
timestep t, zm,l is recovered by performing real-valued Inverse Discrete Fourier
Transform:

                               \label {eq:fourier transform} z_{m,l} = \sum _{i=0}^{k-1} \bm {f}_{i} \cos \left (\frac {i\pi }{N_t}t\right ).                                                                                                (3)

With the proposed model, we encode time information into appearance without
high storage cost.
    The semantic representation of the object model is different from that of
background. The main difference is that the semantic of the object model is a
learnable one-dimensional scalar βo which represents the vehicle semantic class
from the trakcer instead of a M -dimensional vector βb .
Initialization. The SfM [46] point cloud used in 3D Gaussian is suitable for
object centric scene. However, it can not provide good initialization for urban
street scenes with many under-observed or textureless regions. We instead use
aggregated LiDAR point cloud captured by ego vehicle as initialization. The
                           Street Gaussians for Modeling Dynamic Urban Scenes                                                                                                                                                        7




                                                                     Input sequence




         Ours with 4D SH                                                  Ours w/o 4D SH                                                                                                                             Ground Truth

Fig. 3: Effect of 4D SH (spherical harmonics) model. The first row presents
the input sequence, showcasing varying appearances. The second row demonstrates
the impact of utilizing the proposed 4D SH model on the rendering results. Significant
artifacts can be observed if the 4D SH model is absent.



colors of LiDAR point cloud are obtained by projecting to the corresponding
image plane and querying the pixel value.
    To initialize the object model, we first collect aggregated points inside the
3D bounding boxes and transform them into the local coordinate system. For
object with less than 2K LiDAR points, we instead randomly sample 8K points
inside the 3D bounding box as initialization. For the background model, we
perform voxel downsampling for the remaining point cloud and filter out the
points which are invisible to the training cameras. We incorporate SfM point
cloud to compensate for the limited coverage of LiDAR over large areas.


3.2   Rendering of Street Gaussians

To render Street Gaussians, we need to aggregate the contribution of each model
to render the final image. Previous methods [23, 39, 64, 71] require compositional
rendering with complex raymarching because of neural field representation. In-
stead, Street Gaussians can be rendered by contacting all the point clouds and
projecting them to 2D image space. Specifically, given a rendered time step t, we
first compute spherical harmonics with Eq. 3 , and transform the object point
cloud into the world coordinate system using Eq. 2 according to tracked vehicle
pose (Rt , Tt ). Then we concatenate the background point cloud and the trans-
formed object point clouds to form a new point cloud. To project this point
cloud to 2D image space with camera extrinsic W and intrinsic K, we compute
the 2D Gaussian for each point in the point cloud [80]:

                                     \begin {aligned} \bm {\mu }' &= \mathbf {K} \mathbf {W} \bm {\mu } , \\ \bm {\Sigma }' &= \mathbf {J} \mathbf {W} \bm {\Sigma } \mathbf {W}^T \mathbf {J}^T , \end {aligned} 
                                                                                                                                                                                                                                    (4)

where J is the Jacobian matrix of K . µ′ and Σ ′ are the position and covariance
matrix in 2D image space, respectively. Point-based α-blending for each pixel is
8      Y. Yan et al.

used to compute the color C:

                                                                                   \label {eq:rendering} \mathbf {C} = \sum _{i \in N} \mathbf {c}_i \alpha _i \prod _{j=1}^{i-1} (1 - \alpha _j),                                                                                                   (5)

Here αi is the opacity α multiplied by the probability of the 2D Gaussian and
ci is the color computed from spherical harmonics z with the view direction. We
can also render other signals like depth, opacity and semantic. For instance, the
semantic map is rendered by changing color c in Eq. 5 to semantic logits β.
    Since 3D Gaussian is defined in Euclidean space, it is inappropriate for them
to model distant regions like sky. As a result, we utilize a high resolution cube-
map which maps the view direction to sky color Csky . The explicit cubemap
representation helps us recover details in sky regions without sacrificing infer-
ence speed. The final rendering color is obtained by blending Csky and the color
C in Eq. 5. More details can be found in the supplementary.

3.3   Training
Tracking Pose Optimization. Positions and covariance matrices of the object
Gaussians during rendering in Section 3.2 are closely correlated with the tracked
pose parameters as shown in Eq. 2. However, bounding boxes produced by the
tracker model are generally noisy. Directly using them to optimize our scene
representation leads to degradation in rendering quality. As a result, we treat
tracked poses as learnable parameters by adding a learnable transformation to
                                                                                ′
each transformation matrix. Specifically, Rt and Tt in Eq. 2 are replaced by Rt
      ′
and Tt which are defined as:
                                                                                                          \label {eq:tracking pose optimization} \begin {aligned} \mathbf {R}_t' &= \mathbf {R}_t \Delta \mathbf {R}_t, \\ \mathbf {T}_t' &= \mathbf {T}_t + \Delta \mathbf {T}_t, \end {aligned} 
                                                                                                                                                                                                                                                                                                     (6)

where ∆Rt and ∆Tt are the learnable transformation. We represent ∆Tt as a
3D vector and ∆Ri as a rotation matrix converted from yaw offset angle ∆θt .
Gradients of these transformations can be directly obtained without any implicit
function or intermediate processes, which do not require any extra computation
during back-propagation.
Loss function. We jointly optimize our scene representation, sky cubemap and
tracked poses using the following loss function:

                \label {eq:loss function} \begin {aligned} \mathcal {L} = \mathcal {L}_{\text {color}} + \lambda _1 \mathcal {L}_{\text {depth}} + \lambda _2 \mathcal {L}_{\text {sky}} + \lambda _3 \mathcal {L}_{\text {sem}} + \lambda _4 \mathcal {L}_{\text {reg}}. \end {aligned}             (7)
    In Eq. 9, Lcolor is the reconstruction loss between rendered and observed
images following [19]. Ldepth is a L1 loss between rendered depth and the depth
generated by projecting sparse LiDAR points onto the camera plane. Lsky is
a binary cross entropy loss for sky supervision. Lsem is an optional per-pixel
softmax-cross-entropy loss between rendered semantic logits and input 2D se-
mantic segmentation predictions [24] and Lreg is an regularization term used to
remove floaters and enhance decomposition effects. Please refer to the supple-
mentary material for details of each loss term.
                      Street Gaussians for Modeling Dynamic Urban Scenes        9

4     Implementation details

We train Street Gaussians for 30000 iterations with Adam optimizers [20] fol-
lowing the configurations of 3D Gaussians [19]. The learning rate of translation
transformation ∆Tt and rotation transformation ∆Rt are set to 5e−3 and 1e−3 ,
which decay exponentially to 5e−5 and 1e−5 respectively. The resolution of sky
cubemap is set to 1024 with learning rate decays from 1e−2 to 1e−4 exponen-
tially. All the experiments are conducted on one single RTX 4090 GPU.
    We follow [19] to apply adaptive control during optimization. We fix the scale
of background model (20 meters in our experiments) and the scale of each object
model is determined by the bounding box dimensions. In order to prevent object
Gaussians from growing to occluded areas, for each object model we sample a set
of points as a probability distribution function. During optimization, Gaussians
with sampled points outside the bounding box will be pruned.


5     Experiments

5.1   Experimental Setup

Datasets. We conduct experiments on Waymo Open Dataset [51] and KITTI
benchmarks [15]. The frame rates of both datasets are 10 HZ. On the Waymo
Open Dataset, we select 8 recording sequences with large amounts of moving
objects, significant ego-car motion and complex lighting conditions. All sequences
have a length of around 100 frames. We select every 4th image in the sequence as
the test frames and use the remaining for training. As we find that our baseline
methods [39,64] suffer from high memory cost when training with high-resolution
images, we downscale the input images to 1066×1600. On KITTI [15] and Vitural
KITTI 2 [6], we follow the settings of MARS [64] and evaluate our methods with
different train/test split settings. We use the bounding boxes generated by the
detector [62] and tracker [63] on Waymo dataset and use the officially provided
object tracklets from KITTI.
Baseline methods. We compare our methods with four recent methods.
(1) NSG [39] represents background as multi-plane images and use per-object
learned latent codes with a shared decoder to model moving objects. (2) MARS
[64] builds the neural scene graph based on Nerfstudio [53]. (3) 3D Gaussians [19]
models the scene with a set of anisotropy gaussians. (4) EmerNeRF [69] stratifies
scenes into static and dynamic fields, each modeled with a hash grid [36]. Both
NSG and MARS are trained and evaluated using ground truth object tracklets.
Details of baseline implementations can be found in the supplementary.


5.2   Comparisons with the State-of-the-art

Tables 1, 2 present the comparison results of our method with baseline methods
[19,39,64,69] in terms of rendering quality and rendering speed. We adopt PSNR,
SSIM and LPIPS [77] as metrics to evaluate rendering quality. To better evaluate
 10            Y. Yan et al.




Ours




3D GS




NSG




MARS




EmerNeRF




Ground Truth



 Fig. 4: Qualitative comparisons results on the Waymo [51] dataset. NSG [39]
 and MARS [64] often produce blurry and distorted results. 3D GS [19] and EmerN-
 eRF [69] generates ghosting artifacts in regions with moving objects. In contrast, our
 approach significantly outperforms other methods with high fidelity and sharp details.


 Table 1: Quantitative results on the Waymo [51] dataset. The rendering image
 resolution is 1066 × 1600. “PSNR*” denotes the PSNR of moving objects.

                          3D GS [19] NSG [39] MARS [64] EmerNeRF [69] Ours
                 PSNR↑         29.64   28.31   29.75       30.87      34.61
                 SSIM↑         0.918   0.862   0.886       0.905      0.938
                 LPIPS↓        0.117   0.346   0.264       0.133      0.079
                 PSNR*↑        21.25   24.32   26.54       21.67      30.23
                 FPS↑          205     0.47    0.68         0.21       135
                       Street Gaussians for Modeling Dynamic Urban Scenes          11

Table 2: Quantitative results on KITTI [15] and VKITTI2 [6] datasets. We
strictly follow the experimental setting of MARS [64] and borrow results of MARS [64]
and NSG [39] from it. The rendering image resolution is 375 × 1242.

                      KITTI - 75%         KITTI - 50%           KITTI - 25%
                  PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓
      3D GS [19] 19.19   0.737 0.172   19.23   0.739 0.174   19.06   0.730 0.180
      NSG* [39]  21.53   0.673 0.254   21.26   0.659 0.266   20.00   0.632 0.281
      MARS* [64] 24.23   0.845 0.160   24.00   0.801 0.164   23.23   0.756 0.177
      Ours       25.79   0.844 0.081   25.52   0.841 0.084   24.53   0.824 0.090
                     VKITTI2 - 75%       VKITTI2 - 50%         VKITTI2 - 25%
                  PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓
      3D GS [19] 21.12   0.877 0.097   21.11   0.874 0.097   20.84   0.863 0.098
      NSG* [39]  23.41   0.689 0.317   23.23   0.679 0.325   21.29   0.666 0.317
      MARS* [64] 29.79   0.917 0.088   29.63   0.916 0.087   27.01   0.887 0.104
      Ours       30.10   0.935 0.025   29.91   0.932 0.026   28.52   0.917 0.034



the rendering quality of moving objects, we project 3D bounding boxes to 2D
image plane and calculate the loss only on pixels inside the projected box, which
is denoted as PSNR* in our experiments. For all the metrics, our model achieves
the best performance among all the methods with a 12.1% increase in PSNR
and a 13.9% increase in PSNR*. Moreover, our method renders two magnitudes
faster than NeRF-based methods [39, 64, 69]. Although 3D GS is faster than our
method, it can only support static scenes and the rendering result of moving
objects degrades significantly.
    Figure 4 shows the qualitative results of our method and baselines on the
Waymo dataset. 3D GS fails to model dynamic objects and EmerNeRF can
not generates reasonable results in dynamic regions of novel timestep. Although
given ground truth tracking poses, NSG and MARS still suffer from blurry and
distorted results due to the lack of capacity of their model when the scene is
complex. In contrast, our method can generate high-quality novel views with
high fidelity and details.


Table 3: Ablation studies on the Waymo [51] dataset. Metrics are averaged
over all the sequences on the Waymo dataset. “PSNR*” denotes the PSNR of moving
objects. “opt.” denotes optimization. Please refer to Section 5.3 for details.

                                     PSNR↑ PSNR*↑ SSIM↑ LPIPS↓
                Ours w/o LiDAR     34.02       29.53   0.934 0.087
                Ours w/o 4DSH      34.36       29.27   0.937 0.081
                Ours w/o pose opt. 34.18       28.24   0.935 0.081
                Ours w/ GT pose 34.61          29.84   0.937 0.080
                Complete model     34.61       30.23   0.938 0.079
12      Y. Yan et al.




        Ours            Ours w/o pose opt.   Ours w/ GT poses         Ground Truth

Fig. 5: Ablation study on tracking pose optimization. The results indicate that
optimizing tracked poses improves the quality. “opt.” denotes optimization.




               Ours w/ LiDAR                              Ours w/o LiDAR

Fig. 6: Ablation study on LiDAR point cloud. We show the rendered image and
depth of our method with and without LiDAR as input.


5.3   Ablations and Analysis
We validate our algorithm’s design choices on all selected sequences from the
Waymo dataset. Table 3 presents the quantitative results.
Importance of optimizing tracked poses. Experimental results in Table 3
show that our complete model outperforms the model trained without tracking
pose optimization by a large margin, which indicates the effectiveness of our pose
optimization strategy. It is interesting to notice that the result of our method is
even better than the model trained with ground truth poses, a plausible expla-
nation is that there still exists noise in ground truth annotations.
    Visual results of the influence of tracked pose optimization is shown in Figure
5. Treating tracked poses as learnable parameters help the object model synthe-
size more texture details like the rear of the white vehicle or the logo of the black
vehicle and reduce rendering artifacts.
Effectiveness of 4D spherical harmonics. Results in Table 3 indicate that
our 4D spherical harmonics appearance model can refine the rendering qual-
ity. This situation becomes particularly evident when the object interacts with
environmental lighting as shown in Figure 3. Our model can generate smooth
shadows on the car while the rendering results without 4D spherical harmonics
are much noisier.
Influence of incorporating LiDAR points. We evaluate the influence of
LiDAR point cloud by comparing our method to a variant with SfM initializa-
tion for background and random initialization for moving object as described in
Section 3.1. We also disable the LiDAR depth loss in Eq. 9. Table 3 shows that
                         Street Gaussians for Modeling Dynamic Urban Scenes                  13




        (a) Rotation                   (b) Translation                   (c) Swapping

Fig. 7: Editing operations on the Waymo [51] dataset. Images in the first and
second rows represent the results before and after editing. Our method supports various
editing operations, including rotation, translation and swapping.




        Ours                    MARS                      NSG              Reference Image

Fig. 8: Decomposition results on the Waymo [51] dataset. NSG [39] cannot
decompose clean foreground objects while MARS [64] generates floaters in background
regions. Instead, our method successfully decomposes the foreground objects and pro-
duces high fidelity rendering results.


Table 4: Quantitative segmentation results on the KITTI [15] dataset. “VKN
ground-truth” and “VKN rendered” denote semantic prediction results of Video K-Net
with ground-truth images and our rendered images, respectively.

                       Method VKN ground-truth VKN rendered Ours
                       mIoU ↑      57.94                 53.81   58.81




incorporating LiDAR point cloud enhances the results of both background and
moving objects. Figure 6 indicates that using LiDAR points helps our model
recover more accurate scene geometry and reduce blurry artifacts. It is worth
noticing that our method still significantly outperforms baseline methods even
without LiDAR input as shown in Tables 3, 4, which proves the efficiency of
Street Gaussians under different settings.
14        Y. Yan et al.




      Reference image       Ours              Video K-Net         Ground Truth

Fig. 9: Visual semantic segmentation results on the KITTI [15] dataset. It can
be observed that our method achieves better performance, particularly in ambiguous
areas such as shadows, due to our ability to fuse semantic information in 3D.



5.4     Applications

Street Gaussians can be applied to multiple tasks in computer vision including
object decomposition, semantic segmentation and scene editing.
Scene editing. Our instance-aware scene representation enables various types
of scene editing operations. We can rotate the heading of the vehicle (Figure 7
(a)), translate the vehicle (Figure 7 (b)) and swap one vehicle in the scene with
another one (Figure 7 (c)).
Object Decomposition. We compare the decomposition results of our method
with NSG [39] and MARS [64] under the Waymo dataset. As shown in Figure
8, NSG fails to disentangle foreground objects from the background and the
result of MARS is blurry due to the model capacity and lack of regularization.
In contrast, our method can produce high-fidelity and clean decomposed results.
Semantic Segmentation. We compare the quality of our rendered semantic
map with the semantic prediction from Video-K-Net [24] on KITTI dataset.
Our semantic segmentation model is trained with results from Video K-Net.
Qualitative and quantitative results are shown in Figure 9 and Table 4. Our
semantic maps achieve better performance thanks to our representation.


6      Conclusion

This paper introduced Street Gaussians, an explicit scene representation for
modeling dynamic urban street scenes. The proposed representation separately
models the background and foreground vehicles as a set of neural point clouds.
This explicit representation allows easy compositing of object vehicles and back-
ground, enabling scene editing and real-time rendering within half an hour of
training. Furthermore, we demonstrate that the proposed scene representation
can achieve comparable performance to that achieved using precise ground-truth
poses, using only poses from an off-the-shelf tracker. Detailed ablation and com-
parison experiments are conducted on several datasets, demonstrating the effec-
tiveness of the proposed method.
Acknowledgement. The authors would like to acknowledge the support from
NSFC (No. 623B2091), Li Auto and Information Technology Center and State
Key Lab of CAD&CG, Zhejiang University.
                        Street Gaussians for Modeling Dynamic Urban Scenes            15

References
 1. Aliev, K.A., Sevastopolsky, A., Kolos, M., Ulyanov, D., Lempitsky, V.: Neural
    point-based graphics. In: ECCV (2020) 3
 2. Attal, B., Huang, J.B., Richardt, C., Zollhoefer, M., Kopf, J., O’Toole, M., Kim,
    C.: HyperReel: High-fidelity 6-DoF video with ray-conditioned sampling. In: CVPR
    (2023) 3
 3. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srini-
    vasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance
    fields. In: ICCV (2021) 3
 4. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf
    360: Unbounded anti-aliased neural radiance fields. In: CVPR (2022) 3
 5. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Zip-nerf:
    Anti-aliased grid-based neural radiance fields. In: ICCV (2023) 3
 6. Cabon, Y., Murray, N., Humenberger, M.: Virtual kitti 2. arXiv preprint
    arXiv:2001.10773 (2020) 9, 11
 7. Chen, Y., Rong, F., Duggal, S., Wang, S., Yan, X., Manivasagam, S., Xue, S.,
    Yumer, E., Urtasun, R.: Geosim: Realistic video simulation via geometry-aware
    composition for self-driving. In: CVPR (2021) 4
 8. Chen, Y., Gu, C., Jiang, J., Zhu, X., Zhang, L.: Periodic vibration gaussian: Dy-
    namic urban scene reconstruction and real-time rendering. arXiv:2311.18561 (2023)
    4
 9. Cheng, K., Long, X., Yin, W., Wang, J., Wu, Z., Ma, Y., Wang, K., Chen, X.,
    Chen, X.: Uc-nerf: Neural radiance field for under-calibrated multi-view cameras.
    In: ICLR (2024) 3
10. Dai, P., Zhang, Y., Li, Z., Liu, S., Zeng, B.: Neural point cloud rendering via
    multi-plane projection. In: CVPR (2020) 3
11. Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., Koltun, V.: Carla: An open
    urban driving simulator. In: CoRL (2017) 4
12. Fang, J., Zhou, D., Yan, F., Zhao, T., Zhang, F., Ma, Y., Wang, L., Yang, R.: Aug-
    mented lidar simulator for autonomous driving. IEEE Robotics and Automation
    Letters 5(2), 1931–1938 (2020) 4
13. Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A.: K-planes:
    Explicit radiance fields in space, time, and appearance. In: CVPR (2023) 3
14. Fu, X., Zhang, S., Chen, T., Lu, Y., Zhu, L., Zhou, X., Geiger, A., Liao, Y.:
    Panoptic nerf: 3d-to-2d label transfer for panoptic urban scene segmentation. In:
    3DV (2022) 4
15. Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? the kitti
    vision benchmark suite. In: CVPR (2012) 3, 9, 11, 13, 14, 23, 25, 26
16. Guo, J., Deng, N., Li, X., Bai, Y., Shi, B., Wang, C., Ding, C., Wang, D., Li, Y.:
    Streetsurf: Extending multi-view implicit surface reconstruction to street views.
    arXiv preprint arXiv:2306.04988 (2023) 3
17. Huang, S., Gojcic, Z., Wang, Z., Williams, F., Kasten, Y., Fidler, S., Schindler, K.,
    Litany, O.: Neural lidar fields for novel view synthesis. In: ICCV (2023) 4
18. Irshad, M.Z., Zakharov, S., Liu, K., Guizilini, V., Kollar, T., Gaidon, A., Kira, Z.,
    Ambrus, R.: Neo 360: Neural fields for sparse view synthesis of outdoor scenes. In:
    ICCV (2023) 3
19. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
    real-time radiance field rendering. TOG 42(4) (July 2023) 2, 3, 5, 8, 9, 10, 11, 20,
    21
16      Y. Yan et al.

20. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint
    arXiv:1412.6980 (2014) 9
21. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
    Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything.
    In: ICCV (2023) 20
22. Kopanas, G., Philip, J., Leimkühler, T., Drettakis, G.: Point-based neural render-
    ing with per-view optimization. In: CGF. vol. 40, pp. 29–43. Wiley Online Library
    (2021) 3
23. Kundu, A., Genova, K., Yin, X., Fathi, A., Pantofaru, C., Guibas, L., Tagliasacchi,
    A., Dellaert, F., Funkhouser, T.: Panoptic Neural Fields: A Semantic Object-Aware
    Neural Scene Representation. In: CVPR (2022) 2, 3, 4, 7, 23
24. Li, X., Zhang, W., Pang, J., Chen, K., Cheng, G., Tong, Y., Loy, C.C.: Video k-net:
    A simple, strong, and unified baseline for video segmentation. In: CVPR (2022) 8,
    14, 20
25. Li, Y., Lin, Z.H., Forsyth, D., Huang, J.B., Wang, S.: Climatenerf: Physically-based
    neural rendering for extreme climate synthesis. In: ICCV (2023) 4
26. Li, Z., Niklaus, S., Snavely, N., Wang, O.: Neural scene flow fields for space-time
    view synthesis of dynamic scenes. In: CVPR (2021) 3
27. Li, Z., Li, L., Zhu, J.: Read: Large-scale neural scene rendering for autonomous
    driving. In: AAAI (2023) 3
28. Lin, H., Peng, S., Xu, Z., Xie, T., He, X., Bao, H., Zhou, X.: High-fidelity and
    real-time novel view synthesis for dynamic scenes. In: SIGGRAPH (2023) 3
29. Lin, H., Peng, S., Xu, Z., Yan, Y., Shuai, Q., Bao, H., Zhou, X.: Efficient neural
    radiance fields for interactive free-viewpoint video. In: SIGGRAPH (2022) 3
30. Liu, J.Y., Chen, Y., Yang, Z., Wang, J., Manivasagam, S., Urtasun, R.: Neural
    scene rasterization for large scene rendering in real time. In: ICCV (2023) 3
31. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu,
    J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set
    object detection. In: ECCV (2024) 20
32. Lu, F., Xu, Y., Chen, G., Li, H., Lin, K.Y., Jiang, C.: Urban radiance field repre-
    sentation with deformable neural mesh primitives. In: ICCV (2023) 1, 3
33. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking
    by persistent dynamic view synthesis. In: 3DV (2024) 4
34. Manivasagam, S., Wang, S., Wong, K., Zeng, W., Sazanovich, M., Tan, S., Yang,
    B., Ma, W.C., Urtasun, R.: Lidarsim: Realistic lidar simulation by leveraging the
    real world. In: CVPR (2020) 4
35. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
    R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV
    (2020) 2, 3
36. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives
    with a multiresolution hash encoding. In: SIGGRAPH (2022) 3, 9
37. Niemeyer, M., Geiger, A.: Giraffe: Representing scenes as compositional generative
    neural feature fields. In: CVPR (2021) 4
38. Ost, J., Laradji, I., Newell, A., Bahat, Y., Heide, F.: Neural point light fields. In:
    CVPR (2022) 1, 3
39. Ost, J., Mannan, F., Thuerey, N., Knodt, J., Heide, F.: Neural scene graphs for
    dynamic scenes. In: CVPR (2021) 2, 3, 4, 7, 9, 10, 11, 13, 14, 21, 23
40. Park, K., Sinha, U., Hedman, P., Barron, J.T., Bouaziz, S., Goldman, D.B., Martin-
    Brualla, R., Seitz, S.M.: Hypernerf: A higher-dimensional representation for topo-
    logically varying neural radiance fields. TOG 40(6) (dec 2021) 3
                       Street Gaussians for Modeling Dynamic Urban Scenes            17

41. Peng, S., Yan, Y., Shuai, Q., Bao, H., Zhou, X.: Representing volumetric videos as
    dynamic mlp maps. In: CVPR (2023) 3
42. Pun, A., Sun, G., Wang, J., Chen, Y., Yang, Z., Manivasagam, S., Ma, W.C.,
    Urtasun, R.: Neural lighting simulation for urban scenes. In: NeurIPS (2023) 4
43. Rematas, K., Liu, A., Srinivasan, P.P., Barron, J.T., Tagliasacchi, A., Funkhouser,
    T., Ferrari, V.: Urban radiance fields. In: CVPR (2022) 1, 3
44. Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen,
    Y., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., Zhang, L.:
    Grounded sam: Assembling open-world models for diverse visual tasks (2024) 20,
    21
45. Rückert, D., Franke, L., Stamminger, M.: Adop: Approximate differentiable one-
    pixel point rendering. TOG 41(4), 1–14 (2022) 3
46. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: CVPR (2016)
    6, 21
47. Shah, S., Dey, D., Lovett, C., Kapoor, A.: Airsim: High-fidelity visual and physical
    simulation for autonomous vehicles. In: Field and Service Robotics: Results of the
    11th International Conference. pp. 621–635. Springer (2018) 4
48. Shen, B., Yan, X., Qi, C.R., Najibi, M., Deng, B., Guibas, L., Zhou, Y., Anguelov,
    D.: Gina-3d: Learning to generate implicit neural assets in the wild. In: CVPR
    (2023) 4
49. Siddiqui, Y., Porzi, L., Bulò, S.R., Müller, N., Nießner, M., Dai, A., Kontschieder,
    P.: Panoptic lifting for 3d scene understanding with neural fields. In: CVPR (2023)
    20
50. Song, L., Chen, A., Li, Z., Chen, Z., Chen, L., Yuan, J., Xu, Y., Geiger, A.: Nerf-
    player: A streamable dynamic scene representation with decomposed neural radi-
    ance fields. TVCG 29(5), 2732–2742 (2023) 3
51. Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo,
    J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W., Ngiam, J., Zhao, H.,
    Timofeev, A., Ettinger, S., Krivokon, M., Gao, A., Joshi, A., Zhang, Y., Shlens, J.,
    Chen, Z., Anguelov, D.: Scalability in perception for autonomous driving: Waymo
    open dataset. In: CVPR (2020) 2, 3, 9, 10, 11, 13
52. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Bar-
    ron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis.
    In: CVPR (2022) 1, 2, 3
53. Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Kerr, J., Wang, T., Kristoffersen,
    A., Austin, J., Salahi, K., Ahuja, A., McAllister, D., Kanazawa, A.: Nerfstudio: A
    modular framework for neural radiance field development. In: SIGGRAPH 2023
    Conference Proceedings (2023) 9, 21
54. Tonderski, A., Lindström, C., Hess, G., Ljungbergh, W., Svensson, L., Petersson,
    C.: Neurad: Neural rendering for autonomous driving. In: CVPR (2024) 3
55. Turki, H., Ramanan, D., Satyanarayanan, M.: Mega-nerf: Scalable construction of
    large-scale nerfs for virtual fly-throughs. In: CVPR (2022) 3
56. Turki, H., Zhang, J.Y., Ferroni, F., Ramanan, D.: Suds: Scalable urban dynamic
    scenes. In: CVPR (2023) 3
57. Wang, J., Manivasagam, S., Chen, Y., Yang, Z., Bârsan, I.A., Yang, A.J., Ma,
    W.C., Urtasun, R.: Cadsim: Robust and scalable in-the-wild 3d reconstruction for
    controllable sensor simulation. In: CoRL (2022) 4
58. Wang, Z., Chen, W., Acuna, D., Kautz, J., Fidler, S.: Neural light field estimation
    for street scenes with differentiable virtual object insertion. In: ECCV (2022) 4
18      Y. Yan et al.

59. Wang, Z., Shen, T., Gao, J., Huang, S., Munkberg, J., Hasselgren, J., Gojcic,
    Z., Chen, W., Fidler, S.: Neural fields meet explicit geometric representations for
    inverse rendering of urban scenes. In: CVPR (2023) 4
60. Wei, Y., Wang, Z., Lu, Y., Xu, C., Liu, C., Zhao, H., Chen, S., Wang, Y.: Editable
    scene simulation for autonomous driving via collaborative llm-agents. In: CVPR
    (2024) 4
61. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Xinggang,
    W.: 4d gaussian splatting for real-time dynamic scene rendering. In: CVPR (2024)
    4
62. Wu, H., Deng, J., Wen, C., Li, X., Wang, C.: Casa: A cascade attention network
    for 3d object detection from lidar point clouds. IEEE Transactions on Geoscience
    and Remote Sensing (2022) 9
63. Wu, H., Han, W., Wen, C., Li, X., Wang, C.: 3d multi-object tracking in point
    clouds based on prediction confidence-guided data association. IEEE Transactions
    on Intelligent Transportation Systems 23(6), 5668–5677 (2021) 9
64. Wu, Z., Liu, T., Luo, L., Zhong, Z., Chen, J., Xiao, H., Hou, C., Lou, H., Chen,
    Y., Yang, R., Huang, Y., Ye, X., Yan, Z., Shi, Y., Liao, Y., Zhao, H.: Mars: An
    instance-aware, modular and realistic simulator for autonomous driving. In: CICAI
    (2023) 2, 3, 4, 7, 9, 10, 11, 13, 14, 21, 22, 23
65. Xie, Z., Zhang, J., Li, W., Zhang, F., Zhang, L.: S-nerf: Neural radiance fields for
    street views. In: ICLR (2023) 3
66. Xu, L., Xiangli, Y., Peng, S., Pan, X., Zhao, N., Theobalt, C., Dai, B., Lin, D.:
    Grid-guided neural radiance fields for large urban scenes. In: CVPR (2023) 3
67. Xu, Y., Chai, M., Shi, Z., Peng, S., Skorokhodov, I., Siarohin, A., Yang, C., Shen,
    Y., Lee, H.Y., Zhou, B., et al.: Discoscene: Spatially disentangled generative radi-
    ance fields for controllable 3d-aware scene synthesis. In: CVPR (2023) 4
68. Xu, Z., Peng, S., Lin, H., He, G., Sun, J., Shen, Y., Bao, H., Zhou, X.: 4k4d:
    Real-time 4d view synthesis at 4k resolution. In: CVPR (2024) 4
69. Yang, J., Ivanovic, B., Litany, O., Weng, X., Kim, S.W., Li, B., Che, T., Xu,
    D., Fidler, S., Pavone, M., Wang, Y.: Emernerf: Emergent spatial-temporal scene
    decomposition via self-supervision. In: ICLR (2024) 2, 3, 9, 10, 11, 21
70. Yang, Y., Yang, Y., Guo, H., Xiong, R., Wang, Y., Liao, Y.: Urbangiraffe: Repre-
    senting urban scenes as compositional generative neural feature fields. In: ICCV
    (2023) 4
71. Yang, Z., Chen, Y., Wang, J., Manivasagam, S., Ma, W.C., Yang, A.J., Urtasun,
    R.: Unisim: A neural closed-loop sensor simulator. In: CVPR (2023) 2, 3, 4, 7, 20
72. Yang, Z., Manivasagam, S., Chen, Y., Wang, J., Hu, R., Urtasun, R.: Reconstruct-
    ing objects in-the-wild for realistic sensor simulation. In: ICRA (2023) 4
73. Yang, Z., Yang, H., Pan, Z., Zhu, X., Zhang, L.: Real-time photorealistic dynamic
    scene representation and rendering with 4d gaussian splatting. In: ICLR (2024) 4,
    24
74. Yang, Z., Chai, Y., Anguelov, D., Zhou, Y., Sun, P., Erhan, D., Rafferty, S., Kret-
    zschmar, H.: Surfelgan: Synthesizing realistic sensor data for autonomous driving.
    In: CVPR (2020) 4
75. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians
    for high-fidelity monocular dynamic scene reconstruction. In: CVPR (2024) 4
76. Zhang, Q., Baek, S.H., Rusinkiewicz, S., Heide, F.: Differentiable point-based ra-
    diance fields for efficient view synthesis. In: SIGGRAPH. pp. 1–12 (2022) 4
77. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable
    effectiveness of deep features as a perceptual metric. In: CVPR (2018) 9
                       Street Gaussians for Modeling Dynamic Urban Scenes           19

78. Zhang, X., Kundu, A., Funkhouser, T., Guibas, L., Su, H., Genova, K.: Nerflets:
    Local radiance fields for efficient structure-aware 3d scene representation from 2d
    supervision. In: CVPR (2023) 1, 4
79. Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun, D., Yang, M.H.: Drivinggaussian:
    Composite gaussian splatting for surrounding dynamic autonomous driving scenes.
    arXiv preprint arXiv:2312.07920 (2023) 4
80. Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: Ewa volume splatting. In: Pro-
    ceedings Visualization, 2001. VIS’01. pp. 29–538. IEEE (2001) 7
20                Y. Yan et al.

A           More implementation details
A.1            Street Gaussians implementations.
Point cloud initialization. We obtain SfM point cloud of background model
by treating camera poses as known parameters and perform point triangulation.
As moving objects violate the assumption of multi-view consistency, we ignore
these parts by using the mask as shown in Figure 10 during feature extraction.
We can directly concatenate SfM and LiDAR point cloud as they are both defined
in the world coordinate system.
Object semantic. To merge the one-dimensional scalar βo with the M-dimensional
vector βb of background, we convert βo to a M-dimensional one-hot vector for
the vehicle label during rendering.
Sky cubemap. The sky cubemap takes viewing direction d as input and output
sky color Csky . Let the rendered color and opacity of Gaussians as Cg and Og ,
the final rendering color C can be written as:

                                                                                                    \label {eq:blender color} \begin {aligned} \mathbf {C} = \mathbf {C}_g + (1 - \mathbf {O}_g) * \mathbf {C}_\text {sky}. \end {aligned}                                                                         (8)

Loss functions. As we discussed in the main paper, our total loss function is:

                                      \label {eq:loss function} \begin {aligned} \mathcal {L} = \mathcal {L}_{\text {color}} + \lambda _1 \mathcal {L}_{\text {depth}} + \lambda _2 \mathcal {L}_{\text {sky}} + \lambda _3 \mathcal {L}_{\text {sem}} + \lambda _4 \mathcal {L}_{\text {reg}}. \end {aligned}     (9)
1. Lcolor . We apply the L1 and D-SSIM loss between rendered and observed
   images:
                       \label {eq:color loss} \begin {aligned} \mathcal {L}_{\text {color}} = (1 - \lambda _{\text {SSIM}}) \mathcal {L}_1 + \lambda _{\text {SSIM}} \mathcal {L}_{\text {D-SSIM}}. \end {aligned}  (10)
   We set λSSIM to 0.2 following [19].
2. Ldepth . We apply the L1 loss between rendered depth D and the LiDAR
   measurement’s depth Dlidar :

                                                                                                              \label {eq:depth loss} \begin {aligned} \mathcal {L}_{\text {depth}} = \sum || \mathbf {D} - \mathbf {D}^{\text {lidar}} ||_1 \end {aligned}                                                        (11)

   We optimize 95% of the pixels with smallest depth error to prevent noisy
   LiDAR observations from affecting the optimization [71]. λ1 is set to 0.01.
3. Lsky . We apply the binary cross entropy loss between rendered opacity Og
   and predicted sky mask Msky :

                                               \label {eq:sky loss} \begin {aligned} \mathcal {L}_{\text {sky}} = -\sum ((1 - \mathbf {M}_{\text {sky}}) \text {log} \mathbf {O}_g + \mathbf {M}_{\text {sky}} \text {log}(1 - \mathbf {O}_g)) \end {aligned}                                                     (12)

   Msky is generated by Grounded SAM [44]. To be specific, we first get 2D
   boxes by entering text "sky" to Grounding Dino [31]. Then we input the
   boxes as prompt to SAM [21] and obtain the predicted sky mask. λ2 is set
   to 0.05.
4. Lsem . We apply the per-pixel softmax-cross-entropy loss between rendered
   semantic logits and predicted 2D semantic segmentation [24]. In order to
   prevent noisy input semantic labels from influencing the scene geometry [49],
   we only perform backpropagation to semantic logits β for Lsem . λ3 is set to
   0.1.
                                            Street Gaussians for Modeling Dynamic Urban Scenes                                                                                                                                                               21

5. Lreg . The regularization term in our loss function is defined as an entropy loss
   on the accumulated alpha values of decomposed foreground objects Oobj :

                 \label {eq:reg loss} \begin {aligned} \mathcal {L}_{\text {reg}} = -\sum (\mathbf {O}_{\text {obj}} \text {log} \mathbf {O}_{\text {obj}} + (1 - \mathbf {O}_{\text {obj}}) \text {log} (1 - \mathbf {O}_{\text {obj}})) \end {aligned}    (13)

   We add this loss after the adaptive control process to help our model better
   distinguishes foreground and background. Figure 11 shows the qualitative
   results, which demonstrates the effect of this regularization term. λ4 is set
   to 0.1.

Hyperparameters. In practice, we set the number of fourier coefficients k
as 5 to maintain a balance between performance and storage cost. Due to the
relatively less intense view-dependent effect on urban scene compared to dataset
in 3D Gaussian [19], we reduce the SH degree to 1 to prevent overfitting. We set
the voxel size to 0.15m when performing downsampling for LiDAR point cloud.


A.2   Baselines implementations.

We give detailed descriptions of our baseline implementations.

1. Neural Scene Graph [39]. We use the official implementation and try a variant
   where each moving object is modeled by a separate NeRF network instead
   of a shared decoder. The best result is reported for each scene.
2. Mars [64]. We use the official implementation and try a variant where each
   moving object is modeled by a separate Nerfacto [53] model. We choose the
   appearance embedding of nearest training frame as input to each test frame.
   The best result is reported for each scene.
3. 3D Gaussians [19]. We initialize the point cloud by running Colmap [46]
   with known camera parameters from the dataset. We find that the number
   of SfM point cloud generated by Colmap is usually less than 1K when the
   ego-vehicle has little motion. We use LiDAR points to initialize 3D Gaussians
   for these cases to get reasonable results.
4. EmerNeRF [69]. We run the official code under the setting incorporating
   dynamic encoder, flow encoder and feature lifting. We use the same sky
   mask obtained from Grounded SAM [44] for fair comparison.


A.3   Evaluations

Figure 10 visually illustrates the calculation method of the PSNR* metric in our
experiments. We expand each bounding box by 1.5 times in both length and
width dimensions to ensure it fully covers the object. For fair comparison, both
our method and the baselines are evaluated using the mask obtained from object
tracklets provided by the dataset.
22      Y. Yan et al.




                        Image                    Tracked boxes mask

Fig. 10: Illustration of PSNR*. We project the 3D tracked boxes to 2D image plane
and obtain the mask above. We calculate the MSE (Mean Squared Error) for the pixels
within the mask to get the value of PSNR*.




                Ours with reg loss                Ours w/o reg loss

Fig. 11: Effect of regularization loss on decomposition results. “reg loss” de-
notes regularization loss. Adding this term can significantly remove ghosty artifacts
around the vehicle.



B    Additional experiments

Analysis of optimizing tracked poses. As discussed in our main paper, we
observe that our explicit representation facilitates the optimization of tracked
vehicle poses with ease. Herein, we extend our study to explore the impact of an
implicit representation [64] on optimizing tracked vehicle poses. The experimen-
tal results, as presented in Table 5, indicate that while the inclusion of our pose
optimization strategy with implicit representation improves outcomes, there re-
mains a noticeable gap compared to experiments using ground truth tracked
poses. However, the proposed method, employing tracked poses from an off-the-
shelf tracker, achieves results comparable to those using GT poses. This success
can be attributed to the more efficient propagation of gradients through explicit
representations in relation to tracked poses.
Analysis of point cloud initialization. We have demonstrated the impor-
tance of including LiDAR point cloud during initialization. However, the LiDAR
points can not cover the entire scene, especially for far away regions. As a result,
the SfM point cloud is also crucial for the reconstruction of dynamic urban scene.
We perform an ablation study by using only LiDAR point cloud to initialize the
background model. As shown in Figure 12, although LiDAR points can help
recover texture details in near regions like the road, it cannot restore some areas
                      Street Gaussians for Modeling Dynamic Urban Scenes      23

Table 5: More ablation studies on tracking pose optimization. We report the
results of PSNR* on two scenes from Waymo dataset. “opt.” denotes optimization.

                                           Sequence A
                            w/o pose opt. with pose opt. with GT poses
                MARS [64]       25.78         27.68         29.74
                Ours            29.08         31.35         30.84
                                           Sequence B
                            w/o pose opt. with pose opt. with GT poses
                MARS [64]       24.38         25.83         26.94
                Ours            25.86         27.98         28.02



not covered by LiDAR points, such as trees on the other side of the road or
distant road signs. The rendering results without SfM points is even worse than
the one without LiDAR points as illustrated in Table 6. Our approach combines
them as input to leverage their respective strengths.
Analysis of sky modeling. As shown in Figure 13 and Table 6, using a
separate cube map to model the sky can help better recover detail areas, while
avoiding some foreground objects being obscured by the gaussians representing
sky regions.



Table 6: More ablation studies on point cloud initialization and sky model-
ing. We show the quantitative results on two scenes from Waymo dataset with large
scale background and many thin structures.

                                            PSNR↑ SSIM↑ LPIPS↓
                  Ours                  32.63         0.928 0.083
                  Ours w/o LiDAR        30.72         0.920 0.100
                  Ours w/o SfM          29.97         0.911 0.106
                  Ours w/o Sky modeling 31.12         0.921 0.100




Extrapolation results. In Figure 14, we show some qualitative results of novel
view synthesis under the setting of lane changes on Waymo dataset. Our method
can produce high-quality results although the rendering viewpoint is far away
from input sequence.
Qualitative results on the KITTI dataset. In Figure 15, we show the
comparison results with NSG [39] and MARS [64] on the KITTI [15] dataset.
Decomposition results on the KITTI dataset. In Figure 16, we show the
qualitative comparisons of decomposition with NSG [39] and Panoptic Neural
Fields [23] on the KITTI [15] dataset.
24     Y. Yan et al.




                Ours w/o SfM                            Ours

Fig. 12: Effect of incorporating SfM points on novel view synthesis results.




            Ours w/o Sky modeling                       Ours

Fig. 13: Effect of modeling sky with cubemap on novel view synthesis results.



C    Limitations
Street Gaussians also has some known limitations. 1) Our method is limited
to reconstructing rigid dynamic scenes, such as static streets with only moving
vehicles, and cannot handle non-rigid dynamic objects like walking pedestrians.
Future work could consider employing more complex dynamic scene modeling
methods [73], to address this issue. 2) the proposed method is dependent on
the recall rate of off-the-shelf trackers. If some vehicles are missed, our pose
optimization strategy cannot compensate for this. Obtaining continuous track-
lets through methods like 2D tracking can alleviate the problem and modeling
dynamic urban scenes without object tracklets remains an interesting problem.
3) Street Gaussians still requires per-scene optimization. We consider predicting
generalizable 3D Gaussians in feed-forward manner as a future work.
                                     Street Gaussians for Modeling Dynamic Urban Scenes   25




Nearest input view




Extrapolation




 Fig. 14: Qualitative results of novel view synthesis with significant differ-
 ences from the input frames. In each scene we shift the camera by 2 meters.




Ours




NSG




MARS




Ground Truth


                     Fig. 15: Qualitative comparison results on the KITTI [15] dataset.
26   Y. Yan et al.




          Reference Image                         NSG




                 PNF                               Ours

      Fig. 16: Decomposition results on the KITTI [15] dataset.
