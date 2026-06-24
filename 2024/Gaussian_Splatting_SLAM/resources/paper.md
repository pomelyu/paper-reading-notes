# Gaussian Splatting SLAM

Hidenobu Matsuki<sup>1∗</sup> Riku Murai<sup>2∗</sup> Paul H. J. Kelly<sup>2</sup> Andrew J. Davison<sup>1</sup>

<sup>1</sup>Dyson Robotics Laboratory, Imperial College London <sup>2</sup>Software Performance Optimisation Group, Imperial College London {h.matsuki20, riku.murai15, p.kelly, a.davison}@imperial.ac.uk

Website: https://rmurai.co.uk/projects/GaussianSplattingSLAM/ Video: https://youtu.be/x604ghp9R\_Q/

![](images/fad4ae31b1679428a9e6721139d79a3cf4bd85e8f206748b7aa49ebeb05fb7cc.jpg)  
Figure 1. From a single monocular camera, we reconstruct a high fidelity 3D scene live at 3fps. For every incoming RGB frame, 3D Gaussians are incrementally formed and optimised together with the camera poses. We show both the rasterised Gaussians (left) and Gaussians shaded to highlight the geometry (right). Notice the details and the complex material properties (e.g. transparency) captured. Thin structures such as wires are accurately represented by numerous small, elongated Gaussians, and transparent objects are effectively represented by placing the Gaussians along the rim. Our system significantly advances the fidelity a live monocular SLAM system can capture.

## Abstract

We present the first application of 3D Gaussian Splatting in monocular SLAM, the most fundamental but the hardest setup for Visual SLAM. Our method, which runs live at 3fps, utilises Gaussians as the only 3D representation, unifying the required representation for accurate, efficient tracking, mapping, and high-quality rendering. Designed for challenging monocular settings, our approach is seamlessly extendable to RGB-D SLAM when an external depth sensor is available. Several innovations are required to continuously reconstruct 3D scenes with high fidelity from a live camera. First, to move beyond the original 3DGS algorithm, which requires accurate poses from an offline Structure from Motion (SfM) system, we formulate camera tracking for 3DGS using direct optimisation against the 3D Gaussians, and show that this enables fast and robust tracking with a wide basin of convergence. Second, by utilising the explicit nature of the Gaussians, we introduce geometric verification and regularisation to handle the ambiguities occurring in incremental 3D dense reconstruction. Finally, we introduce a full SLAM system which not only achieves state-of-the-art results in novel view synthesis and trajectory estimation but also reconstruction of tiny and even transparent objects.

## 1. Introduction

A long-term goal of online reconstruction with a single moving camera is near-photorealistic fidelity, which will surely allow new levels of performance in many areas of Spatial AI and robotics as well as opening up a whole range of new applications. While we increasingly see the benefit of applying powerful pre-trained priors to 3D reconstruction, a key avenue for progress is still the invention and development of core 3D representations with advantageous properties. Many “layered” SLAM methods exist which tackle the SLAM problem by integrating multiple different 3D representations or existing SLAM components; however, the most interesting advances are when a new unified dense representation can be used for all aspects of a system’s operation: local representation of detail, largescale geometric mapping and also camera tracking by direct alignment.

In this paper, we present the first online visual SLAM system based solely on the 3D Gaussian Splatting (3DGS) representation [11] recently making a big impact in offline scene reconstruction. In 3DGS a scene is represented by a large number of Gaussian blobs with orientation, elongation, colour and opacity. Other previous world/map-centric scene representations used for visual SLAM include occupancy or Signed Distance Function (SDF) voxel grids [24]; meshes [30]; point or surfel clouds [10, 31]; and recently neural fields [35]. Each of these has disadvantages: grids use significant memory and have bounded resolution, and even if octrees or hashing allow more efficiency they cannot be flexibly warped for large corrections [26, 39]; meshes require difficult, irregular topology to fuse new information; surfel clouds are discontinuous and difficult to fuse and optimise; and neural fields require expensive per-pixel raycasting to render. We show that 3DGS has none of these weaknesses. As a SLAM representation, it is most similar to point and surfel clouds, and inherits their efficiency, locality and ability to be easily warped or modified. However, it also represents geometry in a smooth, continuously differentiable way: a dense cloud of Gaussians merge together and jointly define a continuous volumetric function. And crucially, the design of modern graphics cards means that a large number of Gaussians can be efficiently rendered via “splatting” rasterisation, up to 200fps at 1080p. This rapid, differentiable rendering is integral to the tracking and map optimisation loops in our system.

The 3DGS representation has up until now only been used in offline systems for 3D reconstruction with known camera poses, and we present several innovations to enable online SLAM. We first derive the analytic Jacobian on Lie group of camera pose with respect to a 3D Gaussians map, and show that this can be seamlessly integrated into the existing differentiable rasterisation pipeline to enable camera poses to be optimised alongside scene geometry. Second, we introduce a novel Gaussian isotropic shape regularisation to ensure geometric consistency, which we have found is important for incremental reconstruction. Third, we propose a novel Gaussian resource allocation and pruning method to keep the geometry clean and enable accurate camera tracking. Our experimental results demonstrate photorealistic online local scene reconstruction, as well as state-of-the-art camera trajectory estimation and mapping for larger scenes compared to other rendering-based SLAM methods. We further show the uniqueness of the Gaussianbased SLAM method such as an extremely large camera pose convergence basin, which can also be useful for mapbased camera localisation. Our method works with only monocular input, one of the most challenging scenarios in SLAM. To highlight the intrinsic capability of 3D Gaussian for camera localisation, our method does not use any pretrained monocular depth predictor or other existing tracking modules, but relies solely on RGB image inputs in line with the original 3DGS. Since this is one of the most challenging SLAM scenario, we also show our method can easily be extended to RGB-D SLAM when depth measurements are available.

In summary, our contributions are as follows:

• The first near real-time SLAM system which works with a 3DGS as the only underlying scene representation, which can handle monocular only inputs.

• Novel techniques within the SLAM framework, including the analytic Jacobian on Lie group for direct camera pose estimation, isotropic regularisation of the Gaussian shape, and geometric verification.

• Extensive evaluations on a variety of datasets both for monocular and RGB-D settings, demonstrating competitive performance, particularly in real-world scenarios.

## 2. Related Work

Dense SLAM: Dense visual SLAM focuses on reconstructing detailed 3D maps, unlike sparse SLAM methods which excel in pose estimation [5, 6, 22] but typically yield maps useful mainly for localisation. In contrast, dense SLAM creates interactive maps beneficial for broader applications, including AR and robotics. Dense SLAM methods are generally divided into two primary categories: Frame-centric and Map-centric. Frame-centric SLAM minimises photometric error across consecutive frames, jointly estimating per-frame depth and frame-to-frame camera motion. Frame-centric approaches [2, 38] are efficient, as individual frames host local rather than global geometry (e.g. depth maps), and are attractive for long-session SLAM, but if a dense global map is needed, it must be constructed on demand by assembling all of these parts which are not necessarily fully consistent. In contrast, Map-centric SLAM uses a unified 3D representation across the SLAM pipeline, enabling a compact and streamlined system. Compared to purely local frame-to-frame tracking, a map-centric approach leverages global information by tracking against the reconstructed 3D consistent map. Classical map-centric approaches often use voxel grids [3, 24, 27, 42] or points [10, 31, 43] as the underlying 3D representation. While voxels enable a fast look-up of features in 3D, the representation is expensive, and the fixed voxel resolution and distribution are problematic when the spatial characteristics of the environment are not known in advance. On the other hand, a point-based map representation, such as surfel clouds, enables adaptive changes in resolution and spatial distribution by dynamic allocation of point primitives in the 3D space. Such flexibility benefits online applications such as SLAM with deformation-based loop closure [31, 43]. However, optimising the representation to capture high fidelity is challenging due to the lack of correlation among the primitives. Recently, in addition to classical graphic primitives, neural network-based map representations are a promising alternative. iMAP [35] demonstrated the interesting properties of neural representation, such as sensible hole filling of unobserved geometry. Many recent approaches combine the classical and neural representations to capture finer details [9, 29, 48, 49]; however, the large amount of computation required for neural rendering makes the live operation of such systems challenging.

Differentiable Rendering: The classical method for creating a 3D representation was to unproject 2D observations into 3D space and to fuse them via weighted averaging [17, 24]. Such an averaging scheme suffers from over-smooth representation and lacks the expressiveness to capture high-quality details. To capture a scene with photorealistic quality, differentiable volumetric rendering [25] has recently been popularised with Neural Radiance Fields (NeRF) [18]. Using a single Multi-Layer Perceptron (MLP) as a scene representation, NeRF performs volume rendering by marching along pixel rays, querying the MLP for opacity and colour. Since volume rendering is naturally differentiable, the MLP representation is optimised to minimise the rendering loss using multiview information to achieve high-quality novel view synthesis. The main weakness of NeRF is its training speed. Recent developments have introduced explicit volume structures such as multi-resolution voxel grids [7, 15, 36] or hash functions [20] to improve performance. Interestingly, these projects demonstrate that the main contributor to high-quality novel view synthesis is not the neural network but rather differentiable volumetric rendering, and that it is possible to avoid the use of an MLP and yet achieve comparable rendering quality to NeRF [7]. However, even in these systems, per-pixel ray marching remains a significant bottleneck for rendering speed. This issue is particularly critical in SLAM, where immediate interaction with the map is essential for tracking. In contrast to NeRF, 3DGS performs differentiable rasterisation. Similar to regular graphics rasterisations, by iterating over the primitives to be rasterised rather than marching along rays, 3DGS leverages the natural sparsity of a 3D scene and achieves a representation which is expressive to capture high-fidelity 3D scenes while offering significantly faster rendering. Several works have applied 3D Gaussians and differentiable rendering to static scene capture [12, 40], and in particular more recent works utilise 3DGS and demonstrate superior results in vision tasks such as dynamic scene capture [16, 44, 46] and 3D generation [37, 47]. Our method adopts a Map-centric approach, utilising 3D Gaussians as the only SLAM representation. Similar to surfel-based SLAM, we dynamically allocate the 3D Gaussians, enabling us to model an arbitrary spatial distribution in the scene. Unlike other methods such as Elastic-Fusion [43] and PointFusion [10], however, by using differentiable rasterisation, our SLAM system can capture highfidelity scene details and represent challenging object properties by direct optimisation against information from every pixel.

## 3. Method

## 3.1. Gaussian Splatting

Our SLAM representation is 3DGS, mapping the scene with a set of anisotropic Gaussians ${ \mathcal { G } } .$ . Each Gaussian $\mathcal { G } ^ { i }$ contains optical properties: colour $c ^ { i }$ and opacity $\alpha ^ { i }$ . For continuous 3D representation, the mean $\mu _ { W } ^ { i }$ and covariance $\Sigma _ { W } ^ { i } ,$ defined in the world coordinate, represent the Gaussian’s position and its ellipsoidal shape. We omit the spherical harmonics (SHs) representing view-dependent radiance for simplicity but report the ablation with SHs in the supplementary. Since 3DGS uses volume rendering, explicit extraction of the surface is not required. Instead, by splatting and blending $\mathcal { N }$ Gaussians, a pixel colour $\mathcal { C } _ { p }$ is synthesised:

$$
\mathcal { C } _ { p } = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{1}
$$

3DGS performs rasterisation, iterating over the Gaussians rather than marching along the camera rays, and hence, free spaces are ignored during rendering. During rasterisation, the contributions of α are decayed via a Gaussian function, based on the 2D Gaussian formed by splatting a 3D Gaussian. The 3D Gaussians $\mathcal { N } ( \mu _ { W } , \Sigma _ { W } )$ in world coordinates are related to the 2D Gaussians $\mathcal { N } ( \mu _ { I } , \Sigma _ { I } )$ on the image plane through a projective transformation:

$$
{ \pmb \mu } _ { I } = \pi ( { \pmb T } _ { C W } \cdot { \pmb \mu } _ { W } ) , \pmb \Sigma _ { I } = { \bf J } { \bf W } \pmb \Sigma _ { W } { \bf W } ^ { T } { \bf J } ^ { T } ,\tag{2}
$$

where $\pi$ is the projection operation and $T _ { C W } \in S E { ( 3 ) }$ is the camera pose of the viewpoint. J is the Jacobian of the linear approximation of the projective transformation and W is the rotational component of ${ \bf { \mathit { T } } } _ { \mathit { C W } }$ . This formulation enables the 3D Gaussians to be differentiable and the blending operation provides gradient flow to the Gaussians. Using first-order gradient descent [13], Gaussians gradually refines both their optic and geometric parameters to represent the captured scene with high fidelity.

![](images/5aca2f9ea381dcc7d67c0742b47ad44aa1401698fd5c2665fd5688cd682f531a.jpg)  
Figure 2. SLAM System Overview: Our SLAM system uses 3D Gaussians as the only representation, unifying all components of SLAM, including tracking, mapping, keyframe management, and novel view synthesis.

## 3.2. Camera Pose Optimisation

To achieve accurate tracking, we typically require at least 50 iterations of gradient descent per frame. This requirement emphasises the necessity of a representation with computationally efficient view synthesis and gradient computation, making the choice of 3D representation a crucial part of designing a SLAM system.

In order to avoid the overhead of automatic differentiation, 3DGS implements rasterisation with CUDA with derivatives for all parameters calculated explicitly. Since rasterisation is performance critical, we similarly derive the camera Jacobians explicitly.

To the best of our knowledge, we provide the first analytical Jacobian of SE(3) camera pose with respect to the 3D Gaussians used in EWA splatting [50] and 3DGS. This opens up new applications of 3DGS beyond SLAM.

We use Lie algebra to derive the minimal Jacobians, ensuring that the dimensionality of the Jacobians matches the degrees of freedom, eliminating any redundant computations. The terms of Eq. (2) are differentiable with respect to the camera pose ${ \mathbf { } } T _ { C W } ;$ using the chain rule:

$$
\frac { \partial \pmb { \mu } _ { I } } { \partial \pmb { T } _ { C W } } = \frac { \partial \pmb { \mu } _ { I } } { \partial \pmb { \mu } _ { C } } \frac { \pmb { \mathcal { D } } \pmb { \mu } _ { C } } { \pmb { \mathcal { D } } \pmb { T } _ { C W } } \ : ,\tag{3}
$$

$$
\frac { \partial \Sigma _ { I } } { \partial T _ { C W } } = \frac { \partial \Sigma _ { I } } { \partial { \bf J } } \frac { \partial { \bf J } } { \partial \pmb { \mu } _ { C } } \frac { \mathcal { D } \pmb { \mu } _ { C } } { \mathcal { D } T _ { C W } } + \frac { \partial \Sigma _ { I } } { \partial { \bf W } } \frac { \mathcal { D } { \bf W } } { \mathcal { D } T _ { C W } } .\tag{4}
$$

where ${ \pmb T } _ { C W }$ represents the 3D position of Gaussian in the camera coordinate. We take the derivatives on the manifold to derive minimal parameterisation. Borrowing the notation from [32], let $T \in S E { ( 3 ) }$ and $\tau \in { \mathfrak { s e } } ( 3 )$ . We define the partial derivative on the manifold as:

$$
\frac { \mathscr { D } f ( \pmb { T } ) } { \mathscr { D } \pmb { T } } \triangleq \operatorname* { l i m } _ { \tau  0 } \frac { \mathrm { L o g } ( f ( \mathrm { E x p } ( \tau ) \circ \pmb { T } ) \circ f ( \pmb { T } ) ^ { - 1 } ) } { \tau } ,\tag{5}
$$

where ◦ is a group composition, and Exp, Log are the exponential and logarithmic mappings between Lie algebra and Lie Group. With this, we derive the following:

$$
\frac { \mathcal { D } \pmb { \mu } _ { C } } { \mathcal { D } \pmb { T } _ { C W } } = \left[ \pmb { I } _ { \pmb { \tau } } - \pmb { \mu } _ { C } ^ { \times } \right] , \frac { \mathcal { D } \mathbf { W } } { \mathcal { D } \pmb { T } _ { C W } } = \left[ \mathbf { 0 } _ { \pmb { \tau } } - \mathbf { W } _ { : , 2 } ^ { \times } \right]\tag{6}
$$

where <sup>×</sup> denotes the skew symmetric matrix of a 3D vector, and $\mathbf { W } _ { : , i }$ <sub>i</sub> refers to the ith column of the matrix.

## 3.3. SLAM

In this section, we present details of full SLAM framework. The overview of the system is summarised in Fig. 2. Please refer to the supplementary material for the further parameter details.

## 3.3.1 Tracking

In tracking only the current camera pose is optimised, without updates to the map representation. In the monocular case, we minimise the following photometric residual:

$$
E _ { p h o } = \left\| I ( \mathcal { G } , T _ { C W } ) - \bar { I } \right\| _ { 1 } ,\tag{7}
$$

where $I ( \mathcal { G } , \pmb { T } _ { C W } )$ renders the Gaussians G from ${ \bf { \mathit { T } } } _ { \mathit { C W } }$ , and <sup>¯</sup>I is an observed image.

We further optimise affine brightness parameters for varying exposure and penalise non-edge or low-opacity pixels. When depth observations are available, we define the geometric residual as:

$$
E _ { g e o } = \left\| D ( \mathcal { G } , \pmb { T } _ { C W } ) - \bar { D } \right\| _ { 1 } ,\tag{8}
$$

where $D ( \mathcal { G } , \pmb { T } _ { C W } )$ is depth rasterisation and $\bar { D }$ is the observed depth. Rather than simply using the depth measurements to initialise the Gaussians, we minimise both photometric and geometric residuals: $\lambda _ { p h o } E _ { p h o } + ( 1 - \lambda _ { p h o } ) E _ { g e o }$ where $\lambda _ { p h o }$ is a hyperparameter.

As in Eq. (1), per-pixel depth is rasterised by alphablending:

$$
{ \mathcal { D } } _ { p } = \sum _ { i \in { \mathcal { N } } } z _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) \ ,\tag{9}
$$

where $z _ { i }$ is the distance to the mean $\pmb { \mu } _ { W }$ of Gaussian i along the camera ray. We derive analytical Jacobians for the camera pose optimisation in a similar manner to Eq. (3), (4).

## 3.3.2 Keyframing

Since using all the images from a video stream to jointly optimise the Gaussians and camera poses online is infeasible, we maintain a small window $\mathcal { W } _ { k }$ consisting of carefully selected keyframes based on inter-frame covisibility. Ideal keyframe management will select non-redundant keyframes observing the same area, spanning a wide baseline to provide better multiview constraints. The parameters are detailed in the supplementary.

Selection and Management Every tracked frame is checked for keyframe registration based on our simple yet effective criteria. We measure the covisibility by measuring the intersection over the union of the observed Gaussians between the current frame i and the last keyframe j. If the covisibility drops below a threshold, or if the relative translation $t _ { i j }$ is large with respect to the median depth, frame i is registered as a keyframe. For efficiency, we maintain only a small number of keyframes in the current window $\mathcal { W } _ { k }$ following the keyframe management heuristics of DSO [5]. The main difference is that a keyframe is removed from the current window if the overlap coefficient with the latest keyframe drops below a threshold.

Gaussian Covisibility An accurate estimate of covisibility simplifies keyframe selection and management. 3DGS respects visibility ordering since the 3D Gaussians are sorted along the camera ray. This property is desirable for covisibility estimation as occlusions are handled by design. A Gaussian is marked to be visible from a view if used in the rasterisation and if the ray’s accumulated α has not yet reached 0.5. This enables our estimated covisibility to handle occlusions without requiring additional heuristics.

Gaussian Insertion and Pruning At every keyframe, new Gaussians are inserted into the scene to capture newly visible scene elements and to refine the fine details. When depth measurements are available, Gaussian means $\pmb { \mu } _ { W }$ are initialised by back-projecting the depth. In the monocular case, we render the depth at the current frame. For pixels with depth estimates, $\pmb { \mu } _ { W }$ are initialised around those depths with low variance; for pixels without the depth estimates, we initialise $\pmb { \mu } _ { W }$ around the median depth of the rendered image with high variance.

In the monocular case, the positions of many newly inserted Gaussians are incorrect. While the majority will quickly vanish during optimisation as they violate multiview consistency, we further prune the excess Gaussians by checking the visibility amongst the current window $\mathcal { W } _ { k }$ . If the Gaussians inserted within the last 3 keyframes are unobserved by at least 3 other frames, we prune them out as they are geometrically unstable.

## 3.3.3 Mapping

The purpose of mapping is to maintain a coherent 3D structure and to optimise the newly inserted Gaussians. During mapping, the keyframes in $\mathcal { W } _ { k }$ are used to reconstruct currently visible regions. Additionally, two random past keyframes $\mathcal { W } _ { r }$ are selected per iteration to avoid forgetting the global map. Rasterisation of 3DGS imposes no constraint on the Gaussians along the viewing ray direction, even with a depth observation. This is not a problem when sufficient carefully selected viewpoints are provided (e.g. in the novel view synthesis case); however, in continuous SLAM this causes many artefacts, making tracking challenging. We therefore introduce an isotropic regularisation:

$$
E _ { i s o } = \sum _ { i = 1 } ^ { | \mathcal { G } | } \| \mathbf { s } _ { i } - \tilde { \mathbf { s } _ { i } } \cdot \mathbf { 1 } \| _ { 1 }\tag{10}
$$

to penalise the scaling parameters $\mathbf { s } _ { i }$ (i.e. stretch of the ellipsoid) by its difference to the mean s˜<sub>i</sub>. As shown in Fig 3, this encourages sphericality, and avoids the problem of Gaussians which are highly elongated along the viewing direction creating artefacts. Let the union of the keyframes in the current window and the randomly selected one be $\smash { \mathcal { W } = \mathcal { W } _ { k } \cup \mathcal { W } _ { r } }$ For mapping, we solve the following problem:

$$
\operatorname* { m i n } _ { \substack { T _ { C W } ^ { k } \in S E ( 3 ) , \mathcal { G } , \forall k \in \mathcal { W } } } E _ { p h o } ^ { k } + \lambda _ { i s o } E _ { i s o } .\tag{11}
$$

If depth observations are available, as in tracking, geometric residuals Eq. (8) are added to the optimisation problem.

## 4. Evaluation

We conduct a comprehensive evaluation of our system across a range of both real and synthetic datasets. Additionally, we perform an ablation study to justify our design

![](images/98c6af98a80ec5b7ecf962ec6a1f4ee8684a948c5c105ee9f2bcba42e531be85.jpg)  
w/o ? w/ ?Figure 3. Effect of isotropic regularisation: Top: Rendering close to a training view (looking at the keyboard). Bottom: Rendering 3D Gaussians far from the training views (view from a side of the keyboard) without (left) and with (right) the isotropic loss. When the photometric constraints are insufficient, the Gaussians tend to elongate along the viewing direction, creating artefacts in the novel views, and affecting the camera tracking.

choices. Finally, we present qualitative results of our system operating live using a monocular camera, illustrating its practicality and high fidelity reconstruction.

## 4.1. Experimental Setup

Datasets For our quantitative analysis, we evaluate our method on the TUM RGB-D dataset [34] (3 sequences) and the Replica dataset [33] (8 sequences), following the evaluation in [35]. For qualitative results, we use self-captured real-world sequences recorded by Intel Realsense d455. Since the Replica dataset is designed for RGB-D SLAM evaluation, it contains challenging purely rotational camera motions. We hence use the Replica dataset for RGB-D evaluation only. The TUM RGB-D dataset is used for both monocular and RGB-D evaluation.

Implementation Details We run our SLAM on a desktop with Intel Core i9 12900K 3.50GHz and a single NVIDIA GeForce RTX 4090. We present results from our multiprocess implementation aimed at real-time applications. For a fair comparison with other methods on Replica, we additionally report result for single-process implementation which performs more mapping iterations. As with 3DGS, time-critical rasterisation and gradient computation are implemented using CUDA. The rest of the SLAM pipeline is developed with PyTorch. Details of hyperparameters are provided in the supplementary material.

Metrics For camera tracking accuracy, we report the Root Mean Square Error (RMSE) of the Absolute Trajectory Error (ATE) of the keyframes. To evaluate map quality, we report standard photometric rendering quality metrics (PSNR,

SSIM and LPIPS) following the evaluation protocol used in [29]. To evaluate the map quality, on every fifth frame, rendering metrics are computed. We exclude the keyframes (training views). We report the average across three runs for all our evaluations. In the tables, the best result is in bold, and the second best is underlined.

Baseline Methods We primarily benchmark our SLAM method against other approaches that, like ours, do not have explicit loop closure. In monocular settings, we compare with state-of-the-art classical and learning-based direct visual odometry (VO) methods. Specifically, we compare DSO [5], DepthCov [4], and DROID-SLAM [38] in VO configurations. These methods are selected based on their public reporting of results on the benchmark (TUM dataset) or the availability of their source code for getting the benchmark result. Since one of our focuses is the online scale estimation under monocular scale ambiguity, the method which uses ground truth poses for the system initialisation such as [14] is not considered for the comparison. In the RGB-D case, we compare against neural-implicit SLAM methods [8, 9, 29, 35, 41, 45, 48] which are also map-centric, rendering-based and do not perform loop closure.

## 4.2. Quantitative Evaluation

Camera Tracking Accuracy Table 1 shows the tracking results on the TUM RGB-D dataset. In the monocular setting, our method surpasses other baselines without requiring any deep priors. Furthermore, our performance is comparable to systems which perform explicit loop closure. This clearly highlights that there still remains potential for enhancing the tracking of monocular SLAM by exploring fundamental SLAM representations.

Our RGB-D method shows better performance than any other baseline method. Notably, our system surpasses ORB-SLAM in the fr1 sequences, narrowing the gap between Map-centric SLAM and the state-of-the-art sparse frame-centric methods. Table 2 reports results on the synthetic Replica dataset. Our single-process implementation shows competitive performance and achieves the best result in 6 out of 8 sequences. Our multi-process implementation which performs fewer mapping iterations still performs comparably. In contrast to other methods, our system demonstrates higher performance on real-world data (TUM RGB-D), by optimising the Gaussian positions to compensate for the sensor noise.

Novel View Rendering Table 5 summarises the novel view rendering performance of our method with RGB-D input. We consistently show the best performance across most sequences and is least second best. Our rendering FPS is hundreds of times faster than other methods, offering a significant advantage for applications which require real-time map interaction. While Point-SLAM is competitive, that method focuses on view synthesis rather than novel-view synthesis. Their view synthesis is conditional on the availability of depth due to the depth-guided raysampling, making novel-view synthesis challenging. On the other hand, our rasterisation-based approach does not require depth guidance and achieves efficient, high-quality, novel view synthesis. Fig. 4 provides a qualitative comparison of the rendering of ours and Point-SLAM (with depth guidance).

<table><tr><td rowspan="2">Input</td><td rowspan="2">Loop- closure</td><td rowspan="2">Method</td><td rowspan="2">fr1/desk</td><td rowspan="2">fr2/xyz</td><td rowspan="2">fr3/office</td><td rowspan="2">Avg.</td></tr><tr><td></td></tr><tr><td rowspan="6">Mocuular</td><td rowspan="5">w/o</td><td>DSO [5]</td><td>22.4</td><td>1.10</td><td>9.50</td><td>11.0</td></tr><tr><td>DROID-VO [38]</td><td>5.20</td><td>10.7</td><td>7.30</td><td>7.73</td></tr><tr><td>DepthCov-VO [4]</td><td>5.60</td><td>1.20</td><td>68.8</td><td>25.2</td></tr><tr><td>Ours</td><td>3.78</td><td>4.60</td><td>3.50</td><td>3.96</td></tr><tr><td>DROID-SLAM [38]</td><td>1.80</td><td>0.50</td><td>2.80</td><td>1.70</td></tr><tr><td rowspan="10">w/o RGGB-D</td><td>ORB-SLAM2 [21] iMAP [35]</td><td>1.90 4.90</td><td>0.60 2.00</td><td>2.40</td><td>1.60</td></tr><tr><td>NICE-SLAM [48]</td><td></td><td></td><td>5.80</td><td>4.23</td></tr><tr><td>DI-Fusion [8]</td><td>4.26</td><td>6.19</td><td>3.87</td><td>4.77</td></tr><tr><td>Vox-Fusion [45]</td><td>4.40 3.52</td><td>2.00</td><td>5.80</td><td>4.07</td></tr><tr><td>ESLAM [9]</td><td>2.47</td><td>1.49 1.11</td><td>26.01</td><td>10.34</td></tr><tr><td>Co-SLAM [41]</td><td></td><td>1.70</td><td>2.42</td><td>2.00</td></tr><tr><td>Point-SLAM [29]</td><td>2.40</td><td></td><td>2.40</td><td>2.17</td></tr><tr><td></td><td>4.34</td><td>1.31</td><td>3.48</td><td>3.04</td></tr><tr><td>Ours</td><td>1.50</td><td>1.44</td><td>1.49</td><td>1.47</td></tr><tr><td rowspan="3">w/</td><td>BAD-SLAM [31]</td><td>1.70</td><td>1.10</td><td>1.70</td><td>1.50</td></tr><tr><td>Kintinous [42]</td><td>3.70</td><td>2.90</td><td>3.00</td><td>3.20</td></tr><tr><td>ORB-SLAM2 [21]</td><td>1.60</td><td>0.40</td><td>1.00</td><td>1.00</td></tr></table>

Table 1. Camera tracking result on TUM for monocular and RGB-D. ATE RMSE in cm is reported. In both monocular and RGB-D cases, we achieve state-of-the-art performance. In particular, in the monocular case, not only do we outperform systems which use deep prior, but we achieve comparable performance with many of the RGB-D systems.
<table><tr><td>Method</td><td>r0</td><td>r1</td><td>r2</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>Avg.</td></tr><tr><td>iMAP [35]</td><td>3.12</td><td>2.54</td><td>2.31</td><td>1.69</td><td>1.03</td><td>3.99</td><td>4.05</td><td>1.93</td><td>2.58</td></tr><tr><td>NICE-SLAM</td><td>0.97</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.07</td></tr><tr><td>Vox-Fusion [45]</td><td>1.37</td><td>4.70</td><td>1.47</td><td>8.48</td><td>2.04</td><td>2.58</td><td>1.11</td><td>2.94</td><td>3.09</td></tr><tr><td>ESLAM [9]</td><td>0.71</td><td>0.70</td><td>0.52</td><td>0.57</td><td>0.55</td><td>0.58</td><td>0.72</td><td>0.63</td><td>0.63</td></tr><tr><td>Point-SLAM [29]</td><td>0.61</td><td>0.41</td><td>0.37</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td><td>0.53</td></tr><tr><td>Ours</td><td>0.44</td><td>0.32</td><td>0.31</td><td>0.44</td><td>0.52</td><td>0.23</td><td>0.17</td><td>2.25</td><td>0.58</td></tr><tr><td>Ours (sp)</td><td>0.33</td><td>0.22</td><td>0.29</td><td>0.36</td><td>0.19</td><td>0.25</td><td>0.12</td><td>0.81</td><td>0.32</td></tr></table>

Table 2. Camera tracking result on Replica for RGB-D SLAM. ATE RMSE in cm is reported. We achieve best performance across most sequences. Here, Ours is our multi-process implementation and Ours (sp) is the single-process implementation which ensures a certain amount of mapping iteration similar to other works.
<table><tr><td>Input</td><td>Method</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td></tr><tr><td rowspan="3">Mono</td><td> $\overline { { \mathrm { ~ w ~ } / \mathrm { ~ o ~ } E _ { i s o } } }$ </td><td>4.16</td><td>4.66</td><td>5.73</td><td>4.83</td></tr><tr><td>w/o kf selection</td><td>13.2</td><td>4.36</td><td>8.65</td><td>8.73</td></tr><tr><td>Ours</td><td>3.78</td><td>4.60</td><td>3.50</td><td>3.96</td></tr><tr><td>RGG-D</td><td> $\overline { { \mathrm { ~ w ~ } / \mathrm { ~ o ~ } E _ { g e o } } }$ </td><td>2.39</td><td>0.62</td><td>4.98</td><td>2.66</td></tr><tr><td rowspan="2"></td><td>w/o kf selection</td><td>1.64</td><td>1.49</td><td>2.60</td><td>1.90</td></tr><tr><td>Ours</td><td>1.50</td><td>1.44</td><td>1.49</td><td>1.47</td></tr></table>

Table 3. Ablation Study on TUM RGB-D dataset. We analyse the usefulness of isotropic regularisation, geometric residual, and keyframe selection to our SLAM system. Further isotropic regularisation ablation is available in supplementary.
<table><tr><td colspan="5">Memory Usage [MB]</td></tr><tr><td>iMAP [35]</td><td>NICE-SLAM [48]</td><td>Co-SLAM [41]</td><td>Ours (Mono)</td><td>Ours (RGB-D)</td></tr><tr><td>0.8MB</td><td>40.3.4MB</td><td>6.4MB</td><td>2.6MB</td><td>3.97MB</td></tr></table>

Table 4. Memory Analysis on TUM RGB-D dataset. The baseline numbers are computed from the parameter numbers in [41]

<table><tr><td>Method</td><td>PSNR[db]↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Rendering FPS</td></tr><tr><td>NICE-SLAM[48]</td><td>24.42</td><td>0.809</td><td>0.233</td><td>0.54</td></tr><tr><td>Vox-Fusion[45]</td><td>24.41</td><td>0.801</td><td>0.236</td><td>2.17</td></tr><tr><td>Point-SLAM [29]</td><td>35.17</td><td>0.975</td><td>0.124</td><td>1.33</td></tr><tr><td>ours</td><td>38.94</td><td>0.968</td><td>0.070</td><td>769</td></tr></table>

Table 5. Average rendering performance on Replica (RGB-D). Our method outperforms most of the rendering metrics compared to existing methods. Note that Point-SLAM uses ground-truth depth to guide sampling along rays. The full detail is available in supplementary.

Ours  
Point-SLAM  
GT  
![](images/ae2b72478c4db76d416b6e625b01e4cd7d1f47e216706b3e909a51153f923b63.jpg)  
Figure 4. Rendering examples on Replica. Point-SLAM struggle with rendering fine details due to the stochastic ray sampling.

Ablative Analysis In Table 3, we perform ablation to confirm our design choices. Isotropic regularisation and geometric residual improve the tracking of monocular and RGB-D SLAM respectively, as they aid in constraining the geometry when photometric signals are weak. For both cases, keyframe selection significantly improves systems performance, as it automatically chooses suitable keyframes based on our occlusion-aware keyframe selection and management. We further compare the memory usage of different 3D representations in Table 4. MLP-based iMAP is clearly more memory efficient, but it struggles to express high-fidelity 3D scenes due to the limited capacity of small MLP. Compared with a voxel grid of features used in NICE-SLAM, our method uses significantly less memory.

Convergence Basin Analysis In our SLAM experiments, we discovered that 3D Gaussian maps have a notably large convergence basin for camera localisation. To investigate further, we conducted a convergence funnel analysis, an evaluation methodology proposed in [19] and used in [23]. Here, we train a 3D representation (e.g. 3DGS) using 9 fixed views arranged in a square. We set the viewpoint in the middle of the square to be the target view. As shown in Fig 5, we uniformly sample a position, creating a funnel. From the sampled position, given the RGB image of the target view, we perform camera pose optimisation for 1000 iterations. The optimisation is successful if it converges to within 1cm of the target view within the fixed iterations. We compare our Gaussian approach with Co-SLAM [41]’s network (Hash Grid SDF) and iMAP’s [35] network with Co-SLAM’s SDF loss for further geometric accuracy (MLP

Camera Layout  
![](images/1f5a6f5c403d14be52d7a621fc2d87f545c2eedaf76272b015d6a7685dc70439.jpg)

Ours w/ depth  
![](images/eac62a3dd72b79aed4ec9b61f5959294ad16c8da5036eb01434de8f56c9a24b5.jpg)

Hash Grid SDF  
![](images/f6d3a7dcd9890f768e44abd370d346d2791798c87e78693ac945988450847d00.jpg)

Ours w/o depth  
![](images/26f88da9001e42416b5a1fb98170b0069afa048fb786d10cb0d8c1221075d8da.jpg)

MLP SDF  
![](images/25695151cd19126d02b38d875337258e66ff95b28584f499ed0ab1b7d3df6d63.jpg)  
Figure 5. Convergence basin analysis: Left: 3D Gaussian map from training views (Yellow) and visualisation of the test poses (Red) and target pose (Blue). Right: Convergence basin of our method. The green marks success, and the red marks failure.

![](images/9f6126eff7ad22def9dae3c823eb1587ea3d7d8a8de34b772e89832076fad2e9.jpg)

![](images/bda576b3d1d71cd4737d1c342a6ea8558f57fab3dc74a77a2727324cf3175e53.jpg)

Figure 6. Monocular SLAM result on fr1/desk sequence: We show the reconstructed 3D Gaussian maps (Left) and novel view synthesis result (Right).
<table><tr><td>Method</td><td>seq1</td><td>seq2</td><td>seq3</td><td>Avg.</td></tr><tr><td>Neural SDF (Hash Grid)</td><td>0.13</td><td>0.15</td><td>0.16</td><td>0.14</td></tr><tr><td>Neural SDF (MLP)</td><td>0.40</td><td>0.38</td><td>0.22</td><td>0.33</td></tr><tr><td>Ours w/o depth</td><td>0.82</td><td>0.91</td><td>0.65</td><td>0.79</td></tr><tr><td>Ours w/ depth</td><td>0.83</td><td>1.0</td><td>0.65</td><td>0.82</td></tr></table>

Table 6. Camera convergence analysis. We report the ratio of successful camera convergence for the different sequences, across different differentiable 3D representations.

![](images/a13abe2b88440cd198897ffb8c7d85a8be9846eb305e41dee8d2431dbe19e9fc.jpg)

![](images/d4d63a906798490193575f95ed31cc506bdd48a7b3c3729f15a7ffdfedbf092f.jpg)  
Figure 7. Self-captured Scenes: Challenging scenes and objects, for example, transparent glasses and crinkled texture of salad are captured by our monocular SLAM running live.

Neural SDF). We render the training views using a synthetic Replica dataset and create three sequences for testing (seq1, seq2 and seq3). The width of the square formed by the training view is 0.5m, and the test cameras are distributed with radii ranging from 0.2m to 1.2m, covering a larger area than the training view. When training the map, the three methods— Ours w/depth, Hash Grid SDF, and MLP SDF—use RGB-D images, whereas Ours w/o depth utilises only colour images. Fig. 5 shows the qualitative results and Table 6 reports the success rate. For both with and without depth for training, our method shows better convergence. Unlike hashing and positional encoding which can lead to signal conflict, anisotropic Gaussians form a smooth gradient in 3D space, increasing the convergence basin. Further experimental details are available in the supplementary.

## 4.3. Qualitative Results

We report both the 3D reconstruction of the SLAM dataset and self-captured sequences. In Fig. 6, we visualise the monocular SLAM reconstruction of fr1/desk. The placements of the Gaussians are geometrically sensible and are 3D coherent, and our rendering from the different viewpoints highlights the quality of our systems’ novel view synthesis. In Fig. 7, we self-capture challenging scenes for monocular SLAM. By not explicitly modelling a surface, our system naturally handles transparent objects which is challenging for many other SLAM systems.

## 5. Conclusion

We have proposed the first SLAM method using 3D Gaussians as a SLAM representation. Via efficient volume rendering, our system significantly advances the fidelity and diversity of object materials a live SLAM system can capture. Our system achieves state-of-the-art performance across benchmarks for both monocular and RGB-D cases. Interesting directions for future research are the integration of loop closure for handling large-scale scenes and extraction of geometry such as surface normal as Gaussians do not explicitly represent the surface.

## 6. Acknowledgement

Research presented in this paper has been supported by Dyson Technology Ltd. We are very grateful to Eric Dexheimer, Kirill Mazur, Xin Kong, Marwan Taher, Ignacio Alzugaray, Gwangbin Bae, Aalok Patwardhan, and members of the Dyson Robotics Lab for their advice and insightful discussions.

## Supplementary Material

## 7. Implementation Details

## 7.1. System Details and Hyperparameters

## 7.1.1 Tracking and Mapping (Sec. 3.3.1 and 3.3.3)

Learning Rates We use the Adam optimiser for both camera poses and Gaussian parameters optimisation. For camera poses, we used 0.003 for rotation and 0.001 for translation. For 3D Gaussians, we used the default learning parameters of the original Gaussian Splatting implementation [11], apart from in monocular setting where we increase the learning rate of the positions of the Gaussians $\pmb { \mu } _ { W }$ by a factor of 10.

Iteration numbers 100 tracking iterations are performed per frame for across all experiments. However, we terminate the iterations early if the magnitude of the pose update becomes less than $1 0 ^ { - 4 }$ . For mapping, 150 iterations are used for the single-process implementation.

Loss Weights Given a depth observation, for tracking we minimise both photometric Eq. (7) and geometric residual Eq. (8) as:

$$
\operatorname* { m i n } _ { T _ { C W } \in S E ( 3 ) } \lambda _ { p h o } E _ { p h o } + ( 1 - \lambda _ { p h o } ) E _ { g e o } \ : ,\tag{12}
$$

and similarly, for mapping we modify Eq. (11) to:

$$
\begin{array} { r c l } { { } } & { { } } & { { \displaystyle \operatorname* { m i n } } } \\ { { } } & { { } } & { { \pmb { T } _ { C W } ^ { k } \in S E ( 3 ) , \mathcal { G } , \displaystyle \sum _ { \forall k \in \mathcal { W } } ( \lambda _ { p h o } E _ { p h o } ^ { k } + ( 1 - \lambda _ { p h o } ) E _ { g e o } ^ { k } ) } } \\ { { } } & { { } } & { { \qquad \quad + \lambda _ { i s o } E _ { i s o } . } } \end{array}\tag{13}
$$

We set $\lambda _ { p h o } = 0 . 9$ for all RGB-D experiments, and $\lambda _ { i s o } =$ 10 for both monocular and RGB-D experiments.

## 7.1.2 Keyframing (Sec. 3.3.2)

Gaussian Covisibility Check (Sec. 3.3.2) As described in Sec. 3.3.2, keyframe selection is based on the covisibility of the Gaussians. Between two keyframes $i , j ,$ we define the covisibility using the Intersection of Union (IOU) and Overlap Coefficient (OC):

$$
I O U _ { c o v } ( i , j ) = \frac { | \mathcal { G } _ { i } ^ { v } \cap \mathcal { G } _ { j } ^ { v } | } { | \mathcal { G } _ { i } ^ { v } \cup \mathcal { G } _ { j } ^ { v } | } ,\tag{14}
$$

$$
O C _ { c o v } ( i , j ) = \frac { | \mathcal { G } _ { i } ^ { v } \cap \mathcal { G } _ { j } ^ { v } | } { \operatorname* { m i n } ( | \mathcal { G } _ { i } ^ { v } | , | \mathcal { G } _ { j } ^ { v } | ) } ~ ,\tag{15}
$$

where $\mathcal { G } _ { i } ^ { v }$ is the Gaussians visible in keyframe i, based on visibility check described in Section 3.3.2, Gaussian Covisibility. A keyframe i is added to the keyframe window $\mathcal { W } _ { k }$ if given last keyframe j, $I O U _ { c o v } ( i , j ) < k f _ { c o v }$ or if the relative translation $t _ { i j } > k f _ { m } \hat { D } _ { i }$ , where $\hat { D } _ { i }$ is the median depth of frame i. For Replica $k f _ { c o v } = 0 . 9 5 , k f _ { m } = 0 . 0 4$ and for TUM $k f _ { c o v } = 0 . 9 0 , k f _ { m } = 0 . 0 8$ . We remove the registered keyframe j in $\mathcal { W } _ { k }$ if the $O C _ { c o v } ( i , j ) < k f _ { c } .$ , where keyframe i is the latest added keyframe. For both Replica and TUM, we set the cutoff to $k f _ { c } = 0 . 3$ . We set the size of the keyframe window to be for Replica, $| \mathcal { W } _ { k } | = 1 0$ , and for TUM, $| \mathcal { W } _ { k } | = 8$

Gaussian Insertion and Pruning (Sec. 3.3.2) As we optimise the positions of Gaussians and prune geometrically unstable Gaussians, we do not require any strong prior such as depth observation for Gaussian initialisation. When inserting new Gaussians in a monocular setting, we randomly sample the Gaussians position $\pmb { \mu } _ { W }$ using rendered depth $D .$ Since the estimated depth may sometimes be incorrect, we account for this by initialising the Gaussians with some variance. For a pixel p where the rendered depth $\mathcal { D } _ { p }$ exists, we sample the depth from $\mathcal { N } ( \mathcal { D } _ { p } , 0 . 2 \sigma _ { D } )$ . Otherwise, for unobserved regions, we initialise the Gaussians by sampling from $\mathcal { N } ( \hat { D } , 0 . 5 \sigma _ { D } )$ , where D<sup>ˆ</sup> is the median of D. For pruning, as described in Section 3.3.2, we perform visibility-based pruning, where if new Gaussians inserted within the last 3 keyframes are not observed by at least 3 other frames, they are pruned. We only perform visibilitybased pruning once the keyframe window $\mathcal { W } _ { k }$ is full. Additionally, we prune all Gaussians with opacity of less than 0.7.

## 8. Evaluation details

8.1. Camera Tracking Accuracy (Table 1 and Table 2)

## 8.1.1 Evaluation Metric

We measured the keyframe absolute trajectory error (ATE) RMSE. For monocular evaluation, we perform scale alignment between the estimated scale-free and ground-truth trajectories. For RGB-D evaluation, we only align the estimated trajectory and ground truth without scale adjustment.

## 8.1.2 Baseline Results

Table 1 Numbers for monocular DROID-SLAM [38] and ORB-SLAM [21] is taken from [14]. We have locally run DSO [5], DepthCov [4] and DROID-VO [38] – which is DROID-SLAM without loop closure and global bundle adjustment. For the RGB-D case, numbers for NICE-SLAM [48], DI-Fusion [8], Vox-Fusion [45], Point-SLAM [29] are taken from Point-SLAM [29], and numbers for iMAP [35], BAD-SLAM [31], Kintinous [42], ORB-SLAM [21] are from iMAP [35], and ald all the other baselines: ESLAM [9], Co-SLAM [41] are from each individual papers.

Table 2 and 5 We took the numbers from Point-SLAM [29] paper.

Table 4 The numbers are from Co-SLAM [41] paper.

## 8.2. Rendering Performance (Table 5)

We provide the full detail of the rendering performance evaluation in Table 7.

In Table 5, we reported the photometric quality metrics (PSNR, SSIM and LPIPS) and rendering fps of our methods. We demonstrated that our rendering fps (769) is much higher than other existing methods (VoxFusion is the second best with 2.17fps). Here we describe the detail of how we measured the fps. The rendering time refers to the duration necessary for full-resolution rendering (1200 × 680 for the Replica sequence). For each method, we perform 100 renderings and report the average time taken per rendering. The reported rendering fps is found by taking 1 and dividing it by the average rendering time. We summarise the numbers in Table 8. Note that the “rendering fps” means the fps just for the forward rendering, which differs from the end-to-end system fps reported in Table 9 and 10.

## 8.3. The convergence basin analysis (Table 6 and Fig 5)

## 8.3.1 The detail of the benchmark Dataset

For convergence basin analysis, we create three datasets by rendering the synthetic Replica dataset. In addition to the qualitative visualisation in Figure 5, we report more detailed camera pose distributions in Figure 8. Figure 8 shows the camera view frustums of the test (red), training (yellow) and target (blue) views. As we mentioned in the main paper, we set the training view in the shape of a square with a width of 0.5m and test views are distributed with radii ranging from 0.2m to 1.2m, covering a larger area than the training views. We only apply displacements to the camera translation but not to the rotation. For each sequence, we use a total of 67 test views.

## 8.3.2 Training setup

For each method, the 3D representation is trained for 30000 iterations using the training views. Here, we detail the training setup of each of the methods:

Ours We evaluated our method under two settings: $^ { 6 6 } \mathrm { w } /$ depth” and “w/o depth”, where we train the initial 3D Gaussian map $\mathcal { G } _ { i n i t }$ with and without depth supervision. In the “w/o depth” setting, the 3D Gaussians’ positions are randomly initialised, and we minimise the monocular mapping cost Eq. (11) for the 3D Gaussian training, but keeping the camera poses fixed. Specifically, let $k \in \mathbb N$ be a number of training views and 3D Gaussians ${ \mathcal { G } } ,$ , we find $\mathcal { G } _ { i n i t }$ by:

![](images/d0db47655c8917b76b9f061312e0fdb3b1706a6b6a86b09225dceef96e4a0329.jpg)  
Figure 8. 2D Visualisation of the camera pose distributions used for convergence basin analysis in Figure 5.

$$
\mathcal { G } _ { i n i t } = \mathop { \arg \operatorname* { m i n } } _ { \mathcal { G } } \sum _ { \forall k \in \mathcal { W } } E _ { p h o } ^ { k } + \lambda _ { i s o } E _ { i s o } .\tag{16}
$$

Note that training views’ camera poses $\pmb { T } _ { C W } ^ { k }$ are fixed during the optimisation.

In the “w/ depth” setting, we train the Gaussian map by minimising the same cost function as our RGB-D SLAM system:

$$
\begin{array} { r } { \mathcal { G } _ { i n i t } = \underset { \mathcal { G } } { \arg \operatorname* { m i n } } \sum _ { \forall k \in \mathcal { W } } ( \lambda _ { p h o } E _ { p h o } ^ { k } + ( 1 - \lambda _ { p h o } ) E _ { g e o } ^ { k } ) } \\ { + \lambda _ { i s o } E _ { i s o } , \quad \quad ( 1 } \end{array}\tag{17}
$$

where we use $\lambda _ { p h o } = 0 . 9$ and $\lambda _ { i s o } = 1 0$ for all the experiments

Baseline Methods For Hash Grid SDF, we trained the same network architecture as Co-SLAM [41]. For MLP SDF, we trained the network of iMAP [35]. For both baselines, we supervised networks with the same loss functions as Co-SLAM, which are colour rendering loss $L _ { r g b } ,$ depth rendering loss $L _ { d e p t h } ,$ , SDF loss $L _ { f s }$ , free-space loss $L _ { f s } ,$ and smoothness loss $L _ { s m o o t h }$ . Please refer to the original

<table><tr><td>Method</td><td>Metric</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td><td>Avg.</td><td>Rendering FPS</td></tr><tr><td rowspan="3">NICE-SLAM [48]</td><td>PSNR[dB]↑</td><td>22.12</td><td>22.47</td><td>24.52</td><td>29.07</td><td>30.34</td><td>19.66</td><td>22.23</td><td>24.94</td><td>24.42</td><td></td></tr><tr><td>SSIM↑</td><td>0.689</td><td>0.757</td><td>0.814</td><td>0.874</td><td>0.886</td><td>0.797</td><td>0.801</td><td>0.856</td><td>0.809</td><td>0.54</td></tr><tr><td>LPIPS↓</td><td>0.33</td><td>0.271</td><td>0.208</td><td>0.229</td><td>0.181</td><td>0.235</td><td>0.209</td><td>0.198</td><td>0.233</td><td></td></tr><tr><td rowspan="3">Vox-Fusion [45]</td><td>PSNR[dB]↑</td><td>22.39</td><td>22.36</td><td>23.92</td><td>27.79</td><td>29.83</td><td>20.33</td><td>23.47</td><td>25.21</td><td>24.41</td><td></td></tr><tr><td>SSIM ↑</td><td>0.683</td><td>0.751</td><td>0.798</td><td>0.857</td><td>0.876</td><td>0.794</td><td>0.803</td><td>0.847</td><td>0.801</td><td>2.17</td></tr><tr><td>LPIPS↓</td><td>0.303</td><td>0.269</td><td>0.234</td><td>0.241</td><td>0.184</td><td>0.243</td><td>0.213</td><td>0.199</td><td>0.236</td><td></td></tr><tr><td rowspan="3">Point-SLAM [29]</td><td>PSNR[dB] ↑</td><td>32.40</td><td>34.08</td><td>35.5</td><td>38.26</td><td>39.16</td><td>33.99</td><td>33.48</td><td>33.49</td><td>35.17</td><td></td></tr><tr><td>SSIM ↑</td><td>0.974</td><td>0.977</td><td>0.982</td><td>0.983</td><td>0.986</td><td>0.96</td><td>0.960</td><td>0.979</td><td>0.975</td><td>1.33</td></tr><tr><td>LPIPS↓</td><td>0.113</td><td>0.116</td><td>0.111</td><td>0.1</td><td>0.118</td><td>0.156</td><td>0.132</td><td>0.142</td><td>0.124</td><td></td></tr><tr><td rowspan="3">Ours</td><td>PSNR[dB] ↑</td><td>34.83</td><td>36.43</td><td>37.49</td><td>39.95</td><td>42.09</td><td>36.24</td><td>36.7</td><td>36.07</td><td>37.50</td><td></td></tr><tr><td>SSIM ↑</td><td>0.954</td><td>0.959</td><td>0.965</td><td>0.971</td><td>0.977</td><td>0.964</td><td>0.963</td><td>0.957</td><td>0.960</td><td>769</td></tr><tr><td>LPIPS↓</td><td>0.068</td><td>0.076</td><td>0.075</td><td>0.072</td><td>0.055</td><td>0.078</td><td>0.065</td><td>0.099</td><td>0.070</td><td></td></tr></table>

Table 7. Rendering performance comparison of RGB-D SLAM methods on Replica. Our method outperforms most of the rendering metrics compared to existing methods. Note that Point-SLAM uses sensor depth (ground-truth depth in Replica) to guide sampling along rays, which limits the rendering performance to existing views. The numbers for the baselines are taken from [29].

<table><tr><td>Method</td><td>Rendering FPS ↑</td><td>Rendering time per image [s] ↓</td></tr><tr><td>NICE-SLAM [48]</td><td>0.54</td><td>1.85</td></tr><tr><td>Vox-Fusion [45]</td><td>2.17</td><td>0.46</td></tr><tr><td>Point-SLAM [29]</td><td>1.33</td><td>0.75</td></tr><tr><td>Ours</td><td>769</td><td>0.0013</td></tr></table>

Table 8. Further detail of Rendering FPS and Rendering Time comparison based on Table 5.

Co-SLAM paper for the exact formulation (equation (6) - (9)). All the training hyperparameters (e.g. learning rate of the network, number of sampling points, loss weight) are the same as Co-SLAM’s default configuration of the Replica dataset. While Co-SLAM stores training view information by downsampling the colour and depth images, we store the full pixel information because the number of training views is small.

## 8.3.3 Testing Setup

For testing, we localise the camera pose by minimising only the photometric error against the ground-truth colour image of the target view.

Ours Let the camera pose $T _ { C W } \in S E { ( 3 ) }$ and initial 3D Gaussians $\mathcal { G } _ { i n i t } .$ the localised camera pose $\overrightharpoon { \mathbfit { T } _ { C W } ^ { e s t } }$ is found by:

$$
{ \pmb T } _ { C W } ^ { e s t } = \arg \operatorname* { m i n } _ { { \pmb T } _ { C W } } \left| \left| I ( { \mathcal G } _ { i n i t } , { \pmb T } _ { C W } ) - \bar { I } _ { t a r g e t } \right| \right| _ { 1 } .\tag{18}
$$

Note that $\mathcal { G } _ { i n i t }$ is fixed during the optimisation. We initialise ${ \bf { \mathit { T } } } _ { \mathit { C W } }$ at one of the test view’s positions, and optimisation is performed for 1000 iterations. We perform this localisation process for all the test views and measure the success rate. Camera localisation is successful if the estimated pose converges to within 1cm of the target view within the 1000 iterations.

<table><tr><td>Method</td><td>Total Time [s]</td><td>FPS</td></tr><tr><td>Monocular</td><td>798.9</td><td>3.2</td></tr><tr><td>RGB-D</td><td>986.7</td><td>2.5</td></tr></table>

Table 9. Performance Analysis using fr3/office. Both monocular and RGB-D implementations use multiprocessing. We report the total execution time of our system, FPS computed by dividing the total number of processed frames by the total time.
<table><tr><td>Method</td><td>Total Time [s]</td><td>FPS</td></tr><tr><td>RGB-D</td><td>1111.1</td><td>1.8</td></tr><tr><td>RGB-D (sp)</td><td>1904.7</td><td>1.1</td></tr></table>

Table 10. Performance Analysis using replica/office1. RGB-D uses a multi-process implementation and RGB-D-sp is the singleprocess implementation. We report the total execution time of our system, FPS computed by dividing the total number of processed frames by the total time.

Baseline Methods For the baseline methods, the camera localisation is performed by minimising colour volume rendering loss $L _ { r g b } ,$ while all the other trainable network parameters are fixed. The learning rates of the pose optimiser are also the same as Co-SLAM’s default configuration of Replica dataset.

## 9. Further Ablation Analysis (Table 3)

## 9.1. Pruning Ablation (Monocular input)

In Table 9.1, we report the ablation study of our proposed Gaussian pruning, which prunes randomly initialised 3D Gaussians effectively in a monocular SLAM setting. As the result shows, Gaussian pruning plays a significant role in enhancing camera tracking performance. This improvement is primarily because, without pruning, randomly initialised Gaussians persist in the 3D space, potentially leading to incorrect initial geometry for other views.

<table><tr><td>Input</td><td>Method</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td></tr><tr><td rowspan="2">Mono</td><td>w/o pruning</td><td>78.2</td><td>4.5</td><td>57.0</td><td>46.6</td></tr><tr><td>Ours</td><td>3.78</td><td>4.60</td><td>3.50</td><td>3.96</td></tr></table>

Table 11. Pruning Ablation Study on TUM RGB-D dataset (Monocular Input). Numbers are camera tracking error (ATE RMSE) in cm.
<table><tr><td>Input</td><td>Method</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td></tr><tr><td rowspan="2">RGB-D</td><td>w/o  $E _ { i s o }$ </td><td>1.60</td><td>1.42</td><td>1.32</td><td>1.43</td></tr><tr><td>Ours</td><td>1.50</td><td>1.44</td><td>1.49</td><td>1.47</td></tr></table>

Table 12. Isotropic Loss Ablation Study on TUM RGB-D dataset (RGB-D input). Numbers are camera tracking error (ATE RMSE) in cm.

<table><tr><td>Method</td><td>r0</td><td>r1</td><td>r2</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>Avg.</td></tr><tr><td> $\mathrm { w } / \mathrm { o } E _ { i s o }$ </td><td>0.44</td><td>0.86</td><td>0.28</td><td>0.75</td><td>0.99</td><td>0.36</td><td>0.28</td><td>2.6</td><td>0.82</td></tr><tr><td>Ours</td><td>0.44</td><td>0.32</td><td>0.31</td><td>0.44</td><td>0.52</td><td>0.23</td><td>0.17</td><td>2.25</td><td>0.58</td></tr></table>

Table 13. Isotropic Loss Ablation Study on Replica dataset (RGB-D input). Numbers are camera tracking error (ATE RMSE) in cm.

## 9.2. Isotropic Loss Ablation (RGB-D input)

Table 12 and 13 report the ablation study of the effect of isotropic loss $E _ { i s o }$ for RGB-D input. In TUM, as Table 12 shows, isotropic regularisation does not improve the performance but only shows a marginal difference. However, for Replica, as summarised in Table 13, isotropic loss significantly improves camera tracking performance. Even with the depth measurement, since rasterisation does not consider the elongation along the viewing axis. Isotropic regularisation is required to prevent the Gaussians from over-stretching, especially for textureless regions, which are common in Replica.

## 9.3. Effect of Spherical Harmonics (SH)

While we disabled SHs in the main paper for simplicity, here we report the ablation study of the effect of SHs. The 3DGS paper [11] shows that addition of SH leads to small improvements in rendering metrics, and we have found similar improvement with SH enabled in our system (Tab.15a). We did not observe a significant change in runtime with SH enabled, but it notably increases Gaussian map size and hence GPU memory usage. Though an analytical Jacobian propagates the gradients from SH to camera poses, ATE marginally gets worse when SH is enabled (Tab. 16), as SH may incorrectly explain non-view directional effects caused by the camera motion, degrading the trajectory estimate.

## 9.4. Mapping Performance with ORB-SLAM

One of the most straightforward approaches for real-time operation is to combine an existing tracking system and 3DGS. In particular, frame-based SLAM methods have been well-studied for years and is capable of providing reliable tracking. In this section, we compare our unified 3DGS-based method to the combined approach. We have run RGB-D ORB-SLAM to recover the poses and train 3DGS with the poses and sensor depth of the keyframes, equivalent to performing offline splatting. Though ORB-SLAM is best in terms of ATE (Tab.1 main), we find no significant difference across the rendering metrics (Tab.15b). SH is omitted in the synthetic Replica dataset as it contains no view-directional effects. While using an off-theshelf tracker with a 3DGS mapper is possible, this work has focused on the value of the 3DGS throughout the entire algorithms, which is unexplored and therefore provides new insights. Further performance improvement of the unified approach will be an interesting future work.

![](images/15924b96850303570d7e592e61160c1933e4a94fd103adf579af8547df0e707a.jpg)

![](images/4969834af74ad231b7c37512ef576a5f252c7285f02bcf78ed2a7382b7edd51b.jpg)

<table><tr><td></td><td>01-easy</td><td>02-easy</td><td>03-medium</td><td>04-difficult</td><td>05-difficult</td></tr><tr><td>Point-SLAM [29]</td><td>–</td><td>1</td><td>一</td><td>=</td><td>=</td></tr><tr><td>Ours</td><td>0.121</td><td>0.141</td><td>2.197</td><td>4.515</td><td>3.190</td></tr><tr><td>Vins-Fusion [28]</td><td>0.540</td><td>0.460</td><td>0.330</td><td>0.780</td><td>0.500</td></tr><tr><td>SVO [6]</td><td>0.040</td><td>0.070</td><td>0.270</td><td>0.170</td><td>0.120</td></tr><tr><td>ORB-SLAM3 [1]</td><td>0.029</td><td>0.019</td><td>0.024</td><td>0.085</td><td>0.052</td></tr></table>

Table 14. ATE RMSE (meter) on EuRoC Machine Hall with Stereo Depth. Baseline numbers of classical methods are from [1]. The third best result is highlighted with a dash line.
<table><tr><td></td><td></td><td colspan="3">TUM</td><td colspan="3">Replica</td></tr><tr><td></td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td rowspan="3">(a)</td><td>Ours (w/o SH)</td><td>21.89</td><td>0.733</td><td>0.327</td><td>38.94</td><td>0.968</td><td>0.0703</td></tr><tr><td>Ours (w. SH)</td><td>24.37</td><td>0.804</td><td>0.225</td><td></td><td></td><td></td></tr><tr><td>Point-SLAM</td><td>21.39</td><td>0.727</td><td>0.463</td><td>24.37</td><td>0.840</td><td>0.185</td></tr><tr><td rowspan="2">(b)</td><td>ORB+GS (w/o SH)</td><td>25.12</td><td>0.837</td><td>0.161</td><td>37.11</td><td>0.964</td><td>0.040</td></tr><tr><td>ORB+GS (w.SH)</td><td>25.44</td><td>0.842</td><td>0.146</td><td>=</td><td>=</td><td></td></tr></table>

Table 15. Mean Rendering metrics for TUM and Replica (RGBD).
<table><tr><td colspan="5">Memory Usage for RGB-D SLAM</td><td colspan="2">ATE RMSE</td></tr><tr><td>Ours</td><td>Ours</td><td>Point-SLAM</td><td>ORB+GS</td><td>ORB+GS</td><td>Ours</td><td>Ours</td></tr><tr><td>(w/o SH)</td><td>(w. SH)</td><td></td><td>(w/o SH)</td><td>(w. SH)</td><td>(w/o SH)</td><td>(w. SH)</td></tr><tr><td>3.97MB</td><td>11.47MB</td><td>38.0MB</td><td>45.97MB</td><td>186.5MB</td><td>1.47cm</td><td>1.56cm</td></tr></table>

Table 16. Mean Memory and ATE metrics for TUM (RGBD).

## 9.5. Large-scale Scenes with Stereo Inputs:

This work focuses on pioneering 3DGS-based SLAM for live operation in small-scale scenes. However, we tested our method on the large-scale EuRoC Machine Hall dataset with depth from stereo (Tab.14). Fig.1 is a qualitative reconstruction result from our system. Our method is competitive in “easy” sequences, although performance drops for more difficult, longer sequences. Note that Point-SLAM [29] fails on all sequences in this dataset. In future work, we expect to improve our method by incorporating loop closure. In principle, loop closure will be easier to incorporate compared to other representations such as voxel grids (where feature allocations are fixed), via a method similar to surfel-based approaches like ElasticFusion [43].

## 9.6. Memory Consumption and Frame Rate (Table. 4)

## 9.6.1 Memory Analysis

In memory consumption analysis, for Table. 4, we measure the final size of the created Gaussians. The memory footprint of our system is lower than the original Gaussian Splatting, which uses approximately 300-700MB for the standard novel view synthesis benchmark dataset [11], as we only maintain well-constrained Gaussians via pruning and do not store the spherical harmonics.

## 9.6.2 Timing Analysis

To analyse the processing time of our monocular/RGB-D SLAM system, we measure the total time required to process all frames in the TUM-RGBD fr3/office dataset. This approach assesses the performance of our system as a whole, rather than isolating individual components. By adopting this approach, we gain a more realistic understanding of the system’s true performance which better reflects the real-world operating conditions, as it avoids the assumption of an idealised, sequential interleaving of the tracking and mapping processes. As shown in Table 10, our system operates at 3.2 FPS with monocular and 2.5 FPS with depth. The FPS is found by dividing the number of processed frames by the total time. We conducted a similar analysis with the Replica dataset office2. Here, we compare the RGB-D method with and without multiprocessing. As expected, single-process implementation takes longer as it performs more mapping iterations.

## 10. Camera Pose Jacobian

Use of 3D Gaussian as a primitive and performing camera pose optimisation is discussed in [12]; however, the method assumes a smaller number of Gaussians and is based on rayintersection not splatting; hence, is not applicable to 3DGS. While many applications of 3DGS exist, for example, dynamic tracking and 4D scene representation [16, 44], they assume offline application and require accurate camera position. In contrast, we perform camera pose optimisation by deriving the minimal analytical Jacobians on Lie group, and for completeness, we provide the derivation of the Ja-

cobians presented in Eq. (6).

$$
{ \frac { { \mathcal { D } } \pmb { \mu } _ { C } } { { \mathcal { D } } { \pmb T } _ { C W } } } = \operatorname* { l i m } _ { \tau  0 } { \frac { { \mathbf { E x p } } ( \tau ) \cdot \pmb { \mu } _ { C } - \pmb { \mu } _ { C } } { \tau } }\tag{19}
$$

$$
= \operatorname* { l i m } _ { \tau  0 } { \frac { ( I + \tau ^ { \wedge } ) \cdot \pmb { \mu } _ { C } - \pmb { \mu } _ { C } } { \tau } }\tag{20}
$$

$$
= \operatorname* { l i m } _ { \tau \to 0 } \frac { \tau ^ { \wedge } \cdot \pmb { \mu } _ { C } } { \tau }\tag{21}
$$

$$
= \operatorname* { l i m } _ { \tau \to 0 } { \frac { \theta ^ { \times } \mu _ { C } + \rho } { \tau } }\tag{22}
$$

$$
= \operatorname* { l i m } _ { \tau \to 0 } \frac { - \pmb { \mu } _ { C } ^ { \times } \theta + \rho } { \tau }\tag{23}
$$

$$
\begin{array} { r l } { \mathbf { \Pi } } & { { } = \left[ \pmb { I } \quad - \pmb { \mu } _ { C } ^ { \times } \right] } \end{array}\tag{24}
$$

where ${ \mathbf { \nabla } } _ { \mathbf { T } } \cdot \mathbf { x }$ is the group action of $\pmb { T } \in S E ( 3 ) \ \mathrm { o n } \ \mathbf { x } \in \mathbb { R } ^ { 3 }$

Simiarly, we compute the Jacobian with respect to W. Since the translational component is not involved, we only consider the rotational part $R _ { C W }$ of ${ \bf { \mathit { T } } } _ { \mathit { C W } }$

$$
{ \frac { { \mathcal { D } } \mathbf { W } } { { \mathcal { D } } R _ { C W } } } = \operatorname* { l i m } _ { \theta \to 0 } { \frac { { \mathrm { E x p } } ( \theta ) \circ \mathbf { W } - \mathbf { W } } { \theta } }\tag{25}
$$

$$
\mathbf { \partial } = \operatorname* { l i m } _ { \theta  0 } { \frac { ( I + \theta ^ { \wedge } ) \circ \mathbf { W } - \mathbf { W } } { \theta } }\tag{26}
$$

$$
\mathbf { \partial } = \operatorname* { l i m } _ { \theta  0 } \frac { \theta ^ { \wedge } } { \theta } \circ \mathbf { W }\tag{27}
$$

$$
\mathbf { \partial } = \operatorname* { l i m } _ { \theta  0 } \frac { \theta ^ { \times } } { \theta } \circ \mathbf { W }\tag{28}
$$

Since skew symmetric matrix is:

$$
\theta ^ { \times } = \left[ \begin{array} { c c c } { 0 } & { - \theta _ { z } } & { \theta _ { y } } \\ { \theta _ { z } } & { 0 } & { - \theta _ { x } } \\ { - \theta _ { y } } & { \theta _ { x } } & { 0 } \end{array} \right]\tag{29}
$$

The partial derivative of one of the component $( \mathrm { e } . \mathrm { g } . \theta _ { x } )$ is:

$$
{ \frac { \partial \theta ^ { \times } } { \partial \theta _ { x } } } = { \left[ \begin{array} { l l l } { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { - 1 } \\ { 0 } & { 1 } & { 0 } \end{array} \right] } = \mathbf { e } _ { 1 } ^ { \times }\tag{30}
$$

where $\mathbf { e } _ { 1 } = [ 1 , 0 , 0 ] ^ { \top } , \mathbf { e } _ { 2 } = [ 0 , 1 , 0 ] ^ { \top } , \mathbf { e } _ { 3 } = [ 0 , 0 , 1 ] ^ { \top } .$

$$
\frac { \partial \mathbf { W } } { \partial \theta _ { x } } = \mathbf { e } _ { 1 } ^ { \times } \mathbf { W } = \left[ \begin{array} { l } { \mathbf { 0 } _ { 1 \times 3 } } \\ { - \mathbf { W } _ { 3 , : } } \\ { \mathbf { W } _ { 2 , : } } \end{array} \right]\tag{31}
$$

$$
\frac { \partial \mathbf { W } } { \partial \theta _ { y } } = \mathbf { e } _ { 2 } ^ { \times } \mathbf { W } = \left[ \begin{array} { c } { \mathbf { W } _ { 3 , : } } \\ { \mathbf { 0 } _ { 1 \times 3 } } \\ { - \mathbf { W } _ { 1 , : } } \end{array} \right]\tag{32}
$$

$$
\frac { \partial \mathbf { W } } { \partial \theta _ { z } } = \mathbf { e } _ { 3 } ^ { \times } \mathbf { W } = \left[ \begin{array} { c } { - \mathbf { W } _ { 2 , : } } \\ { \mathbf { W } _ { 1 , : } } \\ { \mathbf { 0 } _ { 1 \times 3 } } \end{array} \right]\tag{33}
$$

where $\mathbf { W } _ { i , : }$ refers to the ith row of the matrix. After column-wise vectorisation of Eq. (31), (32), (33), and stacking horizontally we get:

$$
\frac { \mathcal { D } \mathbf { W } } { \mathcal { D } \mathbf { R } _ { C W } } = \left[ \mathbf { - W } _ { : , 2 } ^ { \times } \right] ,\tag{34}
$$

where $\mathbf { W } _ { : , i }$ refers to the ith column of the matrix. Since the translational part is all zeros, with this we get Eq. (6).

## 11. Additional Qualitative Results

We urge readers to view our supplementary video for convincing qualitative results. In Fig. 9 - Fig. 16, we further show additional qualitative results. We visually compare other state-of-the-art SLAM methods using differentiable rendering (Point-SLAM [29] and ESLAM [9]).

## 12. Limitation of this work

Although our novel Gaussian Splatting SLAM shows competitive performance on experimental results, the method also has several limitations.

• Currently, the proposed method is tested only on small room-scale scenes. For larger real-world scenes, the trajectory drift is inevitable. This could be addressed by integrating a loop closure module into our existing pipeline.

• Although we achieve interactive live operation, hard realtime operation on the benchmark dataset (30 fps on TUM sequences) is not achieved in this work. To improve speed, exploring a second-order optimiser would be an interesting direction.

Monocular  
RGB-D  
![](images/2c4f9f1e194084202c3166f62ab5c7b2eadf080f1953d85537dcd40252006ca6.jpg)  
Figure 10. Rendering comparison on TUM fr1/desk

![](images/2e5d39cce9b041c4bb44073f00169fb1c2acb509ac76fa71bf3e2288120c7d4d.jpg)  
Monocular  
RGB-D  
Figure 12. Rendering comparison on TUM fr2/xyz

Monocular  
RGB-D  
![](images/8a946a0ade70d296b89f0ea2ec2d580df401f2798a6c2586fbccd9b008804632.jpg)  
Figure 13. Novel view rendering and Gaussian visualizations on TUM fr3/office

ESLAM  
Point-SLAM  
Ours (Mono)  
Ours (RGBD)  
GT  
![](images/4f6a4bbd3f2053dd413a584954078618ba9573bb534b314ebb9b8059ae5fb926.jpg)  
Figure 14. Rendering comparison on TUM fr3/office

![](images/2556c21230b3ab835b3f11976f2f9f2b7d6808435d6c46c5303fa631cc39be1e.jpg)

![](images/84027462726a8012602f032119e031bc7dd4aaab0271e7b690bc2632b8e234ac.jpg)

![](images/f66426921c32e3e9d1a3f8d79ab7a1055a0e8f93e45093cf5457f446ddc90654.jpg)  
ESLAM  
Point-SLAM

![](images/ddad02bc3f1dc6555ad31c03e5935934b1f4b47f4e19347c25275cbc3971f92a.jpg)  
Figure 15. Novel view rendering and Gaussian visualizations on Replica

Ours  
GT  
![](images/8e7b79b6ced53de366a3d4e4dbf8f43d4403496ab91cd67561c6de7c1134b6bf.jpg)  
Figure 16. Rendering comparison on Replica

## References

[1] Carlos Campos, Richard Elvira, Juan J. Gomez, Jos´ e M. M.´ Montiel, and Juan D. Tardos. ORB-SLAM3: An accurate´ open-source library for visual, visual-inertial and multi-map SLAM. IEEE Transactions on Robotics (T-RO), 37(6):1874– 1890, 2021.

[2] J. Czarnowski, T. Laidlow, R. Clark, and A. J. Davison. Deepfactors: Real-time probabilistic dense monocular SLAM. IEEE Robotics and Automation Letters (RAL), 5(2): 721–728, 2020.

[3] Angela Dai, Matthias Nießner, Michael Zollhofer, Shahram¨ Izadi, and Christian Theobalt. BundleFusion: Real-time Globally Consistent 3D Reconstruction using On-the-fly Surface Re-integration. ACM Transactions on Graphics (TOG), 36(3):24:1–24:18, 2017.

[4] Eric Dexheimer and Andrew J. Davison. Learning a Depth Covariance Function. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[5] J. Engel, V. Koltun, and D. Cremers. Direct sparse odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2017.

[6] C. Forster, M. Pizzoli, and D. Scaramuzza. SVO: Fast Semi-Direct Monocular Visual Odometry. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2014.

[7] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[8] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and Shi-Min Hu. Di-fusion: Online implicit 3d reconstruction with deep priors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[9] M. M. Johari, C. Carta, and F. Fleuret. ESLAM: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[10] M. Keller, D. Lefloch, M. Lambers, S. Izadi, T. Weyrich, and A. Kolb. Real-time 3D Reconstruction in Dynamic Scenes using Point-based Fusion. In Proc. of Joint 3DIM/3DPVT Conference (3DV), 2013.

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023.

[12] Leonid Keselman and Martial Hebert. Approximate differentiable rendering with algebraic surfaces. In Proceedings of the European Conference on Computer Vision (ECCV), 2022.

[13] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2015.

[14] Heng Li, Xiaodong Gu, Weihao Yuan, Luwei Yang, Zilong Dong, and Ping Tan. Dense rgb slam with neural implicit

maps. In Proceedings of the International Conference on Learning Representations (ICLR), 2023.

[15] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. NeurIPS, 2020.

[16] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. 3DV, 2024.

[17] J. McCormac, A. Handa, A. J. Davison, and S. Leutenegger. SemanticFusion: Dense 3D semantic mapping with convolutional neural networks. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2017.

[18] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.

[19] N. J. Mitra, N. Gelfand, H. Pottmann, and L. J. Guibas. Registration of Point Cloud Data from a Geometric Optimization Perspective. In Proceedings of the Symposium on Geometry Processing, 2004.

[20] Thomas Muller, Alex Evans, Christoph Schied, and Alexan-¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG), 2022.

[21] R. Mur-Artal and J. D. Tardos. ORB-SLAM2: An Open-´ Source SLAM System for Monocular, Stereo, and RGB-D Cameras. IEEE Transactions on Robotics (T-RO), 33(5): 1255–1262, 2017.

[22] R. Mur-Artal, J. M. M Montiel, and J. D. Tardos. ORB-´ SLAM: a Versatile and Accurate Monocular SLAM System. IEEE Transactions on Robotics (T-RO), 31(5):1147–1163, 2015.

[23] R. A. Newcombe. Dense Visual SLAM. PhD thesis, Imperial College London, 2012.

[24] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon. KinectFusion: Real-Time Dense Surface Mapping and Tracking. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2011.

[25] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[26] M. Nießner, M. Zollhofer, S. Izadi, and M. Stamminger.¨ Real-time 3D Reconstruction at Scale using Voxel Hashing. In Proceedings of SIGGRAPH, 2013.

[27] Victor Adrian Prisacariu, Olaf Kahler, Ming-Ming Cheng,¨ Carl Yuheng Ren, Julien P. C. Valentin, Philip H. S. Torr, Ian D. Reid, and David W. Murray. A framework for the volumetric integration of depth images. CoRR, abs/1410.0925, 2014.

[28] Tong Qin, Jie Pan, Shaozu Cao, and Shaojie Shen. A general optimization-based framework for local odometry estimation with multiple sensors, 2019.

[29] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R. Os-¨ wald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the International Conference on Computer Vision (ICCV), 2023.

[30] Thomas Schops, Torsten Sattler, and Marc Pollefeys. Sur-¨ felmeshing: Online surfel-based mesh reconstruction. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2020.

[31] Thomas Schops, Torsten Sattler, and Marc Pollefeys. Bad¨ slam: Bundle adjusted direct rgb-d slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[32] J. Sola, J. Deray, and D. Atchuthan. A micro Lie theory for\` state estimation in robotics. arXiv:1812.01537, 2018.

[33] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019.

[34] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. In Proceedings of the IEEE/RSJ Conference on Intelligent Robots and Systems (IROS), 2012.

[35] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison. iMAP: Implici mapping and positioning in real-time. In Proceedings of the International Conference on Computer Vision (ICCV), 2021.

[36] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[37] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[38] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. In Neural Information Processing Systems (NIPS), 2021.

[39] Emanuele Vespa, Nikolay Nikolov, Marius Grimm, Luigi Nardi, Paul HJ Kelly, and Stefan Leutenegger. Efficient octree-based volumetric SLAM supporting signed-distance and occupancy mapping. IEEE Robotics and Automation Letters (RAL), 2018.

[40] Angtian Wang, Peng Wang, Jian Sun, Adam Kortylewski, and Alan Yuille. Voge: a differentiable volume renderer using gaussian ellipsoids for analysis-by-synthesis. 2022.

[41] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[42] T. Whelan, M. Kaess, H. Johannsson, M. F. Fallon, J. J. Leonard, and J. B. McDonald. Real-time large scale dense RGB-D SLAM with volumetric fusion. International Journal of Robotics Research (IJRR), 34(4-5):598–626, 2015.

[43] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison. ElasticFusion: Dense SLAM without a pose graph. In Proceedings of Robotics: Science and Systems (RSS), 2015.

[44] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[45] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2022.

[46] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[47] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[48] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[49] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-slam: Neural implicit scene encoding for rgb slam. International Conference on 3D Vision (3DV), 2024.

[50] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. Ewa splatting. IEEE Transactions on Visualization and Computer Graphics, 8(3):223–238, 2002.