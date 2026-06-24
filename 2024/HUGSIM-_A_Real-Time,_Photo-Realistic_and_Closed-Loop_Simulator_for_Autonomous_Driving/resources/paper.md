# HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving

Hongyu Zhou

Longzhong Lin

Jiabao Wang

Yichong Lu Dongfeng Bai

Bingbing Liu

Yue Wang Andreas Geiger Yiyi Liao<sup></sup>

Abstract—In the past few decades, autonomous driving algorithms have made significant progress in perception, planning, and control. However, evaluating individual components does not fully reflect the performance of entire systems, highlighting the need for more holistic assessment methods. This motivates the development of HUGSIM, a closed-loop, photo-realistic, and real-time simulator for evaluating autonomous driving algorithms. We achieve this by lifting captured 2D RGB images into the 3D space via 3D Gaussian Splatting, improving the rendering quality for closed-loop scenarios, and building the closed-loop environment. In terms of rendering, We tackle challenges of novel view synthesis in closed-loop scenarios, including viewpoint extrapolation and 360-degree vehicle rendering. Beyond novel view synthesis, HUGSIM further enables the full closed simulation loop, dynamically updating the ego and actor states and observations based on control commands. Moreover, HUGSIM offers a comprehensive benchmark across more than 70 sequences from KITTI-360, Waymo, nuScenes, and PandaSet, along with over 400 varying scenarios, providing a fair and realistic evaluation platform for existing autonomous driving algorithms. HUGSIM not only serves as an intuitive evaluation benchmark but also unlocks the potential for fine-tuning autonomous driving algorithms in a photorealistic closed-loop setting.

Index Terms—Gaussian Splatting, Holistic Understanding, Dynamic Urban Scenes, Simulator, Autonomous Driving

## 1 INTRODUCTION

ing route to achieving full autonomy at low cost. Nonetheless, rigorously testing a vision-based AD system in real-world environments poses significant challenges, primarily due to safety concerns and the inherent inefficiencies of such evaluations. Moreover, the risk associated with assessing safety-critical scenarios in real settings is substantial, yet these situations are precisely where AD algorithms must demonstrate their reliability and effectiveness.

Existing methods [8], [26], [44], [58], [69] primarily assess vision-based AD algorithms using collected datasets, offering open-loop evaluations of specific components like perception and planning. However, this approach falls short of capturing closed-loop performance [15], [16], [17], as it evaluates AD algorithms solely in pre-collected, safe and normal driving environments conducted by experienced drivers. This limitation underscores the urgent need for advanced simulators capable of conducting closed-loop evaluations of vision-based AD systems, enabling the safe exploration of out-of-domain and safety-critical scenarios.

To tackle this issue, closed-loop AD simulators have been investigated. Some studies [9], [27], [66] implement a closed-loop system for planning and control, assuming ground truth perception results are available, neglecting the importance of perception. Other works [19], [42] offer an end-to-end closed-loop system based on game engines, resulting in video-game-style simulators. Although these simulators provide numerous features for testing and training AD algorithms, significant domain gaps still exist between the real world and video games. Another downside of these simulators is the considerable time and manpower investment in manually creating assets and scenarios.

In this paper, we bridge the gap between real-world evaluation and simulation by building a novel photorealistic AD simulator leveraging advanced novel view synthesis techniques. Although many previous works [39], [70], [83], [84] tackle novel view synthesis of urban scenes, there remains a significant gap between urban novel view synthesis and an AD simulator.

First, a closed-loop simulator with interactive actors poses new challenges for the rendering process: 1) The trajectories of dynamic objects are originally observed from discrete timestamps, whereas a closed-loop simulator requires continuous trajectories. 2) Unlike typical novel view synthesis tasks, which evaluate on interpolated views, simulators require photo-realistic rendering even for extrapolated views, especially for lane areas. 3) Since actors are observed under unpredictable viewpoints due to the interaction between the ego vehicle and the actors, we should ensure that actors appear realistic from 360-degree viewpoints. Moreover, there are extra implementation steps for building a full-fledged closed-loop simulator. For instance, a communication bridge between the simulator and AD algorithms is necessary; waypoints retrieved from AD algorithms must be converted into control commands to update the ego vehicle’s position and actor trajectories need to be generated efficiently.

Recently, a concurrent work called NeuroNCAP [45] introduced, to the best of our knowledge, the first publicly available photorealistic closed-loop simulator that offers high-fidelity rendering. However, NeuroNCAP is not specifically designed to address the challenges of viewpoint extrapolation, 360-degree high-fidelity actor rendering, or the efficient simulation of interactive safety-critical scenarios.

![](images/074c3d20dd04d8d48f67080733cc3386c0c0464e2800d2a89d203baa104add98.jpg)  
Fig. 1: Overview of HUGSIM. We propose HUGSIM, a photorealistic closed-loop simulator for AD, supporting a variety of scenes from different datasets and scenarios with varying levels of difficulty. HUGSIM also provides a closed-loop benchmark and presents challenges for existing AD algorithms.

In this paper, we propose HUGSIM, a novel vision-based AD simulator to address the aforementioned challenges, see Fig. 1. We start by extending 3D Gaussian Splatting (3DGS) [37] to model dynamic 3D scenes with multiple modalities, from RGB images and noisy 2D and 3D predictions. This approach enables holistic scene reconstruction, including appearance, semantics, flow, depth, and motion of rigidly moving objects. Note that semantic information is essential for collision detection, while flow and depth can be helpful for AD algorithms [82]. Importantly, we apply physical constraints to both dynamic vehicles and the ground to enhance rendering quality. Specifically, we regularize the trajectory of dynamic vehicles using a unicycle model, allowing for improving the rendering quality of dynamic vehicles given noisy 3D bounding box predictions. We further introduce a multi-plane ground model to address lane distortion in extrapolated views, which works across different types of ground, including sloped surfaces. In addition, we insert non-native actors reconstructed from 3DRealCar [20], ensuring both faithful geometry and appearance from arbitrary observation views, in contrast to partially observed native vehicles in typical driving scenes. Moreover, we close the simulation loop by querying waypoints from AD algorithms, applying a linear quadratic regulator (LQR) for control, extracting the multiple ground planes, and updating the pose of ego vehicles and actors. By leveraging HD maps, we deploy the Intelligent Driver Model (IDM) planning strategy [62] to simulate normal actor behavior. To simulate safety-critical scenarios, we design an attack planning strategy to generate aggressive actor behavior, which works effectively even in scenes without HD maps.

Additionally, we encapsulate the entire scene into an interactive closed-loop simulator and provide Gymnasium APIs [6] to facilitate AD evaluation and the potential application of reinforcement learning in our simulator. Our simulator operates in real-time, as it introduces minimal computational overhead compared to 3DGS. It supports scenes from KITTI-360 [44], Waymo [58], nuScenes [8], and

PandaSet [69]. To fairly evaluate different AD algorithms, we propose an evaluation metric named HD-Score, inspired by NAVSIM [18] and DriveArena [73]. The HD-Score reflects AD performance based on NC (No Collision), DAC (Drivable Area Compliance), T T C (Time to Collision), COM (Comfort), and $\bar { R } _ { c }$ (Route Completion).

We summarize the contributions of this paper as follows:

We present a novel simulator for autonomous driving, characterized by being closed-loop, photorealistic, and real-time, bridging the gap between urban scene novel view synthesis and autonomous driving simulators.

To address the specific rendering challenges in the simulator, we leverage physical constraints and nonnative actors to improve fidelity, surpassing previous novel view synthesis approaches.

We propose an efficient actor trajectory generation strategy, enabling the simulation of aggressive driving behaviors even without HD maps.

We introduce a new benchmark for fair evaluation of AD algorithms, offering more photorealistic simulation environments compared to existing closed-loop vision-based AD simulators. Additionally, the benchmark provides a variety of scenes built upon multiple datasets, with varying scenarios with different levels of difficulty.

This journal paper is an extension of a conference paper published at CVPR 2024, HUGS [83]. In comparison to HUGS, we 1) extend the novel view synthesis task by constructing a closed-loop simulator; 2) design an efficient strategy for simulating safety-critical scenarios without relying on HD maps; 3) propose a multi-plane ground Gaussian model and real captured vehicle models to enhance appearance fidelity; 4) introduce a novel benchmark for fair evaluation of vision-based AD algorithms.

## 2 RELATED WORK

In this section, we first review methods in urban scene reconstruction, a core task for enabling AD simulators. Then we discuss existing datasets, benchmarks and simulators used for training or evaluating AD algorithms. We compare various features of representative relative work in Table 1.

<table><tr><td></td><td></td><td></td><td>Closed-Loop RGB MV-Consistency P-Realistic FV-Actors IA-Actors Ext-Views Real-Time</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MARS [68]</td><td>X</td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>StreetGaussian [70]</td><td>X</td><td></td><td></td><td></td><td>X</td><td>X</td><td>x</td><td></td></tr><tr><td>KITTI [26]</td><td></td><td></td><td></td><td></td><td>X</td><td></td><td>X</td><td></td></tr><tr><td>NAVSÍM [18]</td><td></td><td></td><td></td><td></td><td>x</td><td></td><td>X</td><td></td></tr><tr><td>nuplan [9]</td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td>X</td><td></td></tr><tr><td>Carla [19]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NeuroNCAP [45]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DriveArena [73]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HUGSIM (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE 1: Comparison of Related Work, including autonomous driving-related reconstruction methods open-loop benchmarks , and closed-loop benchmarks . We compare various features like closed-loop evaluation (Closed-Loop), RGB images (RGB), multi-view image consistency (MV-Consistency), photo-realistic rendering (P-Realistic), fullview actors (FV-Actors), interactive actors (IA-Actors), fidelity of extrapolated views (Ext-Views), and real-time simulation (Real-Time) capabilities. Note that only one representative work is shown for overlapping features.

## 2.1 Urban Scene Reconstruction

Static Scenes: Numerous studies have been conducted to reconstruct urban scenes using various methods. These methods can be categorized into four classes: point-based [1], [55], mesh-based [24], [40], NeRF-based [28], [46], [47], [49], [53], [59], [67], [80] and 3DGS-based [29], [41], [56]. While point-based and mesh-based methods demonstrate faithful reconstructions, they struggle to recover all aspects of the scene, especially when it comes to high-quality appearance modeling. In contrast, NeRF-based models not only reconstruct scene appearance but also enable highquality rendering from novel viewpoints, while 3DGS-based models support real-time rendering without sacrificing quality. However, these approaches are primarily designed for static scenes, lacking the ability to handle dynamic urban environments. In this study, our focus lies in addressing the challenges of dynamic urban scenes.

Dynamic Scenes: Several methods have also been developed to address the reconstruction of dynamic urban scenes. Many of these approaches rely on the availability of accurate 3D bounding boxes for moving objects in order to separate the dynamic elements from the static components, as seen in [22], [52], [61], [68], [74], [84]. PNF [39], NeuRAD [61] and StreetGaussians [70] takes a different approach by leveraging monocular-based or LiDAR-based 3D bounding box predictions and proposes a joint optimization of object poses during the reconstruction process. However, our experimental observations indicate that the straightforward optimization of object poses yields unsatisfactory results due to the absence of physical constraints. Another method, SUDS [63], avoids the use of 3D bounding boxes by grouping the scene based on learned feature fields. However, the accuracy of this approach lags behind. In parallel, EmerNeRF [71] and PVG [12] follows a similar idea to SUDS by decomposing the scene purely into static and dynamic components. In our research, we have the capability to further decompose individual dynamic objects within the scene and estimate their motion. This reconstructed motion can then be used to simulate driving behavior within the simulator.

Extrapolated View Rendering: Although the urban reconstruction methods mentioned above can render highfidelity images in interpolated views, most struggle with rendering in extrapolated views, where artifacts particularly on the lanes are common. Some methods, such as [64], [77], leverage diffusion models to add additional views for supervision, addressing the problem of sparse input views in urban scene reconstruction. However, diffusion models introduce challenges such as inconsistency between views, increased training complexity, and a slowdown in training speed due to the heavy computational burden of these models. GGS [29] presents a generalizable model based on MVSplat [11], incorporating a virtual lane generation strategy during training to address the extrapolated view problem. While this approach significantly improves fidelity in extrapolated views, MVSplat allows only a few frames as input, which may restrict scalability and lead to multiview misalignment. In comparison, the concurrent works AutoSplat [38] and RoGS [21] apply physical priors to constrain ground Gaussians, similar to our approach. However, AutoSplat relies on LiDAR data for initialization, and RoGS uniformly distributes Gaussians across the ground plane, with both approaches fixing Gaussian positions on the plane. These methods use a large number of Gaussians to model non-textured areas, which we find unnecessary and inefficient. By optimizing additional parameters such as position and scale, HUGSIM achieves better performance without requiring as many Gaussians.

The related work mentioned above primarily focus on novel view synthesis but still fall short of being fully developed closed-loop simulator, as shown in Table 1.

## 2.2 Benchmarks

Open-Loop: Most existing datasets and benchmarks [8], [26], [44], [58], [69] for AD algorithms follow an openloop approach, evaluating individual components of the algorithms separately. For instance, in the perception component, tasks such as semantic segmentation, bounding box detection, and lane detection are assessed, while the planning component evaluates tasks like route planning, behavior prediction, and trajectory prediction. Although these open-loop benchmarks provide detailed and convincing performance evaluations of each part, they evaluate the performance of AD algorithms using sensory data collected by experts, lacking scenarios deviated from the expertcollected sensory data. Without closed-loop feedback, the long-term consequences of these deviations remain unexplored, which could be critical for understanding the robustness and safety of AD systems in real-world conditions. NAVSIM [18] provides a benchmark positioned between open-loop and closed-loop evaluations. Although it is a nonreactive simulator and lacks the capability for novel view synthesis, it computes closed-loop metrics by forecasting the planning trajectory for a few seconds into the future. However, NAVSIM is limited to short time horizons and does not address how accumulated deviations over time impact driving safety.

![](images/4f4b2e9ea98c19615eb91777067e0a31b95b717d53000435191f484bf326d89e.jpg)  
Fig. 2: The Pipeline of HUGSIM. We reconstruct urban scenes by disentangling them as ground, non-ground static background, and dynamic objects, while also enabling multi-modal rendering. Subsequently, we implement a closed-loop simulator based on the reconstructed results, providing a benchmark for evaluating autonomous driving algorithms.

Closed-Loop: A number of closed-loop simulators have attempted to address the limitations of open-loop benchmarks in autonomous driving. Some simulators [9], [27], [66] provide ground-truth positions and rotations of other vehicles, lacking the perception aspect of the AD system, which is crucial for a comprehensive evaluation. Other works [19], [42] use game engines to create simulators, offering high edit-ability but often lacking photorealism and requiring extensive manual design of scenes, making them costly and less scalable. Rapid advances in video diffusion models [25], [31], [65], [72] have shown promise in generating photorealistic driving videos. DriveArena [73] uses a video diffusion model to build a closed-loop simulator, controlling scene generation through scene layouts. However, these models still suffer from issues like temporal inconsistency and significant computational demands. UniSim [74] and NeuroNcap [45] takes a different approach by creating a NeRF-based simulator that enables photorealistic, closedloop simulations. However, UniSim is not an open-source work, while NeuroNCAP has several shortcomings, including non-real-time rendering, sub-optimal quality in extrapolated views, and fully manual actor behavior design. In contrast, HUGSIM addresses these challenges by offering realtime performance, improved fidelity in extrapolated views, and efficient, automated generation of actor behaviors.

## 3 URBAN SCENE RECONSTRUCTION

In this section, we begin by introducing the preliminaries. We then present our decomposed urban scene representation, designed to address rendering challenges in visionbased AD simulation. Next, we describe our holistic urban Gaussian splatting approach, covering appearance, semantics, optical flow, and depth. Finally, we detail the loss functions employed in our scene reconstruction.

## 3.1 Preliminaries

3D Gaussian Splatting: 3DGS [37] represents scenes using a set of anisotropic 3D Gaussians with differentiable properties, allowing for efficient image rendering through tile-based rasterization. Each 3D Gaussian is defined by its position $\mu ,$ rotation R (also represented as quaternion q during training), scale S (comprising $s _ { x } , s _ { y } , s _ { z } )$ , opacity $\alpha ,$ and spherical harmonic coefficient SH. The 3D covariance matrix $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ of each Gaussian is defined as:

$$
\pmb { \Sigma } = R S S ^ { T } \mathbf  \} R ^ { T }\tag{1}
$$

A 3D Gaussian is defined as follow:

$$
G ( \mathbf { x } ) = \alpha \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \boldsymbol { \mu } ) ^ { T } \Sigma ^ { - 1 } ( \mathbf { x } - \boldsymbol { \mu } ) \right)\tag{2}
$$

The color c of each Gaussian can be computed based on the view direction and its corresponding spherical harmonics (SH). Given a set of sorted 3D Gaussians N along the ray, we obtain the accumulated color via volume rendering:

$$
\pi : \quad \mathbf { C } = \sum _ { i \in \mathcal { N } } \mathbf { c } _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{3}
$$

Here $\alpha ^ { \prime }$ is determined by the projected 2D Gaussian and the 3D opacity α following [85], see the appendix for details.

Coordinate System Definition: Given a sequence of urban images, we define the world coordinates based on the first frame throughout this paper. Specifically, the origin is set at the front camera of the first frame. The coordinate axes are aligned with OpenCV conventions [5], with the x axis pointing to the right, the y axis pointing downward, and the z axis pointing forward.

![](images/79e5dfdc24c87e1c6c090651a4f2d2a45eb006f29e4094d5d99d4b60da27460c.jpg)

Fig. 3: Lane Distortion In the left image of the extrapolated view, the lane appears highly distorted due to incorrect geometry of ground gaussians. In contrast, the right image, which applies the multi-plane ground model, shows significantly fewer artifacts in the same view.  
![](images/e3d69b78500d549aba3809e13e4369c2e7459b7b70ac148ef7508bfbe81f8e73.jpg)  
Fig. 4: The Multi-Plane Ground Model. Our ground model is initialized by multiple planes, each containing a set of flat 3D Gaussians. During training, we sample small patches of these Gaussians and constrain the height variance.

## 3.2 Decomposed Scene Representation

We assume the scene consists of static regions and dynamic vehicles exhibiting rigid motion. Static regions are decomposed into ground and non-ground regions, allowing the application of planar constraints to the ground to preserve lane structure in extrapolated views. We consider two categories of dynamic vehicles: native dynamic vehicles present in the original driving dataset and non-native dynamic vehicles reconstructed from 360-degree captured images.

Non-Ground Static Gaussians: Following 3DGS [37], we model all regions of urban scenes using 3D Gaussians. In addition to the original definition of 3D Gaussians, We propose to additionally model semantic logits $\mathbf { s } \in \mathbb { R } ^ { S }$ of each 3D Gaussian, allowing for rendering 2D semantic labels. Furthermore, we can naturally obtain a rendered optical flow $\mathbf { f } _ { t _ { 1 }  t _ { 2 } } \in \mathbb { R } ^ { 2 }$ for each 3D Gaussian by projecting the 3D position $\mu$ to the image space at two different timestamps, $t _ { 1 }$ and $t _ { 2 } ,$ and calculating the motion. We provide details of the multi-modality rendering in Section 3.3.

Ground Gaussians: Lanes play a crucial role in the perception of AD algorithms. However, most existing reconstruction methods struggle to accurately render lane geometry in extrapolated views, as shown in Fig. 3. The cause of these distortions is that ground Gaussians tend to overfit to the training views, failing to reconstruct the correct ground geometry. Our preliminary experiments show that directly supervising the render depth fails to solve the problem, as Gaussians with incorrect geometry can still render accuratelooking 2D depth maps. Insteand, we regularize ground Gaussians to form planar structures, yielding correct geometry as shown in Fig. 4.

A naive assumption would be to consider the ground of scenes as a single planar surface, allowing ground Gaussians to be distributed at the same height. However, this assumption overlooks more complex scenarios, such as sloped roads. To tackle this problem, we propose a multiplane ground model, where we assume the ground to be planar only within a limited distance, denoted as $\Delta Z .$ In this model, each local plane is assumed to have a fixed height relative to the nearest camera. Since the camera poses reflect the surface slopes, this multi-plane approach effectively models such complex scenarios. Specifically, we optimize ground Gaussians and constrain the distribution of Gaussians in 3D space by limiting the height variance of sampled Gaussian patches within a small $\Delta Z$ in corresponding camera coordinates. Note that the local planes overlap with each other, hence avoiding boundary artifacts. More formally, the constraints of our ground model can be expressed as the optimization target:

$$
\begin{array} { l l } { \displaystyle \operatorname* { m i n i m i z e } _ { \{ \mu _ { x , y , z } , s _ { x , z } , \mathbf { c } , \alpha \} } } & { ( 1 - \lambda _ { S S I M } ) \| \hat { \mathbf { I } } - \tilde { \mathbf { I } } \| _ { 1 } + \lambda _ { S S I M } \mathrm { S S I M } ( \hat { \mathbf { I } } , \tilde { \mathbf { I } } ) } \\ { \displaystyle \mathrm { s u b j e c t ~ t o } \quad } & { \displaystyle \operatorname* { l i m } _ { \Delta Z \to 0 } \sqrt { \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } ( \mu _ { y _ { i } } ^ { c a m } - \bar { \mu } _ { y } ^ { c a m } ) ^ { 2 } } = 0 } \end{array}\tag{4}
$$

where ${ \hat { \mathbf { I } } } ,$ <sup>˜</sup>I represent the rendered image and the ground truth image, respectively. N denotes the number of Gaussians in one local plane. $\mu _ { y _ { i } } ^ { c a m }$ and $\bar { \mu } _ { y } ^ { c a m }$ represent heights of 3D Gaussians in camera coordinates and average heights in a patch. Note that the $s _ { y }$ and q of the Gaussians are kept constant to ensure they remain flat and oriented upward.

Unlike previous approaches that use densely tiled or LiDAR-initilaized Gaussians, as seen in RoGS [21] and AutoSplat [38], we find that the ground can be efficiently represented using sparser distributed Gaussians since the ground textures are not uniformly distributed. We therefore retain color, position, opacity, two dimensions of scale as optimizable parameters, while also incorporating the density control strategy [37]. Our approach enables high-quality ground rendering without the need for an excessive number of 3D Gaussians, as demonstrated in our experiments.

Native Dynamic Vehicle Gaussians and Unicycle Model: For dynamic vehicles, we assume 3D bounding boxes are predicted from the input RGB images, enabling 3D Gaussian modeling in object coordinate space. To address noise in predictions, we jointly optimize them by regularizing with a unicycle model. Specifically, we parameterize the transformations $( \mathbf { R } _ { t } , \mathbf { t } _ { t } )$ of each dynamic vehicle following the unicycle model<sup>1</sup>. The state of a unicycle model is parameterized by three elements: $\left( x _ { t } , z _ { t } , \theta _ { t } \right)$ , where $x _ { t }$ and $z _ { t }$ represent horizontal coordinates of t with $\mathbf { t } _ { t } = [ x _ { t } , y _ { t } , z _ { t } ]$ , and $\theta _ { t }$ is the yaw angle of $\mathbf { R } _ { t } .$ . To adapt the continuous unicycle model to discrete frames, we derive the calculus of the unicycle model for the vehicle transition from timestamp t to t + 1 as follows:

$$
\begin{array} { l } { x _ { t + 1 } = x _ { t } + \frac { v _ { t } } { \omega _ { t } } ( \sin \theta _ { t + 1 } - \sin \theta _ { t } ) } \\ { z _ { t + 1 } = z _ { t } - \frac { v _ { t } } { \omega _ { t } } ( \cos \theta _ { t + 1 } - \cos \theta _ { t } ) } \\ { \theta _ { t + 1 } = \theta _ { t } + \omega _ { t } } \end{array}\tag{5}
$$

1. While it is more accurate to model vehicles using a bicycle model, we observe that using the simpler unicycle model is sufficient here.

![](images/0b8b99c03a0d6654f2d26aa49816856ca2d8b37e160f98180c88b285356aa798.jpg)  
Fig. 5: Non-Native Vehicle Reconstruction We reconstruct more than 100 vehicles from 3DRealCar [20] to enable 360-degree high-fidelity actor observation. Additionally, we place shadow Gaussians with gradually changing opacity inside a rounded rectangle to enhance the visual realism of the inserted actor.

Here, $v _ { t }$ represents the forward velocity, and $\omega _ { t }$ is the angular velocity. This model integrates physical constraints when compared to directly optimizing the transformations of dynamic vehicles at every frame independently, thus enabling smoother motion modeling of moving objects and making them less prone to local minima.

While it is possible to define an initial state $( x _ { 1 } , z _ { 1 } , \theta _ { 1 } )$ and derive the following states recursively based on velocities, $v _ { t }$ and $\omega _ { t } ,$ such a recursive parameterization is challenging to optimize. In practice, we define a set of trainable states $\{ ( x _ { t } , \dot { z } _ { t } , \theta _ { t } ) \} _ { t = 1 } ^ { T }$ along with trainable velocities $\{ v _ { t } , \omega _ { t } \} _ { t = 1 } ^ { T - 1 }$ , and add a regularization term to ensure that the vehicle’s states adhere to the characteristics of a unicycle model in Eq. 5. The regularization terms will be described in Section 3.4. Additionally, we retrieve the vertical locations of the vehicle $\{ y _ { t } \} _ { t = 1 } ^ { T }$ from our multi-plane ground model. During simulation, we interpolate the states from Eq. 5 by given timestamps.

Non-Native Full-Observed Vehicle Gaussians: The AD simulator requires rendering high-fidelity actors from all 360 degrees, particularly when integrating interactive actors into closed-loop simulations. However, vehicles in the original reconstructed scenes are only captured from a limited set of viewpoints, resulting in noticeable artifacts when viewed from angles far outside the training perspectives. To address this, we reconstruct vehicles using a densely captured realworld dataset, 3DRealCar [20], which provides 360-degree observations of real-world vehicles. Our experiments show that real-world captured vehicles outperform vehicles from original scenes when inserted into the simulation scenes with random viewing angles.

The 3DRealCar dataset provides masks of the vehicles. We leverage the mask information to ensure that the 3D Gaussians only model the car foreground. This is achieved by considering an alpha mask loss in addition to the vanilla rendering loss. Importantly, directly inserting the foreground vehicles without shadows often appear as if they are floating in the air. However, inverse rendering requires an accurate environment map, which is difficult to obtain from perspective cameras. Although some work [43], [50], [51], [57] has addressed the challenge of inverse rendering in Gaussian Splatting, it remains a computationally expensive operation. To simplify the problem, we assume the light source (the sun) is directly overhead, meaning shadows should appear beneath the vehicles. To render vehicle shadows, we place flat Gaussians at the bottom of the vehicle in canonical space, as shown in Fig. 5. The α attribute of these Gaussians decreases smoothly based on their distance from the bottom center. Although this is a simplified assumption, we observe that the inserted nonnative vehicles appear plausible in many scenarios, striking a good balance between efficiency and photorealism.

![](images/0b5442c2e4f552063b8e494f76aa600ce267f35efd9439f67a470b975e8ebe10.jpg)  
Fig. 6: 3D Semantic Reconstruction. Comparison between applying softmax to accumulated 2D semantic logits (left) and to 3D semantic logits (right). Normalizing semantic logits in 3D space clearly reduces floaters and yields better 3D semantic reconstruction than the 2D normalization counterpart. This improvement is also crucial for our simulator collision detection.

## 3.3 Holistic Urban Gaussian Splatting

Given the representation specified above, we are able to render images, semantic maps and optical flow to supervise the model or make predictions at inference time. We now elaborate on the rendering of each modality.

Novel View Synthesis: The combination of static and dynamic Gaussians can be sorted and projected onto the image plane via α-blending π in Eq. 3.

In contrast to single-object scenes, urban scenes typically involve more complex lighting conditions and the images are usually captured with auto white balance and auto exposure. NeRF-based methods [47] typically feed a per-frame appearance embedding along with the 3D positions into a neural network to compute the color, thereby compensating exposure. However, when working with 3D Gaussians, there is no neural network capable of processing appearance embeddings. Inspired by Urban Radiance Field [53], we generate an exposure affine matrix for each camera by mapping the camera’s extrinsic parameters to an affine matrix $\overset { \cdot } { \mathbf { A } } \in \mathbb { R } ^ { 3 \times 3 }$ and vector $\mathbf { b } \in \mathbb { R } ^ { 3 }$ via a small MLP:

$$
\tilde { \mathbf { C } } = \mathbf { A } \times \mathbf { C } + \mathbf { b }\tag{6}
$$

We demonstrate that modeling the exposure improves rendering quality in the experimental section.

Semantic Reconstruction: Similarly to Eq. 3, we can obtain 2D semantic labels via α-blending based on the 3D semantic logit s:

$$
\pi _ { \mathbf { S } } : \quad \mathbf { S } = \sum _ { i \in \mathcal { N } } \mathrm { s o f t m a x } ( \mathbf { s } _ { i } ) \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{7}
$$

Note that we perform the softmax operation on 3D semantic logits s<sub>i</sub> prior to α blending, in contrast to most existing methods that apply softmax to 2D semantic logits S<sup>¯</sup> obtained by accumulating unnormalized 3D semantic logits $\mathbf { s } _ { i }$ [23], [39], [81]. As shown in Fig. $^ { 6 , }$ applying softmax in 2D space leads to noisy 3D semantic labels. This is due to the fact that 2D space softmax can produce accurate 2D semantics by adjusting the scale of the 3D semantic logits, allowing a single sampled point with a substantial logit value to significantly influence the volume rendering outcome. For example, an undesired floating point labeled with “car” may not be penalized despite the target rendered label being “tree”, as long as a 3D Gaussian is providing a large logit value of “tree” along this ray. Our solution instead removes such floaters by normalizing logits in 3D space. See the appendix for more quantitative and qualitative details. The removal of these floaters is also essential for building our simulator, as the background collision detection is implemented based on Gaussian semantics and geometry.

Optical Flow: The 3D Gaussian representation also enables the rendering of optical flow. Given two timestamps $t _ { 1 }$ and $t _ { 2 } ,$ we first calculate the optical flow of each 3D Gaussian’s center $\mu$ as $\mathbf { f } _ { t _ { 1 }  t _ { 2 } }$ . Specifically, we project $\mu$ to the 2D image space based on the camera’s intrinsic and extrinsic parameters:

$$
\begin{array} { r } { \mu _ { 1 } ^ { \prime } = \mathbf { K } [ \mathbf { R } _ { t _ { 1 } } ^ { \mathrm { c a m } } ; \mathbf { t } _ { t _ { 1 } } ^ { \mathrm { c a m } } ] \boldsymbol { \mu } , \quad \mu _ { 2 } ^ { \prime } = \mathbf { K } [ \mathbf { R } _ { t _ { 2 } } ^ { \mathrm { c a m } } ; \mathbf { t } _ { t _ { 2 } } ^ { \mathrm { c a m } } ] \boldsymbol { \mu } , } \end{array}\tag{8}
$$

and then calculate the motion vector as $\mathbf { f } _ { t _ { 1 }  t _ { 2 } } = \mu _ { 2 } ^ { \prime } - \mu _ { 1 } ^ { \prime }$ Next, we render the optical flow via accumulate the optical flows via volume rendering:

$$
\pi _ { \mathbf { F } } : \quad \mathbf { F } = \sum _ { i \in \mathcal { N } } \mathbf { f } _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{9}
$$

Note that this rendering process assumes that any pixel of a rendered 2D Gaussian splat shares the same optical flow direction as the corresponding Gaussian center but with scaled magnitude. While this is indeed a simplified approximation, we observe this to work well in practice.

Depth: Depth maps are also provided during rendering. Each Gaussian’s depth distance relative to the observed camera is defined as ${ \bf d } _ { i }$ . The depth map is rendered by accumulating d<sub>i</sub> using volume rendering:

$$
\pi _ { \mathbf { D } } : \quad \mathbf { D } = \sum _ { i \in \mathcal { N } } \mathbf { d } _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{10}
$$

## 3.4 Loss Functions

Our loss functions are divided into two parts. The first is image-based losses, which aim to fit the observed ground truth or pseudo-ground truth 2D images. The second is geometry-based losses, which constrain the reconstructed result to align with physical priors from the real world.

We leverage the pre-trained perception model [4] to provide noisy 2D semantic. To compensate for the absence of manually labeled 3D bounding boxes, we use pre-trained recognition models [32] to provide noisy 3D tracking results. These easy-to-obtain predictions are crucial for enabling RGB-only holistic scene understanding in both 2D and 3D space, without relying on LiDAR input or 3D semantic supervision.

Image-Based Losses: Our model is supervised with the ground truth images using a combination of L1 and SSIM losses. Our rendering loss is defined as follows:

$$
\mathcal { L } _ { \mathbf { I } } = \big ( 1 - \lambda _ { S S I M } \big ) \| \hat { \mathbf { I } } - \tilde { \mathbf { I } } \| _ { 1 } + \lambda _ { S S I M } \mathrm { S S I M } ( \hat { \mathbf { I } } , \tilde { \mathbf { I } } )\tag{11}
$$

where <sup>ˆ</sup>I and <sup>˜</sup>I represent the rendered and the ground truth image, respectively. We additionally apply the cross-entropy loss to the rendered semantic label wrt. pseudo-2D semantic segmentation ground truth $\hat { \bf S } \colon$

$$
\mathcal { L } _ { \mathbf { S } } = - \sum _ { k = 0 } ^ { S - 1 } \hat { \mathbf { S } } _ { k } \log ( \mathbf { S } _ { k } )\tag{12}
$$

During the reconstruction of the non-native vehicles, we apply an alpha loss to prevent Gaussians from representing the masked background and to ensure the reconstructed vehicle appears as a non-translucent entity:

$$
\mathcal { L } _ { A } = \| \mathcal { A } - \tilde { \mathbf { I } } _ { \mathcal { M } } \| _ { 2 }\tag{13}
$$

where $\tilde { \mathbf { I } } _ { \mathcal { M } }$ denotes the ground truth mask and A the rendered alpha map.

Physical-Based Regularizations: A 3D regularization loss is applied to the multi-plane ground model to limit the variance in the height of the Gaussians, forcing the model learns an almost flat ground patch. We express the optimization problem $\operatorname { E q }$ . 4 as the multi-plane ground model regularization loss during training:

$$
\mathcal { L } _ { g r o u n d } = \frac { 1 } { N - 1 } \sum _ { z _ { i } - z _ { 0 } < \Delta z } ( \mu _ { y _ { i } } ^ { c a m } - \bar { \mu } _ { y } ^ { c a m } ) ^ { 2 }\tag{14}
$$

We use a unicycle model to guide the noisy 3D bounding box predictions:

$$
\mathcal { L } _ { \mathbf { t } } = \sum _ { t } \| x _ { t } - \hat { x } _ { t } \| _ { 2 } + \sum _ { t } \| z _ { t } - \hat { z } _ { t } \| _ { 2 }\tag{15}
$$

where $\hat { x } _ { t }$ and $\hat { y } _ { t }$ are the $x$ and $y$ locations of a noisy 3D bounding box at timestamp t.

As mentioned earlier, we parameterize the vehicle’s states $( x _ { t } , y _ { t } , \theta _ { t } )$ and the velocities v<sub>t</sub>, ω<sub>t</sub> as learnable parameters. Hence, we add the following regularization to make the states adhere to the unicycle model as follows:

$$
\begin{array} { c } { \displaystyle \mathcal { L } _ { u n i } = \sum _ { t } \| x _ { t + 1 } - x _ { t } - \frac { v _ { t } } { \omega _ { t } } ( \sin \theta _ { t + 1 } - \sin \theta _ { t } ) \| + } \\ { \displaystyle \sum _ { t } \| z _ { t + 1 } - z _ { t } + \frac { v _ { t } } { \omega _ { t } } ( \cos \theta _ { t + 1 } - \cos \theta _ { t } ) \| + } \\ { \displaystyle \sum _ { t } \| \theta _ { t + 1 } - \theta _ { t } - \omega _ { t } \| } \end{array}\tag{16}
$$

![](images/72fd583ae295a10fc85b4e02fc17bdf774b6c11860b87cdc62c13d375c01ea14.jpg)  
Fig. 7: The Aggressive Driving Behavior Model. From (a) to (e), we illustrate the behavior of an aggressive actor (orange) attempting to collide the ego-vehicle (green) over a time period. In (a), the attacker selects the trajectory with the highest probability of collision by estimating the future trajectory of the ego vehicle based on its current state. In (c), the attacker re-estimates and updates the trajectory, successfully colliding with the ego vehicle in (e).

In addition, we regularize the acceleration of the forward velocity $v _ { t }$ and angular velocity $\omega _ { t }$ to be smooth:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { { r e g } } = \sum _ { t } \| v _ { t + 1 } + v _ { t - 1 } - 2 v _ { t } \| _ { 2 } + } \\ { \displaystyle \sum _ { t } \| \theta _ { t + 1 } + \theta _ { t - 1 } - 2 \theta _ { t } \| _ { 2 } } \end{array}\tag{17}
$$

## 4 SIMULATION

In this section, we first elaborate on the construction of a closed-loop simulator based on the reconstruction of the previous section. Next, we present our method for efficiently generating actor behaviors, including both normal and aggressive driving patterns. Finally, we introduce the evaluation metrics used to assess AD algorithms within our simulator.

## 4.1 Graphicial Configuration Interface

We develop a graphicial user interface (GUI) to facilitate the configuration of testing scenarios in our simulator. The GUI configuration involves several steps. The first step is configuring the camera settings, including the number of cameras, camera intrinsics, and and the extrinsic parameters wrt. the vehicle. The second step is configuring the ego vehicle parameters, including specifying the kinematic model, the control frequency, and the start state of the ego vehicle. The final step involves configuring the actors, including native and non-native vehicles with different behaviors specified in Section 4.3. The appearance of all these actors can be selected from a pool of more than 100 candidate 3D vehicles reconstructed from 3DRealCar [20].

## 4.2 Closed-Loop Simulation

Simulator-User communication: Given the 3D Gaussian reconstruction results, we encapsulate the scenes as Gymnasium environments [6], providing functions such as environment resetting, retrieving camera observations, accessing information about the ego vehicle and environment, and advancing to the next state with specified control actions for the ego vehicle. Ideally, the simulator should support evaluating various AD algorithms without requiring code adaptation. To achieve this, we maintain separate repositories for the simulator and AD algorithms, implementing them independently while enabling parallel execution and communication. When both the simulator and AD algorithms are run on the same machine, we use named pipes to establish a communication bridge, minimizing communication costs. In other cases, web sockets are available for communication.

Controller: By providing observations and ego vehicle information to the AD algorithms, our simulator expects feedback in the form of either planned waypoints or a sequence of control commands for the next several seconds. When the feedback is planning waypoints, these waypoints need to be converted into a sequence of control commands, including steering angle (δ) and acceleration (a˙ ). To achieve this conversion, we leverage a Linear Quadratic Regulator (LQR) control following [36], [45], [73].

Ego-Vehicle Kinematic Model: Given the control commands of steering angle (δ) and acceleration (a˙ ), the next states S of the ego vehicle can be computed by applying a discrete version of the kinematic bicycle model, following [36], [54]. In this model, L denotes the length of the ego vehicle:

$$
\mathcal { S } = \left( \begin{array} { c } { x } \\ { y } \\ { \theta } \\ { v } \end{array} \right) , \frac { d \mathcal { S } } { d t } = \left( \begin{array} { c } { v \cos \theta } \\ { v \sin \theta } \\ { \frac { v \tan \delta } { \mathbf { L } } } \\ { \dot { a } } \end{array} \right)\tag{18}
$$

Collision Detection: We define two types of collisions for the ego vehicle: collisions with the foreground (actors) and collisions with the background. Collisions with the foreground are detected by checking if the Bird’s Eye View (BEV) bounding boxes of the ego vehicle and actors overlap. Collisions with the background are identified by counting the number of background Gaussians inside the ego vehicle’s 3D bounding box, excluding Gaussians with ground semantic labels and low opacities. If the number exceeds a preset threshold, a collision is considered to have occurred.

## 4.3 Actor Driving Behaviors

We design three patterns of driving behavior: replayed, normal and aggressive.

Replayed Driving Behavior: For native dynamic vehicles in scenes, we reconstruct distinct actor pose observations into a continuous trajectory using our unicycle models. The reconstructed trajectory can be integrated into the HUGSIM closed-loop simulation by providing corresponding timestamps. It is important to note that this replayed driving behavior does not interact with the ego vehicle and is only applied in some easy scenarios.

Normal Driving Behavior: Normal driving behavior follows IDM [62], a simple car-following model that relies on HD maps for lane tracking and vehicle following. If no vehicle is present in front, the model drives at a constant speed. Among the datasets we consider in this work, NuScenes is the only one that provides paired RGB images and HD maps. Therefore, we apply the IDM-based normal driving behavior on nuScenes alone. For other datasets, we use a constant speed following a predefined driving direction as a simple alternative.

Aggressive Driving Behavior: The design of aggressive driving behavior is illustrated in Fig. $^ { 7 , }$ this adversarial model performs effectively both with and without the use of HD maps. Inspired by KING [30] and CAT [78], we design an optimization-based generative trajectory planner. Unlike KING, we generate adversarial behavior during simulation rather than in a post-processing manner. In contrast to CAT, our approach can generate candidate trajectories both with and without HD maps. First, based on road information or current actor states, we generate a set of feasible candidate trajectories $\{ s _ { 1 : T } ^ { a ( i ) } \} _ { i = } ^ { N }$ based on a spline-planner for the attacking actor. $T$ is the planning horizon and N is the number of candidate trajectories $\mathbf { \mathcal { S } } _ { 1 : T } ^ { a ( i ) }$ . The spline-planner generates feasible trajectories through grid-based terminal state sampling, spline-based trajectory interpolation, and feasibility filtering using dynamic constraints. This process enables the generation of diverse and robust trajectory sets while ensuring adherence to physical and environmental constraints. Next, we select from the feasible trajectories with the goal of attacking the ego vehicle while avoiding collision with other actors, achieved by predicting the future trajectory of the ego vehicle and other actors. More formally, we select the aggressive actor trajectory by minimizing the composite cost:

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m i n } _ { i } C _ { t o t a l } \big ( s _ { 1 : T } ^ { a ( i ) } \big ) = C _ { a t t a c k } \big ( s _ { 1 : T } ^ { a ( i ) } \big ) + \lambda C _ { c o l l i s i o n } \big ( s _ { 1 : T } ^ { a ( i ) } \big ) , } \\ & { C _ { a t t a c k } \big ( s _ { 1 : T } ^ { a ( i ) } \big ) = \displaystyle \operatorname* { m i n } _ { t = 1 : T } \Big \| s _ { t } ^ { e } - s _ { t } ^ { a ( i ) } \Big \| , } \\ & { C _ { c o l l i s i o n } \big ( s _ { 1 : T } ^ { a ( i ) } \big ) = \displaystyle \sum _ { j = 1 } ^ { M } \mathbb { 1 } \big ( \operatorname* { m i n } _ { t = 1 : T } \Big \| s _ { t } ^ { n ( j ) } - s _ { t } ^ { a ( i ) } \Big \| < t o l e r a n c e \big ) . } \end{array}\tag{19}
$$

where $s _ { 1 : T } ^ { e }$ denotes the ego future trajectory and $s _ { 1 : T } ^ { n ( j ) }$ denote the trajectories of other inserted actors, both predicted based on their current states. $C _ { a t t a c k }$ denotes the distance score for successful attacks, and $C _ { c o l l i s i o n }$ penalizes collision with other actors. Furthermore, we can adjust the attack intensity by randomly choosing from the top-k trajectories sorted by $\dot { C } _ { t o t a l } ( s _ { 1 : T } ^ { a ( i ) } )$ , instead of choosing the most aggressive one. Additionally, we can adjust the attack planning frequency to further control the intensity of the adversarial attack.

![](images/e79f357a10b7d946416470bee1326ab2bf15c428dc2cf1bb54dca568263de521.jpg)  
Fig. 8: Illustration of sub-scores combined in HUGSIM. We present cases that result in the failure or reduction of the corresponding sub-scores.

## 4.4 Evaluation

Inspired by NAVSIM [18] and DriveArena [73], we propose the HD-Score (HUGSIM Driving Score) to measure the performance of AD algorithms in our simulator. At each timestamp, the HD-Score is defined as:

$$
\begin{array} { r l } { \mathrm { H D - S c o r e } _ { t } = \underbrace { \left( \displaystyle \prod _ { m \in \{ N \cup C , D A C \} } s c o r e _ { m } \right) } _ { \mathrm { d r i v i n g p o i i c y ~ i t e m s } } \times } & { } \\ { \underbrace { \left( \frac { \displaystyle \sum _ { w \in \{ T T C , C O M \} } w e i g h t _ { w } \times s c o r e _ { w } } { \displaystyle \sum _ { w \in \{ T T C , C O M \} } w e i g h t _ { w } } \right) } _ { \mathrm { c o r t r i b u t o r y ~ i t e m s } } } \end{array}\tag{20}
$$

The driving policy items include driving with no collisions (NC) and drivable area compliance (DAC), these subscores are crucial for driving safety. The contributory items include time-to-collision $( T \overset { \vartriangle } { T } C )$ and comfort (COM ), which may not directly cause failure cases when they are low. Still, they can lead to feelings of insecurity and discomfort for passengers. We illustrate each sub-score in Fig. 8. The detailed definition of these sub-scores can be found in [18]. In the calculation of N C and T T C, we consider collisions with static background entities such as buildings, fences, vegetation, etc., leveraging our semantic information. This background entity collision is not included in NAVSIM nor DriveArena.

Additionally, unlike NAVSIM and DriveArena, which evaluate ego progress (EP ) using pseudo ground truth generated by planning algorithms such as the Predictive Driver Model (PDM) [17], we argue that in a closed-loop simulator, other metrics alone can sufficiently reflect the performance of AD algorithms. Furthermore, the PDM is not a flawless algorithm, even provided with ground-truth perception results. Additionally, there is often multiple acceptable driving style in most scenarios. For these reasons, evaluating AD algorithms using $E P$ in a closed-loop manner becomes unsuitable. Instead of measuring ego progress at each frame, we introduce a global route completion score $R _ { c } \in [ 0 , 1 ]$ , which represents the percentage of the driving distance completed by AD algorithms.

PNF  
![](images/2adfabaa5c62e7f49cb50d0a72c344e1f22adbfcbad06e1e0ead59a38f9c2338.jpg)  
Fig. 9: Details Qualitative Comparison with MARS on KITTI-360 Leaderboard.

![](images/2a3b6a78795be819302d3f46176eaa4bda9dfa53eab61bf77c5a2e650f918394.jpg)  
Ours  
Fig. 10: Qualitative Comparison with PNF on KITTI-360 Leaderboard.

The final HD-Score is averaged across all simulation timestamps and multiplied by the global route completion score $R _ { c }$ :

$$
{ \mathrm { H D - S c o r e } } = R _ { c } \times { \frac { \sum _ { t = 0 } ^ { T } { \mathrm { H D - S c o r e } } _ { t } } { T } }\tag{21}
$$

## 5 RENDERING EVALUATION

In this section, we evaluate HUGSIM from several perspectives. In Section 5.1, we focus on evaluating interpolated views in both dynamic and static scenes. In Section 5.2, we assess the semantics and geometry of reconstructed scenes. In Section 5.3, we evaluate extrapolated views, which are significant in a closed-loop driving simulator. Finally, in Section 5.4, we conduct ablation studies on each component of HUGSIM.

## 5.1 Novel View Synthesis on Interpolated Views

We first evaluate HUGSIM for interpolated novel view synthesis on KITTI [26], Virtual KITTI2 [7] and KITTI-360 [44] including dynamic and static scenes. For dynamic scenes, we leverage noisy 3D bounding box predictions as input, instead of using the ground truth. We evaluate the dynamic scene rendering results on KITTI and vKITTI in this section because previous dynamic urban reconstruction baselines have been tested on these two datasets. We apply 50% dropout rate following existing evaluation protocols [44], [68] on all of these datasets. We evaluate the dynamic scene novel view synthesis task by comparing our method with NSG [52] and MARS [68], which are two open-source methods for dynamic urban scenes. Additionally, we compare the static novel view appearance task with mip-NeRF [2], PNF [39], and MARS [68]. We adopt the default setting for interpolated views quantitative assessments, including the evaluation of PSNR, SSIM and LPIPS [79]. Although not our main foucus, we include a comparison using ground truth 3D bounding boxes in the appendix.

<table><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>|mIoUcls ↑mIoUcat ↑</td></tr><tr><td>mip-NeRF [2]</td><td>21.54</td><td>0.778</td><td>0.365 48.25</td><td>67.47</td></tr><tr><td>PNF [39]</td><td>22.07</td><td>0.820</td><td>0.221</td><td>73.06 84.97</td></tr><tr><td>MARS [68]</td><td>23.09</td><td>0.857</td><td>0.174</td><td></td></tr><tr><td>Ours</td><td>23.38</td><td>0.870</td><td>0.121</td><td>72.65 85.64</td></tr></table>

TABLE 2: Novel View Semantic and Appearance Synthesis on KITTI-360.
<table><tr><td></td><td>acc.↓</td><td>comp.↓</td><td> $\mathrm { \underline { { { m I o U } _ { \mathrm { { c l s } } } \uparrow } } }$ </td></tr><tr><td>Semantic Nerfacto</td><td>1.508</td><td>24.28</td><td>0.055</td></tr><tr><td>Ours</td><td>0.233</td><td>0.214</td><td>0.505</td></tr></table>

TABLE 3: 3D Semantic Reconstruction on KITTI-360. Note that all metrics are calculated in 3D space.

Dynamic Scene with Noisy 3D Bounding Boxes: Following [52], [68], we evaluate our performance on dynamic scenes of the KITTI and vKITTI datasets. In contrast to these methods that leverage ground truth poses, we investigate a more practical scenario where the bounding boxes are generated by a monocular-based 3D tracking algorithm, QD-3DT [32], in Table 4. Here, the predicted 3D bounding boxes are only provided for training views, as testing views should not be used as inputs for the tracking model. In experiments where the unicycle model is not utilized, the bounding boxes of testing views are obtained through linear interpolation from neighbour training views. Where the unicycle model is used, the bounding boxes of testing views are computed using Eq. 5. For vKITTI, there is no pretrained monocular tracking algorithm. We hence jitter the ground truth poses to simulate noisy monocular predictions, with an average noise of 0.5 meters in translation and 5 degrees in rotation. Our model’s robustness wrt. various levels of noise will be analyzed in the ablation study.

Table 4 demonstrate that our method consistently outperforms against the baselines. Note, that QD-3DT yields reasonable predictions on the KITTI dataset<sup>2</sup>. Hence, NSG and MARS reconstruct the dynamic objects reasonably well, but with more blurriness and artifacts (see Fig. 11), as they do not model the optimization of the object poses. In contrast, our method allows for reconstructing dynamic objects with sharp details, not only in cases of minor pose error on the KITTI dataset but also on the vKITTI dataset with more severe noise. We present the optimization progress of noisy bounding boxes and a comparison before and after optimization in the appendix.

Static Scene Leaderboard: We further evaluate our performance on the KITTI-360 leaderboard, which contains 5 static sequences. Our method achieves state-of-the-art performance on the leaderboard as in Table 2 (left), demonstrating the effectiveness of the 3D Gaussian representation in modeling complex urban scenes. As we will discuss in the ablation study, incorporating the affine transform to model camera exposure is important for reaching high fidelity. Fig. 9 shows the qualitative comparison of our proposed method to another top-ranking method, MARS, on the leaderboard.

<table><tr><td></td><td colspan="3">KITTI Scene02</td><td colspan="3">KITTI Scene06</td><td colspan="3">vKITTI Scene02</td><td colspan="3">vKITTI Scene06</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>NSG [52]</td><td>23.00</td><td>0.664</td><td>0.373</td><td>23.78</td><td>0.717</td><td>0.234</td><td>21.40</td><td>0.689</td><td>0.376</td><td>20.60</td><td>0.719</td><td>0.255</td></tr><tr><td>MARS [68]</td><td>23.30</td><td>0.731</td><td>0.139</td><td>25.09</td><td>0.856</td><td>0.083</td><td>22.67</td><td>0.882</td><td>0.128</td><td>21.67</td><td>0.856</td><td>0.134</td></tr><tr><td>Ours</td><td>25.42</td><td>0.821</td><td>0.092</td><td>28.20</td><td>0.919</td><td>0.027</td><td>26.21</td><td>0.911</td><td>0.040</td><td>26.65</td><td>0.921</td><td>0.030</td></tr></table>

TABLE 4: Novel View Synthesis on Dynamic Scenes with predicted or noisy 3D trackings.

![](images/b7a44cd4637c7a7fb66b0260e046b14028a8c33a42015558d133b80e119a1033.jpg)  
Fig. 11: Qualitative Comparison on KITTI and vKITTI. We use monocular-based 3D bounding box predictions for KITTI, and manually jittered 3D bounding boxes for vKITTI. We zoom in on a patch of a dynamic object for each KITTI scene.

## 5.2 Semantic and Geometric Scene Understanding

Next, we evaluate our model on various semantic and geometric scene understanding tasks on the KITTI-360 dataset [44] as it provides dense 2D and 3D labels. We compare the semantic synthesis task with mip-NeRF [2], PNF [39], and MARS [68]. Furthermore, we assess the quality of 3D semantic scene reconstruction by comparing it with Semantic Nerfacto [60]. We follow KITTI-360, which reports the mean Intersection over Union on class $\mathrm { ( m I o U _ { c l s } ) }$ and category $\mathrm { ( m I o U _ { \mathrm { c a t } } ) }$ , respectively. Further, we evaluate our performance on 3D Semantic Segmentation against a ground truth semantic LiDAR point cloud, measuring both geometric reconstruction quality and semantic accuracy. The geometric quality is evaluated as the chamfer distance between two point clouds, including completeness and accuracy, whereas the semantic accuracy is also measured using mIoU . Correct 3D semantics, the reduction of floaters, and the reconstruction of accurate geometry are essential for building our closed-loop simulator and benchmark.

Novel View Semantic Synthesis: Our holistic representation also enables novel view semantic synthesis. Hence, we submit our novel view semantic synthesis performance to the KITTI-360 leaderboard for comparison as well, see Table 2 (right). Despite not leveraging category-level prior as done in previous work [39], our approach achieves comparable performance to the SOTA [39] as shown in Fig. 10.

3D Semantic Scene Reconstruction: While existing 2Dto-3D semantic lifting methods solely evaluate their performance in the 2D image space, we further evaluate our performance in the 3D space to examine the underlying 3D geometry. To this goal, we leverage the ground truth LiDAR points provided by the KITTI-360 dataset for evaluation. With each Gaussian possessing semantic information, we can obtain a semantic point cloud by extracting the Gaussian’s center $\mu$ and its semantic label. We evaluate the geometric quality and semantic accuracy of this semantic point cloud in Table 3. We compare our method with Semantic Nerfacto [60], a Semantic NeRF implemented using a more advanced backbone, as the state-of-the-art novel view semantic synthesis method, PNF, in Table 2 is not open-source. For this baseline, we extract a semantic point cloud by specifying a threshold to the density field. While Semantic Nerfacto enables rendering faithful 2D semantic labels as shown in the appendix, the underlying 3D semantic point cloud is significantly worse in comparison. The Gaussian based representation instead allows for extracting a much more accurate semantic point cloud in comparison. Accurate geometry is crucial for implementing the closedloop simulator.

## 5.3 Novel View Synthesis on Extrapolated Views

The ego vehicle can change lanes or view scenes from a wide range of perspectives beyond the training views, which may lead to artifacts in novel view synthesis, particularly causing distortion in lanes and other ground-level signals. However, since most extrapolated camera poses are not directly observed, quantitative metrics like PSNR that require ground-truth images cannot be computed directly. Instead, we use the Kernel Inception Distance (KID) [3] to measure the similarity in distribution between extrapolated views and real captured images.

<table><tr><td colspan="5">Waymo</td><td colspan="4">NuScenes</td><td colspan="3">Attributes</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> ${ \mathrm { K I D } } _ { e x t r a p } ~ .$  →</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $\mathrm { K I D } _ { e x t r a p } \downarrow$ </td><td>FPS↑</td><td>#GS↓</td><td>Inputs</td></tr><tr><td>NeuRAD [61]</td><td>29.18</td><td>0.839</td><td>0.120</td><td>0.094</td><td>24.49</td><td>0.692</td><td>0.287</td><td>0.082</td><td>2.61</td><td></td><td>LiDAR+RGB</td></tr><tr><td>StreetGaussian [70]</td><td>27.20</td><td>0.868</td><td>0.113</td><td>0.151</td><td></td><td></td><td></td><td></td><td>66.50</td><td>6.85M</td><td>LiDAR+RGB</td></tr><tr><td>Ours</td><td>28.97</td><td>0.872</td><td>0.110</td><td>0.077</td><td>25.36</td><td>0.776</td><td>0.171</td><td>0.062</td><td>89.15</td><td>4.45M</td><td>RGB</td></tr></table>

TABLE 5: Quantitative Comparison with NeuRAD and StreetGaussian. All the experiments are conducted on a NVIDIA RTX 3090.

![](images/ca07c7cb247c877624272b2416c8bcafb80fac07264d9b9a959a8a5a14805fdd.jpg)  
Fig. 12: Extrapolated Views Qualitative Comparison with StreetGaussian and NeuRAD [70] on Waymo.

We evaluate extrapolated views on the Waymo Open Dataset [58] and nuScenes [8] by comparing our method to NeuRAD [61] and StreetGaussian [70], two recent approaches for novel view synthesis in urban scenes. Street-Gaussian supports the Waymo dataset, while NeuRAD supports both Waymo and nuScenes datasets. Notably, NeuRAD is also the reconstruction and rendering method used in NeuroNCAP [45]. We further compare our results to RoGS [21], a method designed specifically for ground reconstruction in nuScenes.

Full Scene Rendering: We selected two sequences from the Waymo dataset and two from nuScenes for comparison. During training, HUGSIM, StreetGaussian [70], and Neu-RAD [61] drop every 5th frame. To generate extrapolated views, we shift the original views horizontally to simulate lane changes and slightly adjust the yaw angle to simulate varied observation angles. In this experiment, we evaluate only the front camera rendering results, as the front view plays the most crucial role for driving. For a fair comparison, all methods use the same extrapolated views within the same sequences.

Table 5 shows that our method achieves comparable or better performance in interpolated views and state-ofthe-art performance on extrapolated views. Additionally, while both StreetGaussian and NeuRAD require LiDAR point clouds as input, HUGSIM relies solely on RGB images. HUGSIM also outperforms the compared methods in rendering speed and number of Gaussians. Fig. 12 and

![](images/1949fe75ccc105e88ec26e413c300b0fec55b994a679ee3d2d1a88786e97fde3.jpg)  
Fig. 13: Extrapolated Views Qualitative Comparison with NeuRAD [70] on nuScenes.

<table><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $\mathrm { K I D } _ { e x t r a p } \downarrow$ </td><td>#GS↓</td></tr><tr><td>ROGS [21]</td><td>18.70</td><td>0.795</td><td>0.158</td><td>0.223</td><td>1.02M</td></tr><tr><td>Ours</td><td>27.44</td><td>0.924</td><td>0.061</td><td>0.196</td><td>0.71M</td></tr></table>

TABLE 6: Quantative Comparison with RoGS. Note that we evaluate all the ground areas, which differs from the evaluation in the RoGS paper, where only sparse colored LiDAR points are considered.

Fig. 13 provides a qualitative comparison of HUGSIM and other methods. Without ground constraints, significant distortion is evident in NeuRAD and StreetGaussian, while our method produces realistic results in extrapolated views.

Ground Rendering: We further compare our ground model with RoGS [21], an approach specifically designed for ground reconstruction using 2D Gaussian Splatting [34]. The primary difference between RoGS and our method is that RoGS uniformly distributes Gaussians on the ground plane and fixes their attributes, allowing only spherical harmonics and height to be learned. In contrast, our method fixes only the y-axis scale and rotations of Gaussians and applies $\mathcal { L } _ { g r o u n d }$ to constrain the ground flatness, as demonstrated in Section 3.2.

The experiments are conducted on four sequences from nuScenes. We utilize a semantic mask to color non-ground pixels black. Since the top half of the images does not contain any ground pixels, we focus our evaluation on the lower half of the images for this experiment. Table 6 and Fig. 14 show our method surpasses RoGS both quantitatively and qualitatively, while also using fewer Gaussians to reconstruct the scene and exhibiting greater flexibility without the need to assign the hyperparameter of road width. We attribute the advantages of our method to its learnable position, scaling, and opacity, allowing it to allocate more Gaussians to texture-rich areas and fewer to regions with minimal texture. This leads to to a reduced number of Gaussians while maintaining high texture detail. In contrast, the uniform distribution of large numbers of Gaussians, as implemented in RoGS, still exists holes and results in less

![](images/f1841da8565116f773f85033cbf7eae69dc10941df45b2bab67862f01cf5b5b2.jpg)  
Fig. 14: Qualitative Comparison with RoGS. Compared to our results, RoGS exhibits grid-style aliasing and limited rendering areas. Both methods show no obvious distortion in extrapolated views.

<table><tr><td></td><td colspan="5">KITTI (5% noise)</td><td colspan="5">KITTI (10% noise)</td><td colspan="4">KITTI (20% noise)</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td> $\begin{array} { r l } { e _ { \mathbf { R } } \downarrow } & { { } e _ { \mathbf { t } } \downarrow } \end{array}$ </td><td></td><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓</td><td> $\begin{array} { r l } { e _ { \mathbf { R } } \downarrow } & { { } e _ { \mathbf { t } } \downarrow } \end{array}$ </td><td></td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td> $\begin{array} { r l } { e _ { \mathbf { R } } \downarrow } & { { } e _ { \mathbf { t } } \downarrow } \end{array}$ </td></tr><tr><td>w/o opt., w/o uni.</td><td>23.83</td><td>0.878</td><td>0.062</td><td>0.031 0.027</td><td></td><td>22.16</td><td>0.861</td><td>0.079</td><td>0.063 0.106</td><td></td><td>20.28</td><td>0.835</td><td>0.101</td><td>0.125 0.425</td></tr><tr><td>w/ opt., w/o uni.</td><td>24.80</td><td>0.897</td><td>0.038</td><td>0.022 0.051</td><td></td><td>22.75</td><td>0.879</td><td>0.056</td><td>0.0540.130</td><td></td><td>20.56</td><td>0.855</td><td>0.081</td><td>0.1350.612</td></tr><tr><td>w/ opt., w/ uni. (Ours)</td><td>28.78</td><td>0.928</td><td>0.023</td><td>0.017 0.022</td><td></td><td>26.66</td><td>0.908</td><td>0.032</td><td>0.037 0.035</td><td></td><td>23.59</td><td>0.875</td><td>0.061</td><td>0.081 0.176</td></tr></table>

TABLE 7: Ablation Study of Dynamic Scenes of KITTI.

<table><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td></td><td>LPIPS↓ Depth ↓</td></tr><tr><td>w/o Affine transform</td><td>24.18</td><td>0.827</td><td>0.083</td><td></td></tr><tr><td> ${ \bf w } / { \bf o } ~ \mathcal { L } _ { \bf S }$ </td><td>24.47</td><td>0.831</td><td>0.081</td><td>0.892</td></tr><tr><td>Ours</td><td>24.52</td><td>0.833</td><td>0.081</td><td>0.872</td></tr></table>

TABLE 8: Ablation Study of Static Scenes on KITTI-360.
<table><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> ${ \mathrm { K I D } } _ { i n t e r p } \downarrow$ </td><td> $\mathrm { K I D } _ { e x t r a p } \downarrow$ </td></tr><tr><td> $\mathbf { \Sigma } - \mathcal { L } _ { g r o u n d }$ </td><td>28.03</td><td>0.928</td><td>0.059</td><td>0.041</td><td>0.249</td></tr><tr><td> $\mathrm { ~ \bf ~ \underline { ~ } { ~ l ~ r ~ } ~ } s _ { x } , s _ { z }$ </td><td>25.84</td><td>0.909</td><td>0.079</td><td>0.050</td><td>0.266</td></tr><tr><td> $- \ln \mu$ </td><td>26.91</td><td>0.916</td><td>0.065</td><td>0.039</td><td>0.242</td></tr><tr><td>Ours</td><td>27.44</td><td>0.924</td><td>0.061</td><td>0.039</td><td>0.196</td></tr></table>

TABLE 9: Quantitative Ablation of Ground Model. While the model without $\mathcal { L } _ { g r o u n d }$ demonstrates better metrics in interpolated views, noticeable artifacts are rendered in extrapolated views.

![](images/5078a9b768d418216a7e687246fd6e99127a561628e909766d590b19d6855ece.jpg)  
w/o opt., w/o uni. w/ opt., w/o uni. Ours

Fig. 15: Detail Qualitative Comparison on KITTI with Noisy Bounding Boxes.

satisfactory rendering outcomes.

## 5.4 Ablation

We conduct ablation studies on dynamic, static scenes, ground model and inserted vehicles.

Dynamic Scene: As KITTI provides accurate 3D bounding box ground truth, we ablate the effectiveness of our unicycle model on KITTI by manually adding noise to the 3D bounding boxes and evaluate both the novel view synthesis results and the tracking performance, see Table 7. In this experiment, we compare our full model to two variants, i.e., using the noises without optimization (w/o opt., w/o uni.), and performing na¨ıve per-frame optimization without using the unicycle model (w/ opt., w/o uni.). We evaluate 3D tracking performance by measuring the rotation and translation error $e _ { \mathbf { R } }$ and $e _ { \mathbf { t } }$ of our optimized 3D bounding boxes wrt. the ground truth following [10]. The results validate the effectiveness of the unicycle model, which obviously improves the rendering quality and 3D tracking accuracy. Qualitative results in Fig. 15 further verify the effectiveness of our unicycle model in enabling accuracy object reconstruction given noisy 3D bounding boxes.

Static Scene: We study the effect of different components on three static scenes of KITTI-360 in Table 8. This allows us to ablate design choices without mixing up the impact of dynamic objects. The results indicate the significance of exposure modeling, which is particularly important for scenes with strong exposure variance. The semantic loss has little contribution to improving novel view synthesis. It is rational as imposing a constraint on the semantics does not necessarily contribute to appearance. However, note that incorporating the semantic supervision improves the underlying geometry, see the appendix for qualitative comparison. Moreover, we utilize the semantic information for collision detection of our closed-loop simulator.

Ground Model: We ablate the effectiveness of $\mathcal { L } _ { g r o u n d }$ and importance of learnable position, scaling and opacity, as shown Table 9 and Fig. 16. PSNR, SSIM and LPIPS are evaluated in interpolated views, while KID is evaluated in extrapolated views. The results demonstrate the importance of $\bar { \mathcal { L } _ { g r o u n d } }$ in significantly reducing floating Gaussians. Additionally, the learnable position, scaling, and opacity improve the reconstruction fidelity of our road model.

Inserted Vehicles: To compare simulation quality with native versus non-native vehicles, we inserted vehicles with random poses into a sequence, as shown in Fig. 17. The results indicate that native vehicles yield less satisfactory rendering results due to limited observations.

![](images/5c5853eadd58c41ae2fafc2cce355248e894f7a2308b02e9b1f82565ffaca5e2.jpg)  
Fig. 16: Qualitatively Ablation of Ground Model

## 6 CLOSED-LOOP BENCHMARK

In this section, we first establish the HUGSIM benchmark, comprising a variety of scenes and scenarios with varying difficulty levels for evaluating AD algorithms. Next, we use the HUGSIM benchmark to evaluate several existing AD algorithms, providing a fair, closed-loop evaluation. Furthermore, we provide a brief case study on the failure cases of existing AD algorithms.

## 6.1 HUGSIM Benchmark

We establish the HUGSIM benchmark based on KITTI-360 [44], Waymo [58], nuScenes [8] and PandaSet [69]. Our benchmark contains more than 70 scenes and over 400 scenarios with varying difficulty levels for testing AD algorithms.

Scenes: The HUGSIM benchmark consists of scenes from KITTI-360, Waymo, nuScenes, and PandaSet. These scenes feature a variety of street layouts, lighting conditions, and imaging styles.

As AD algorithms like UniAD [33], VAD [35] and Latent-Transfuser (LTF) [14] demand for a relative large field on view, KITTI [26] is not included in HUGSIM, as it only provides two front-facing cameras. For scene reconstruction, we use the two front-facing perspective cameras and two side fisheye cameras converted to perspective [48] in KITTI-360, the three front perspective cameras in Waymo, and all six perspective cameras in both nuScenes and PandaSet.

For our simulator rendering, we use the same camera configuration as nuScenes. For scenes from nuScenes and PandaSet, we provide rendered images from all six cameras as observations to the AD algorithms. For scenes from KITTI-360 and Waymo, we provide renderings from only the front three cameras, as these two datasets primarily focus on front and side views. Our experiments show that although UniAD and VAD are trained on a setup with six surrounding cameras, they exhibit robust generalizability even when only the front three camera views are provided.

![](images/8ca017293017f362bdabbdb81e61722fd84df6e778099f4a204061bb9c4f8df3.jpg)  
Fig. 17: Qualitative Comparison with native and non-native vehicles insertion.

Scenarios: For each reconstructed scene, we design scenarios across various difficulty levels: easy, medium, hard and extreme. Each reconstructed scene consists of more than four scenarios, as each difficulty level may contain multiple scenarios. Altogether, the HUGSIM benchmark comprises over 400 scenarios, offering a diverse set of training and testing environments for AD algorithms with interactive actors.

Easy scenarios feature mostly static scenes or only include replayed driving actors modeled with a unicycle model. Medium scenarios include IDM and constant-speed driving actors with typical driving behaviors, where IDM actors yield to the ego vehicle. IDM is implemented only in nuScenes due to its HD map availability. Hard and extreme scenarios introduce aggressive actors that attempt to collide with the ego vehicle, with extreme scenarios increasing both the number of aggressive actors, a higher frequency of replanned attack routes and a greater likelihood of selecting the most aggressive trajectories. We provide details of the scenario designs in the appendix.

## 6.2 Evaluation on HUGSIM Benchmark

Evaluated AD Algorithms: The HUGSIM benchmark is well-suited for testing AD algorithms that require only RGB images and ego vehicle status as inputs. In this paper, we evaluate UniAD [33], VAD [35] and Latent-Transfuser (LTF) [14]. LTF is the LiDAR free version of Transfuser, which replaces the LiDAR input as a template positional encoding. It is worth noting that only LTF requires data from the front three cameras, while UniAD and VAD demands for input from six surrounding cameras. For all these three methods, we implement evaluation APIs that interface with HUGSIM Gymnasium environments [6]. These APIs receive data, parse it into the required format for model inference, and send back the planned future waypoints to HUGSIM.

Closed-Loop Evaluation Results: The evaluation results on our HUGSIM benchmark are presented in Fig. 18. Detailed evaluation results can be found in the appendix. UniAD and LTF demonstrate similar performance on the HUGSIM benchmark. While LTF performs better at easier scenarios, UniAD exhibits more powerful capabilities in handling most of the complex scenarios. We attribute this advantage of UniAD to its trajectory post-processing strategy based on predicted occupancy. However, this post-processing strategy also introduces non-smooth trajectories, negatively impacting the COM score. Another notable point is that in KITTI-360 and Waymo scenes, only the front three camera views are provided, while UniAD is trained using surrounding six views. The similar performance on Waymo and nuScenes indicates the generalizability of UniAD. VAD shows less satisfying performance compared to UniAD and LTF on the HUGSIM benchmark, particularly on KITTI-360, Waymo, and PandaSet. This could potentially be due to VAD overfitting to the scene and scenario styles in nuScenes, making it challenging to handle out-of-domain observations in our closed-loop simulator. All these algorithms face challenges with KITTI-360 scenes due to narrow streets and numerous parked vehicles, along with an imaging style that differs significantly from nuScenes and nuPlan, which are training datasets of these algorithms. Furthermore, the HUGSIM benchmark poses significant challenges for existing AD algorithms. Even the easy level, which consists of original scenes, can be difficult for these algorithms, highlighting the lack of generalization and the need for systematic analysis using a closed-loop simulator as presented in this paper. The greater challenges presented at the medium, hard, and extreme levels indicate that substantial efforts are still required to achieve the goal of full autonomy.

![](images/d058c3576b268f11af1d1dbe6813cd4014f0319319dd881fbf69e4709d21f6a6.jpg)

![](images/84c5d333f2b9925b80a580aec6182ce258b9a0d81d65e56141a4f42e2a5b5744.jpg)

![](images/2907a4bdd5474725d62c2fc94f0e0ba0db6d720acc8b75d20fd9c6686934c90b.jpg)  
Fig. 18: Evaluation Results on HUGSIM Benchmark. The histograms indicate performance of models on each type of dataset, while the lines represent the averaged sub-scores across datasets.

![](images/feb7cac216a98f338f67d3cb5b797340d7dd55421911182cc5347e3922c76c50.jpg)  
Fig. 19: Case Study. The left part shows the RGB observations, perception results, and planning trajectories, the right part displays the predicted BEV information of UniAD.

Case Study: We outline some common failure cases encountered by AD algorithms in our closed-loop simulator as shown in Fig. 19: (a) Since collisions never occur in openloop datasets, models rarely encounter situations where there is no drivable area in front of the vehicle. As a result, they consistently predict a drivable area ahead, regardless of the RGB observations. (b) AD algorithms face difficulties with turning, often steering at inappropriate angles relative to the street structure. (c) Although a leading vehicle is detected, AD algorithms often do not take evasive action in advance; instead, they attempt to make a sharp turn at a very close distance, which can easily result in a collision. (d) The planned trajectory of AD algorithms can be unstable, which may lead to collisions in narrow streets and a reduction in COM score.

## 7 CONCLUSION

We present HUGSIM, a novel photorealistic closed-loop simulator for autonomous driving, featuring real-time, highquality rendering in extrapolated views and efficiently generated actor behavior. Specifically, we reconstruct urban scenes using 3D Gaussians and introduce a ground model, along with single-vehicle reconstruction, to improve the rendering quality of extrapolated views. For actor behavior, we propose an attack-cost-based trajectory interactively search to simulate aggressive driving behaviors of actors.

Furthermore, we establish the HUGSIM benchmark across multiple datasets including variance sequences, designing more than 300 scenarios for evaluating and training AD algorithms. We evaluate several baselines on our benchmark. Our results show that the HUGSIM benchmark presents significant challenges for existing AD algorithms. This closed-loop benchmark reveals substantial room for improvement in autonomous driving performance. We hope that our dataset and benchmark will fertilize new research across communities, fostering progress towards the ultimate goal of full autonomy.

For future work, HUGSIM can be enhanced in several ways. First, we assume all dynamic objects follow rigid motion, which may cause blurring for non-rigidly moving objects, such as pedestrians. This can be addressed by incorporating non-rigid dynamic reconstruction methods [13], [75] into our framework. While our method improves rendering at extrapolated viewpoints, it struggles with highfidelity rendering at views that are far from the input or very close to objects. These challenges could be mitigated by leveraging priors from 2D generative models [64], [76]. Furthermore, since our approach opens up the possibility for fine-tuning AD algorithms in a photorealistic closedloop setting, this presents a promising avenue for future exploration.

## ACKNOWLEDGMENTS

The authors thank Kashyap Chitta for providing details of Transfuser and NAVSIM, and Chaojie Ji, Yuanbo Yang, Xianxu Xiang, and Jiahao Shao for proofreading. This work is supported by NSFC under grant 62202418, U21B2004 and the National Key R&D Program of China under Grant 2021ZD0114501. Yiyi Liao is with the Zhejiang Provincial Key Laboratory of Information Processing, Communication and Networking (IPCAN). Andreas Geiger was supported by the ERC Starting Grant LEGO-3D (850533) and the DFG EXC number 2064/1 - project number 390727645.

## REFERENCES

[1] S. Agarwal, Y. Furukawa, N. Snavely, I. Simon, B. Curless, S. M. Seitz, and R. Szeliski, “Building rome in a day,” Communications of the ACM, vol. 54, no. 10, pp. 105–112, 2011. 3

[2] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields,” in Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2021, pp. 5855–5864. 10, 11

[3] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “De-´ mystifying mmd gans,” arXiv preprint arXiv:1801.01401, 2018. 11

[4] S. Borse, Y. Wang, Y. Zhang, and F. Porikli, “Inverseform: A loss function for structured boundary-aware segmentation,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 5901–5911. 7,

[5] G. Bradski, “The opencv library,” Dr. Dobb’s Journal of Software Tools, 2000. 4

[6] G. Brockman, “Openai gym,” arXiv preprint arXiv:1606.01540, 2016. 2, 8, 14

[7] Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” arXiv preprint arXiv:2001.10773, 2020. 10,

[8] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 621– 11 631. 1, 2, 3, 12, 14,

[9] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari, “nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles,” arXiv preprint arXiv:2106.11810, 2021. 1, 3, 4,

[10] X. Chen, Z. Dong, J. Song, A. Geiger, and O. Hilliges, “Category level object pose estimation via neural analysis-by-synthesis,” in Proc. of the European Conf. on Computer Vision (ECCV). Springer, 2020, pp. 139–156. 13,

[11] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images,” arXiv preprint arXiv:2403.14627, 2024. 3

[12] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering,” arXiv:2311.18561, 2023. 3

[13] Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic, O. Litany, Z. Gojcic, S. Fidler, M. Pavone et al., “Omnire: Omni urban scene reconstruction,” arXiv preprint arXiv:2408.16760, 2024. 16

[14] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “Transfuser: Imitation with transformer-based sensor fusion for autonomous driving,” IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), 2023. 14,

[15] F. Codevilla, M. Muller, A. L¨ opez, V. Koltun, and A. Dosovitskiy,´ “End-to-end driving via conditional imitation learning,” in Proc. IEEE International Conf. on Robotics and Automation (ICRA). IEEE, 2018, pp. 4693–4700. 1

[16] F. Codevilla, E. Santana, A. M. Lopez, and A. Gaidon, “Exploring ´ the limitations of behavior cloning for autonomous driving,” in Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2019, pp. 9329–9338. 1

[17] D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta, “Parting with misconceptions about learning-based vehicle motion planning,” in Proc. Conf. on Robot Learning (CoRL). PMLR, 2023, pp. 1268–1281. 1, 9

[18] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone et al., “Navsim: Datadriven non-reactive autonomous vehicle simulation and benchmarking,” arXiv preprint arXiv:2406.15349, 2024. 2, 3, 4, 9,

[19] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “Carla: An open urban driving simulator,” in Proc. Conf. on Robot Learning (CoRL). PMLR, 2017, pp. 1–16. 1, 3, 4,

[20] X. Du, H. Sun, S. Wang, Z. Wu, H. Sheng, J. Ying, M. Lu, T. Zhu, K. Zhan, and X. Yu, “3drealcar: An in-the-wild rgb-d car dataset with 360-degree views,” arXiv preprint arXiv:2406.04875, 2024. 2, 6, 8

[21] Z. Feng, W. Wu, and H. Wang, “Rogs: Large scale road surface reconstruction based on 2d gaussian splatting,” arXiv preprint arXiv:2405.14342, 2024. 3, 5, 12,

[22] T. Fischer, J. Kulhanek, S. R. Bulo, L. Porzi, M. Pollefeys, and\` P. Kontschieder, “Dynamic 3d gaussian fields for urban areas,” arXiv preprint arXiv:2406.03175, 2024. 3

[23] X. Fu, S. Zhang, T. Chen, Y. Lu, L. Zhu, X. Zhou, A. Geiger, and Y. Liao, “Panoptic nerf: 3d-to-2d label transfer for panoptic urban scene segmentation,” in Proc. of the International Conf. on 3D Vision (3DV). IEEE, 2022, pp. 1–11. 7

[24] D. Gallup, J.-M. Frahm, and M. Pollefeys, “Piecewise planar and non-planar stereo for urban scene reconstruction,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). IEEE, 2010, pp. 1418–1425. 3

[25] S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li, “Vista: A generalizable driving world model with high fidelity and versatile controllability,” arXiv preprint arXiv:2405.17398, 2024. 4

[26] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). IEEE, 2012, pp. 3354–3361. 1, 3, 10, 14,

[27] C. Gulino, J. Fu, W. Luo et al., “Waymax: An accelerated, datadriven simulator for large-scale autonomous driving research,” in Advances in Neural Information Processing Systems (NIPS), 2023. 1, 4

[28] J. Guo, N. Deng, X. Li, Y. Bai, B. Shi, C. Wang, C. Ding, D. Wang, and Y. Li, “Streetsurf: Extending multi-view implicit surface reconstruction to street views,” arXiv preprint arXiv:2306.04988, 2023. 3

[29] H. Han, K. Zhou, X. Long, Y. Wang, and C. Xiao, “Ggs: Generalizable gaussian splatting for lane switching in autonomous driving,” arXiv preprint arXiv:2409.02382, 2024. 3

[30] N. Hanselmann, K. Renz, K. Chitta, A. Bhattacharyya, and A. Geiger, “King: Generating safety-critical driving scenarios for robust imitation via kinematics gradients,” in Proc. of the European Conf. on Computer Vision (ECCV), 2022. 9

[31] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “Gaia-1: A generative world model for autonomous driving,” arXiv preprint arXiv:2309.17080, 2023. 4

[32] H.-N. Hu, Y.-H. Yang, T. Fischer, T. Darrell, F. Yu, and M. Sun, “Monocular quasi-dense 3d object tracking,” IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), vol. 45, no. 2, pp. 1992– 2008, 2022. 7, 10,

[33] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang et al., “Planning-oriented autonomous driving,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 17 853–17 862. 14,

[34] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in ACM Trans. on Graphics, 2024, pp. 1–11. 12,

[35] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “Vad: Vectorized scene representation for efficient autonomous driving,” in Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2023, pp. 8340–8350. 14,

[36] N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo et al., “Towards learning-based planning: The nuplan benchmark for real-world autonomous driving,” arXiv preprint arXiv:2403.04133, 2024. 8

[37] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaus-¨ sian splatting for real-time radiance field rendering.” ACM Trans. on Graphics, vol. 42, no. 4, pp. 139–1, 2023. 2, 4, 5,

[38] M. Khan, H. Fazlali, D. Sharma, T. Cao, D. Bai, Y. Ren, and B. Liu, “Autosplat: Constrained gaussian splatting for autonomous driving scene reconstruction,” arXiv preprint arXiv:2407.02598, 2024. 3, 5

[39] A. Kundu, K. Genova, X. Yin, A. Fathi, C. Pantofaru, L. J. Guibas, A. Tagliasacchi, F. Dellaert, and T. Funkhouser, “Panoptic neural fields: A semantic object-aware neural scene representation,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12 871–12 881. 1, 3, 7, 10, 11,

[40] F. Lafarge, R. Keriven, M. Bredif, and H.-H. Vu, “A hybrid multi-´ view stereo algorithm for modeling urban scenes,” IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), vol. 35, no. 1, pp. 5–17, 2012. 3

[41] H. Li, Y. Gao, D. Zhang, C. Wu, Y. Dai, C. Zhao, H. Feng, E. Ding, J. Wang, and J. Han, “Ggrt: Towards generalizable 3d gaussians without pose priors in real-time,” arXiv preprint arXiv:2403.10147, 2024. 3

[42] Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou, “Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning,” IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), vol. 45, no. 3, pp. 3461–3475, 2022. 1, 4

[43] Z. Liang, Q. Zhang, Y. Feng, Y. Shan, and K. Jia, “Gs-ir: 3d gaussian splatting for inverse rendering,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 644–21 653. 6

[44] Y. Liao, J. Xie, and A. Geiger, “Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d,” IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), vol. 45, no. 3, pp. 3292–3310, 2022. 1, 2, 3, 10, 11, 14,

[45] W. Ljungbergh, A. Tonderski, J. Johnander, H. Caesar, K. Astr<sup>˚</sup> om,¨ M. Felsberg, and C. Petersson, “Neuroncap: Photorealistic closedloop safety testing for autonomous driving,” arXiv preprint arXiv:2404.07762, 2024. 1, 3, 4, 8, 12,

[46] F. Lu, Y. Xu, G. Chen, H. Li, K.-Y. Lin, and C. Jiang, “Urban radiance field representation with deformable neural mesh primitives,” in Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2023, pp. 465–476. 3

[47] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields for unconstrained photo collections,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 7210– 7219. 3, 6

[48] C. Mei and P. Rives, “Single view point omnidirectional camera calibration from planar grids,” in Proc. IEEE International Conf. on Robotics and Automation (ICRA). IEEE, 2007, pp. 3945–3950. 14,

[49] S. Miao, J. Huang, D. Bai, W. Qiu, B. Liu, A. Geiger, and Y. Liao, “Efficient depth-guided urban view synthesis,” arXiv preprint arXiv:2407.12395, 2024. 3

[50] N. Moenne-Loccoz, A. Mirzaei, O. Perel, R. de Lutio, J. M. Esturo, G. State, S. Fidler, N. Sharp, and Z. Gojcic, “3d gaussian ray tracing: Fast tracing of particle scenes,” arXiv preprint arXiv:2407.07090, 2024. 6

[51] ——, “3d gaussian ray tracing: Fast tracing of particle scenes,” ACM Trans. on Graphics, 2024. 6

[52] J. Ost, F. Mannan, N. Thuerey, J. Knodt, and F. Heide, “Neural scene graphs for dynamic scenes,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 2856–2865. 3, 10, 11,

[53] K. Rematas, A. Liu, P. P. Srinivasan, J. T. Barron, A. Tagliasacchi, T. Funkhouser, and V. Ferrari, “Urban radiance fields,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12 932–12 942. 3, 6

[54] C. V. Samak, T. V. Samak, and S. Kandhasamy, “Control strategies for autonomous vehicles,” in Autonomous driving and advanced driver-assistance systems (ADAS). CRC Press, 2021, pp. 37–86. 8

[55] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113. 3

[56] X. Shi, L. Chen, P. Wei, X. Wu, T. Jiang, Y. Luo, and L. Xie, “Dhgs: Decoupled hybrid gaussian splatting for driving scene,” arXiv preprint arXiv:2407.16600, 2024. 3

[57] Y. Shi, Y. Wu, C. Wu, X. Liu, C. Zhao, H. Feng, J. Liu, L. Zhang, J. Zhang, B. Zhou et al., “Gir: 3d gaussian inverse rendering for relightable scene factorization,” arXiv preprint arXiv:2312.05133, 2023. 6

[58] P. Sun, H. Kretzschmar, X. Dotiwalla et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), June 2020. 1, 2, 3, 12, 14,

[59] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, “Block-nerf: Scalable large scene neural view synthesis,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 8248–8258. 3

[60] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja et al., “Nerfstudio: A modular framework for neural radiance field development,” in ACM Trans. on Graphics, 2023, pp. 1–12. 11,

[61] A. Tonderski, C. Lindstrom, G. Hess, W. Ljungbergh, L. Svensson, ¨ and C. Petersson, “Neurad: Neural rendering for autonomous driving,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 895–14 904. 3, 12,

[62] M. Treiber, A. Hennecke, and D. Helbing, “Congested traffic states in empirical observations and microscopic simulations,” Physical review E, vol. 62, no. 2, p. 1805, 2000. 2, 9

[63] H. Turki, J. Y. Zhang, F. Ferroni, and D. Ramanan, “Suds: Scalable urban dynamic scenes,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 12 375–12 385. 3

[64] Q. Wang, L. Fan, Y. Wang, Y. Chen, and Z. Zhang, “Freevs: Generative view synthesis on free driving trajectory,” arXiv preprint arXiv:2410.18079, 2024. 3, 16

[65] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu, “Drivedreamer: Towards real-world-driven world models for autonomous driving,” arXiv preprint arXiv:2309.09777, 2023. 4

[66] L. Wenl, D. Fu, S. Mao, P. Cai, M. Dou, Y. Li, and Y. Qiao, “Limsim: A long-term interactive multi-scenario traffic simulator,” in Proc. IEEE Conf. on Intelligent Transportation Systems (ITSC). IEEE, 2023, pp. 1255–1262. 1, 4

[67] F. Wimbauer, N. Yang, C. Rupprecht, and D. Cremers, “Behind the scenes: Density fields for single view reconstruction,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 9076–9086. 3

[68] Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou, H. Lou, Y. Chen, R. Yang et al., “Mars: An instance-aware, modular and realistic simulator for autonomous driving,” in International Conference on Artificial Intelligence (CICAI). Springer, 2023, pp. 3–15. 3, 10, 11,

[69] P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, K. Jiang et al., “Pandaset: Advanced sensor suite dataset for autonomous driving,” in Proc. IEEE Conf. on Intelligent Transportation Systems (ITSC). IEEE, 2021, pp. 3095–3101. 1, 2, 3, 14,

[70] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng, “Street gaussians: Modeling dynamic urban scenes with gaussian splatting,” in Proc. of the European Conf. on Computer Vision (ECCV), 2024. 1, 3, 12,

[71] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu, S. Fidler, M. Pavone et al., “Emernerf: Emergent spatialtemporal scene decomposition via self-supervision,” arXiv preprint arXiv:2311.02077, 2023. 3

[72] J. Yang, S. Gao, Y. Qiu, L. Chen, T. Li, B. Dai, K. Chitta, P. Wu, J. Zeng, P. Luo et al., “Generalized predictive model for autonomous driving,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 662–14 672. 4

[73] X. Yang, L. Wen, Y. Ma, J. Mei, X. Li, T. Wei, W. Lei, D. Fu, P. Cai, M. Dou, B. Shi, L. He, Y. Liu, and Y. Qiao, “Drivearena: A closedloop generative simulation platform for autonomous driving,” arXiv preprint arXiv:2408.00415, 2024. 2, 3, 4, 8, 9

[74] Z. Yang, Y. Chen, J. Wang, S. Manivasagam, W.-C. Ma, A. J. Yang, and R. Urtasun, “Unisim: A neural closed-loop sensor simulator,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1389–1399. 3, 4

[75] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene

reconstruction,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 331–20 341. 16

[76] W. Yu, J. Xing, L. Yuan, W. Hu, X. Li, Z. Huang, X. Gao, T.-T. Wong, Y. Shan, and Y. Tian, “Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis,” arXiv preprint arXiv:2409.02048, 2024. 16

[77] Z. Yu, H. Wang, J. Yang, H. Wang, Z. Xie, Y. Cai, J. Cao, Z. Ji, and M. Sun, “Sgd: Street view synthesis with gaussian splatting and diffusion prior,” arXiv preprint arXiv:2403.20079, 2024. 3

[78] L. Zhang, Z. Peng, Q. Li, and B. Zhou, “Cat: Closed-loop adversarial training for safe end-to-end driving,” in Proc. Conf. on Robot Learning (CoRL), 2023. 9

[79] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595. 10,

![](images/992dd64678dd79991b22f5dc26ed954d975189967dde397a297bc11cfca7170a.jpg)

[80] X. Zhang, A. Kundu, T. Funkhouser, L. Guibas, H. Su, and K. Genova, “Nerflets: Local radiance fields for efficient structure-aware 3d scene representation from 2d supervision,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 8274– 8284. 3

[81] S. Zhi, T. Laidlow, S. Leutenegger, and A. J. Davison, “In-place scene labelling and understanding with implicit scene representation,” in Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2021, pp. 15 838–15 847. 7,

[82] B. Zhou, P. Krahenb ¨ uhl, and V. Koltun, “Does computer vision¨ matter for action?” Science Robotics, vol. 4, no. 30, p. eaaw6661, 2019. 2

[83] H. Zhou, J. Shao, L. Xu, D. Bai, W. Qiu, B. Liu, Y. Wang, A. Geiger, and Y. Liao, “Hugs: Holistic urban 3d scene understanding via gaussian splatting,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 21 336–21 345. 1, 2,

[84] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 634– 21 643. 1, 3

![](images/cd92a0105471935b8460e55e563c351002ce73395f498ffa0b88354c8835f38a.jpg)

![](images/7461dc9bebc0f6c2875f5652378a43c62b907f2d226522c2943f1e8bc8219a19.jpg)

[85] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Ewa splatting,” IEEE Transactions on Visualization and Computer Graphic (TVCG), vol. 8, no. 3, pp. 223–238, 2002. 4,

![](images/9a28f9dda6b6fee78c7edf3522f51f28215bf743be2e7a0f6b2e934b62ae53ed.jpg)  
Hongyu Zhou is a Ph.D. student at Zhejiang University. Before that, he received his B.S. degree from College of Software Engineering, Tongji University, China. His research interests include 3D computer vision and 3D reconstruction and generative models.

![](images/492f422b591cb639dc8e271ba55d73e31283e7fdb06e348605bb038318861f20.jpg)

![](images/cb40b86cf5adaf298c7b7bbc1dc2b67a31d51fbffd759b6bc1fe1040db8985b9.jpg)  
Longzhong Lin is a Ph.D. student at Zhejiang University. Before that, he received his B.S. degree from the College of Control Science and Engineering, Zhejiang University, China. His research interests include autonomous robots and robot learning.

![](images/7e07f54f3ce18371e466b9fda7759df2a823cbb8ce23b7ef260dbb32319bb499.jpg)

![](images/d8ce94f7c7c683cb2f8a9341ad8d36b0d0299e27a799344649de521e8e32d7ef.jpg)

Bingbing Liu (M09) is a Technology Specialist with Noah’s Ark Lab, Huawei. Previously he worked as a Research Scientist with Institute for Infocomm Research, A\*STAR, Singapore. He received his Bachelor degree from Harbin Institute of Technology, China and the Ph.D degree from Nanyang Technological University, Singapore respectively. His research interests are computer vision technologies for autonomous driving and robotics.

Yichong Lu Yichong Lu is a Master’s student in Zhejiang University. Before that, he obtained bachelor degree in the College of Information Science and Electronic Engineering, Zhejiang University. His research interest lies in 3D computer vision, including scene understanding, 3D reconstruction and 3D generative models.

Yue Wang received the Ph.D. degree from the Department of Control Science and Engineering, Zhejiang University, Hangzhou, China, in 2016. He is currently working as a Professor with the Department of Control Science and Engineering, Zhejiang University. His current research interests include autonomous robots and robot learning.

Dongfeng Bai received the M.S. degree in the Department of Artificial Intelligence from Xi’an Jiaotong University, Xi’an, China, in 2020. He is a researcher in Noah’s Ark Lab, Huawei. His research interests include 3D reconstruction, neural rendering and 3D generation.

Andreas Geiger received his Diploma in computer science and his Ph.D. degree from Karlsruhe Institute of Technology in 2008 and 2013. Currently, he is leading the Autonomous Vision Group at the University of Tubingen. He is also a¨ core faculty member of the Tubingen AI Center.¨ His research interests include computer vision, machine learning and scene understanding with a focus on self-driving vehicles.

![](images/14c915c6dc311061adfc6b3b3da2ba0146ad1647d060092766247fd6bbfdbf01.jpg)

Jiabao Wang is an undergraduate student at Zhejiang University in the College of Information Science and Electronic Engineering. His research interests include 3D computer vision and reinforcement learning.

Yiyi Liao is an assistant professor at Zhejiang University, leading the X-Dimensional Representations Lab. She received her Ph.D. in Control Science and Engineering from Zhejiang University in June 2018 and her B.S. degree from Xi’an Jiaotong University in 2013. Her research interests include 3D vision and scene understanding.

## APPENDIX A IMPLEMENTATION

In this section, we begin by discussing our 3D Gaussian details, encompassing semantic, opacity and depth implementation (A.1). Subsequently, we discuss the difference between 3D softmax and 2D softmax in 3D Semantic Scene Reconstruction (A.2). Finally, we elucidate the evaluation metrics we utilize (A.3).

## A.1 3D Gaussian Details

Following [37], each Gaussian has the following attributes: rotation $( \mathbf { \tilde { R } } _ { g } \in \mathbb { R } ^ { 3 \times 3 } )$ , scale $( \mathbf { S } _ { g } \in \mathbb { R } ^ { 3 \times 1 } )$ , opacity (α) and spherical harmonics (SH). The corresponding 3D covariance matrix $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ can be calculated using the following formula:

$$
\boldsymbol { \Sigma } = \mathbf { R } _ { g } \mathbf { S } _ { g } \mathbf { S } _ { g } ^ { T } \mathbf { R } _ { g } ^ { T }\tag{22}
$$

When provided with a viewing transformation $\mathbf { W } \in \mathbb { R } ^ { 3 \times 3 }$ and the Jacobian of the affine approximation of the projective transformation $\textbf { J } \in \ \mathbb { R } ^ { 3 \times \mathbf { \hat { 3 } } }$ , the covariance matrix $\mathbf { \bar { \mathbf { Z } } ^ { \prime } } \in \mathbb { R } ^ { 3 \times 3 }$ in camera coordinates can be expressed as:

$$
\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \mathbf { J } \mathbf { W } \pmb { \Sigma } \mathbf { W } ^ { T } \mathbf { J } ^ { T } } \end{array}\tag{23}
$$

Following EWA splatting [85], we can skip the third row and column of $\pmb { \Sigma } ^ { \prime }$ to obtain $\textsf { a 2 } \times 2$ covariance matrix with the same structure and properties. For brevity, we still use the notation $ { \Sigma } ^ { \prime } \in  { \mathbb { R } } ^ { 2 \times 2 }$ to denote the 2D covariance matrix.

By considering the projected 3D Gaussian center $\pmb { \mu } \in$ $\mathbb { R } ^ { 2 \times 1 ^ { \smile } }$ and an arbitrary point $\mathbf { x } \in \mathbb { R } ^ { 2 \times 1 }$ on camera coordinates, the opacity $\alpha ^ { \prime }$ of x contributed by this 3D Gaussian can be computed as follows:

$$
\alpha ^ { \prime } = \alpha \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } ) ^ { T } ( \pmb { \Sigma } ^ { \prime } ) ^ { - 1 } ( \mathbf { x } - \pmb { \mu } ) \right)\tag{24}
$$

The color c of each Gaussian can be computed based on the view direction and its corresponding spherical harmonics (SH). Given a set of sorted 3D Gaussians $\mathcal { N }$ along the ray, we obtain the accumulated color via volume rendering:

$$
\pi : \quad \mathbf { C } = \sum _ { i \in \mathcal { N } } \mathbf { c } _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{25}
$$

The same volume rendering technique can be applied to obtain semantic S, depth D and optical flow F. With the given semantic feature $\mathbf { s } _ { i } ,$ depth value $d _ { i } ,$ and Gaussian motion $\mathbf { f } _ { i }$ relative to the camera pose, we can define the semantic rendering, depth rendering, and flow rendering as follows:

$$
\mathbf { S } = \sum _ { i \in \mathcal { N } } \mathrm { s o f t m a x } ( \mathbf { s } _ { i } ) \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{26}
$$

$$
\mathbf { D } = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{27}
$$

$$
\mathbf { F } = \sum _ { i \in \mathcal { N } } \mathbf { f } _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{28}
$$

Note that all the projections and volume rendering techniques mentioned are implemented in CUDA. Calculating the projected 2D opacity $\alpha ^ { \prime }$ on each pixel and sorting

Gaussians based on their distances from the camera takes the majority of computations in the rendering process. These computations need to be performed only once for rendering all modalities, thus maintaining the real-time rendering property of the original 3D Gaussian Splatting.

## A.2 3D Semantic Scene Reconstruction

We utilize Eq. 26, referred to as 3D softmax, to render semantic maps. This is in contrast to most existing NeRF-based semantic reconstruction methods that perform softmax to the accumulated 2D logits [23], [81], described in Eq. 29, referred to as 2D softmax. The fundamental difference between these two rendering techniques lies in the fact that 3D softmax normalizes the logits of each 3D point. This normalization process helps prevent a single point with a significantly high logit value from imposing an overwhelming influence on the overall volume rendering outcome. On the other hand, it also prevents placing 3D points of low logit values in empty space. As a result, 3D softmax is effective in reducing floaters and enhancing the geometry of the reconstruction results. In Eq. E.3, we present a comprehensive analysis of the qualitative and quantitative comparison results between these two rendering methods.

$$
\mathbf { S } _ { \mathrm { 2 D \mathrm { - } n o r m } } = \mathrm { s o f t m a x } \left( \sum _ { i \in \mathcal { N } } \mathbf { s } _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } ) \right)\tag{29}
$$

In the following sections, we refer to our default setting obtained by Eq. 26 as $\mathrm { \bf S } _ { 3 \mathrm { D } _ { - } }$ <sub>norm</sub>.

## A.3 Metrics

Novel View Appearance Synthesis: To assess the quality of novel view appearance synthesis, we utilize the Peak Signalto-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) [79] following the common practice.

Novel View Semantic Synthesis: Following KITTI-360 [44], we evaluate the quality of novel view semantic synthesis via the mean Intersection over Union (mIoU) metric.

3D Semantic Reconstruction: We evaluate 3D semantic reconstruction quality by extracting a 3D semantic point cloud and comparing it with the ground truth LiDAR points. We evaluate both geometric and semantic metrics in the 3D space. Specifically, we evaluate geometric reconstruction quality by measuring the accuracy (acc.) and completeness (comp.). Accuracy measures the average distance from reconstructed points to the nearest LiDAR point, while completeness measures the average distance from LiDAR points to the nearest reconstructed points. In order to measure the semantic quality of the reconstructed point cloud, we map the predicted 3D semantics to the LiDAR points. Concretely, for each point in the LiDAR point cloud, we identify its closest counterpart in the predicted semantic point cloud and allocate a semantic label based on this nearest neighbor. The assigned semantic labels of all LiDAR points are then compared with the 3D semantic segmentation ground truth provided by KITTI-360, evaluated via the mIoU metric. Note that we only use the LiDAR point clouds for evaluation.

![](images/6390d86d26d92ba445c303d0fed99198c5fb501c9a99311d8615c85570f26e4c.jpg)  
(a)

![](images/2d86204098fbb2470418ec166824cc8f16f6b7787cd8a7443f38728a0923a9fd.jpg)  
(b)

![](images/a229546c104de45b6273dd572bef6935f18f464eed63074d0c31ced544e396d1.jpg)

H<sup>ard</sup>

![](images/2a19c8c4feecb077b02e90924469dac3acee69b6a8eabc357f40481dda497e7f.jpg)

(c)  
![](images/6f6f05d4406c98026bb92b47b87a1641f4ba017410f0d091fb894bd8fb28e3dc.jpg)

![](images/b7527db3f4b035b573de262ae9ef70e98cd4f6e5bfbaa7663d284db00dd1b402.jpg)  
(a)

Extreme

![](images/651c1658bdbaf637ddac2bb396ff46dc4daaddec7790bf450cb3273a4bdbb124.jpg)  
(a)

(b)  
![](images/acfbb27accb797775d786804965810e1989a65a5ae8dc71cb1538f3196779824.jpg)  
(b)

(c)  
![](images/902c58cab1a0be444dcce3c10db927bd5c8dafdcbbcd8e993178b4fdbbf6fe08.jpg)  
(c)  
Fig. 20: Scenarios Design. Green, blue, gray, and orange represent the ego vehicle, static actors, normal actors, and aggressive actors, respectively.

3D Tracking: To demonstrate the effectiveness of our model in rectifying noisy 3D tracking results, we evaluate the accuracy of predicted poses compared to ground truth poses in our ablation study. Considering the rotation and translation parameters of a ground truth bounding box denoted as R<sup>ˆ</sup> and <sup>ˆ</sup>t, respectively, and the corresponding parameters of predicted poses, represented as R and t, we employ two metrics for this evaluation following [10]: e<sub>R</sub> quantifies the rotation accuracy, while $e _ { \mathbf { t } }$ assesses the translation accuracy as follows

$$
e _ { \mathbf { R } } = \operatorname { a r c c o s } { \frac { T r ( { \hat { \mathbf { R } } } \cdot \mathbf { R } ^ { - 1 } ) - 1 } { 2 } }
$$

$$
e _ { \mathbf { t } } = \lVert \hat { \mathbf { t } } - \mathbf { t } \rVert _ { 2 }\tag{30}
$$

(31)

where T r represents the trace of a matrix.

Depth Estimation: In our ablation study, we evaluate the depth estimation quality of our different variants. This is achieved by first projecting the LiDAR points acquired at the same frame to the 2D image space, followed by measuring the L2 distance between the projected LiDAR depth and our method. Considering the projected LiDAR depth is sparse, our assessment focuses solely on pixels with valid LiDAR projections when calculating the L2 distance.

## APPENDIX B

## DATA

In this section, we present details of datasets on which we conducted our experiments, including KITTI [26], Virtual

![](images/3e2f12261544947cec6b50f8dabb61c63835c14a22d117e54d07bb343b38de98.jpg)

![](images/a0303da34a85cba9dec55cc0ffb5779ab12e9ddc4c0fd97f5751174badfe864f.jpg)

![](images/f1822e78ce94d71518b6f9b13c18114d5e01dca313db0c65e9b1e693c1efd088.jpg)  
Ours w/ S<sub>2D norm</sub>  
Ours w/ S<sub>3D norm</sub>

Fig. 21: Qualitative Comparison of 3D and 2D softmax results. Note that normalizing semantic logits in 3D space (Ours $\mathrm { ~ \bf ~ w / \Sigma } \mathbf { S } _ { 3 \mathrm { D \_ n o r m } } )$ clearly reduces floaters and yields better 3D semantic reconstruction than the 2D normalization counterpart (Ours $\mathbf { w } / \mathbf { S } _ { 2 \mathrm { D } \_ \mathrm { n o r m } } )$

KITTI 2 (vKITTI) [7], KITTI-360 [44], Waymo [58], nuScenes [8] and PandaSet [69].

<table><tr><td></td><td>Pre.</td><td>+πRGB</td><td>+ Affine</td><td>+ π Semantic</td><td>+ π Flow</td></tr><tr><td>Speed (ms)</td><td>6.25</td><td>8.13 (+1.88)</td><td>8.54 (+0.41)</td><td>9.70 (+1.16)</td><td>10.17 (+0.47)</td></tr></table>

TABLE 10: Time consumption breakdown of our method.
<table><tr><td></td><td colspan="3">KITTI Scene02</td><td colspan="3">KITTI Scene06</td><td colspan="3">vKITTI Scene02</td><td colspan="3">vKITTI Scene06</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>NSG [52]</td><td>22.51</td><td>0.653</td><td>0.397</td><td>23.38</td><td>0.717</td><td>0.243</td><td>23.50</td><td>0.718</td><td>0.352</td><td>26.42</td><td>0.811</td><td>0.170</td></tr><tr><td>MARS [68]</td><td>22.95</td><td>0.728</td><td>0.145</td><td>27.01</td><td>0.883</td><td>0.062</td><td>29.80</td><td>0.950</td><td>0.034</td><td>32.71</td><td>0.959</td><td>0.023</td></tr><tr><td>Ours</td><td>25.89</td><td>0.829</td><td>0.092</td><td>28.90</td><td>0.925</td><td>0.016</td><td>30.73</td><td>0.955</td><td>0.018</td><td>33.31</td><td>0.963</td><td>0.010</td></tr></table>

TABLE 11: Novel View Appearance on Dynamic Scenes with ground truth 3D trackings.

![](images/ce3544ae67e38d3aae4e78ea8b2d752b266770267e6815f7975f123c9851da7f.jpg)  
Fig. 22: Qualitative Comparison with Nerfacto on 2D space. The Pseudo GT column represents the semantic maps that are predicted by [4] on GT RGB images.

![](images/5d5b48fd83af928209dba122046be70c6d609690521cec3d246f7623840deb89.jpg)  
Fig. 23: Qualitative Comparison with Nerfacto on 3D space. The semantic point cloud extracted from Semantic Nerfacto struggles to faithfully represent the geometry.

KITTI: Following NSG [52] and MARS [68], we select frames 140 to 224 from Scene02 and frames 65 to 120 from Scene06 on KITTI for conducting our experiments.

vKITTI: Virtual KITTI 2 is a synthetic dataset that closely resembles the scenes present in KITTI. In line with the settings outlined in NSG and MARS, we conduct experiments on exactly the same frames from Scene02 and Scene06.

<table><tr><td rowspan="2">02</td><td></td><td>eR ↓</td><td>et ↓</td></tr><tr><td>QD-3DT Ours</td><td>0.027 0.018</td><td>0.215 0.108</td></tr><tr><td>06</td><td>QD-3DT Ours</td><td>0.017 0.012</td><td>0.046 0.033</td></tr></table>

TABLE 12: Quantitative Pose Comparison on two KITTI sequences.

KITTI-360: We perform experiments on KITTI-360, encompassing both static and dynamic scenes. For the tasks of novel view synthesis and novel semantic synthesis on the leaderboard, we conduct experiments on the sequences provided by the official dataset. We also explore dynamic scenes, such as frames 11322 to 11381 from sequence 00, as showcased in our teaser. Additionally, We construct our closed-loop benchmark based on KITTI-360, using both the front two stereo cameras and the side two fisheye cameras. We convert images from the fisheye cameras to perspective images using the model of [48] as the projection model.

Waymo: We perform experiments on Waymo to compare with StreetGaussian [70] and NeuRAD [61] in extrapolated views. We select all frames from sequences segment-10676267326664322837 and segment-9385013624094020582 for the experiments. Every 5th frame is dropped as the test frame, and use the front, front-left and front-right cameras as inputs. Additionally, we construct HUGSIM based on

![](images/73e9c66e3e941a63d943570a272a796b2fd330ddaaa3210129e4d9ebea869c87.jpg)  
Fig. 24: Visualization of Optimization Progress. Our method jointly optimizes the static background and the trajectory of the dynamic foreground objects. By integrating physical constraints using the unicycle model, our method allows for recovering a smooth trajectory from noisy 3D bounding boxes. To prevent visual clutter, we exclude point clouds of the dynamic object and only visualize the bounding boxes.

![](images/2cecb80997a3a22a56971a357d01314cfacea0fc1e4a79b20ecb3f3c14dbe316.jpg)  
Fig. 25: Pose comparison with QD-3DT.

Waymo using the same settings.

Nuscenes: NuScenes is used to compare our method with StreetGaussian, NeuRAD, and RoGS [21] in extrapolated views. We select all frames from sequences 0051, 0411, and 0655. Every 5th frame is dropped as the test frame, and we use all 6 surrounding cameras as inputs. HUGSIM benchmark is also constructed based on nuScenes. It is important to note that the camera extrinsics of nuScenes are not well refined, so we apply a rigid bundle adjustment to obtain more accurate camera extrinsics.

PandaSet: Scenes from PandaSet are also part of the HUGSIM benchmark. We use all 6 surrounding cameras in PandaSet and drop every 5th frame as the test frame during scene reconstruction training.

## APPENDIX C

## HUGSIM BENCHMARK

## C.1 Scenario Design

We design easy, medium, hard and extreme levels for our HUGSIM benchmark. The easy scenarios consist of static scenes or replayed dynamic vehicle behaviors. The design of scenarios for the medium, hard, and extreme levels is shown in Fig. 20.

Medium: Scenario (a) simulates a stationary vehicle in the front of ego vehicle. Scenario (b) simulates a vehicle driving ahead the ego but may slower than ego vehicle. Scenario (c) simulates a vehicle driving ahead but suddenly stops.

Hard: Scenario (a) simulates a vehicle driving in reverse. Scenario (b) simulates a stationary vehicle ahead and a vehicle reversing in the side lane. Scenario (c) simulates an oncoming aggressive vehicle attempting to attack the ego vehicle.

Extreme: Scenario (a) simulates a situation similar to Hard scenario (c), but with the aggressive vehicle having a larger attack frequency and being more likely to select the most aggressive trajectory. Scenario (b) simulates an aggressive vehicle suddenly changing lanes in front of the ego vehicle. Scenario (c) simulates multiple reverse-driving vehicles.

## C.2 Implementation of Tested AD Algorithms

We implement inference scripts for UniAD [33], VAD [35], and LTF [14] to integrate these AD algorithms with HUGSIM. We build communication bridges based on named pipelines in memory, which transfer data at ultrahigh speed, with nearly no additional time consumption.

UniAD: UniAD proposes an end-to-end AD algorithm, enabling flexible intermediate representations and exchanging multi-task knowledge toward planning. The inference code of the official UniAD implementation couples observations, CAN-bus data, and ground-truth information together. We re-implement the inference script, which only requires RGB image observations and sensor poses, based on the official code.

VAD: VAD proposes an end-to-end vectorized paradigm for AD and is developed on the UniAD codebase. We reimplemented the inference script for VAD in a similar way to UniAD.

LTF: Latent TransFuser (LTF) is an image-only version of Transfuser [14], replacing the LiDAR BEV histogram input with a positional encoding. The original version of LTF is trained on CARLA [19], we use the version implemented in NAVSIM [18] which is trained on nuplan [9], offering better alignment with the real world.

<table><tr><td></td><td></td><td colspan="4">UniAD [33]</td><td colspan="4">VAD [35]</td><td colspan="4">LTF [14]</td></tr><tr><td></td><td></td><td>Easy</td><td>Medium</td><td>Hard</td><td>Extreme</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Extreme</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Extreme</td></tr><tr><td rowspan="5">KITTI-360 [44]</td><td>NC↑</td><td>0.459</td><td>0.379</td><td>0.315</td><td>0.242</td><td>0.333</td><td>0.322</td><td>0.253</td><td>0.171</td><td>0.330</td><td>0.170</td><td>0.181</td><td>0.133</td></tr><tr><td>DAC ↑</td><td>0.712</td><td>0.711</td><td>0.754</td><td>0.750</td><td>0.603</td><td>0.604</td><td>0.609</td><td>0.657</td><td>0.646</td><td>0.627</td><td>0.622</td><td>0.633</td></tr><tr><td>TTC↑</td><td>0.271</td><td>0.193</td><td>0.131</td><td>0.091</td><td>0.204</td><td>0.162</td><td>0.105</td><td>0.090</td><td>0.290</td><td>0.125</td><td>0.132</td><td>0.080</td></tr><tr><td>COM↑</td><td>0.672</td><td>0.652</td><td>0.555</td><td>0.527</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.996</td><td>1.000</td><td>1.000</td></tr><tr><td>Rc ↑</td><td>0.166</td><td>0.117</td><td>0.122</td><td>0.094</td><td>0.138</td><td>0.125</td><td>0.122</td><td>0.110</td><td>0.245</td><td>0.138</td><td>0.121</td><td>0.111</td></tr><tr><td rowspan="7">Waymo [58]</td><td>HD-Score↑</td><td>0.047</td><td>0.019</td><td>0.017</td><td>0.006</td><td>0.029</td><td>0.014</td><td>0.014</td><td>0.008</td><td>0.080</td><td>0.028</td><td>0.017</td><td>0.003</td></tr><tr><td>NC↑</td><td>0.900</td><td>0.903</td><td>0.792</td><td>0.512</td><td>0.698</td><td>0.493</td><td>0.526</td><td>0.416</td><td>0.855</td><td>0.544</td><td>0.451</td><td>0.333</td></tr><tr><td>DAC↑</td><td>0.863</td><td>0.903</td><td>0.771</td><td>0.915</td><td>0.568</td><td>0.679</td><td>0.694</td><td>0.726</td><td>0.779</td><td>0.830</td><td>0.861</td><td>0.852</td></tr><tr><td>TTC↑</td><td>0.862</td><td>0.778</td><td>0.711</td><td>0.461</td><td>0.641</td><td>0.363</td><td>0.437</td><td>0.299</td><td>0.833</td><td>0.518</td><td>0.428</td><td>0.258</td></tr><tr><td>COM↑</td><td>0.890</td><td>0.767</td><td>0.712</td><td>0.483</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.992</td><td>0.995</td><td>0.995</td><td>1.000</td></tr><tr><td>Rc ↑</td><td>0.784</td><td>0.547</td><td>0.590</td><td>0.326</td><td>0.323</td><td>0.266</td><td>0.259</td><td>0.272</td><td>0.698</td><td>0.371</td><td>0.357</td><td>0.261</td></tr><tr><td>NC↑</td><td>HD-Score↑ 0.664</td><td>0.419</td><td>0.401</td><td>0.171</td><td>0.154</td><td>0.093</td><td>0.110</td><td>0.085</td><td>0.546</td><td>0.200</td><td>0.219</td><td>0.071</td></tr><tr><td rowspan="5">Nuscenes [8]</td><td></td><td>0.823</td><td>0.761</td><td>0.781</td><td>0.688</td><td>0.765</td><td>0.486</td><td>0.516 0.467</td><td></td><td>0.835 0.587</td><td></td><td>0.513</td><td>0.401</td></tr><tr><td>DAC↑</td><td>0.967</td><td>0.925</td><td>0.929</td><td>0.973</td><td>0.824</td><td>0.917</td><td>0.941</td><td>0.971</td><td>0.858</td><td>0.872</td><td>0.872</td><td>0.928</td></tr><tr><td>TTC↑</td><td>0.787</td><td>0.652</td><td>0.673</td><td>0.519</td><td>0.692</td><td>0.365</td><td>0.376</td><td>0.396</td><td>0.781</td><td>0.527</td><td>0.453</td><td>0.353</td></tr><tr><td>COM↑</td><td>0.880</td><td>0.728</td><td>0.729</td><td>0.642</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.994</td><td>0.995</td><td>1.000</td><td>0.993</td></tr><tr><td>Rc ↑</td><td>0.703</td><td>0.510</td><td>0.516</td><td>0.291</td><td>0.540</td><td>0.340</td><td>0.383</td><td>0.311</td><td>0.847</td><td>0.542</td><td>0.498</td><td>0.357</td></tr><tr><td rowspan="7">Pandaset [69]</td><td>HD-Score↑</td><td>0.589</td><td>0.378</td><td>0.365</td><td>0.168</td><td>0.348</td><td>0.132</td><td>0.204</td><td>0.158</td><td>0.616</td><td>0.307</td><td>0.226</td><td>0.171</td></tr><tr><td>NC↑</td><td>0.914</td><td>0.851</td><td>0.771</td><td>0.732</td><td>0.847</td><td>0.529</td><td>0.474</td><td>0.388</td><td>0.945</td><td>0.597</td><td>0.481</td><td>0.265</td></tr><tr><td>DAĆ↑</td><td>0.998</td><td>0.933</td><td>0.987</td><td>0.947</td><td>0.963</td><td>0.990</td><td>0.995</td><td>0.993</td><td>0.987</td><td>1.000</td><td>0.999</td><td>0.991</td></tr><tr><td>TTC↑</td><td>0.913</td><td>0.792</td><td>0.685</td><td>0.644</td><td>0.793</td><td>0.270</td><td>0.235</td><td>0.251</td><td>0.919</td><td>0.560</td><td>0.467</td><td>0.182</td></tr><tr><td>COM↑</td><td>0.871</td><td>0.734</td><td>0.692</td><td>0.689</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>Rc ↑</td><td>0.691</td><td>0.473</td><td>0.389</td><td>0.329</td><td>0.548</td><td>0.349</td><td>0.257</td><td>0.229</td><td>0.947</td><td>0.579</td><td>0.498</td><td>0.293</td></tr><tr><td>HD-Score↑</td><td>0.647</td><td>0.366</td><td>0.308</td><td>0.226</td><td>0.442</td><td>0.158</td><td>0.087</td><td>0.079</td><td>0.871</td><td>0.450</td><td>0.331</td><td>0.080</td></tr><tr><td rowspan="7">Average</td><td>NC↑</td><td>0.774</td><td>0.724</td><td>0.665</td><td>0.544</td><td>0.661</td><td>0.457</td><td>0.443 0.361</td><td></td><td>0.741</td><td>0.474</td><td>0.407</td><td>0.283</td></tr><tr><td>DAC ↑</td><td>0.885</td><td>0.868</td><td>0.860</td><td>0.896</td><td>0.739</td><td>0.798</td><td>0.810</td><td>0.837</td><td>0.817</td><td>0.832</td><td>0.839</td><td>0.851</td></tr><tr><td>TTC↑</td><td>0.708</td><td>0.604</td><td>0.550</td><td>0.429</td><td>0.582</td><td>0.290</td><td>0.288</td><td>0.259</td><td>0.706</td><td>0.433</td><td>0.370</td><td>0.218</td></tr><tr><td>COM↑</td><td>0.828</td><td>0.720</td><td>0.672</td><td>0.585</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.996</td><td>0.997</td><td>0.999</td><td>0.998</td></tr><tr><td>Rc ↑</td><td>0.586</td><td>0.412</td><td>0.404</td><td>0.260</td><td>0.387</td><td>0.270</td><td>0.255</td><td>0.230</td><td>0.684</td><td>0.407 0.246</td><td>0.369 0.198</td><td>0.255 0.081</td></tr><tr><td>HD-Score↑</td><td>0.487</td><td>0.295</td><td>0.273</td><td>0.143</td><td>0.243</td><td>0.099</td><td>0.104</td><td>0.082</td><td>0.528</td></table>

TABLE 13: AD Algorithms Evaluating on the HUGSIM Benchmark

## APPENDIX D

## BASELINES

In this section, we discuss the baselines against which we compare our approach, including NSG [52], MARS [68], PNF [39], Semantic Nerfacto [60], StreetGaussian [70], Neu-RAD [61] and RoGS [21].

NSG: NSG is the pioneering method that introduces the decomposition of dynamic scenes into static background and dynamic foreground components. They propose a learned scene graph representation that enables efficient rendering of novel scene arrangements and viewpoints. However, the official source code provided by NSG often encounters issues when training on KITTI Scene02. Therefore, we utilize the version implemented by the authors of MARS, which is more stable and yields slightly improved results compared to the original version.

MARS: We utilize the latest version of the code provided by the official MARS repository. This latest version incorporates bug fixes and includes additional training iterations, resulting in improved performance. In fact, the updated version achieves a notable improvement of 3 to 4 dB on PSNR compared to the numbers reported in the original paper.

PNF: Since PNF is not open-source, we directly compare our method to their submission on the KITTI-360 leaderboard regarding novel view appearance & semantic synthesis. To the best of our knowledge, PNF is the only work that considers the optimization of noisy 3D bounding boxes of dynamic objects. In our ablation study, we conduct a na¨ıve baseline that optimizes the 3D bounding boxes of each frame independently, which can be considered as a reimplementation of PNF’s bounding box optimization in our framework.

Semantic Nerfacto: For the evaluation of 3D semantic point cloud geometry, we compare our results with Semantic Nerfacto [60] as an alternative to PNF [39]. Nerfacto [60] is an integration of several successful methods that demonstrate strong performance on real data. It incorporates camera pose refinement, per-image appearance embedding, proposal sampling, scene contraction, and hash encoding within its pipeline. Additionally, Nerfacto includes a semantic head in its framework, enabling the generation of meaningful semantic maps, as demonstrated in E.2.

StreetGaussian: StreetGaussian is a concurrent work to HUGS [83], proposing a dynamic urban scene reconstruction algorithm also based on 3D Gaussian Splatting [37], with primary experiments conducted on Waymo. Street-Gaussian requires both LiDAR and RGB data as inputs.

NeuRAD: NeuRAD is also a concurrent work to HUGS, proposing a NeRF-based dynamic urban scene reconstruction method. NeuRAD integrates various strategies, achieving state-of-the-art performance across many AD datasets.

<table><tr><td></td><td> $\mathrm { S e q 0 1 \ m I o U _ { \mathrm { c l s } } \uparrow }$ </td><td> $\mathrm { S e q 0 2 m I o U _ { \mathrm { c l s } } \uparrow }$ </td><td> $\mathrm { S e q 0 3 \ m I o U _ { \mathrm { c l s } } \uparrow }$ </td><td> $\mathrm { \underline { { A v e r a g e ~ m I o U _ { c l s } } } \uparrow }$ </td></tr><tr><td>Ours  $\overline { { \mathbf { w } / \mathbf { \Lambda } } } \mathbf { S } _ { 2 \mathrm { D _ { \overline { { \alpha } } n o r m } } }$ </td><td> $\overline { { 0 . 4 2 7 } }$ </td><td> $\overline { { 0 . 3 6 3 } }$ </td><td> $\overline { { 0 . 4 1 6 } }$ </td><td> $\overline { { 0 . 4 0 2 } }$ </td></tr><tr><td>Ours  $\mathrm { \bf w } / \mathrm { \bf S } _ { 3 \mathrm { D \_ n o r m } }$ </td><td>0.544</td><td>0.452</td><td>0.520</td><td>0.505</td></tr></table>

TABLE 14: Comparison on 3D and 2D Semantic Softmax on KITTI-360.

![](images/bc5dbcee71e81c57a1bee0ad265640e85d49c4684d83d42d159bbf59fba74902.jpg)  
Fig. 26: Qualitative Comparison on depth. In the presence of the semantic loss $\mathcal { L } _ { \mathbf { S } } ,$ , We set the sky region’s depth infinite based on its semantic label.

Like StreetGaussian, NeuRAD requires both LiDAR and RGB data as inputs. It is worth noting that NeuRAD is the reconstruction rendering method used in NeuroNCAP [45].

RoGS: RoGS is a concurrent work to HUGSIM, proposing an efficient method tailored to ground reconstruction based on 2D Gaussian Splatting [34] in nuScenes. It requires only RGB images for ground reconstruction and uniformly tiles Gaussians on the ground plane.

## APPENDIX E

## ADDITIONAL EXPERIMENT RESULTS

## E.1 Time Consumption Breakdown

Table 10 shows our detailed runtime breakdown as various components are incrementally enabled. Preparation (Pre.) contains operations like tile partition and Gaussian sorting. π denotes volume rendering, and affine denotes affine transform. Other components like unicycle model, dynamic decomposition, and depth rendering are excluded as they hardly consume any additional time.

## E.2 Additional Comparison Experiments

Dynamic Scene with GT 3D Bounding Boxes: Despite not being our primary focus, we additionally provide a comparison with NSG and MARS using ground truth 3D trackings. In this setting, our approach demonstrates superior performance across all test scenes, see Table 11.

Details of Comparison with Semantic Nerfacto: While Semantic Nerfacto excels at rendering meaningful novel view semantic images (as seen in Fig. 22), Fig. 23 shows it struggling to accurately reconstruct correct geometry. Following the common practice of NeRF-based semantic reconstruction methods [60], we apply 2D softmax to Semantic Nerfacto. when we attempted to apply the 3D Softmax technique to Nerfacto, it did not yield better results compared to using 2D softmax. The results can be attributed to the incorrect of Nerfacto’s 3D geometry. Instead of adjusting 2D logits with large-scale logits in 3D, the use of 3D softmax prevents the ”cheating” approach by normalizing logits in 3D space. However, this normalization requirement necessitates sufficiently accurate geometry for satisfactory results.

Comparisons with Tracking Methods: To further compare with off-the-shelf tracking methods, we show the performance of QD-3DT [32] and our optimized pose initialized with [32] in Table 12 and qualitatively illustrate the poses of one vehicle in Fig. 25. Our method consistently improves [32] across two KITTI scenes.

## E.3 Additional Ablation Experiments

3D and 2D Semantic Softmax: We provide more 3D and 2D semantic logits softmax comparison in Fig. 21 and Fig. 14. As can be seen, normalizing semantic logits in 3D space leads to notable qualitative and quantitative improvement compared to 2D space normalization.

Improvements on Geometry: We now qualitatively examine how the semantic loss $\mathcal { L } _ { \mathbf { S } }$ impact the geometry, as shown in Fig. 26. Th figure reveals that incorporating the semantic loss improves the underlying geometry. It’s important to note that when the semantic loss $\mathcal { L } _ { \mathbf { S } }$ is active, the sky region of the depth maps in Fig. 26 is set to infinite.

## E.4 Visualization of Optimization Progress

We present the visualization of the optimization progress for both the noisy bounding boxes and the background semantic point cloud in Fig. 24. Using noisy 3D bounding boxes as input, our approach optimizes both the background and the poses of the bounding boxes simultaneously. As evident, the application of physical constraints derived from the unicycle model results in a smooth trajectory for the bounding boxes.

## E.5 HUGSIM Benchmark evaluation results

We present the results of HD-Score and sub-scores N C, $D A \bar { C } , T T C , C O M ,$ , and R<sub>c</sub> for evaluating UniAD [33], VAD [35], and LTF [14] on our HUGSIM benchmark in Table 13. These results are identical to those in Fig. 18, but presented in a more detailed format.