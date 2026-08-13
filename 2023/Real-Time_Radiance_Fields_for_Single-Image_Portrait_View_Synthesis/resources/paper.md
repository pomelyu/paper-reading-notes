                                        Real-Time Radiance Fields for Single-Image Portrait View Synthesis
                                        ALEX TREVITHICK∗ , University of California San Diego, USA
                                        MATTHEW CHAN and MICHAEL STENGEL, NVIDIA, USA
                                        ERIC R. CHAN∗ , Stanford University, USA
                                        CHAO LIU, ZHIDING YU, and SAMEH KHAMIS, NVIDIA, USA
                                        MANMOHAN CHANDRAKER and RAVI RAMAMOORTHI, University of California San Diego, USA
                                        KOKI NAGANO, NVIDIA, USA




arXiv:2305.02310v1 [cs.CV] 3 May 2023
                                        Fig. 1. Given a single RGB input image, our method generates 3D-aware images and geometry of an object (e.g., faces [top row] and cats [bottom row, left]) in
                                        real-time, while the state-of-the-art 3D GAN inversion [Chan et al. 2022] does not generate a satisfactory result after 20 mins of fine-tuning [Roich et al.
                                        2021] (top right). Our method can also be applied to a video frame-by-frame for video-based novel view synthesis (bottom row, right). Ours (LT) refers to a
                                        lightweight faster version of our model that has almost the same quality as the full model. Credits to Erik (HASH) Hersman and 2017 Canada Summer Games.

                                        We present a one-shot method to infer and render a photorealistic 3D rep-                    augmentation strategy, and a well-designed loss function for synthetic data
                                        resentation from a single unposed image (e.g., face portrait) in real-time.                  training. We benchmark against the state-of-the-art methods, demonstrating
                                        Given a single RGB input, our image encoder directly predicts a canonical                    significant improvements in robustness and image quality in challenging
                                        triplane representation of a neural radiance field for 3D-aware novel view                   real-world settings. We showcase our results on portraits of faces (FFHQ)
                                        synthesis via volume rendering. Our method is fast (24 fps) on consumer                      and cats (AFHQ), but our algorithm can also be applied in the future to other
                                        hardware, and produces higher quality results than strong GAN-inversion                      categories with a 3D-aware image generator.
                                        baselines that require test-time optimization. To train our triplane encoder
                                        pipeline, we use only synthetic data, showing how to distill the knowledge                   CCS Concepts: • Computing methodologies → Image-based render-
                                        from a pretrained 3D GAN into a feedforward encoder. Technical contribu-                     ing.
                                        tions include a Vision Transformer-based triplane encoder, a camera data
                                                                                                                                     Additional Key Words and Phrases: View Synthesis, Inverse Rendering,
                                        ∗ This project was initiated and substantially carried out during an internship at NVIDIA.
                                                                                                                                     Neural Radiance Field
                                        Authors’ addresses: Alex Trevithick, University of California San Diego, La Jolla, USA;      ACM Reference Format:
                                        Matthew Chan; Michael Stengel, NVIDIA, Santa Clara, USA; Eric R. Chan, Stanford
                                        University, Stanford, USA; Chao Liu; Zhiding Yu; Sameh Khamis, NVIDIA, Santa Clara,          Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu,
                                        USA; Manmohan Chandraker; Ravi Ramamoorthi, University of California San Diego,              Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi,
                                        La Jolla, USA; Koki Nagano, NVIDIA, Santa Clara, USA.                                        and Koki Nagano. 2023. Real-Time Radiance Fields for Single-Image Portrait
                                                                                                                                     View Synthesis. ACM Trans. Graph. 1, 1, Article 1 (January 2023), 15 pages.
                                        Permission to make digital or hard copies of part or all of this work for personal or
                                        classroom use is granted without fee provided that copies are not made or distributed
                                                                                                                                     https://doi.org/10.1145/3592460
                                        for profit or commercial advantage and that copies bear this notice and the full citation
                                        on the first page. Copyrights for third-party components of this work must be honored.
                                        For all other uses, contact the owner/author(s).                                             1   INTRODUCTION
                                        © 2023 Copyright held by the owner/author(s).
                                        0730-0301/2023/1-ART1                                                                        Digitally reproducing the 3D appearance of an object from a single
                                        https://doi.org/10.1145/3592460                                                              image is a long-standing goal for computer graphics and vision.
                                                          Not like typical StyleGAN encoder, this work predicts
                                                          the tri-plane directly instead of the latent code                                   ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:2   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. 2. Comparison to the state-of-the-art 3D GAN[Chan et al. 2022] with test-time fine tuning[Roich et al. 2021] (EG3D-PTI). Single-view 3D GAN inversion
approaches trade off the 2D reconstruction quality and the 3D effects. When fine tuned longer (7500 iterations), EG3D-PTI can capture the same fine-scale
details as ours (𝐿𝑃𝐼 𝑃𝑆 = 0.199), but the quality of another view starts to degrade. On the other hand, our method captures out-of-domain details (e.g.,
emblem) in one-shot while producing realistic rendering of another view, and operating in real-time. Credit to Obama White House.


Interactive synthesis of photorealistic novel views opens new possi-                   Table 1. Time taken to lift the input image to 3D (Encoding) and render
bilities for AR/VR, and for 3D telepresence and videoconferencing                      (Render) a 3D representation given an input image on a single RTX 3090
                                                                                       GPU. The end-to-end runtime with our model and our lightweight model
when applied to humans. In this work, we propose a technique to
                                                                                       (LT) is significantly faster than NeRF-based baselines. † ROME employs
infer a 3D representation for real-time view synthesis given a single                  2D-based neural rendering with mesh-based neural textures, producing
portrait-style input image (e.g., of a human face, see Fig. 1).                        the output at 256x256 resolution; it also requires a segmentation mask and
   Recently, 3D aware-image generation approaches (e.g., [Chan                         detected keypoints from off-the-shelf models which requires around 200ms.
et al. 2022; Deng et al. 2022; Skorokhodov et al. 2022]) demonstrated
unconditional generation of photorealistic 3D representations from
                                                                                             Time        H.NeRF     ROME      EG3D-PTI      Ours     Ours (LT)
a collection of single-view 2D images by combining NeRF-based
representations [Mildenhall et al. 2020] and GANs [Goodfellow                                Encoding       60s      60ms†      2 mins      40ms       16ms
et al. 2014]. Notably, EG3D [Chan et al. 2022] proposed an efficient                         Render        58ms      31ms        24ms       24ms       24ms
triplane 3D representation and demonstrated real-time 3D-aware
image rendering with quality comparable to 2D GANs. Once trained,
                                                                                       available. Our work may motivate applications such as temporally
the 3D GAN generators can be frozen and used for single-image
                                                                                       consistent view synthesis; Fig. 1 (bottom right) shows our method
3D reconstruction tasks via GAN inversion [Karras et al. 2020] and
                                                                                       applied to a video in a frame-by-frame fashion without any special
test-time fine tuning [Roich et al. 2021]. However, there are a few
                                                                                       handling.
challenges in this 3D-GAN inversion-based methods. (1) Due to the
                                                                                         In summary, contributions of our work include:
multi-view nature of training a NeRF, it needs careful optimization
objectives and additional 3D priors [Xie et al. 2022a; Yin et al. 2022]                    • We propose a feed forward encoder model to directly infer a
in the single view setting to avoid unsatisfactory results on novel                          triplane 3D representation from an input image. No test-time
views and corrupted geometry (see Fig. 6). Fig. 2 shows the tradeoff                         optimization is needed.
in the SOTA single-view 3D GAN inversion pipeline. (2) The test-                           • We present a new strategy for training a feed forward triplane
time optimization requires an accurate camera pose as input or to                            encoder for 3D inversion using only synthetic data generated
be jointly optimized [Ko et al. 2023]. (3) The above optimization for                        from a pre-trained 3D-aware image generator.
every single image is time-consuming, limiting the technique for                           • We demonstrate that our method can infer a photorealistic
real-time video applications.                                                                3D representation in real-time given a single unposed image.
   In this paper, we present a one-shot approach to lift an input 2D                         Together with our Transformer-based encoder and on-the-
portrait image to 3D in real-time (24fps on consumer hardware, see                           fly augmentation strategy, our method can robustly handle
Tab. 1). Unlike previous work that reuses a pre-trained generator,                           challenging input images of side views and occlusions.
we train an encoder end-to-end that directly predicts the triplane 3D
features from a single input image. In contrast to prior works that                    2    RELATED WORK
use multiview real image acquisition setups, we do not need any real                   Our work touches on light fields, few-shot view synthesis, learn-
images at all, nor do we require time-consuming physically-based                       ing with synthetic data, 3D-aware portrait generation, and GAN
rendering of high-quality and expensive face assets.                                   inversion. Our focus is on real-time view synthesis from a single
   Instead, we fully supervise the training of our triplane encoder                    image, and we do not address portrait relighting or editing. Tab. 1
for novel view synthesis using multiview-consistent synthetic data                     summarizes runtime for inferring 3D representations from an input
generated from a pre-trained 3D GAN. Together with our data aug-                       and rendering. Our one-shot method is three orders of magnitude
mentation strategies and Transformer-based encoder, we present                         faster than the NeRF-based state-of-the-art methods for inference,
a model which can handle challenging real-world input images in-                       enabling a real-time pipeline.
cluding occlusion and three-quarter views. We showcase our results
                                                                                          Light Fields and Image-Based Rendering. View synthesis or image-
on human and cat face categories in this paper, but the methodology
                                                                                       based rendering has a long history in computer graphics and vi-
can apply to any category for which 3D-aware image generators are
                                                                                       sion [Chen and Williams 1993; McMillan and Bishop 1995], and has

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                   Real-Time Radiance Fields for Single-Image Portrait View Synthesis    • 1:3


often been framed in terms of reconstructing the light field [Gortler       2011], they don’t generalize beyond humans. Recently, 3D aware-
et al. 1996; Levoy and Hanrahan 1996]. However, those methods typ-          image generation approaches [Chan et al. 2021; Nguyen-Phuoc et al.
ically required hundreds of views. Subsequent light field approaches        2019; Niemeyer and Geiger 2021; Schwarz et al. 2020] started to
demonstrated few-shot general [Kalantari et al. 2016] and even sin-         tackle the problem of unconditional generation of photorealistic
gle image view synthesis for categories [Srinivasan et al. 2017], but       3D representations from a collection of single-view 2D images. By
required light field camera training data. More recently, neural-field      combining neural volumetric rendering[Mildenhall et al. 2020] and
based approaches [Mildenhall et al. 2020; Xie et al. 2022b] combine         generative adversarial networks (GANs)[Goodfellow et al. 2014],
recent neural implicit 3D representations [Chen and Zhang 2019;             recent 3D GAN approaches[Chan et al. 2022; Deng et al. 2022; Gu
Mescheder et al. 2019; Park et al. 2019; Sitzmann et al. 2019] with vol-    et al. 2021; Or-El et al. 2022; Rebain et al. 2022; Skorokhodov et al.
ume rendering for novel view synthesis, but require a large number          2022; Xiang et al. 2022; Xu et al. 2022b; Zhang et al. 2022; Zhou et al.
of input images per scene.                                                  2021] started to demonstrate an ability to generate high-resolution
                                                                            multi-view consistent images and geometry of a category of objects.
   Few-shot novel view synthesis. Some recent work extends NeRF             We adapt the efficient triplane 3D representation from EG3D [Chan
for training from even a single view [Xu et al. 2022a] or for few-shot      et al. 2022] and demonstrate single-view novel view synthesis on
novel view synthesis using fully implicit 3D representations [Jang          similar categories.
and Agapito 2021; Li et al. 2022; Trevithick and Yang 2021; Yu et al.
2021], 3D convolutions [Chen et al. 2021; Yu et al. 2022], or Trans-           3D GAN inversion. Following the success of GAN inversion in
formers [Lin et al. 2023; Wang et al. 2021b]. However these ap-             2D domains for image editing and manipulations[Alaluf et al. 2021;
proaches do not generate novel views in real-time. Moreover, all of         Dinh et al. 2022; Richardson et al. 2021; Tov et al. 2021; Wang et al.
the above approaches need multi-view images to train their mod-             2022c], existing 3D GAN inversion methods [Ko et al. 2023; Lin
els. Our method, on the other hand, only needs synthetic images             et al. 2022; Sun et al. 2022] project a given image to variants of the
generated from a pre-trained 3D GAN, which is trained by a collec-          pre-trained StyleGAN2 latent space [Abdal et al. 2019; Karras et al.
tion of single-view images. FWD [Cao et al. 2022] builds on top of          2020]. Assuming multiview images, FreeStyleGan [Leimkühler and
SynSin[Wiles et al. 2020] for real-time novel view synthesis using          Drettakis 2021] proposes to map projected camera coordinates to a
depth-based image warping, but requires depth data from multi-              subject-specific StyleGAN2 latent space which allows the subject
view stereo or a depth sensor. Yet another family of approaches             to be rendered from specified cameras under the constraints of the
is the geometry-free method[Ren and Wang 2022; Rombach et al.               StyleGAN prior. While this global latent space provides an additional
2021; Sajjadi et al. 2022], but they need a large number of images to       ability for 3D-aware portrait editing, the StyleGAN2 latent space
learn precise ray transformations; otherwise it may lead to blurry          trades off reconstruction fidelity for editability, making the exact
or multiview inconsistent results.                                          reconstruction of the input image challenging. Thus, existing 3D
                                                                            GAN inversion approaches require an approximate camera pose
   Learning with synthetic data. Synthetic data provides useful su-
                                                                            and slight generator weight tuning [Feng et al. 2022; Roich et al.
pervision for training a deep learning model when ground truth data
                                                                            2021] at test time to reconstruct out-of-domain input images. Our
is not available. Previous methods used synthetic data for various
                                                                            feed forward encoder takes an unposed image as input and does
deep learning-based tasks such as dense visual alignment [Peebles
                                                                            not need test-time optimization for camera poses unlike concurrent
et al. 2022], 3D face reconstruction [Pan et al. 2021; Wood et al. 2022]
                                                                            work [Ko et al. 2023].
and analysis [Wood et al. 2021], portrait normalization [Nagano
et al. 2019; Zhang et al. 2020], and semantic segmentation [Tritrong           Talking-head generators. Given a single target portrait and a driv-
et al. 2021; Zhang et al. 2021]. Some previous work used synthetic          ing video, recent talking-head generators can reenact the portrait
face portrait images generated by rendering 3D face assets using            by transferring facial expressions and head poses from the driver
a phycally-based pathtracer to train a model for portrait relight-          video [Doukas et al. 2021; Drobyshev et al. 2022; Hong et al. 2022b;
ing [Yeh et al. 2022] or relighting and view synthesis [Sun et al. 2021].   Wang et al. 2021a, 2022b; Zakharov et al. 2020; Zhao and Zhang
Since the CG rendering exhibits a synthetic look, they need an addi-        2022]. Trained by video datasets, their methods mainly focuse on
tional step to adapt to real images. Other concurrent work [Ko et al.       talking-head video generation by manipulating avatar poses and
2023] uses a discrete number of pre-generated synthetic images              expressions within a 2D portrait. As such, they do not predict volu-
from a 3D-aware generator [Chan et al. 2022] for 3D GAN inversion           metric representations that allow free viewpoint rendering including
tasks. Instead, we generate an unlimited amount of synthetic data in        background and do not provide dense 3D geometry like our method.
the training loop and show that on-the-fly camera augmentation is           Therefore, we do not compare to these approaches.
critical for generalization to real images for synthetic data training.

   3D-aware portrait generation and manipulation. For a well-known          3   PRELIMINARIES: TRIPLANE-BASED 3D GAN
category of object, such as human faces, previous work [Athar               We first give an overview of the state-of-the-art 3D GAN method,
et al. 2022; Gao et al. 2020; Groueix et al. 2018; Hong et al. 2022a;       EG3D, [Chan et al. 2022] from which our method will distill knowl-
Khakhulin et al. 2022; Kim et al. 2018; Mihajlovic et al. 2022; Nagano      edge. EG3D learns unconditional 3D-aware image generation from
et al. 2018; Wang et al. 2022a] used 3D face priors for few-shot            a collection of single-view images and corresponding noisy camera
portrait synthesis. While the face priors provide additional capabili-      poses, where each image has resolution 512 × 512. As mentioned
ties for facial manipulations and expression retargeting [Seol et al.       in Sec. 2, EG3D makes use of a hybrid triplane representation to

                                                                                    ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:4   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. 3. Inference and training outline for our pipeline. At inference, we take an unposed image, and extract low resolution features 𝑭low with a DeepLabv3
backbone. These features are fed to a ViT yielding 𝑭 and then concatenated after convolution with high-resolution features 𝑭high before being decoded
with a ViT and convolutions to a triplane representation. These features condition the volumetric rendering process which yields depth, feature, color, and
superresolved images. During training, we sample an identity from EG3D and then render two supervision views. The first serves as the input to our encoder,
which predicts a triplane, which conditions volume rendering from the same two views. The rendering results are compared with those of EG3D as outlined in
Sec. 4. Feature maps are visualized for illustration.


condition the neural volumetric rendering process, whereby three                       The neural rendered images 𝑰128 and 𝑰 𝑓 are then fed to a 2D super-
2D feature grids are stored along each of the three canonical planes–                  resolution network, which yields the final superresolved rendering
𝑥𝑦, 𝑥𝑧, 𝑦𝑧. Using a StyleGAN2 generator [Karras et al. 2020], EG3D                     output:
maps a noise vector and conditioning camera poses to a triplane                                     SuperRes(𝑰 𝑓 , 𝑰128 ) = 𝑰512 ∈ R3×512×512 .        (2)
representation 𝑻 ∈ R256×256×96 which corresponds to the 3 axis-
                                                                                       This 3D GAN pipeline is trained end-to-end following 2D GAN
aligned planes, each with 32 channels. These features condition the
                                                                                       training with a 2D (dual) discriminator. The reader is referred to the
neural volumetric rendering.
                                                                                       original paper [Chan et al. 2022] for full details.
   To assign a point 𝒙 ∈ R3 with its feature, color and volume
                                                                                          The efficient design of EG3D allows rendering from a triplane at
density, (f, c, 𝜎), a lightweight MLP decodes the three feature vectors
                                                                                       42 fps on the RTX 3090. At the same time, EG3D provides comparable
gathered by projecting 𝒙 to each of the canonical planes:
                                                                                       quality to even the state-of-the-art 2D GANs by FID. These attributes
                                                                                       provide a strong basis for supervising our encoder-based method
                      (f, c, 𝜎) = MLP(Φ(𝒇𝑥 𝑦 , 𝒇𝑥𝑧 , 𝒇𝑦𝑧 )),                   (1)     using EG3D-generated synthetic data.

where 𝒇𝑖 𝑗 are the features gathered by projecting 𝒙 to the 𝑖 𝑗 plane                  4    METHOD
and bilinearly interpolating the nearby features, and Φ is the mean                    Our goal is to distill the knowledge of a fully trained EG3D gen-
operator. Note that output values including the color are indepen-                     erative model (learned over a category or set of categories) into
dent of viewing direction and only depend on 𝒙. By accumulating                        a feedforward encoder pipeline that can directly map an unposed
many points along rays, and performing volume rendering [Max                           image to a canonical triplane 3D representation∗ which can be de-
1995] as in NeRF [Mildenhall et al. 2020], one may render a feature                    coded with a NeRF. This pipeline requires only a single feedforward
image 𝑰 𝑓 ∈ R32×128×128 and a raw neural rendering RGB image                           network pass, thus avoiding the expensive GAN inversion process,
𝑰128 ∈ R3×128×128 from a given camera pose. In practice, 𝑰128                          while allowing free viewpoint re-rendering of the input in real-time.
corresponds to the first three channels of the feature image 𝑰 𝑓 .
                                                                                       ∗ Note that each category has a different notion of canonical representation: for human
   We additionally extract a dense depth map 𝑰𝐷 ∈ R128×128 from                        faces, the center of the head is the origin, and planes orthogonally intersect the head
this volume rendering, which we use later to supervise our model.                      up-to-down, left-to-right, and front-to-back.


ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                        Real-Time Radiance Fields for Single-Image Portrait View Synthesis    • 1:5




Fig. 4. FFHQ and AFHQ qualitative results from our model (left) and our lightweight model (LT) (right). We showcase reconstructed input and novel views, or
the learned geometry. In the bottom-right, note our model’s ability to infer structure consistent with the input behind occlusion. Credits to YuChen Cheng,
Montclair Film, Lydia Liu.


Note that our contribution focuses on the image-to-triplane encoder
and associated synthetic training method, as shown in the pipeline                                      𝑭 = Conv(ViT(𝑭low )),                       (3)
of Fig. 3. We make use of the MLP volume renderer and superreso-                where Conv is a CNN and ViT is the Vision Transformer Block
lution architectures from EG3D as per Eqns 1 and 2 and train all the            from Segformer [Xie et al. 2021] with efficient self-attention. We
components end-to-end. In Tab. 1, the top row shows that our image              choose the Segformer ViT for two reasons: 1) it was designed to
to triplane inference runs at up to 60 fps (16 ms), while rendering             quickly map to a high-resolution output space similar to a triplane,
has identical performance to EG3D (bottom row of Tab. 1).                       and 2) the efficient self-attention mechanism allows the use of high-
                                                                                resolution intermediate feature maps so that all information flows
                                                                                from input to triplane.
                                                                                   We consider the ViT features as having successfully created a
4.1   Triplane encoder                                                          canonicalized 3D representation of the subject (completing the step
We note that inferring a canonicalized 3D reprentation (i.e., the               1 above), and found during our experimentation that this shallow
inferred 3D representation is frontalized and aligned) from an arbi-            encoder is sufficient to reasonably canonicalize a subject, yet cannot
trary RGB image while simultaneously synthesizing precise subject-              represent important high-frequency or subject-specific details like
specific details from the input is a highly non-trivial task. We break          strands of hair or birthmarks.
this challenge into the two-fold goals: 1) to create a canonicalized               In order to simultaneously complete the second step (adding
3D representation of the subject from an image, and 2) to render                high-frequency detail), we next reincorporate high-resolution image
high-frequency person-specific details. We note these goals are                 features. We convolutionally encode the image again with only a
often at odds with one another, and exemplify the bias-variance                 single downsampling stage with encoder Ehigh to obtain features
tradeoff whereby the output will resemble the input well, but may               𝑭high = Ehigh (𝑰 ). These are concatenated with the extracted global
not be correctly canonicalized in 3D (see Fig. 12), or the output               features and passed through another Vision Transformer, which
will have the correct 3D structure, but not resemble the 2D input               is finally decoded to a triplane with convolutions as seen in Fig. 3.
image (see Fig. 11). Our encoder manages to accomplish both of                  Thus, the output of our encoder has the following form:
these goals simultaneously. Specifically, we develop and train a hy-
                                                                                                  𝑻 = E(𝑰 ) = Conv(ViT(𝑭 ⊕ 𝑭high )),                             (4)
brid convolutional-Transformer encoder, E, which maps from an
unposed RGB image, 𝑰 , to the canonical triplane representation.                where ⊕ denotes concatenation along the channel axis, and 𝑻 is
   As seen in the upper half of Fig. 3, the architecture of our en-             triplane feature representation used in Sec. 3.
coder begins with a fast convolutional backbone, DeepLabV3 [Chen
et al. 2017], which extracts robust low-resolution features, 𝑭low =              4.2   Training with synthetic data
DeepLabV3(𝑰 ). These features are then fed to a Vision Transformer              As seen in Fig. 3 in the training step, we train our triplane encoder
(and CNN) which gives a global inductive bias to the intermediate               with synthetic data. Sampling a latent vector and passing it through
output features,                                                                the EG3D generator yields a corresponding triplane, 𝑻 . Given camera

                                                                                         ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:6   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano


parameters 𝑷 (a focal length, principal point, camera orientation and                     Implementation details. Before training with the full adversarial
position), we can render any image from the frozen EG3D generator                      objectives in Eqn. 5, we warm up the model by training over 30k
and 𝑻 . At each gradient step, we synthesize two images of the same                    iterations without the adversarial loss and continue to train the
identity (same latent code) from a reference (input) camera 𝑷ref                       model with the full loss functions in Eqn. 5 over 220k iterations.
and another camera 𝑷mv for multiview supervision. Using the same                       Since we sample two camera poses per iteration (with batch size
notation as in Sec. 3, each rendering pass will give us four images:                   32), we effectively use over 16 million images during the training,
𝑰 𝑓 , 𝑰128 , 𝑰512 , and 𝑰𝐷 as seen in Fig. 3.                                          which is not obtainable from real images (nor even physically-based
    Again as shown in Fig. 3, the input to our encoder is the high-                    rendered images) in practice. For full implementation details, please
resolution image 𝑰512 (highlighted in red) rendered from the input                     refer to the supplement. We train two encoders with two different
camera 𝑷ref , so that 𝑻 = E(𝑰512 ). We then use 𝑻 to condition the                     compute budgets: "Ours", which has 87M parameters and "Ours
volume rendering process from both camera 𝑷ref and 𝑷mv , to get                        (LT)", a lightweight model (LT) which has 63M parameters. The
two more sets of four images, which we denote as 𝑰 𝑓 , 𝑰 128 , 𝑰 512 ,                 main difference between the two is in resolution of the intermediate
and 𝑰 𝐷 . Our loss intuitively compares those quantities synthesized                   feature maps, which result in fewer parameters in the LT model,
by EG3D and those created by our encoder, along with a generative                      but both contain the same structure outlined above. "Ours" runs in
adversarial objective as follows:                                                      22ms on a single A100 GPU (where rendering takes 15ms) and 40ms
                                                                                       on RTX 3090 as seen in Table 1. "Ours (LT)" runs in just 16ms on
                                                                                       RTX 3090, while retaining strong performance (see the numerical
          𝐿 = 𝐿tri + 𝐿col + 𝐿LPIPS + 𝐿feat + 𝜆1 𝐿adv + 𝜆2 𝐿cate                (5)
                                                                                       evaluations in Tabs. 2 and 3). Figures 1 and 4 show the qualitative
                                                                                       outputs from both models.
𝐿tri is the L1 loss between 𝑻 and 𝑻 ; 𝐿col is the mean L1 loss computed
between both sets of pairs (𝑰128, 𝑰 128 ) and (𝑰512, 𝑰 512 ); 𝐿LPIPS is                5     RESULTS
the LPIPS perceptual loss [Zhang et al. 2018] computed over both                       We evaluate methods for single-view novel view synthesis on 3
sets of pairs (𝑰128, 𝑰 128 ) and (𝑰512, 𝑰 512 ); 𝐿feat is the mean L1 loss             main aspects (1) 2D image reconstruction (LPIPS [Zhang et al. 2018],
computed between the pairs (𝑰 𝑓 , 𝑰 𝑓 ); 𝐿adv is the adversarial loss us-              DISTS [Ding et al. 2022], SSIM [Wang et al. 2004]) and likeness
ing a pretrained dual discriminator from EG3D which is fine-tuned                      (identity consistency) (2) general image quality (FID [Heusel et al.
during training; 𝜆1 is 0.1 for the reference image or 0.025 for the                    2017]) and (3) 3D reconstruction quality (depth, and pose estimation).
multiview image; and 𝐿cate is an optional category-specific loss. For                  For the reconstruction tasks, we need to re-render our outputs to
human faces, we use 𝜆2 to be 1 with face identity features from                        the input views for the purpose of the evaluation using a camera
ArcFace [Deng et al. 2019a] following practice in 2D GAN inver-                        pose estimated using an off-the-shelf pose predictor [Deng et al.
sion [Richardson et al. 2021; Tov et al. 2021]. For cat faces, we set                  2019b]. However, we noticed that errors present in the estimated
𝜆2 to 0. This objective is optimized end-to-end, i.e., with respect                    poses create a small image misalignment between the ground truth
to all of the parameters of the encoder, rendering and upsampling                      and our feedforward results (as opposed to inversion models which
modules. Note that the rendering, upsampling, and dual discrimina-                     directly optimize for the given view), making the raw pixel metrics
tor modules are all fine-tuned from the pretrained EG3D. However,                      like PSNR and SSIM unreliable. For this reason, we mainly rely on
the dual discriminator in our pipeline doesn’t rely on any real data;                  the deep perceptual image metrics such as LPIPS and DISTS, which
instead, we train this discriminator to differentiate between images                   judge that the given images are of the same perceptual quality for our
rendered from our encoder model and images rendered from the                           evaluation. Nonetheless, we report SSIM results in the main paper
frozen EG3D. An ablation showing its effectiveness is provided in                      and include PSNR results in the supplement along with an analysis
Tab. 5 and Fig. 13.                                                                    of alignment issues. In the end, our experiments qualitatively and
                                                                                       quantitatively support that our method achieves the state-of-the-
   On-the-fly augmentation. Naively optimizing this objective will                     art results on in-the-wild portraits as well as multiview 3D scan
yield a model which performs almost perfectly on synthetic data,                       datasets. For more results, please refer to the supplement video.
but lacks the ability to generalize to real images (see Fig. 12). In order
to remedy this, we augment the standard EG3D rendering method                             Datasets. Our method is evaluated on FFHQ [Karras et al. 2019], a
which assumes a fixed camera roll, focal length, principal point                       representative dataset for high-quality in-the-wild human portraits,
and distance from subject. In contrast, we sample all four of these                    H3DS [Ramon et al. 2021], which has high resolution ground truth
values from random distributions to choose the camera parameters                       3D scans and 360º images of 23 human heads with associated camera
𝑷ref . The details of these distributions for each dataset are given in                calibrations, and AFHQv2 Cats [Choi et al. 2020; Karras et al. 2021],
the supplement. For 𝑷mv , we choose fixed values as in the EG3D                        a collection of high-resolution in-the-wild portraits of cats.
model. For 𝑷ref , we sample the cameras from a pitch range of ±26◦
and yaw range of ±49◦ relative to the front of a human face. For                       5.1    Comparisons
𝑷mv , we sample the cameras from a pitch range of ±26◦ and yaw                            Baselines. We compare our methods against three state-of-the
range of ±36◦ relative to the front of a human face. This allows the                   art methods for 3D aware-image generation from a single image:
supervision of our model to happen with highly variable camera                         ROME [Khakhulin et al. 2022], HeadNeRF [Hong et al. 2022a], and
poses, forcing the model to learn to effectively canonicalize and                      EG3D-PTI, which combines an unconditional EG3D generator [Chan
infer from challenging images as seen in Fig. 4.                                       et al. 2022] and Pivotal Tuning Inversion (PTI) [Roich et al. 2021]. We

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                     Real-Time Radiance Fields for Single-Image Portrait View Synthesis    • 1:7




Fig. 5. Qualitative results displaying our model’s reconstruction on the input view, and the learned geometry from the frontal view. The reconstructed
geometry remains faithful to the input image. Credits to Devon Weller, Jamie, SupportPDX, Mary Sawatzky, map, Herzliya Conference, Helse Midt-Norge,
Tom Munnecke, pter tr, UGA CAES/Extension, Rare Cancers Australia, Vladimir Agafonkin, Michael E. Macmillan, Nguyen Hung Vu.


                                                                                      ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:8   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. 6. Qualitative results displaying our model in comparison to baseline methods HeadNeRF, ROME, and EG3D-PTI, comparing the image quality (left) and
reconstructed geometry (right). EG3D-PTI occasionally exhibits corrupted 3D geometry (2nd and 4th rows) when the input is side view, indicating that the
learned 3D prior alone is not enough to ensure robust reconstruction. Credit to U.S. Dept. of HUD, Cristina Cifuentes, Rainforest Action Network, CENA
MINEIRA.


                                                                                       demonstrating high-quality novel views and 3D geometry recon-
                                                                                       structed by our method from a single portrait. Fig. 6 provides a qual-
                                                                                       itative comparison against baselines. While HeadNeRF and ROME
                                                                                       provide adequate shapes and images, they need image segmentation
                                                                                       as a preprocess, and struggle with obtaining photorealistic results.
                                                                                       Despite the 20 mins of fine tuning, EG3D-PTI does not ensure the
                                                                                       reconstruction looks photorealistic when viewed from a non-input
                                                                                       view (see Fig. 2). In contrast, our method reconstructs the entire
                                                                                       portrait with accurate photorealistic details. Fig. 7 provides com-
                                                                                       parisons to the ground truth validation view and 3D scan on H3DS.
                                                                                       The synthesized image and 3D geometry of ROME and HeadNeRF
                                                                                       generally lack the fidelity and reconstruct only a part of the head.
                                                                                       EG3D-PTI occasionally outputs a degenerate 3D shape due to the
                                                                                       highly unconstrained nature of single-view training of the NeRF
                                                                                       representation (see Figs 1, 6 and 7). Our geometry retains overall the
Fig. 7. Ground truth comparisons on the H3DS dataset including ground                  3D shape as well as person-specific facial details. We also provide
truth geometry (second row) and unseen validation view (third row). Since              results on lifting 2D drawings and paintings into 3D in Fig. 8. While
the H3DS ground truth data has inconsistent lighting, the lighting discrep-            our method is never trained with stylized images, it can reasonably
ancy is expected for the validation view.                                              well handle those out-of-domain input images. Finally, we also show
                                                                                       the outputs of our method in comparison to baselines at varying
                                                                                       pitch and yaw in Figs 9 and 10, displaying the benefit of our method
                                                                                       for photorealistic facial frontalization of challenging images. In com-
also compare with EG3D itself as an unconditional reference on FID.
                                                                                       parison to baselines, our method’s geometry does not collapse for
We additionally provide extensive evaluations on our lightweight
                                                                                       challenging yaws as EG3D-PTI, and shows a significantly higher
model (LT), which is introduced in Sec. 4.
                                                                                       degree of photorealism than ROME and HeadNeRF.
  Qualitative results. Fig. 1 shows our qualitative results on FFHQ
and AFHQ. Fig. 4 and Fig. 5 show selected examples from FFHQ,

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                          Real-Time Radiance Fields for Single-Image Portrait View Synthesis     • 1:9




Fig. 8. Qualitative results displaying our model’s ability to lift StyleGAN2-generated drawings and paintings to 3D. These results display the generalizability
of our model, as it can canonicalize out-of-domain drawings and portraits, lifting them to 3D.


Table 2. Quantitative evaluation using LPIPS, DISTS, SSIM, pose accuracy           Table 3. Scale- and translation-invariant depth evaluation using ground
(Pose) and identity consistency (ID) on 500 FFHQ images. † Evaluated only          truth geometry from H3DS datasets. † Evaluated only using the face region.
using the foreground on 2562 images. ‡ Evaluated only using the face region.
                                                                                         Depth      H.NeRF      ROME       EG3D-PTI        Ours     Ours (LT)
                      LPIPS ↓    DISTS ↓    SSIM ↑    Pose ↓    ID ↑
                                                                                         L1 ↓        0.108†      0.054        0.071       0.048        0.049
      HeadNeRF‡        .2502      .2427     .7514     .0644    .2031                     RMSE ↓      0.147†      0.084        0.101       0.074        0.075
      Ours‡            .1240      .0770     .8246     .0490    .5481
      ROME (256)†      .1158      .1058     .8257     .0637    .3231
                                                                                   our models evaluated on the same masked region. Tab. 2 shows
      Ours†            .0468      .0407     .8981     .0486    .5410
                                                                                   that our model significantly outperforms the baselines on all the
      EG3D-PTI         .3236      .1277     .6722     .0575    .4650               metrics except SSIM; our SSIM score is only marginally lower than
      Ours             .2692      .0904     .6598     .0485    .5426               EG3D-PTI despite the aforementioned issue of the image misalign-
      Ours (LT)        .2750      .1021     .6655     .0448    .5404               ment and the fact that EG3D-PTI directly optimizes the pixels for
                                                                                   the evaluation view. The geometry evaluation in Tab. 3 on H3DS in
                                                                                   which we compare the depths of the ground truth from the input
   Quantitative evaluations. Tab. 2 shows numerical comparisons of                 view as predicted by each model validates that our models produce
our method against baselines on 500 randomly selected images from                  more accurate 3D geometry.
FFHQ. We measure the 2D image reconstruction quality in the input
view using LPIPS, DISTS, and SSIM. We evaluate multiview con-                      5.2   Ablation study
sistency using poses (Pose) estimated from synthesized images by
                                                                                   We provide ablation studies comparing variants of our architecture
an off-the-shelf pose detector [Deng et al. 2019b] following similar
                                                                                   and different training strategies. All variants are evaluated after
protocols as in previous work [Chan et al. 2022; Shi et al. 2021], and
                                                                                   training with 3M images.
identity (ID) consistency by computing the mean of MagFace [Meng
et al. 2021] (not used in our training) cosine similarity scores be-                  Inference time and number of parameters. We compare the per-
tween the input view and synthesized view from a random camera                     formance of two variants of our model, which have the same ar-
pose. Since HeadNeRF and ROME only produce the face region and                     chitecture but have different numbers of parameters and resolu-
the foreground respectively, we also provide the same metrics from                 tion of intermediate feature maps: "Ours" (87M params) and "Ours

                                                                                            ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:10   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. 9. Comparison to baselines at various input pitch angles. Credits to Bjørnar Tollaksen, Juliana Martuscelli, The 621st Contingency Response Wing, U.S.
Army Security Assistance Command, Sam Wadman, Laity Lodge Family Camp, U.S. President’s Malaria Initiative, SickKids Foundation.


(LT)" (63M params). Tab. 1 provide runtime comparisons of the two.                     this variant, we replaced the ViT module with CNN with matching
Tabs. 2 and 3 provide several comparisons of the two on image re-                      number of parameters. Tab. 4 provides numerical comparisons of the
construction, the accuracy of 3D shapes and identity consistencies.                    two variants on image and 3D quality metrics. These quantitative
These extensive evaluations suggest that our lightweight model                         and qualitative comparisons show that the ViT layers are important
retains very close performance to our full model despite running                       for creating more accurate 3D representations as well as achieving
significantly faster. Figs. 1 and 4 show qualitative samples from both                 more accurate 2D image reconstruction.
our full model and our lightweight model.
                                                                                          Effects of camera augmentation. Fig. 12 compares the models
  Effects of Transformers. Fig. 11 compares results obtained with                      trained with or without the camera augmentation for robustness
or without the proposed Transformer layers in the encoder. For                         to camera noise (also see the first row of Fig. 6 for the results of

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                       Real-Time Radiance Fields for Single-Image Portrait View Synthesis    •   1:11




Fig. 10. Comparison to baselines at various input yaw angles. Credit to Lionel AZRIA, justinkim1, nonorganical, Paradox Wolf, Agência Senado, Ademir Brito,
Ariana Vincent, Jay Weenig, John Benson, Seong Bae.




                                                                                         ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:12   •    Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano


the same subject without the camera noise). We fix the camera
calibration and apply image space rotation, translation, and zoom
to the input image, emulating the effect of inaccurate camera ex-
trinsics and intrinsics. Although our model does not rely on any
camera information for canonicalization, the result is not robust
without the proposed camera augmentation. EG3D-PTI assumes
a fixed image alignment used to train the GAN model and is very
sensitive to small image misalignment in the input. Tab. 4 provides
numerical comparisons of our model with and without the proposed
                                                                                        Fig. 11. Ablation study comparing our model with and without the proposed
augmentation.
                                                                                        Transformer modules. The model w/o Transformer replaces all Transformer
   Effects of fine-tuned synthetic discriminator. We provide an addi-                   Blocks with resolution-preserving residual CNNs with similar parameters.
tional ablation on the discriminator loss (𝐿adv in Eqn. 5), which fine-                 Credit to Kirill Chebotar.
tunes the pre-trained EG3D discriminator with EG3D-generated
images. As seen in Tab. 5, removing this discriminator loss results
in a worse FID score. Moreover, as seen in Fig. 13, the renderings of
our proposed method are significantly sharper with the synthetic
discriminator tuning. Please see Sec. A1 and Tab. A2 for attempts
to train the discriminator with real images.

Table 4. Ablation studies evaluating the proposed camera augmentation                   Fig. 12. Camera augmentation ablation study. Note that this is the same
and the Transformer module. Without augmentation, the model acts as                     image as the first row of Fig. 6 except rotated and cropped non-centrally.
an autoencoder, mapping real images to arbitrary 3D representations that                Without augmentation, our result exhibits artifacts when the input image
resemble the input (good ID score), but are not actually 3D (poor Pose                  has zoom or camera roll. Similarly, EG3D-PTI is also sensitive to the image
score). Without a transformer, the encoder can canonicalize the inputs well             misalignment, as the camera pose becomes noisy, while our method correctly
(good Pose score), but cannot represent the details of the input (poor ID               canonicalizes the face. Images are cropped and aligned for visual consistency.
score). Our full method achieves both good Pose and ID scores with high                 Credit to U.S. Dept. of HUD.
reconstruction quality.


                          LPIPS ↓    DISTS ↓     Pose ↓       ID ↑     FID ↓
   No aug.                0.3846      .1286      0.1758     0.5359      3.42
   No Transformer         0.5419      .1650      0.0426     0.1906      11.5
   Ours                   0.2894      .1053      0.0461     0.5230      4.45



Table 5. Comparison in FID between our model and an ablated model
without the synthetic discriminator.


                           FID ↓                    FFHQ
                           w/o synthetic disc.       7.71
                           Ours                      4.45
                                                                                        Fig. 13. Comparison between our model and an ablated model trained with-
                                                                                        out the synthetic discriminator. Note the blurriness without the adversarial
                                                                                        loss. Credit to Mohd Fazlin Mohd Effendy Ooi.
5.3        Application: real-time 3D telepresence
We apply our method for lifting a monocular RGB video input
to 3D in real-time, as would be needed for 3D telepresence. Our
method processes the video frame by frame. Despite being trained on                     6    DISCUSSION
individual frames of synthetic data and processing the input video                         Limitation. When the input is a strong profile view (e.g., 60 de-
in a frame to frame fashion, our method can provide reasonable                          grees yaw angle), our method may struggle with properly canon-
temporal consistency. Please refer to the teaser Fig. 1 (bottom right)                  icalizing the input, as it is highly out-of-distribution with respect
for the output from our lightweight model as well as video examples                     to EG3D-generated images and FFHQ. Please see Figs. 9 and 10 for
from the supplement. Fig. 14 shows our system set up and running                        various challenging levels of pitch and yaw for input images. While
off of a desktop with a single RTX 4090. Our method can lift a                          our method can predict a canonicalized 3D representation without
monocular RGB video frame from a mobile phone to 3D in real-                            requiring camera poses as input, the rendered image may be slightly
time.                                                                                   misaligned when compared to the input view (see Fig. A8 in the

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                     Real-Time Radiance Fields for Single-Image Portrait View Synthesis       •   1:13


                                                                              Eric Chan were partially supported by DARPA’s Semantic Forensics
                                                                              (SemaFor) contract (HR0011-20-3-0005). The views and conclusions
                                                                              contained in this document are those of the authors and should
                                                                              not be interpreted as representing the official policies, either ex-
                                                                              pressed or implied, of the U.S. Government. Distribution Statement
                                                                              “A” (Approved for Public Release, Distribution Unlimited).


                                                                              REFERENCES
                                                                              Rameen Abdal, Yipeng Qin, and Peter Wonka. 2019. Image2StyleGAN: How to Em-
                                                                                  bed Images Into the StyleGAN Latent Space?. In IEEE International Conference on
                                                                                  Computer Vision (ICCV).
                                                                              Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. 2021. ReStyle: A Residual-Based
                                                                                  StyleGAN Encoder via Iterative Refinement. In IEEE International Conference on
Fig. 14. Our system applied to create a 3D telepresence live from a monocu-       Computer Vision (ICCV).
lar RGB input. Please see the supplement video for the live demonstration.    ShahRukh Athar, Zexiang Xu, Kalyan Sunkavalli, Eli Shechtman, and Zhixin Shu. 2022.
                                                                                  RigNeRF: Fully Controllable Neural 3D Portraits. In IEEE Conference on Computer
                                                                                  Vision and Pattern Recognition (CVPR).
                                                                              Ang Cao, Chris Rockwell, and Justin Johnson. 2022. FWD: Real-time Novel View
supplement for the detailed analysis) possibly due to the combina-                Synthesis with Forward Warping and Depth. In IEEE Conference on Computer Vision
                                                                                  and Pattern Recognition (CVPR).
tion of the imperfect canonicalization and noisy camera poses from            Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De
an off-the-shelf pose estimator. Finally, although our method can                 Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero
provide reasonable temporal consistency when applied to a video                   Karras, and Gordon Wetzstein. 2022. Efficient Geometry-aware 3D Generative
                                                                                 Adversarial Networks. In IEEE Conference on Computer Vision and Pattern Recognition
in a frame-by-frame fashion, temporal inconsistencies remain as                  (CVPR).
the canonicalizations change slightly per frame, and the predicted            Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. 2021.
camera poses are entirely independent.                                            pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image
                                                                                  Synthesis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
                                                                              Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu,
   Future work. In the future, combining our method with camera                   and Hao Su. 2021. Mvsnerf: Fast generalizable radiance field reconstruction from
pose optimization [Ko et al. 2023] may lead to more accurate 3D                   multi-view stereo. In IEEE International Conference on Computer Vision (ICCV).
reconstruction and camera pose estimation. Additionally, jointly              Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. 2017.
                                                                                  Rethinking atrous convolution for semantic image segmentation. arXiv preprint
predicting the camera poses and triplanes in an autoregressive or re-             arXiv:1706.05587 (2017).
current context [Kalchbrenner et al. 2017; Shi et al. 2015; Srivastava        S Chen and L Williams. 1993. View Interpolation for Image Synthesis. In SIGGRAPH 93.
                                                                                  279–288.
et al. 2015] may result in more consistent frame-by-frame results.            Zhiqin Chen and Hao Zhang. 2019. Learning implicit fields for generative shape
Next, it would be interesting to incorporate real images in the train-            modeling. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
ing as our preliminary attempts did not yield improvements. Finally,          Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. 2020. StarGAN v2: Diverse
                                                                                  Image Synthesis for Multiple Domains. In IEEE Conference on Computer Vision and
as our pipeline does not necessarily assume any category-specific                 Pattern Recognition (CVPR).
priors, we can view it as a general method to distill the knowledge           Jiankang Deng, Jia Guo, Xue Niannan, and Stefanos Zafeiriou. 2019a. ArcFace: Additive
of a 3D GAN into a feedforward encoder. Thus, extending 3D GANs                  Angular Margin Loss for Deep Face Recognition. In IEEE Conference on Computer
                                                                                  Vision and Pattern Recognition (CVPR).
to more general scenes [Skorokhodov et al. 2023] may allow our                Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong. 2022. GRAM: Generative
pipeline to create 3D representations of arbitrary scenes in the fu-              Radiance Manifolds for 3D-Aware Image Generation. In IEEE Conference on Computer
                                                                                  Vision and Pattern Recognition (CVPR).
ture. Specifically extending our work to handle hands or the full             Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. 2019b.
body, is of interest for real-time telepresence applications.                    Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single
                                                                                  Image to Image Set. In IEEE Computer Vision and Pattern Recognition Workshops.
   Conclusion. We proposed a one-shot encoder-based framework                 Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. 2022. Image Quality
                                                                                 Assessment: Unifying Structure and Texture Similarity. IEEE Transactions on Pattern
to lift a single RGB image to 3D in real-time and demonstrated our                Analysis and Machine Intelligence (2022).
method, trained entirely from synthetic data, can handle challeng-            Tan M. Dinh, Anh Tuan Tran, Rang Nguyen, and Binh-Son Hua. 2022. HyperInverter:
ing (even out-of-domain) real-world images. We believe that this                  Improving StyleGAN Inversion via Hypernetwork. In IEEE Conference on Computer
                                                                                  Vision and Pattern Recognition (CVPR).
opens up possibilities for accessible 3D reconstructions of real-world        Michail Christos Doukas, Stefanos Zafeiriou, and Viktoriia Sharmanska. 2021.
objects and interactive 3D visualization from a picture.                          HeadGAN: One-shot Neural Head Synthesis and Editing. In IEEE International
                                                                                  Conference on Computer Vision (ICCV).
                                                                              Nikita Drobyshev, Jenya Chelishev, Taras Khakhulin, Aleksei Ivakhnenko, Victor Lem-
ACKNOWLEDGEMENTS                                                                  pitsky, and Egor Zakharov. 2022. Megaportraits: One-shot megapixel neural head
We thank David Luebke, Jan Kautz, Peter Shirley, Alex Evans, Towaki               avatars. arXiv preprint arXiv:2207.07621 (2022).
                                                                              Qianli Feng, Viraj Shah, Raghudeep Gadde, Pietro Perona, and Aleix Martinez. 2022.
Takikawa, Ekta Prashnani and Aaron Lefohn for feedback on drafts                  Near perfect gan inversion. arXiv preprint arXiv:2202.11833 (2022).
and early discussions. We acknowledge the significant efforts and             Chen Gao, Yichang Shih, Wei-Sheng Lai, Chia-Kai Liang, and Jia-Bin Huang. 2020.
                                                                                  Portrait Neural Radiance Fields from a Single Image. arXiv preprint arXiv:2012.05903
suggestions of the reviewers. For allowing the use of video, we thank            (2020).
Elys Muda. This work was funded in part at UCSD by ONR grants                 Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
N000142012529, N000142312526, an NSF graduate Fellowship, a Ja-                   Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In
                                                                                  Advances in Neural Information Processing Systems (NeurIPS).
cobs Fellowship, and the Ronald L. Graham chair. Manmohan Chan-               S Gortler, R Grzeszczuk, R Szeliski, and M Cohen. 1996. The Lumigraph. In SIGGRAPH
draker acknowledges support of NSF IIS 2110409. Koki Nagano and                   96. 43–54.


                                                                                        ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:14   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano


Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, and Mathieu            Koki Nagano, Jaewoo Seo, Jun Xing, Lingyu Wei, Zimo Li, Shunsuke Saito, Aviral
    Aubry. 2018. AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation.           Agarwal, Jens Fursund, and Hao Li. 2018. PaGAN: Real-Time Avatars Using Dynamic
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).                       Textures. ACM Transactions on Graphics (SIGGRAPH ASIA) (2018).
Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. 2021. StyleNeRF: A Style-        Thu Nguyen-Phuoc, Chuan Li, Lucas Theis, Christian Richardt, and Yong-Liang Yang.
    based 3D-Aware Generator for High-resolution Image Synthesis. arXiv preprint                2019. HoloGAN: Unsupervised learning of 3D representations from natural images.
    arXiv:2110.08985 (2021).                                                                    In IEEE International Conference on Computer Vision (ICCV).
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klam-          Michael Niemeyer and Andreas Geiger. 2021. GIRAFFE: Representing Scenes as Com-
    bauer, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update                   positional Generative Neural Feature Fields. In IEEE Conference on Computer Vision
    Rule Converge to a Nash Equilibrium. In Advances in Neural Information Processing           and Pattern Recognition (CVPR).
    Systems (NeurIPS).                                                                      Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira
Fa-Ting Hong, Longhao Zhang, Li Shen, and Dan Xu. 2022b. Depth-Aware Generative                 Kemelmacher-Shlizerman. 2022. StyleSDF: High-Resolution 3D-Consistent Im-
    Adversarial Network for Talking Head Video Generation. IEEE Conference on                   age and Geometry Generation. In IEEE Conference on Computer Vision and Pattern
    Computer Vision and Pattern Recognition (CVPR).                                             Recognition (CVPR).
Yang Hong, Bo Peng, Haiyao Xiao, Ligang Liu, and Juyong Zhang. 2022a. HeadNeRF:             Xingang Pan, Bo Dai, Ziwei Liu, Chen Change Loy, and Ping Luo. 2021. Do 2D GANs
    A Real-time NeRF-based Parametric Head Model. In IEEE Conference on Computer                Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs. In
    Vision and Pattern Recognition (CVPR).                                                      International Conference on Learning Representations (ICLR).
Wonbong Jang and Lourdes Agapito. 2021. Codenerf: Disentangled neural radiance              Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Love-
    fields for object categories. In IEEE International Conference on Computer Vision           grove. 2019. DeepSDF: Learning Continuous Signed Distance Functions for Shape
    (ICCV).                                                                                     Representation. In IEEE Conference on Computer Vision and Pattern Recognition
N. Khademi Kalantari, T. Wang, and R. Ramamoorthi. 2016. Learning-based view                    (CVPR).
    synthesis for light field cameras. ACM Transactions on Graphics (SIGGRAPH Asia          William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Torralba, Alexei Efros, and Eli
    16) 35, 6 (2016), 193:1–193:10.                                                             Shechtman. 2022. GAN-Supervised Dense Visual Alignment. In IEEE Conference on
Nal Kalchbrenner, Aäron Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex                Computer Vision and Pattern Recognition (CVPR).
    Graves, and Koray Kavukcuoglu. 2017. Video pixel networks. In International             Eduard Ramon, Gil Triginer, Janna Escur, Albert Pumarola, Jaime Garcia, Xavier Giro-i
    Conference on Machine Learning. PMLR, 1771–1779.                                            Nieto, and Francesc Moreno-Noguer. 2021. H3D-Net: Few-Shot High-Fidelity 3D
Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehti-          Head Reconstruction. In IEEE International Conference on Computer Vision (ICCV).
    nen, and Timo Aila. 2021. Alias-Free Generative Adversarial Networks. In Advances       Daniel Rebain, Mark Matthews, Kwang Moo Yi, Dmitry Lagun, and Andrea Tagliasacchi.
    in Neural Information Processing Systems (NeurIPS).                                         2022. LOLNeRF: Learn from One Look. In IEEE Conference on Computer Vision and
Tero Karras, Samuli Laine, and Timo Aila. 2019. A Style-Based Generator Architecture            Pattern Recognition (CVPR).
    for Generative Adversarial Networks. In IEEE Conference on Computer Vision and          Xuanchi Ren and Xiaolong Wang. 2022. Look Outside the Room: Synthesizing A
    Pattern Recognition (CVPR).                                                                 Consistent Long-Term 3D Scene Video from A Single Image. In IEEE Conference on
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo             Computer Vision and Pattern Recognition (CVPR).
    Aila. 2020. Analyzing and Improving the Image Quality of StyleGAN. In IEEE              Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro,
    Conference on Computer Vision and Pattern Recognition (CVPR).                               and Daniel Cohen-Or. 2021. Encoding in Style: a StyleGAN Encoder for Image-to-
Taras Khakhulin, Vanessa Sklyarova, Victor Lempitsky, and Egor Zakharov. 2022. Re-              Image Translation. In IEEE Conference on Computer Vision and Pattern Recognition
    alistic One-shot Mesh-based Head Avatars. In European Conference on Computer                (CVPR).
    Vision (ECCV).                                                                          Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. 2021. Pivotal
Hyeongwoo Kim, Pablo Garrido, Ayush Tewari, Weipeng Xu, Justus Thies, Matthias                  Tuning for Latent-based Editing of Real Images. arXiv preprint arXiv:2106.05744
    Nießner, Patrick Pérez, Christian Richardt, Michael Zollöfer, and Christian Theobalt.       (2021).
    2018. Deep Video Portraits. ACM Transactions on Graphics (SIGGRAPH) (2018).             R. Rombach, P. Esser, and B. Ommer. 2021. Geometry-Free View Synthesis: Transformers
Jaehoon Ko, Kyusun Cho, Daewon Choi, Kwangrok Ryoo, and Seungryong Kim. 2023.                   and no 3D Priors. In IEEE International Conference on Computer Vision (ICCV).
    3D GAN Inversion with Pose Optimization. In IEEE Winter Conference on Applications      Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Rad-
    of Computer Vision (WACV).                                                                  wan, Suhani Vora, Mario Lučić, Daniel Duckworth, Alexey Dosovitskiy, et al. 2022.
Thomas Leimkühler and George Drettakis. 2021. Freestylegan: Free-view editable                  Scene representation transformer: Geometry-free novel view synthesis through
    portrait rendering with the camera manifold. arXiv preprint arXiv:2109.09378 (2021).        set-latent scene representations. In IEEE Conference on Computer Vision and Pattern
M Levoy and P Hanrahan. 1996. Light Field Rendering. In SIGGRAPH 96. 31–42.                     Recognition (CVPR).
Xingyi Li, Chaoyi Hong, Yiran Wang, Zhiguo Cao, Ke Xian, and Guosheng Lin. 2022.            Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. 2020. GRAF: Genera-
    SymmNeRF: Learning to Explore Symmetry Prior for Single-View View Synthesis.                tive Radiance Fields for 3D-Aware Image Synthesis. In Advances in Neural Informa-
    In Asian Conference on Computer Vision (ACCV).                                              tion Processing Systems (NeurIPS).
C.Z. Lin, D.B. Lindell, E.R. Chan, and G. Wetzstein. 2022. 3D GAN Inversion for             Yeongho Seol, Jaewoo Seo, Paul Hyunjin Kim, J. P. Lewis, and Junyong Noh. 2011. Artist
    Controllable Portrait Image Animation. In ECCV Workshop on Learning to Generate             Friendly Facial Animation Retargeting. In ACM Transactions on Graphics (SIGGRAPH
    3D Shapes and Scenes.                                                                       ASIA).
Kai-En Lin, Lin Yen-Chen, Wei-Sheng Lai, Tsung-Yi Lin, Yi-Chang Shih, and Ravi              Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-
    Ramamoorthi. 2023. Vision Transformer for NeRF-Based View Synthesis from a                  chun Woo. 2015. Convolutional LSTM network: A machine learning approach for
    Single Input Image. In IEEE Winter Conference on Applications of Computer Vision            precipitation nowcasting. In Advances in Neural Information Processing Systems
    (WACV).                                                                                     (NeurIPS).
N. Max. 1995. Optical models for direct volume rendering. IEEE Transactions on              Yichun Shi, Divyansh Aggarwal, and Anil K Jain. 2021. Lifting 2D StyleGAN for
    Visualization and Computer Graphics (TVCG) (1995).                                          3D-Aware Face Generation. In IEEE Conference on Computer Vision and Pattern
L McMillan and G Bishop. 1995. Plenoptic Modeling: An Image-Based Rendering                     Recognition (CVPR).
    System. In SIGGRAPH 95. 39–46.                                                          Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. 2019. Scene Representa-
Qiang Meng, Shichao Zhao, Zhida Huang, and Feng Zhou. 2021. MagFace: A universal                tion Networks: Continuous 3D-Structure-Aware Neural Scene Representations. In
    representation for face recognition and quality assessment. In IEEE Conference on           Advances in Neural Information Processing Systems (NeurIPS).
    Computer Vision and Pattern Recognition (CVPR).                                         Ivan Skorokhodov, Aliaksandr Siarohin, Yinghao Xu, Jian Ren, Hsin-Ying Lee, Peter
Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas               Wonka, and Sergey Tulyakov. 2023. 3D generation on ImageNet. arXiv preprint
    Geiger. 2019. Occupancy Networks: Learning 3D Reconstruction in Function Space.             arXiv:2303.01416 (2023).
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).                   Ivan Skorokhodov, Sergey Tulyakov, Yiqun Wang, and Peter Wonka. 2022. EpiGRAF:
Marko Mihajlovic, Aayush Bansal, Michael Zollhoefer, Siyu Tang, and Shunsuke Saito.             Rethinking training of 3D GANs. In Advances in Neural Information Processing
    2022. KeypointNeRF: Generalizing Image-based Volumetric Avatars using Relative              Systems (NeurIPS).
    Spatial Encoding of Keypoints. In European Conference on Computer Vision (ECCV).        P. Srinivasan, T. Wang, A. Sreelal, R. Ramamoorthi, and R. Ng. 2017. Learning to
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-                Synthesize a 4D RGBD Light Field from a Single Image. In International Conference
    mamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields            on Computer Vision (ICCV). 2262–2270.
    for View Synthesis. In European Conference on Computer Vision (ECCV).                   Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. 2015. Unsupervised
Koki Nagano, Huiwen Luo, Zejian Wang, Jaewoo Seo, Jun Xing, Liwen Hu, Lingyu Wei,               learning of video representations using lstms. In International conference on machine
    and Hao Li. 2019. Deep face normalization. ACM Transactions on Graphics (TOG)               learning. PMLR, 843–852.
    38, 6 (2019), 1–16.                                                                     Jingxiang Sun, Xuan Wang, Yichun Shi, Lizhen Wang, Jue Wang, and Yebin Liu. 2022.
                                                                                                IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-aware Portrait



ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                                   Real-Time Radiance Fields for Single-Image Portrait View Synthesis      •   1:15


    Synthesis. ACM Transactions on Graphics (SIGGRAPH ASIA) (2022).                             Conference on Computer Vision (ECCV).
Tiancheng Sun, Kai-En Lin, Sai Bi, Zexiang Xu, and Ravi Ramamoorthi. 2021. NeLF: Neu-       Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018.
    ral Light-transport Field for Portrait View Synthesis and Relighting. In Eurographics      The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In IEEE
    Symposium on Rendering.                                                                     Conference on Computer Vision and Pattern Recognition (CVPR).
Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. 2021.              Xuaner Zhang, Jonathan T. Barron, Yun-Ta Tsai, Rohit Pandey, Xiuming Zhang, Ren
    Designing an Encoder for StyleGAN Image Manipulation. ACM Transactions on                   Ng, and David E. Jacobs. 2020. Portrait Shadow Manipulation. ACM Transactions on
    Graphics (SIGGRAPH) (2021).                                                                 Graphics (SIGGRAPH).
Alex Trevithick and Bo Yang. 2021. GRF: Learning a General Radiance Field for 3D            Xuanmeng Zhang, Zhedong Zheng, Daiheng Gao, Bang Zhang, Pan Pan, and Yi Yang.
    Scene Representation and Rendering. In IEEE International Conference on Computer            2022. Multi-View Consistent Generative Adversarial Networks for 3D-aware Image
    Vision (ICCV).                                                                              Synthesis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Nontawat Tritrong, Pitchaporn Rewatbowornwong, and Supasorn Suwajanakorn. 2021.             Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Bar-
    Repurposing GANs for One-shot Semantic Part Segmentation. In IEEE Conference                riuso, Antonio Torralba, and Sanja Fidler. 2021. DatasetGAN: Efficient Labeled Data
    on Computer Vision and Pattern Recognition (CVPR).                                          Factory with Minimal Human Effort. In IEEE Conference on Computer Vision and
Daoye Wang, Prashanth Chandran, Gaspard Zoss, Derek Bradley, and Paulo Gotardo.                 Pattern Recognition (CVPR).
    2022a. MoRF: Morphable Radiance Fields for Multiview Neural Head Modeling. In           Jian Zhao and Hui Zhang. 2022. Thin-plate spline motion model for image animation.
    ACM SIGGRAPH 2022 Conference Proceedings.                                                   In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 3657–3666.
Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul Srinivasan, Howard Zhou,                  Peng Zhou, Lingxi Xie, Bingbing Ni, and Qi Tian. 2021. CIPS-3D: A 3D-Aware Genera-
   Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser.             tor of GANs Based on Conditionally-Independent Pixel Synthesis. arXiv preprint
    2021b. IBRNet: Learning Multi-View Image-Based Rendering. In IEEE Conference on             arXiv:2110.09788 (2021).
    Computer Vision and Pattern Recognition (CVPR).
Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, and Qifeng Chen. 2022c. High-
    Fidelity GAN Inversion for Image Attribute Editing. In IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR).
Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. 2021a. One-Shot Free-View Neural
   Talking-Head Synthesis for Video Conferencing. In IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR).
Yaohui Wang, Di Yang, Francois Bremond, and Antitza Dantcheva. 2022b. Latent Image
   Animator: Learning to Animate Images via Latent Space Navigation. In International
    Conference on Learning Representations (ICLR).
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. 2004. Image Quality Assessment:
    From Error Visibility to Structural Similarity. TIP (2004).
Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. 2020. SynSin:
    End-to-end View Synthesis from a Single Image. In IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR).
Erroll Wood, Tadas Baltrusaitis, Charlie Hewitt, Matthew Johnson, Jingjing Shen, Nikola
    Milosavljevic, Daniel Wilde, Stephan Garbin, Chirag Raman, Jamie Shotton, Toby
    Sharp, Ivan Stojiljkovic, Tom Cashman, and Julien Valentin. 2022. 3D face recon-
    struction with dense landmarks. In European Conference on Computer Vision (ECCV).
Erroll Wood, Tadas Baltrušaitis, Charlie Hewitt, Sebastian Dziadzio, Thomas J. Cashman,
    and Jamie Shotton. 2021. Fake It Till You Make It: Face Analysis in the Wild Using
    Synthetic Data Alone. In IEEE International Conference on Computer Vision (ICCV).
Jianfeng Xiang, Jiaolong Yang, Yu Deng, and Xin Tong. 2022. Gram-hd: 3d-consistent
    image generation at high resolution with generative radiance manifolds. arXiv
    preprint arXiv:2206.07255 (2022).
Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping
    Luo. 2021. SegFormer: Simple and efficient design for semantic segmentation with
    transformers. In Advances in Neural Information Processing Systems (NeurIPS).
Jiaxin Xie, Hao Ouyang, Jingtan Piao, Chenyang Lei, and Qifeng Chen. 2022a. High-
    fidelity 3D GAN Inversion by Pseudo-multi-view Optimization. arXiv preprint
    arXiv:2211.15662 (2022).
Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan,
    Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. 2022b.
    Neural fields in visual computing and beyond. In Computer Graphics Forum, Vol. 41.
   Wiley Online Library.
Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang.
    2022a. SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single
    Image. In European Conference on Computer Vision (ECCV).
Yinghao Xu, Sida Peng, Ceyuan Yang, Yujun Shen, and Bolei Zhou. 2022b. 3D-aware
    Image Synthesis via Learning Structural and Textural Representations. In IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR).
Yu-Ying Yeh, Koki Nagano, Sameh Khamis, Jan Kautz, Ming-Yu Liu, and Ting-Chun
   Wang. 2022. Learning to Relight Portrait Images via a Virtual Light Stage and
    Synthetic-to-Real Adaptation. ACM Transactions on Graphics (SIGGRAPH ASIA)
   (2022).
Fei Yin, Yong Zhang, Xuan Wang, Tengfei Wang, Xiaoyu Li, Yuan Gong, Yanbo Fan,
   Xiaodong Cun, Öztireli Cengiz, and Yujiu Yang. 2022. 3D GAN Inversion with Facial
    Symmetry Prior. arxiv:2211.16927 (2022).
Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. 2021. pixelnerf: Neural
    radiance fields from one or few images. In IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR).
Xianggang Yu, Jiapeng Tang, Yipeng Qin, Chenghong Li, Xiaoguang Han, Linchao Bao,
    and Shuguang Cui. 2022. PVSeRF: Joint Pixel-, Voxel- and Surface-Aligned Radiance
    Field for Single-Image Novel View Synthesis. In ACM International Conference on
    Multimedia.
Egor Zakharov, Aleksei Ivakhnenko, Aliaksandra Shysheya, and Victor Lempitsky. 2020.
    Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars. In European



                                                                                                      ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                        Supplementary Material
                                        Real-Time Radiance Fields for Single-Image Portrait View Synthesis
                                        ALEX TREVITHICK∗ , University of California San Diego, USA
                                        MATTHEW CHAN and MICHAEL STENGEL, NVIDIA, USA
                                        ERIC R. CHAN∗ , Stanford University, USA
                                        CHAO LIU, ZHIDING YU, and SAMEH KHAMIS, NVIDIA, USA
                                        MANMOHAN CHANDRAKER and RAVI RAMAMOORTHI, University of California San Diego, USA
                                        KOKI NAGANO, NVIDIA, USA




arXiv:2305.02310v1 [cs.CV] 3 May 2023
                                        ACM Reference Format:                                                                         A1.2   Qualitative comparisons to [Ko et al. 2023]
                                        Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu,                       In Fig. A3, we provide comparisons to the state-of-the-art 3D GAN
                                        Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi,
                                                                                                                                      inversion work by [Ko et al. 2023]. While their method needs test-
                                        and Koki Nagano. 2023. Supplementary Material Real-Time Radiance Fields
                                        for Single-Image Portrait View Synthesis. ACM Trans. Graph. 1, 1, Article 1
                                                                                                                                      time optimization for the camera parameters and generator tuning,
                                        (January 2023), 10 pages. https://doi.org/10.1145/3592460                                     our method can process an unposed input in one-shot.

                                                                                                                                      A1.3   Additional comparisons
                                        In this supplement, we first provide the additional results including                         We provide additional comparisons to HeadNeRF, ROME, and EG3D-
                                        additional evaluations and comparisons in Sec. A1. We provide the                             PTI in Fig. A6. HeadNeRF only reconstructs the head region and
                                        implementation details of our models, including architecture de-                              struggles to reconstruct out-of-domain hair color (first row). ROME
                                        tails, camera augmentation, training details, and hyper parameters                            reconstructs the foreground image well, but requires background
                                        in Sec. A2. We also provide further experiment details in Sec. A3.                            segmentation and the geometry does not fully capture the hairstyles
                                        Finally, we discuss the limitations of our work in Sec. A4. We encour-                        and eyeglasses (second and third rows). EG3D-PTI reconstructs full
                                        age the readers to view our accompanying videos in the supplement,                            RGB images and geometry, but occasionally produces distorted 3D
                                        which include the additional visual comparisons, results, and live                            shapes (first row, better viewed in 3D in the accompanying video)
                                        demonstration of the novel view synthesis from a video input.                                 when the input view is non-frontal. Our method produces consistent
                                                                                                                                      image and geometry reconstruction quality across the variety of
                                        A1 ADDITIONAL RESULTS                                                                         inputs including a non-realistic human image (fourth row).
                                        A1.1 Additional qualitative results
                                        We provide additional qualitative results generated from a single
                                                                                                                                      A1.4   Percentile results based on LPIPS
                                        input image from FFHQ in Fig. A1 and AFHQ in Fig. A2. Fig. A1                                 In Fig. A7, we show our results on FFHQ and AFHQ shown in the
                                        shows that our method can handle complex hairstyles (first row),                              order of the LPIPS percentile scores. For FFHQ, we use the same
                                        and asymmetric facial expressions (second and third rows). Fig. A2                            randomly selected 500 FFHQ test set described in the main manu-
                                        shows our method can handle unconstrained poses of cats present                               script and for AFHQ, we randomly selected 485 images for which
                                        in the portraits as well as a wide variety of their textures.                                 we computed the LPIPS scores. The percentile results preferred by
                                                                                                                                      the LPIPS scores show that our method can demonstrate consistent
                                                                                                                                      quality for the large portion of the test images.
                                        ∗ This project was initiated and substantially carried out during an internship at NVIDIA.

                                                                                                                                         PSNR and SSIM on misaligned images. We provide our analysis
                                                                                                                                      on PSNR and SSIM metrics on images when images are aligned
                                        Authors’ addresses: Alex Trevithick, University of California San Diego, La Jolla, USA;       and when images have a small misalignment in Fig. A8. While
                                        Matthew Chan; Michael Stengel, NVIDIA, Santa Clara, USA; Eric R. Chan, Stanford               LPIPS scores can tolerate a small image misalignment (little change
                                        University, Stanford, USA; Chao Liu; Zhiding Yu; Sameh Khamis, NVIDIA, Santa Clara,
                                        USA; Manmohan Chandraker; Ravi Ramamoorthi, University of California San Diego,               when images are aligned or misaligned), the PSNR and SSIM scores
                                        La Jolla, USA; Koki Nagano, NVIDIA, Santa Clara, USA.                                         significantly change, which make these metrics unreliable for our
                                                                                                                                      tasks when the reconstructed images are not perfectly pixel-to-
                                                                                                                                      pixel aligned. The issues of PSNR and SSIM scores sensitivity under
                                        Permission to make digital or hard copies of all or part of this work for personal or
                                        classroom use is granted without fee provided that copies are not made or distributed
                                                                                                                                      geometry transformation are reported by previous work [Ding et al.
                                        for profit or commercial advantage and that copies bear this notice and the full citation     2022]. The DISTS [Ding et al. 2022] metric can also tolerate slight
                                        on the first page. Copyrights for components of this work owned by others than ACM            misalignment.
                                        must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
                                        to post on servers or to redistribute to lists, requires prior specific permission and/or a
                                        fee. Request permissions from permissions@acm.org.                                            A1.5   Evaluation of FID
                                        © 2023 Association for Computing Machinery.
                                        0730-0301/2023/1-ART1                                                                         Tab. A1 provides comparisons on FID calculated over 50K images
                                        https://doi.org/10.1145/3592460                                                               from FFHQ and 10K images from AFHQ. Our lightweight model

                                                                                                                                              ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:2   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. A1. Additional qualitative results generated by our method on FFHQ. Credits to USAID | Southern Africa, TimothyJ, toan đào song, Travis Rock, Curt
Mills, UGA CAES/Extension.


ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                                 Supplementary Material
                                              Real-Time Radiance Fields for Single-Image Portrait View Synthesis • 1:3




Fig. A2. Additional qualitative results generated by our method on AFHQ.


                                                ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:4   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano


                                                                                       Table A2. Additional ablation study comparing variants of our models.
                                                                                       "10K" refers to when we pre-compute 10K triplanes (subjects) and generate
                                                                                       supervising views on the fly using EG3D. "w/real data" shows preliminary
                                                                                       results of our initial attempt to incorporate real images in the training.


                                                                                                              LPIPS ↓   DISTS ↓     Pose ↓     ID ↑    FID ↓
                                                                                               10K            0.2797     0.995     0.0458    0.5000     4.60
                                                                                               w/real data    0.3060     0.1125    0.0539    0.4556     6.15
                                                                                               Ours           0.2894     .1053     0.0461    0.5230     4.45

Fig. A3. Qualitative comparisons to the concurrent work[Ko et al. 2023]
that relies on test-time camera optimization and generator weights tuning.             A1.6     Ablation study
                                                                                       We provide additional ablation studies concerning the importance
                                                                                       of the training dataset size and describe our preliminary attempt to
                                                                                       incorporate real images into the training.
                                                                                          Adding real data to the training. We attempted to incorporate real
                                                                                       data into the training pipeline in a variety of ways, but each one
                                                                                       proved unsuccessful. Our most succesful attempt was to train the
                                                                                       real part of the discriminator with images from FFHQ (using the
                                                                                       same conditioning as the original EG3D) and add significant noise
                                                                                       to the discriminator pose conditioning. Tab. A2 shows the results of
                                                                                       our preliminary attempt to incorporate real images in the training.
                                                                                       As can be seen in Fig. A9, even this method fails to reconstruct the
                                                                                       input image faithfully.
                                                                                          Size of training data. We additionally performed an ablation on
Fig. A4. Visualization of the limits in pitch and yaw of the camera pose
                                                                                       the number of subjects in the training set. To do so, we chose 10k
distribution for synthetic input images used to supervise our model.
                                                                                       latent codes from EG3D and rendered images from only these. We
                                                                                       found that training with 10k subjects with on the fly supervising
                                                                                       view generation (theoretically each subject has infinite views to
                                                                                       supervise) performs similarly to our method which synthesizes new
                                                                                       identities on the fly, as seen in Tab. A2. We hypothesize that this is
                                                                                       because the number of subjects is similar to the datasets, such as
                                                                                       VGGFace2 [Cao et al. 2018] (9K subjects) and CASIA-WebFace [Yi
                                                                                       et al. 2014] (10K subjects), used to train a one-shot face recognition
                                                                                       model.

                                                                                       A1.7     Additional Applications
                                                                                          Portrait frontalization. Our method can be applied to portrait face
Fig. A5. Visualization of two sigmas of noise in the principal point and focal         frontalization, which is useful for 3D reconstruction and avatar
length used for camera augmentation during our training.                               digitization [Nagano et al. 2019]. Please see the examples in Figs. A1
                                                                                       (4th column), A2 (4th column), A7 (3rd and 7th columns).

Table A1. Comparisons to an unconditional reference on FID evaluated over              A2     IMPLEMENTATION DETAILS
50K images of FFHQ and 10k images of AFHQ (including horizontal flips).
† Using transfer learning from a pretrained FFHQ model.                                We implement our framework in PyTorch on top of the official EG3D
                                                                                       codebase (https://github.com/NVlabs/eg3d).

                          FID ↓          FFHQ       AFHQ                                  EG3D pre-trained model. For human faces, we use the EG3D model
                                                                                       trained on the FFHQ dataset (ffhqrebalanced512-128.pkl). To sim-
                          EG3D            4.05      2.88†
                                                                                       plify our encoder training supervision, we replaced the latent code
                          Ours            3.48      2.39†
                          Ours (LT)       4.25      2.11†                              𝑊 injected to the StyleGAN2-based super-resolution layer with a
                                                                                       constant 1 and fine tuned the entire EG3D models on FFHQ for
                                                                                       additional 6.8 M images of training. This resulted in the FID score
                                                                                       4.05 for FFHQ as reported in the main manuscript. For cats faces,
"Ours (LT)" produces competitive FID scores to our full model                          we performed transfer learning [Karras et al. 2020] from this FFHQ
("Ours").                                                                              checkpoint and trained additional 3.2M images on the cat split of

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                                                                           Supplementary Material
                                                                                        Real-Time Radiance Fields for Single-Image Portrait View Synthesis • 1:5




Fig. A6. Additional qualitative comparisons against baselines on input view reconstruction and geometry. Credit to Steffen Geyer, Force Ouvrière, Matt Hamm,
scarlett1854, The Society of Motion Picture and Television Engineers.


AFHQv2, which resulted in the FID score 2.88, again reported in                  not technically 𝑭 , and instead 𝑭 after being processed by additional
the main manuscript. Please refer to the samples of synthetic data               convolutional layers.
generated by the EG3D model in Figs. A4 and A5.
                                                                                    Encoder for 𝑭high . We then encode the image again (with its
                                                                                 stacked pixel coordinates) with Ehigh with architectures given in
   Encoder for 𝑭low . We modify the first layer of the DeepLabV3
                                                                                 A11. Note that the input to the LT model’s high-resolution encoder
[Chen et al. 2017] architecture from the Pytorch Segmentation Mod-
                                                                                 is the second layer output features of DeepLabV3, rather than the
els repo [Iakubovskii 2019] by concatenating the 2D pixel coordinate
                                                                                 raw conditioning image.
of each pixel, so that the input is 5 channels. We also remove all
instances of batch norm (reintroducing the biases in all of the con-                Final triplane encoder. Finally, 𝑭 and 𝑭high are concatenated and
volutional layers). Otherwise, we use the standard encoder-decoder               decoded to the triplane 𝑻 with the architectures seen in Fig. A12,
as implemented with a ResNet34 encoder. We take the feature map                  completing the final encoding stage seen in the main paper’s pipeline
output of the decoder of DeepLabV3 (the layer before bilinear up-                figure.
sampling and segmentation head). As seen in the top half of the
                                                                                    Misc. For super-resolution, we used the same super-resolution
pipeline figure in the main paper, this gives us a feature map 𝑭low .
                                                                                 network architecture as EG3D, but replaced the 𝑤 to be constant
                                                                                 1 as mentioned earlier. For volume rendering and decoding the
   Encoder for 𝑭 with Conv Layers. 𝑭low is fed to a hybrid convolutional-
                                                                                 triplane, we follow EG3D; specifically, we use 48 depth samples for
transformer architecture. We will denote OverLapPatchEmbed as
                                                                                 coarse and fine passes for training. For discriminator, we use 2D
the patchwise embedding from Segformer [Xie et al. 2021] with
                                                                                 dual-disciriminator from EG3D.
patch_size=3, and TransformerBlock as the efficient self-attention
block from Segformer [Xie et al. 2021] without dropout, with kqv_bias,             Training. In practice, we alternate between taking gradient steps
and with layer normalization. Then the DeepLabV3 decoder fea-                    with reference view supervision and multiview supervision. To do so,
tures are fed to the module given in Fig. A10, which outputs the                 we begin by synthesizing synthetic input images for our encoder by
low-resolution canonical features 𝑭 as seen in the top half of the               sampling from the distribution for 𝑷ref as detailed in the main paper.
main paper’s pipeline figure. Note that the output of this module is             We render these cameras from triplanes from the frozen, pretrained

                                                                                          ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:6   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. A7. Results generated by our method on FFHQ and AFHQ shown in the order of percentile. Note that percentiles for FFHQ are calculated with alignment,
whereas AFHQ percentiles are calculated without alignment from a test set. Credits to yasminehabib, Rutgers Council on Public and Internation Affairs,
davitydave, Laity Lodge Family Camp, Houston Marsh, Ordiziako Jakintza Ikastola, Edgar Caraballo, NGÁO STUDIO, Debbie, WorldSkills UK, Craig Duffy.

ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                                                                       Supplementary Material
                                                                                    Real-Time Radiance Fields for Single-Image Portrait View Synthesis • 1:7


                                                                             EG3D. These are then fed to our encoder, whereby a triplane is then
                                                                             predicted. We can then render the same input cameras to take a
                                                                             gradient step for a loss computed only over the input views. We
                                                                             can additionally sample some cameras 𝑷mv , render ground truth
                                                                             from the EG3D triplanes, and render from the predicted triplanes
                                                                             for a multiview loss as well. In particular, at every gradient step, we
                                                                             always render 32 input cameras and 32 multiview cameras from the
                                                                             aforementioned distributions from the frozen EG3D. However, we
                                                                             do not always perform a gradient step for the input view loss.
                                                                                In the first stage of the training, we compute losses for the refer-
                                                                             ence set of cameras once every 10 triplane syntheses and perform
                                                                             gradient steps with respect to multiview supervision at every gradi-
                                                                             ent step. We additionally do not incorporate any adversarial loss,
                                                                             nor category loss, and do not train the MLP decoder and super-
                                                                             resolution network parameters at all. We train for 30k iterations
                                                                             without these objectives in this first stage. In the second stage, we
                                                                             add the adversarial and category losses and backpropagate to all pa-
                                                                             rameters in the pipeline, computing losses for the reference cameras
                                                                             every 2 gradient passes. In this stage, we remove the feature loss,
                                                                             and set the weight of the triplane loss to 0.01. After 37.5k iterations,
                                                                             we begin to compute multiview supervision and reference view
                                                                             supervision at every EG3D triplane synthesis step and continued to
                                                                             reach 220k iters in total (including the first 37.5k). We use a learning
                                                                             rate of 1e-4 for the encoder parameters, except for the transformer
                                                                             parameters, which have a learning rate of 5e-5. We use the same
                                                                             settings as EG3D for the the discriminator. We train for about 10
                                                                             days on 8 A100 GPUs or 8 A40 GPUs, for about 220k iterations in
                                                                             total.
                                                                                For training our model for cat faces, we used transfer learning
                                                                             following [Chan et al. 2022; Karras et al. 2020]. We initialized our
                                                                             cat face model with our human face model that is already trained,
                                                                             and ran training for additional 5 million images using the AFHQv2
                                                                             checkpoint from EG3D.

                                                                                Camera augmentation. EG3D assumes a fixed camera radius of
Fig. A8. LPIPS, DISTS, PSNR, and SSIM scores computed on images that         2.7, focal length of 18.83, zero camera roll, and a central principal
have a small misalignment. Previous work [Ding et al. 2022] reported that
                                                                             point. For the FFHQ experiments, we sample the focal length from
LPIPS can tolerate small geometric misalignment better than PSNR and
SSIM; DISTS is most robust to small image misalignment. Credits to Dong
                                                                             a normal distribution with standard deviation 1 centered at 18.83,
Quang, Presidencia de la Republica Mexicana.                                 the camera radius from a normal distribution centered at 2.7 with
                                                                             standard deviation 0.1, the principal point from a normal distribution
                                                                             with standard deviation 14 and centered at 256, and camera roll with
                                                                             a normal distribution of mean 0 and standard deviation 2 degrees.
                                                                                For the AFHQ experiments, we sample the focal length from a
                                                                             normal distribution with standard deviation 1.5 centered at 18.83,
                                                                             the camera radius from a normal distribution centered at 2.7 with
                                                                             standard deviation 0.1, the principal point from a normal distribution
                                                                             with standard deviation 25 and centered at 256, and camera roll with
                                                                             a normal distribution of mean 0 and standard deviation 6 degrees.

                                                                                Training data. We visualize the distribution of synthetic training
Fig. A9. Comparisons showing the output of our initial attempt in incorpo-   data in two figures. Fig. A4 visualizes the limits of the input image
rating real images. Credit to Mario Krajčír.
                                                                             poses in pitch and yaw for two subjects. Fig. A5 visualizes two
                                                                             sigmas of noise in the focal length and in the principal point used
                                                                             to augment camera information during our training.

                                                                               Inference. To calculate the timings of our method, we wrap the
                                                                             forward calls of the encoder (not the rendering) in autocast, which

                                                                                     ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:8   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano




Fig. A10. Details of the hybrid convolutional-transformer architecture which decodes the DeepLabV3 features before being concatenated with the high-
resolution image features later on.




                                       Fig. A11. Details of 𝐸 high which maps the input image to a high-resolution feature map.




Fig. A12. Details of the hybrid convolutional-transformer architecture which decodes the concatenated transformer features and high-resolution image
features directly into a triplane representation.


we use for real-time applications. For renderings, we use 48 depth                     A3 EXPERIMENT DETAILS
samples for real-time applications and 96 depth samples for the                        A3.1 Baselines
offline videos, following EG3D.
                                                                                       For all the baselines we used, we used official code from the authors
                                                                                       with released pre-trained checkpoints.


ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
                                                                                                                                    Supplementary Material
                                                                                 Real-Time Radiance Fields for Single-Image Portrait View Synthesis • 1:9


   For HeadNeRF, we used the highest resolution model model_Reso64       (LT), EG3D-PTI, and ROME, we compute the scale- and translation-
on the official website (https://github.com/CrisHY1995/headnerf),        invariant L1 and RMSE errors only on the valid depth pixels from
which produces the final output at 512 resolution using a feature        the ROME prediction. For HeadNeRF, we use only the valid depth
map of resolution 64.                                                    pixels from its prediction. We found that the geometry of HeadNeRF
   For ROME, we use the pre-trained model from the official code         can collapse to a plane in front of the predicted 3D face.
release (https://github.com/SamsungLabs/rome), which produces
the output at 256 resolution.                                            A4     DISCUSSION
   For EG3D models, we used the FFHQ and AFHQv2 fine tuned
models as described in Sec. A2, which are derived from the official        Ethical considerations. Since our method does not predict a latent
                                                                         space for portrait editing, it offers limited capabilities for portrait
EG3D models. The baseline "EG3D-PTI" combines the unconditional
                                                                         manipulations for malicious uses. However, it may be used to ma-
EG3D model with the lightweight generator tuning at test time
using Pivotal Tuning Inversion (PTI) [Roich et al. 2021] for 3D GAN      nipulate the viewpoint of a portrait. Potential solutions include
inversion from a single image. For the PTI inversion experiment,         detection of unseen image generators [Corvi et al. 2022; Nagano
we follow the hyperparameter settings from the original PTI paper        and Luebke 2021] and image watermarking [Yu et al. 2021, 2022].
and the PTI experiment done in the EG3D paper, and optimize the
latent code for 600 iterations, followed by fine tuning the generator      Adding real images to the training. Intuitively, incorporating real
weights for an additional 350 iterations. Unless noted otherwise, we     data into the training pipeline may be desirable in order to maximize
used this setting for all our experiments.                               the photorealism of rendered images and robustness in the most
                                                                         challenging settings. Future work may investigate the best way to
   FFHQ. For the comparisons on FFHQ between our model and               use both synthetic and real data in conjunction with one another.
other baselines, we postprocess our images with a rigid 2D warp.
To accomplish this, we estimate 2D landmarks with an off-the-shelf          Extension to handling a video input. The framework of our model
facial landmark model [Bulat and Tzimiropoulos 2017] for both the        is such that we require only a single image at inference time. Our
ground truth and our predicted image. We then solve the Orthog-          single-image method can be extended to handle a video input in a
onal Procrustes problem [Virtanen et al. 2020] to find the optimal       frame by frame fashion, but this may lead to flickering and temporal
orthogonal matrix to rigidly align our image onto the target image       inconsistency when rendering videos due to the single-image nature
approximately around the face region using the facial landmarks.         of our model. In such cases where multiple images of a subject are
Examples of this alignment can be seen in Fig. A8. We found that         available at inference, it is desirable to incorporate all such available
this alignment resulted in worse performance for EG3D-PTI and            information. Further work may investigate making the triplane
HeadNeRF, so we do not postprocess these methods’ renderings.            autoregressive or recurrent, conditioned on the previous frames so
For comparison to ROME, we align our renderings and ROME’s               that occlusions are handled in a consistent way, and there is greater
renderings to ROME’s warped input (lower-resolution) image with          temporal coherence.
the same process before computing the metrics. In any cases where
the warp produced black pixels on the border (out of bounds), we
                                                                         REFERENCES
set the ground truth, and the baselines pixels to black there as well,   Adrian Bulat and Georgios Tzimiropoulos. 2017. How far are we from solving the 2D
to ensure that we are comparing the exact same pixels between the           & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks). In
methods. For ROME and HeadNeRF, we compare only on their valid              International Conference on Computer Vision.
                                                                         Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman. 2018. Vggface2:
pixels, using the provided masks from these methods.                        A dataset for recognising faces across pose and age. In 2018 13th IEEE international
   To ensure fairness in the ID and Pose comparisons of our model           conference on automatic face & gesture recognition (FG 2018). IEEE, 67–74.
against HeadNeRF and ROME, we postprocess our images to align            Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De
                                                                            Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero
with the output of each baseline. For HeadNeRF, we mask both our            Karras, and Gordon Wetzstein. 2022. Efficient Geometry-aware 3D Generative
results and the ground truth results to the non-empty region using          Adversarial Networks. In IEEE Conference on Computer Vision and Pattern Recognition
the provided HeadNeRF masks, and calculate the Pose and ID losses           (CVPR).
                                                                         Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. 2017.
on the modified images. For ROME, we first downsample both our              Rethinking atrous convolution for semantic image segmentation. arXiv preprint
output and the ground truth images from 5122 to 2562 to align               arXiv:1706.05587 (2017).
                                                                         Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, and
with the ROME output, then align the ROME output to the ground              Luisa Verdoliva. 2022. On the detection of synthetic images generated by diffusion
truth images using the same landmark detection and Procrustes               models.
alignment as described for PTI. Again, we mask both our output           Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. 2022. Image Quality
                                                                            Assessment: Unifying Structure and Texture Similarity. IEEE Transactions on Pattern
and ground truth to the non-empty region predicted by ROME, then            Analysis and Machine Intelligence (2022).
calculate Pose and ID on the processed images.                           Pavel Iakubovskii. 2019. Segmentation Models Pytorch. https://github.com/qubvel/
                                                                            segmentation_models.pytorch.
   H3DS.. For the depth evaluations on the H3DS dataset [Ramon           Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila.
                                                                            2020. Training Generative Adversarial Networks with Limited Data. In Advances in
et al. 2021], we select a frontal image from all 23 subjects, then          Neural Information Processing Systems (NeurIPS).
render the ground truth depth from the corresponding camera pose         Jaehoon Ko, Kyusun Cho, Daewon Choi, Kwangrok Ryoo, and Seungryong Kim. 2023.
and the ground truth mesh. We normalize each depth to lie within            3D GAN Inversion with Pose Optimization. In IEEE Winter Conference on Applications
                                                                            of Computer Vision (WACV).
[0,1]. We then feed the RGB images as input to all baselines, and        Koki Nagano and David Luebke. 2021. StyleGAN3 Detector. https://github.com/
compute each method’s corresponding depth maps. For ours, ours              NVlabs/stylegan3-detector.


                                                                                   ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
1:10   •   Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano


Koki Nagano, Huiwen Luo, Zejian Wang, Jaewoo Seo, Jun Xing, Liwen Hu, Lingyu Wei,             Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors.
   and Hao Li. 2019. Deep face normalization. ACM Transactions on Graphics (TOG)              2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature
   38, 6 (2019), 1–16.                                                                        Methods 17 (2020), 261–272. https://doi.org/10.1038/s41592-019-0686-2
Eduard Ramon, Gil Triginer, Janna Escur, Albert Pumarola, Jaime Garcia, Xavier Giro-i      Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping
   Nieto, and Francesc Moreno-Noguer. 2021. H3D-Net: Few-Shot High-Fidelity 3D                Luo. 2021. SegFormer: Simple and efficient design for semantic segmentation with
   Head Reconstruction. In IEEE International Conference on Computer Vision (ICCV).           transformers. In Advances in Neural Information Processing Systems (NeurIPS).
Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. 2021. Pivotal               Dong Yi, Zhen Lei, Shengcai Liao, and S. Li. 2014. Learning Face Representation from
   Tuning for Latent-based Editing of Real Images. arXiv preprint arXiv:2106.05744            Scratch. ArXiv (2014).
   (2021).                                                                                 Ning Yu, Vladislav Skripniuk, Sahar Abdelnabi, and Mario Fritz. 2021. Artificial Finger-
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David          printing for Generative Models: Rooting Deepfake Attribution in Training Data. In
   Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright,            IEEE International Conference on Computer Vision (ICCV).
   Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay        Ning Yu, Vladislav Skripniuk, Dingfan Chen, Larry S. Davis, and Mario Fritz. 2022.
   Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan       Responsible Disclosure of Generative Models Using Scalable Fingerprinting. In
   Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert      International Conference on Learning Representations (ICLR).
   Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald,




ACM Trans. Graph., Vol. 1, No. 1, Article 1. Publication date: January 2023.
