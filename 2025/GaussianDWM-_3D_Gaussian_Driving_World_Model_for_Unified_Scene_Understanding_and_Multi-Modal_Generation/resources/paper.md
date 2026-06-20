# GaussianDWM: 3D Gaussian Driving World Model for Unified Scene Understanding and Multi-Modal Generation

Tianchen Deng1\*, Xuefeng Chen2\*, Yi Chen1\*, Qu Chen3,4, Yuyao Xu3,4, Lijin Yang3,4, Le Xu3,4, Yu Zhang3,4, Bo Zhang3,4, Wuxiong Huang3,4, Hesheng Wang1 1 Shanghai Jiao Tong University 2 Tsinghua University 3 MEGVII Technology 4 Mach Drive

![](images/0910939d57e5e392eb500bb00b1afabd5a05585f576cb95df869461bdb0b176f.jpg)  
Figure 1. We propose the first unified 3D Gaussian-based world model framework that achieves comprehensive scene understanding and scene generation for driving scenarios. It efficiently encodes complex scenes, samples task-relevant information, and handles diverse question-answering tasks. Moreover, by leveraging the extracted world knowledge, our framework guides the generative model to perform accurate spatial and temporal scene generation.

## Abstract

Driving World Models (DWMs) have been developing rapidly with the advances of generative models. However, existing DWMs lack 3D scene understanding capabilities and can only generate content conditioned on input data, without the ability to interpret or reason about the driving environment. Moreover, current approaches represent 3D spatial information with point cloud or BEV features do not accurately align textual information with the underlying 3D scene. To address these limitations, we propose a novel unified DWM framework based on 3D Gaussian scene representation, which enables both 3D scene understanding and multi-modal scene generation, while also enabling

contextual enrichment for understanding and generation tasks. Our approach directly aligns textual information with the 3D scene by embedding rich linguistic features into each Gaussian primitive, thereby achieving early modality alignment. In addition, we design a novel task-aware language-guided sampling strategy that removes redundant 3D Gaussians and injects accurate and compact 3D tokens into LLM. Furthermore, we design a dual-condition multi-modal generation model, where the information captured by our vision-language model is leveraged as a highlevel language condition in combination with a low-level image condition, jointly guiding the multi-modal generation process. We conduct comprehensive studies on the nuScenes, and NuInteract datasets to validate the effectiveness of our framework. Our method achieves state-ofthe-art performance. We will release the code publicly on

GitHub https://github.com/dtc111111/GaussianDWM .

## 1. Introduction

Driving World Model (DWM) [3, 20, 28, 57, 63] have become essential for autonomous driving for their ability to predict future scene generation. These models predict environmental changes and synthesize simulation data for risk forecasting, route optimization, and corner case training. Most existing approaches achieve this by predicting modalities such as images [70, 76] and point clouds [36, 37, 74], which represent the visual and geometric properties of the environment. While these DWMs excel at forecasting how the environment may evolve, they are difficult to interpret, describe, or query, and cannot easily provide contextual information (e.g., visual question answering or scene description). With the rapid progress of Large-Language Models and Vision-Language Models (VLMs) [56, 59–61, 69, 81], remarkable advancements have been achieved in general vision tasks by leveraging world knowledge and causal reasoning. This highlights the potential of combining the scene understanding capability of VLMs with the generative power of DWMs as a promising future direction. Notably, pioneering efforts such as HERMES [80] and UniFuture [38] have first achieved the unification of scene generation and understanding for autonomous driving world models. They adopts BEV/Depth features to represent spatial information, aligns them with the text space, and incorporates them into the generative model. However, this BEV-based scene representation only achieves feature-level alignment between textual and spatial inputs, which is not sufficiently accurate. To overcome this limitation, we propose a novel 3DGS-based scene representation that unifies scene understanding and generation.

First, we directly embed linguistic features into the 3D Gaussians, thereby achieving explicit spatial alignment between text and the 3D scene. This improves the accuracy of cross-modal alignment. Second, given the redundancy of Gaussians (i.e., extremely dense representations with tens of thousands of tokens per scene), it is impractical for VLMs to process such a large number of tokens effectively. To address this, we introduce a novel task-aware language-guided sampling strategy. By calculating the similarity between the input text and the 3D Gaussians, our method selects the most relevant Gaussians for the query and projects them into the context space through our projector, injecting accurate and compact 3D tokens into the textual understanding. Finally, we design a dual-condition multi-modal generation model. In this framework, the understanding from the VLM provides a high-level language condition, while image features serve as a low-level image condition. Together, they guide the generation of multiple modalities, including RGB, depth, and language. Moreover, our framework supports both spatial and temporal generation. Overall, our contributions are shown as follows:

• We propose the first 3D Gaussian-based unified world model framework that supports both scene understanding and scene generation.

• We introduce a novel token extraction and projection module for 3D Gaussian scene representations. Due to the redundancy of 3D Gaussians, we further develop a task-aware language-guided sampling strategy that overcomes token length limitations while preserving essential spatial information.

• We design a novel dual-condition multi-modal scene generation framework with high-level feature from world knowledge and low-level feature from images.

• Extended experiments on the Nuscenes and NuInteract datasets demonstrate that our method effectively bridges the gap between understanding and generation, enabling both accurate scene comprehension and more coherent future scene prediction.

## 2. Related Work

Novel View Synthesis for Urban Scene With the emergence of NeRF [44] and 3D Gaussian Splatting (3DGS) [30], many methods have adopted these representations across a wide range of tasks, including robotic mapping and localization [9, 11, 12, 14–16], VR [13, 21– 23, 34, 83], and autonomous driving [5, 45]. Early NeRFbased methods such as NSG [45], SUDS [53], ProSGNeRF [8], EmerNeRF [71], and FreeDriveRF [62] achieved dynamic–static disentangled reconstruction through the use of scene graphs, optical flow, or other motion cues. More recently, several 3DGS-based approaches have been proposed to further improve rendering efficiency. PVG [4] introduces periodic vibration-based temporal dynamics to unify static and dynamic elements without manual annotations. Methods such as Street Gaussians [68], Driving Gaussian [79], and DeSiRe-GS [48] also explicitly separate dynamic and static components for reconstruction. LESSON [47] proposes a teacher-guided diffusion strategy for generating 3D Gaussian splats using only 2D supervision. STORM [72] proposes a feed-forward Transformer architecture to infer dynamic Gaussians and their velocities, enabling efficient large-scale outdoor scene reconstruction. DrivingForward [51] achieves feed-forward reconstruction from sparse surround-view inputs using self-supervised pose and depth estimation. MUDG [84] and Dist-4D [25] propose multimodal novel view synthesis frameworks that generate both RGB and depth modalities.

Driving World Model Driving world models [18, 32, 40, 52, 67] have attracted considerable attention in autonomous driving due to their ability to provide comprehensive environmental representations and predict future scenarios. Current methods primarily rely on 2D and 3D conditions for scene generation. GAIA-1 [29] introduces an autoregressive model for video generation in driving scenarios. DriveDreamer [76] proposes a scene generation framework conditioned on 3D structure to provide geometric representations that benefit downstream autonomous driving tasks. MagicDrive [19] presents a street-view synthesis framework with precise 3D controls (e.g., camera poses, road maps) using cross-view attention, improving 3D object detection and BEV segmentation. DreamDrive [43] further combines video diffusion with hybrid Gaussian scene representations to synthesize 4D scenes with 3D-consistent dynamic video rendering. Most recently, UniScene [33] proposes an occupancy-centric approach that unifies semantic, visual, and LiDAR data generation. However, existing DWMs overlook the scene understanding ability of the driving environment.

Large Language Models for Driving Large Language Models (LLMs) have demonstrated impressive generalization ability and extensive world knowledge across various tasks, including scene understanding [27, 58, 75, 78, 82], visual question answering (VQA), and both 2D and 3D visual grounding. DriveGPT4 [66] processes front-view video inputs to predict vehicle actions while providing natural language justifications via an LLM. DriveLM [50] leverages LLMs for graph-based VQA and end-to-end autonomous driving. NuInteract [77] further integrates large vision-language models (LVLMs) with a spatial processor using a set of learnable queries, trained on the large-scale NuInteract dataset containing over 1.5M multi-view image–language pairs, covering dense scene captioning and diverse interactive tasks. GaussianVLM [26] introduces a 3D Gaussian-based visual question answering (VQA) framework, where a SceneSplat-style variational autoencoder (VAE) is employed to directly encode 3D Gaussian scenes. Hermes [80] proposes a BEV-based world model that integrates scene representation with a Vision-Language Model (VLM) for joint scene understanding and generation, which is the most closely related work to ours. However, existing approaches typically rely on image, point cloud, or BEV representations. In contrast, our method leverages a 3D Gaussian scene representation [10] that achieves explicit spatial alignment between language features and 3D geometry, resulting in more accurate multimodal correspondence and improved representation of both texture and structural information in complex environments.

## 3. Method

In this paper, we propose GaussianDWM, a unified framework with 3D gaussian scene representation for driving scenarios understanding and generation. The pipeline of our method is illustrated in Fig. 2. The input to our method consists of images $\{ I _ { i } \}$ , Gaussian ellipsoids $\left\{ { G } _ { i } \right\}$ , and query text $\{ t _ { i } \}$ . The framework is composed of three main modules:(i) World tokenizer (Sec. 3.1); (ii) Scene Understanding(Sec. 3.2); (iii) Multi-modal Generation (Sec. 3.3). We elaborate on the entire pipeline of our system in the following subsections.

## 3.1. World Tokenizer

Our world tokenizer encodes the world observations, i.e., the current multi-view images, into a compact continuous 3D Gaussian representation. We then apply a gaussian projector to align the selected 3D features to the text space before processing by the VLM.

3D Gaussian Tokenizer To preserve texture, 3D structure, and language alignment, we adopt 3D Gaussians as the scene representation for LLM input. We build upon LangSplat [49] to construct a 3DGS language field, where each Gaussian is augmented with a language embedding $f _ { i } .$ These embeddings are obtained from CLIP features, which inherit hierarchical semantics extracted via SAM [31]. We then follow the standard 3DGS rendering strategy, incorporating language information directly into the Gaussian primitives.

$$
\pmb { F } ( \pmb { v } ) = \sum _ { i \in \mathcal { N } } f _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{1}
$$

where $\boldsymbol { F } ( \boldsymbol { v } )$ represents the language embedding rendered at pixel $v ,$ and $\alpha _ { i } = o _ { i } G _ { i } ^ { 2 D } ( v )$ . Here $o _ { i }$ is the opacity of the i th Gaussian and $G _ { i } ^ { 2 D } ( \cdot )$ represents the function of the ith Gaussian projected onto 2D. To further reduce memory consumption and improve efficiency, we introduce a scenewise language autoencoder E, which maps the CLIP embeddings $\pmb { F } ( v ) \in \mathbb { R } ^ { D }$ to $\pmb { H } ( \boldsymbol { v } ) = E \left( \pmb { F } ( \boldsymbol { v } ) \right) \in \mathbb { R } ^ { d }$ , where $d \ll D$ . We select $d = 3 , D = 5 1 2$ . Then we learn a decoder Ψ to reconstruction CLIP feature. Our autoencoder can significantly decrease memory requirements while retaining semantic fidelity.

3D Gaussian Projector We first align the extracted 3D Gaussian tokens to the text space. For each Gaussian primitive $G _ { i }$ , we represent its attributes as $G _ { i } = ( x _ { i } , o _ { i } , s _ { i } , r _ { i } , f _ { i } )$ where $x _ { i } \in \mathbb { R } ^ { 3 }$ denotes the 3D spatial position, $o _ { i }$ the opacity, $s _ { i }$ the scale, $r _ { i }$ the rotation, and $f _ { i }$ the associated CLIP feature. For the Gaussian tokenizer, we first apply learnable Fourier embeddings [44] to the 3D coordinates $x _ { i } { : }$

$$
\gamma ( x _ { i } ) = \left[ \sin ( 2 ^ { k } \pi x _ { i } ) , \cos ( 2 ^ { k } \pi x _ { i } ) \right] _ { k = 0 } ^ { L - 1 } ,\tag{2}
$$

where L is set to 10. For the opacity $o _ { i } ,$ we apply a sigmoid activation $\hat { o } _ { i } = \sigma ( o _ { i } )$ to constrain the value to [0, 1]. For the CLIP feature $f _ { i } ,$ we use a pre-trained scene-wise decoder to project it to a 512 dimension $\tilde { f } _ { i } = \Psi ( f _ { i } ) \in \mathbb { R } ^ { N \times 5 1 2 }$ N denotes the number of 3D Gaussian ellipsoids. Then, we employ a set of MLP projectors to map Gaussian attribute into a shared 4096-dimensional feature space, i.e., $h _ { i } ^ { x } = \phi _ { x } ( \gamma ( x _ { i } ) ) , h _ { i } ^ { o } = \phi _ { o } ( \hat { o } _ { i } ) , h _ { i } ^ { s } = \phi _ { s } ( s _ { i } ) , h _ { i } ^ { r } = \phi _ { r } ( r _ { i } )$ , and $h _ { i } ^ { f } = \phi _ { f } ( \tilde { f } _ { i } )$ , where $\gamma ( \cdot )$ is the Fourier embedding and each ϕ·(·) is a learnable MLP. Finally, we fuse the projected features via learnable weights to obtain the Gaussian scene token $\begin{array} { r } { \mathcal G _ { i } = \sum _ { p \in \{ x , o , s , r , f \} } \alpha _ { p } \cdot h _ { i } ^ { p } } \end{array}$ , where each $\alpha _ { p }$ is a trainable scalar normalized by a softmax. For text queries, we tokenize the input prompts into vocabulary indices and text tokens T for LLM processing.

![](images/4307d82ccd856db6084992f0ca45a7c366c1fa3ba46f9d39fd19aabf885a6eac.jpg)  
Figure 2. System Overview. We propose the first unified 3D Gaussian-based world model framework that simultaneously supports both scene understanding and scene generation. We first employ a scene encoder to align the language information with the 3D Gaussians, resulting in language-augmented 3D Gaussian representations. Then, a designed Gaussian projector aligns the 3D Gaussian tokens, 2D image tokens, and text tokens into a unified latent space. Subsequently, a task-aware hybrid sampling strategy is applied to select the most relevant 3D Gaussian tokens for the current query, which are then fed into the LLM. The LLM produces both textual answers and high-level language features that encapsulate world knowledge, which are later used to guide multi-modal scene generation.

## 3.2. Scene Understanding

This section introduces the world understanding module. The Large Language Model (LLM) interprets driving scenarios from the world tokenizer outputs $\mathbf { \bar { \mathcal { G } } } _ { i } \in \mathbb { R } ^ { N \times \mathbf { \bar { C } } }$ according to user instructions. Then, the LLM parses the user instruction $\mathcal { T } _ { i }$ and extracts world knowledge from the driving scene, generating both a textual answer $t _ { i }$ and a language feature representation $C _ { L }$ , which is later used as a condition signal for scene generation. This feature encodes high-level world knowledge as well as spatial information and is later used as a condition for scene generation. We implement the LLM using the widely adopted Qwen3 model [69]. The overall architecture is:

$$
\{ t _ { i } , C _ { i } ^ { l } \} = L L M ( \mathcal { G } _ { i } , \mathcal { T } _ { i } )\tag{3}
$$

Task-aware Language-guided Sampling However, directly converting all Gaussians into tokens would exceed the maximum token length limits of LLMs, and the high degree of redundancy in the Gaussian set would make it difficult for the LLM to reason about spatial interactions across different views. To address this, we propose a task-aware hybrid sampling strategy tailored for 3D Gaussian scene representations. For scene understanding tasks, we adopt a global sampling strategy that preserves holistic scene information by selecting a representative subset of Gaussians.

We apply both uniform sampling and top-k sampling to select $N = 4 0 9 6$ gaussian tokens from the hundreds of thousands in the scene, which are then fed into the LLM. In contrast, for 2D and 3D visual grounding tasks, we further introduce a language-guided sampling module, which retokenizes the dense scene representation into a more compact and sparse form conditioned on the text query. Specifically, we apply similarity calculation between the 3D Gaussian features and the text query to identify and retain only those Gaussians that are most relevant to the query. This design is highly effective for both 2D and 3D grounding tasks, as it enables the model to selectively inject the most relevant 3D spatial information into the language reasoning process. Compared with the previous Hermes framework, our 3D Gaussian–based LLM model can effectively respond to user queries about the driving environment, providing scene descriptions and answers to visual questions. In addition, it supports both 2D and 3D visual grounding tasks, achieving state-of-the-art performance in terms of averaged metrics.

Training Strategy Similar to many VLM training protocols, we adopt a two-stage training strategy consisting of an alignment stage and a fine-tuning stage. In the first stage, we freeze the entire VLM and train the aligner with full parameters for 5k warm-up steps to align the visual representations with the textual space. In the second stage, we further adapt the LLM using LoRA for 30k steps. Both stages share the same training objective. Following, we use a prefix language modeling, where the model is conditioned on an input prefix and trained to autoregressively generate the target continuation:

$$
\mathcal { L } ( \theta , \mathcal { B } ) = - \sum _ { \{ t _ { \mathrm { p r e f i x } } , t _ { \mathrm { g t } } \} \in B } \sum _ { i = 1 } ^ { | t _ { \mathrm { g t } } | } \log p _ { \theta } \left( t _ { \mathrm { g t } } ^ { ( i ) } \mid t _ { \mathrm { g t } } ^ { ( < i ) } , t _ { \mathrm { p r e f i x } } \right) ,\tag{4}
$$

where θ are the model parameters, B denotes a batch of samples of prefix input $t _ { \mathrm { p r e f i x } }$ (text tokens, image tokens and 3D Gaussian tokens), and ground truth response $t _ { \mathrm { g t } } . t _ { \mathrm { g t } } ^ { ( t ) }$ denotes the t-th token in the ground truth response sequence.

## 3.3. Multi-modal Scene Generation

In this section, we propose a dual condition multi-modal scene generation model. Our model consists a denoising UNet [1], and a frozen pre-trained VAE [64] to encode RGB images $I _ { i } .$ , and depth maps $D _ { i }$ into a unified latent space. To satisfy the VAE input specifications, we convert depth maps into pseudo-RGB images via channel replication. The VAE encoding process can be written as: $z _ { I } =$ $\begin{array} { r } { \mathcal { E } \left( I _ { i } \right) , \quad z _ { D } = \mathcal { E } \left( D _ { i } \right) } \end{array}$ During decoding, the VAE decoder D reconstructs RGB, and depth from the latent variable $z _ { i } .$ For the depth map, we average the three output channels of the decoded result to obtain the final single-channel depth prediction: $\begin{array} { r } { \hat { I } _ { i } = \mathcal { D } \left( z _ { I } \right) , \quad \hat { D } _ { i } = \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } \mathcal { D } \left( z _ { D } \right) _ { c } } \end{array}$ In the training phase, we random sample on the known camera trajectory, and get the latent code of corresponding color and depth images with the VAE encoder $z _ { I } , z _ { D }$ . At each timestep t, we add noise to the sampled data. Then, we use the projection matrix to project surrounding point cloud at time t to time $t + n$ to serve as low-level image conditions $\{ C _ { I } , C _ { D } \}$ Then we concatenate the noisy latent representations of each modality with the low-level image control signal and high-level language control signal $C _ { L }$ from LLM as the input to the denoising diffusion network. The model is using a v-prediction objective. The target $\mathbf { v } _ { t }$ is defined as: $\mathbf v _ { t } = \alpha _ { t } \epsilon _ { t } - \sigma _ { t } d _ { t }$ , where $\epsilon _ { t } \sim \mathcal { N } ( 0 , I )$ denotes the sampled gaussian noise, $\alpha _ { t }$ and $\sigma _ { t }$ represent the time-dependent noise scheduling coefficients, and $\mathbf { } d _ { t }$ corresponds to the noisy input modality requiring denoising. The training objective is defined as:

$$
\mathcal { L } = \mathbb { E } _ { d , \epsilon , t , s } \left. \mathcal { F } _ { \theta } \left( d _ { t } , d _ { \mathrm { r e f } } , C _ { I } , C _ { D } , C _ { L } , s \right) - \mathbf { v } _ { t } \right. _ { 2 } ^ { 2 } ,\tag{5}
$$

The low-level conditions $C _ { I }$ and $C _ { D }$ represent the scene’s texture and geometric information, guiding the generation process. Meanwhile, the high-level language feature $C _ { L }$ encapsulates comprehensive world knowledge extracted from the LLM. By conditioning the generation on multiple levels of information, our model achieves more accurate and consistent temporal and spatial synthesis.

For spatial generation, we project the surrounding point cloud using the spatial transformation of the query frame to obtain a sparse condition map. For temporal generation, we utilize the trajectory predicted by the front-end LLM to project the point cloud and construct a temporal sparse condition map, enabling temporally coherent scene generation.

Our generation model supports both spatial scene generation, i.e., novel view synthesis with spatial shifts of 1m or 2m, and temporal scene generation, i.e., future scene prediction at 1s and 2s into the future.

## 4. Experiments

## 4.1. Dataset and Evaluation Metric

We evaluate our method on two datasets for both scene understanding and scene generation. NuScenes [2] is a widely used autonomous driving dataset. We use six surrounding camera images as inputs. NuInteract [77] are recent benchmark for scene understanding. NuInteract provides ∼1.5M annotations and supports multiple tasks, including 2D perception and 3D visual grounding. For scene understanding, we adopt standard language and captioning metrics, including gCIDEr [54], BLEU-4 [46], ROUGE [39]. For scene generation, we use FID for images, and FVD for videos as evaluation metrics.

## 4.2. Implementation Details

Our training pipeline consists of three stages. In the first stage, we train the Gaussian tokenizer, projector, and the proposed sampling strategy independently. We then integrate these components with the LLM and perform joint fine-tuning. All models in this stage are trained using 16 NVIDIA A100 GPUs. In the second stage, we train the multi-modal generation module. We start by training a low-resolution (224 × 400) RGB video generation model, extend it to low-resolution RGB-D generation, and finally refine it into a high-resolution (424 × 800) RGB-D video generator using a mixed-frame-length strategy. The model is optimized using simulation-free rectified flow and a vprediction loss [17]. In the final stage, we perform end-toend joint optimization over all components to ensure consistency between scene understanding and scene generation.

## 4.3. Scene Understanding

In this section, we present the results of scene understanding, which are divided into four tasks: region description and perception, planning, 2D visual grounding, and 3D visual grounding. The results are shown in Tab. 1. The experimental settings strictly follow Drivemonkey [77]. We primarily compare two categories of methods: (1) LLMbased visual grounding approaches such as LLaVA [41] and InternVL [6], and (2) specialized 3D detection models such as BEVFormer [35], PETR [42], and CAPE [65]. As shown in Tab. 1, GaussianDWM significantly outperforms previous LVLMs in terms of average metrics across all tasks on the NuInteract test dataset. It achieves state-of-the-art performance in terms of averaged metrics on the RDP, 2D VG, 3D VG, and Planning tasks, outperforming previous state-of-the-art methods by a relative margin of 13.6%, effectively demonstrating the importance of introducing a 3D Gaussian scene representation for enabling LLMs to better comprehend 3D spatial information. Compared with previous point cloud– and BEV-based scene representations, our 3D Gaussian formulation allows for explicit alignment between 3D structural, texture, and language features, leading to more efficient and semantically consistent information injection into LLMs. Moreover, our task-aware hybrid sampling strategy efficiently selects Gaussian tokens most relevant to the query text while filtering out redundant 3D Gaussians. Compared with the current state-of-the-art VQA method Drivemonkey, our model shows clear advantages in both 2D and 3D visual grounding tasks, while also achieving comparable performance to specialized 3D detectors designed specifically for 3D VG.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Years</td><td rowspan="2">LLM</td><td colspan="3">2D RD&amp;Pre↑</td><td colspan="3">2DVG↑</td><td colspan="3">3DVG↑</td><td rowspan="2">Plan↑ Avg.↑</td></tr><tr><td>BLEU</td><td>Rouge_L</td><td>CIDEr</td><td>mAP</td><td>F1</td><td>MIoU</td><td>Pr mAP</td><td>F1</td><td>Acc</td></tr><tr><td>LLaVA1.5</td><td>2024</td><td>Vicuna-7B</td><td>64.23</td><td>76.69</td><td>74.82</td><td>0.10</td><td>0.16</td><td>14.31</td><td>6.51</td><td>5.33 3.12</td><td>36.20</td><td>28.16</td></tr><tr><td>MiniCPM-V 2</td><td>2024</td><td>MiniCPM-2B</td><td>47.43</td><td>63.16</td><td>69.88</td><td>0.11</td><td>0.13</td><td>13.34</td><td>0.97 1.55</td><td>0.86</td><td>36.69</td><td>23.41</td></tr><tr><td>MiniCPM-V 2.6</td><td>2024</td><td>QWen2-7B</td><td>47.92</td><td>69.11</td><td>70.20</td><td>0.36</td><td>0.49</td><td>18.74</td><td>1.97</td><td>1.61 0.93</td><td>36.42</td><td>24.78</td></tr><tr><td>IntermVL1.5-2B</td><td>2024</td><td>InternLM2-7B</td><td>67.14</td><td>81.10</td><td>79.83</td><td>14.74</td><td>17.64</td><td>55.43</td><td>28.05 21.73</td><td>12.92</td><td>53.96</td><td>43.25</td></tr><tr><td>IntermVL1.5-4B</td><td>2024</td><td>Phi3-4B</td><td>66.63</td><td>80.64</td><td>79.24</td><td>14.27</td><td>17.60</td><td>53.52</td><td>25.14 19.46</td><td>11.63</td><td>40.25</td><td>40.84</td></tr><tr><td>QWen2VL</td><td>2024</td><td>Qwen2-2B</td><td>67.92</td><td>80.24</td><td>78.51</td><td>17.11</td><td>20.87</td><td>57.24</td><td>12.82 10.20</td><td>6.12</td><td>45.59</td><td>39.66</td></tr><tr><td>Qwen2VL</td><td>2024</td><td>QWen2-7B</td><td>66.65</td><td>78.57</td><td>77.97</td><td>16.06</td><td>20.04</td><td>55.51</td><td>20.64</td><td>16.26 9.82</td><td>49.33</td><td>41.09</td></tr><tr><td>InternVL2-1B</td><td>2024</td><td>Qwen2-0.5B</td><td>66.89</td><td>81.00</td><td>79.59</td><td>16.70</td><td>20.21</td><td>55.94</td><td>23.36 18.35</td><td>10.94</td><td>44.08</td><td>41.71</td></tr><tr><td>InternVL2-2B</td><td>2024</td><td>InternLM2-2B</td><td>66.77</td><td>80.87</td><td>79.62</td><td>16.12</td><td>19.49</td><td>55.29 27.83</td><td>21.09</td><td>12.58</td><td>44.61</td><td>42.43</td></tr><tr><td>InternVL2-4B</td><td>2024</td><td>Phi3-4B</td><td>66.88</td><td>80.76</td><td>79.29</td><td>19.14</td><td>23.47</td><td>59.07 25.28</td><td>20.12</td><td>11.97</td><td>40.43</td><td>42.64</td></tr><tr><td>IntermVL2-8B</td><td>2024</td><td>InternLM2.5-7B</td><td>67.32</td><td>81.39</td><td>80.01</td><td>20.61</td><td>25.24</td><td>61.90</td><td>31.47 24.67</td><td>14.70</td><td>46.93</td><td>45.42</td></tr><tr><td>DriveMonkey</td><td>2025</td><td>InternLM2.5-7B</td><td>67.50</td><td>81.15</td><td>79.79</td><td>19.47</td><td>24.02</td><td>59.36 51.90</td><td>34.53</td><td>20.86</td><td>82.64</td><td>52.12</td></tr><tr><td>Bevformer</td><td>2022</td><td></td><td></td><td></td><td></td><td></td><td></td><td>44.50</td><td>23.69</td><td>1.576</td><td></td><td></td></tr><tr><td>PETR</td><td>2022</td><td></td><td></td><td></td><td> Unsupported</td><td></td><td></td><td>55.80</td><td>31.34</td><td>20.58</td><td></td><td> Unsupported</td></tr><tr><td>CAPE</td><td>2023</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.02</td><td>32.94</td><td>21.33</td><td></td></tr><tr><td>GaussianDWM</td><td>2025</td><td>Qwen3-8B</td><td>68.78</td><td>81.06</td><td>78.72</td><td>34.9540.4971.85</td><td></td><td></td><td>50.66</td><td>52.78</td><td>32.05 80.95</td><td>59.23</td></tr></table>

Table 1. The comparison between our GaussianDWM and other state-of-the-art models on the NuInteract dataset [77]. The scene understanding task includes four subtasks: region description and perception, 2D visual grounding, 3D visual grounding, and planning. Our method achieves state-of-the-art average performance across all four tasks, which fully demonstrates the effectiveness of introducing a 3D Gaussian scene representation for enhancing the LLM’s capability to understand 3D spatial information.
<table><tr><td rowspan="2">Finetune</td><td rowspan="2">Gaussian</td><td rowspan="2">Sampling</td><td colspan="3">2D RD&amp; Pre↑ Rouge_L</td><td colspan="3">2D VG↑</td><td colspan="3">3D VG个</td><td rowspan="2">Plan↑</td><td rowspan="2">Avg.↑</td></tr><tr><td>BLEU</td><td></td><td>CIDEr</td><td>mAP</td><td>F1</td><td>MIoU</td><td>Pr</td><td>mAP</td><td>Acc</td></tr><tr><td>zeroshot</td><td>w/o</td><td></td><td>2.91</td><td>12.68</td><td>0.59</td><td>0.00</td><td>0.00</td><td>12.24</td><td>48.75</td><td>47.59</td><td>F1 29.12</td><td>0.00</td><td>15.39</td></tr><tr><td>finetuned</td><td>w/o</td><td></td><td>65.09</td><td>78.35</td><td>76.56</td><td>30.01</td><td>35.45</td><td>69.01</td><td>50.36</td><td>51.88</td><td>31.43</td><td>45.09</td><td>53.32</td></tr><tr><td>finetuned</td><td>W</td><td>Random</td><td>66.19</td><td>79.00</td><td>76.97</td><td>33.94</td><td>39.37</td><td>71.40</td><td>50.94</td><td>52.85</td><td>32.03</td><td>49.43</td><td>55.21</td></tr><tr><td>finetuned</td><td>W</td><td>Top-k + Uniform</td><td>68.78</td><td>81.06</td><td>78.82</td><td>33.89</td><td>39.31</td><td>71.37</td><td>51.16</td><td>52.87</td><td>32.05</td><td>80.95</td><td>58.93</td></tr><tr><td>finetuned</td><td>W</td><td> Top-k + Uniform + similarity</td><td>68.78</td><td>81.06</td><td>78.82</td><td>34.95</td><td>40.49</td><td>71.85</td><td>50.66</td><td>52.78</td><td>32.05</td><td>80.95</td><td>59.23</td></tr></table>

Table 2. We conduct ablation studies to validate the effectiveness of each proposed component, including the 3D Gaussian scene representation, the top-k and uniform sampling strategies, and the similarity–based sampling module. Note that the similarity sampling strategy is applied only to grounding tasks requiring focused attention (e.g., 2DVG and 3DVG).

## 4.4. Scene Generation

In this section, we present the evaluation of multi-modal scene generation, which includes both spatial scene generation and temporal scene generation. All experiments are conducted on the nuScenes dataset [2]. Following previous works [19, 25], we interpolate the 2Hz keyframe annotations to a higher frame rate of 12Hz, and the experimental

settings strictly follow [25].

We compare our method with representative street-view synthesis approaches, including PVG [4], EmerNeRF [71], StreetGaussian [68], OmniRe [7], FreeVS [55], DiST-4D [24]. As shown in Tab. 3, our method outperforms all existing methods and achieves state-of-the-art performance. We use FID and FVD as evaluation metrics since ground-truth data is not available after spatial shifts. Our method demonstrates superior consistency and photorealism under extreme viewpoint shifts (±4m), outperforming direct reconstruction-based approaches. This indicates that our framework effectively combines the advantages of 3D Gaussian scene representation and diffusion-based generative modeling. Under the guidance of our dual-condition mechanism—leveraging both high-level world knowledge and low-level geometric cues—our method achieves stateof-the-art spatial and texture fidelity under large viewpoint variations.

We also visualize the qualitative results of spatial scene generation in Fig. 3. We further compare our method with several existing novel view synthesis approaches, and the qualitative results are presented in Fig. 4. As shown, our RGB and depth generations exhibit photorealistic qual-

![](images/4c31dbfcc08c234c0df628d361e7bb63f08cdf462c7aa9b742b7aeb765da4d75.jpg)

3D Gaussians  
![](images/1d45764bc1c6175cf2daac0e98ffd7eb6243c56bfc90344d98e98cf042d250f2.jpg)

Q: Based on <gauss>, find all car in this <CAM\_BACK>. For each car, provide its 3D bounding box. The output format required is JSON: \`[{\"bbox\_3d\":[x\_center, y\_center, z\_center, x\_size, y\_size, z\_size, roll, pitch, yaw],\"label\":\"category\"}] A: [{\"bbox\_3d\": [-9.98, 0.36, 18.22, 4.12, 1.65, 1.7, -0.0, 0.51, 1.0], \"label\": \"car\"}, {\"bbox\_3d\": [-9.11, 0.65, 12.62, 4.49, 1.64, 1.69, -0.0, 2.48, 1.0], \"label\": \"car\"}]

Q: Based on <gauss>, what is the appropriate driving course for the ego vehicle under these conditions? A: Turn

Q: Based on <gauss>, what information does the <CAM\_BACK> <box>(130,228),(242,268)</box> contain? Could you provide insights into the future motion patterns of the parked car at <CAM\_BACK> <box>(130,228),(242,268)</box>?   
A: Parked car. It will remain stationary.

Q: Based on <gauss>, what does the object in the <CAM\_FRONT> <box>(475,260),(634,446)</box> stand for? A: White line on the road, it is about 11 meters away from the car.

Spatial Generation  
![](images/d44338f5eb3b5e21a5da787a8b4e3a99d7c6bfd7bc8f582c63000b6ce634bdc5.jpg)  
Figure 3. Qualitative results for scene understanding and scene generation. From top to bottom, we display the multi-view input of the current scene and the 3D Gaussian ellipsoids, the scene understanding results, and the spatial and temporal scene generation results.

ity, demonstrating the complementary strengths of world knowledge, diffusion-based generation, and 3D Gaussian scene representation.

The proposed dual-condition design enables high-level world knowledge to effectively guide the model’s temporal reasoning for future scene prediction, while the lowlevel sparse condition constrains 2D texture and style consistency. This demonstrates the importance of incorporating explicit world knowledge into world models to improve both semantic and spatial coherence during temporal generation. Overall, GaussianDWM is capable of simultaneously understanding and generating complex driving scenarios, highlighting a promising research direction toward unified scene understanding and generation in world modeling.

## 4.5. Ablation Study

In this section, we conduct comprehensive ablation studies to validate the effectiveness of each component in our proposed framework. We systematically analyze the impact of the key modules, including the 3D Gaussian scene representation, the hybrid sampling strategies, and the dualcondition generation design, to demonstrate their respective contributions to the overall performance.

![](images/1083793e3c4dff7fae9c85d5c388bad2bb8b18c596a2bd0d2135ddb5d8988c7f.jpg)

Figure 4. Qualitative comparison of RGB-D NVS with 2m shift. Compared with state-of-the-art reconstruction-based methods for spatial NVS [4, 7, 68, 73], our method reduce artifacts of dynamic objects and preserves temporal-spatial consistency across large viewpoint shifts.
<table><tr><td rowspan="2">Method</td><td> $\mathrm { S h i f t } \pm 1$ </td><td> ${ \mathrm { S h i f t } } \pm 2$ </td><td> $\mathrm { S h i f t } \pm 4$ </td><td rowspan="2"></td></tr><tr><td>FID↓FVD↓</td><td>FID↓</td><td>FVD↓ FID↓FVD↓</td></tr><tr><td>PVG</td><td>48.15 246.74</td><td>60.44</td><td>356.23</td><td>84.50 501.16</td></tr><tr><td>EmerNeRF</td><td>37.57 171.47</td><td>52.03</td><td>294.55</td><td>76.11 497.85</td></tr><tr><td>StreetGaussian OmniRe</td><td>32.12 153.45 31.48 152.01</td><td>43.24 43.31</td><td>256.91 254.52</td><td>67.44 4429.98 67.36 428.20</td></tr><tr><td>FreeVS</td><td>51.26 431.99</td><td>62.04</td><td>497.37</td><td>77.14 556.14</td></tr><tr><td>DiST-S</td><td></td><td>12.97</td><td>68.80</td><td></td></tr><tr><td>Ours</td><td>10.12 8.36</td><td>45.14 44.50 11.27</td><td>68.17</td><td>17.57 105.29 18.81 116.40</td></tr></table>

FRTable 3. Quantitive comparison between our method and other AMnovel view synthesis methods on nuScenes dataset.

3D Gaussian Representation Compared with other scene representations, the 3D Gaussian scene representation provides a better ability to encode environmental features, including both texture and geometric information, and can be directly aligned to the 3D space through language features. The results are shown in Tab. 2. Through ablation studies, we fully verify that the introduction of 3D Gaussian representations enhances the LLM’s 3D scene understanding capability, enabling it to better perceive the geometric and semantic information of the scene.

Hybrid Sampling Based on the observed redundancy in the 3D Gaussian scene representation, we design a hybrid sampling strategy to extract the most informative components of the environment. The results are shown in Tab. 2. By comparing random sampling, uniform+top-k sampling, and similarity-based sampling, we observe that when similaritybased sampling is applied between the text query and the 3D Gaussian tokens, the model can more effectively select task-relevant Gaussian tokens, leading to improved scene understanding performance.

<table><tr><td>Low-Level High-level</td><td></td><td>FID±1mFID±2m</td></tr><tr><td>X</td><td>√</td><td>= 1</td></tr><tr><td>√</td><td>X</td><td>10.12 45.14</td></tr><tr><td>√</td><td>√</td><td>8.36 44.5</td></tr></table>

Table 4. Ablation Study of dual-condition generation mechanism. “–” denotes failure under the setting.

High-level World Knowledge Compared with other world model and scene generation frameworks, we design a dualcondition guidance framework that controls scene generation at different levels. The ablation results for this part are shown in Tab. 4. It can be observed that our method performs better in long-sequence prediction and under large viewpoint variance, demonstrating the effectiveness of incorporating world knowledge.

## 5. Conclusion

This paper introduces a novel unified driving world model framework that supports both scene understanding and generation within a single architecture. We bridge the gap between these two tasks by proposing a new 3D Gaussian scene representation combined with a task-aware hybrid Gaussian sampling strategy, enabling effective worldquery injection into the LLM. Extensive experiments validate the effectiveness of the proposed framework, demonstrating significant improvements in both scene understanding and generation. We believe this work an important step toward unified driving world models with 3D Gaussian scene representations.

Acknowledgement This work was supported by National Key R&D Program of China (Grant No.2024YFB4708900). It was also supported in part by the Natural Science Foundation of China under Grant 62225309,U24A20278, 62361166632. We sincerely thank Yikang Ding from Kuaishou Kling AI for his kind help and support.

## References

[1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 5

[2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 5, 6

[3] Ruidong Chen, Yancheng Bai, Xuanpu Zhang, Jianhao Zeng, Lanjun Wang, Dan Song, Lei Sun, Xiangxiang Chu, and Anan Liu. Layer-wise instance binding for regional and occlusion control in text-to-image diffusion transformers. arXiv preprint arXiv:2603.05769, 2026. 2

[4] Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, and Li Zhang. Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering. arXiv preprint arXiv:2311.18561, 2023. 2, 6, 8

[5] Yi Chen, Tianchen Deng, Wentao Zhao, Xiaoning Wang, Wenqian Xi, Weidong Chen, and Jingchuan Wang. Sn-lidar: Semantic neural fields for novel space-time view lidar synthesis. arXiv preprint arXiv:2504.08361, 2025. 2

[6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024. 5

[7] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, Li Song, and Yue Wang. Omnire: Omni urban scene reconstruction. In The Thirteenth International Conference on Learning Representations, 2025. 6, 8

[8] Tianchen Deng, Yanbo Wang, Yejia Liu, Chenpeng Su, Jingchuan Wang, Hesheng Wang, Danwei Wang, and Weidong Chen. Prosgnerf: Progressive dynamic neural scene graph with frequency modulated auto-encoder in urban scenes. arXiv preprint arXiv:2312.09076, 2023. 2

[9] Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wentao Zhao, Jingchuan Wang, Danwei Wang, and Weidong Chen. Plgslam: Progressive neural scene represenation with local to global bundle adjustment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19657–19666, 2024. 2

[10] Tianchen Deng, Yue Pan, Shenghai Yuan, Dong Li, Chen Wang, Mingrui Li, Long Chen, Lihua Xie, Danwei Wang, Jingchuan Wang, Javier Civera, Hesheng Wang, and Weidong Chen. What is the best 3d scene representation for robotics? from geometric to foundation models. arXiv preprint arXiv:2512.03422, 2025. 3

[11] Tianchen Deng, Guole Shen, Xun Chen, Shenghai Yuan, Hongming Shen, Guohao Peng, Zhenyu Wu, Jingchuan Wang, Lihua Xie, Danwei Wang, Hesheng Wang, and Weidong Chen. Mcn-slam: Multi-agent collaborative neural slam with hybrid implicit neural scene representation. arXiv preprint arXiv:2506.18678, 2025. 2

[12] Tianchen Deng, Guole Shen, Chen Xun, Shenghai Yuan, Tongxin Jin, Hongming Shen, Yanbo Wang, Jingchuan Wang, Hesheng Wang, Danwei Wang, et al. Mne-slam: Multi-agent neural slam for mobile robots. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1485–1494, 2025. 2

[13] Tianchen Deng, Yanbo Wang, Hongle Xie, Hesheng Wang, Rui Guo, Jingchuan Wang, Danwei Wang, and Weidong Chen. Neslam: Neural implicit mapping and self-supervised feature tracking with depth completion and denoising. IEEE Transactions on Automation Science and Engineering, 22: 12309–12321, 2025. 2

[14] Tianchen Deng, Wenhua Wu, Junjie He, Yue Pan, Xirui Jiang, Shenghai Yuan, Danwei Wang, Hesheng Wang, and Weidong Chen. Vpgs-slam: Voxel-based progressive 3d gaussian slam in large-scale scenes. arXiv preprint arXiv:2505.18992, 2025. 2

[15] Yinan Deng, Jiahui Wang, Jingyu Zhao, Jianyu Dou, Yi Yang, and Yufeng Yue. Openobj: Open-vocabulary objectlevel neural radiance fields with fine-grained understanding. IEEE Robotics and Automation Letters, 10(1):652–659, 2024.

[16] Yinan Deng, Yufeng Yue, Jianyu Dou, Jingyu Zhao, Jiahui Wang, Yujie Tang, Yi Yang, and Mengyin Fu. Omnimap: A general mapping framework integrating optics, geometry, and semantics. IEEE Transactions on Robotics, 2025. 2

[17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 5

[18] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-toend autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24823– 24834, 2025. 2

[19] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601, 2023. 3, 6

[20] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li.

Vista: A generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems, 37:91560–91596, 2024. 2

[21] Ziren Gong, Xiaohan Li, Fabio Tosi, Jiawei Han, Stefano Mattoccia, Jianfei Cai, and Matteo Poggi. Ov3r: Openvocabulary semantic 3d reconstruction from rgb videos. arXiv preprint arXiv:2507.22052, 2025. 2

[22] Ziren Gong, Xiaohan Li, Fabio Tosi, Youmin Zhang, Stefano Mattoccia, Jun Wu, and Matteo Poggi. Dino-slam: Dinoinformed rgb-d slam for neural implicit and explicit representations. arXiv preprint arXiv:2507.19474, 2025.

[23] Ruocheng Gu, Sen Jia, Yule Ma, Jinqin Zhong, Jenq-Neng Hwang, and Lei Li. Mocount: Motion-based repetitive action counting. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9026–9034, 2025. 2

[24] Jiazhe Guo, Yikang Ding, Xiwu Chen, Shuo Chen, Bohan Li, Yingshuang Zou, Xiaoyang Lyu, Feiyang Tan, Xiaojuan Qi, Zhiheng Li, and Hao Zhao. Dist-4d: Disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), page 27231–27241, 2025. 6

[25] Jiazhe Guo, Yikang Ding, Xiwu Chen, Shuo Chen, Bohan Li, Yingshuang Zou, Xiaoyang Lyu, Feiyang Tan, Xiaojuan Qi, Zhiheng Li, et al. Dist-4d: Disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation. arXiv preprint arXiv:2503.15208, 2025. 2, 6

[26] Anna-Maria Halacheva, Jan-Nico Zaech, Xi Wang, Danda Pani Paudel, and Luc Van Gool. Gaussianvlm: Scene-centric 3d vision-language models using languagealigned gaussian splats for embodied reasoning and beyond. arXiv preprint arXiv:2507.00886, 2025. 3

[27] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023. 3

[28] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 2

[29] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 3

[30] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 2

[31] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3

[32] Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu,

Mingkai Jia, et al. 3d and 4d world modeling: A survey. arXiv preprint arXiv:2509.07996, 2025. 2

[33] Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11971–11981, 2025. 3

[34] Lei Li, Sen Jia, Jianhao Wang, Zhongyu Jiang, Feng Zhou, Ju Dai, Tianfang Zhang, Zongkai Wu, and Jenq-Neng Hwang. Human motion instruction tuning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17582–17591, 2025. 2

[35] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 5

[36] Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, and Xiang Bai. Pointmamba: A simple state space model for point cloud analysis. Advances in neural information processing systems, 37:32653–32677, 2024. 2

[37] Dingkang Liang, Tianrui Feng, Xin Zhou, Yumeng Zhang, Zhikang Zou, and Xiang Bai. Parameter-efficient fine-tuning in spectral domain for point cloud learning. IEEE transactions on pattern analysis and machine intelligence, 47(12): 10949–10966, 2025. 2

[38] Dingkang Liang, Dingyuan Zhang, Xin Zhou, Sifan Tu, Tianrui Feng, Xiaofan Li, Yumeng Zhang, Mingyang Du, Xiao Tan, and Xiang Bai. Unifuture: A 4d driving world model for future generation and perception. In IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2026. 2

[39] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 5

[40] Hongkai Lin, Dingkang Liang, Mingyang Du, Xin Zhou, and Xiang Bai. More than generation: Unifying generation and depth estimation via text-to-image diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2

[41] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024. 5

[42] Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. In European conference on computer vision, pages 531–548. Springer, 2022. 5

[43] Jiageng Mao, Boyi Li, Boris Ivanovic, Yuxiao Chen, Yan Wang, Yurong You, Chaowei Xiao, Danfei Xu, Marco Pavone, and Yue Wang. Dreamdrive: Generative 4d scene modeling from street view images. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 367–374. IEEE, 2025. 3

[44] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:

Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 3

[45] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2856–2865, 2021. 2

[46] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002. 5

[47] Chensheng Peng, Ido Sobol, Masayoshi Tomizuka, Kurt Keutzer, Chenfeng Xu, and Or Litany. A lesson in splats: Teacher-guided diffusion for 3d gaussian splats generation with 2d supervision. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 28707– 28717, 2025. 2

[48] Chensheng Peng, Chengwei Zhang, Yixiao Wang, Chenfeng Xu, Yichen Xie, Wenzhao Zheng, Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan. Desire-gs: 4d street gaussians for static-dynamic decomposition and surface reconstruction for urban driving scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6782–6791, 2025. 2

[49] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051–20060, 2024. 3

[50] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European conference on computer vision, pages 256–274. Springer, 2024. 3

[51] Qijian Tian, Xin Tan, Yuan Xie, and Lizhuang Ma. Drivingforward: Feed-forward 3d gaussian splatting for driving scene reconstruction from flexible surround-view input. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7374–7382, 2025. 2

[52] Sifan Tu, Xin Zhou, Dingkang Liang, Xingyu Jiang, Yumeng Zhang, Xiaofan Li, and Xiang Bai. The role of world models in shaping autonomous driving: A comprehensive survey. arXiv preprint arXiv:2502.10498, 2025. 2

[53] Haithem Turki, Jason Y Zhang, Francesco Ferroni, and Deva Ramanan. Suds: Scalable urban dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12375–12385, 2023. 2

[54] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015. 5

[55] Qitai Wang, Lue Fan, Yuqi Wang, Yuntao Chen, and Zhaoxiang Zhang. Freevs: Generative view synthesis on free driving trajectory. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. 6

[56] Yongxin Wang, Meng Cao, Haokun Lin, Mingfei Han, Liang Ma, Jin Jiang, Yuhao Cheng, and Xiaodan Liang. Eaco: En-

hancing alignment in multimodal llms via critical observation. arXiv preprint arXiv:2412.04903, 2024. 2

[57] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024. 2

[58] Yanbo Wang, Zipeng Fang, Lei Zhao, and Weidong Chen. Learning to tune like an expert: Interpretable and sceneaware navigation via mllm reasoning and cvae-based adaptation. arXiv preprint arXiv:2507.11001, 2025. 3

[59] Yuhui Wang, Changjiang Li, Guangke Chen, Jiacheng Liang, and Ting Wang. Reasoning or retrieval? a study of answer attribution on large reasoning models, 2025. 2

[60] Yongxin Wang, Zhicheng Yang, Meng Cao, Mingfei Han, Haokun Lin, Yingying Zhu, Xiaojun Chang, and Xiaodan Liang. Care what fails: Contrastive anchored-reflection for verifiable multimodal. arXiv preprint arXiv:2512.19554, 2025.

[61] Yuhui Wang, Rongyi Zhu, and Ting Wang. Self-destructive language model, 2025. 2

[62] Yue Wen, Liang Song, Yijia Liu, Siting Zhu, Yanzi Miao, Lijun Han, and Hesheng Wang. Freedriverf: Monocular rgb dynamic nerf without poses for autonomous driving via point-level dynamic-static decoupling. arXiv preprint arXiv:2505.09406, 2025. 2

[63] Yichen Xie, Chensheng Peng, Mazen Abdelfattah, Yihan Hu, Jiezhi Yang, Eric Higgins, Ryan Brigden, Masayoshi Tomizuka, and Wei Zhan. Raynova: Scale-temporal autoregressive world modeling in ray space. arXiv e-prints, pages arXiv–2602, 2026. 2

[64] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2024. 5

[65] Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, and Xiang Bai. Cape: Camera view position embedding for multi-view 3d object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21570–21579, 2023. 5

[66] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 2024. 3

[67] Tianyi Yan, Dongming Wu, Wencheng Han, Junpeng Jiang, Xia Zhou, Kun Zhan, Cheng-zhong Xu, and Jianbing Shen. Drivingsphere: Building a high-fidelity 4d world for closedloop simulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27531–27541, 2025. 2

[68] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban

scenes with gaussian splatting. In European Conference on Computer Vision, pages 156–173. Springer, 2024. 2, 6, 8

[69] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2, 4

[70] Changzhi Yang, Huihui Pan, Jue Wang, and Yuanduo Hong. Trajdiff: Trajectory prediction with diffusion probabilistic models. IEEE Transactions on Image Processing, pages 1– 14, 2025. 2

[71] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, et al. Emernerf: Emergent spatial-temporal scene decomposition via self-supervision. arXiv preprint arXiv:2311.02077, 2023. 2, 6

[72] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Apoorva Sharma, Maximilian Igl, Peter Karkus, Danfei Xu, et al. Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes. arXiv preprint arXiv:2501.00602, 2024. 2

[73] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023. 8

[74] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14673–14684, 2024. 2

[75] Pingrui Zhang, Yifei Su, Pengyuan Wu, Dong An, Li Zhang, Zhigang Wang, Dong Wang, Yan Ding, Bin Zhao, and Xuelong Li. Cross from left to right brain: Adaptive text dreamer for vision-and-language navigation. arXiv preprint arXiv:2505.20897, 2025. 3

[76] Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10412–10420, 2025. 2, 3

[77] Zongchuang Zhao, Haoyu Fu, Dingkang Liang, Xin Zhou, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Extending large vision-language model for diverse interactive tasks in autonomous driving. arXiv preprint arXiv:2505.08725, 2025. 3, 5, 6

[78] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024. 3

[79] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21634–21643, 2024. 2

[80] Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and

Xiang Bai. Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. arXiv preprint arXiv:2501.14729, 2025. 2, 3

[81] Chunzheng Zhu, Yangfang Lin, Shen Chen, Yijun Wang, and Jianxin Lin. Medeyes: Learning dynamic visual focus for medical progressive diagnosis. arXiv preprint arXiv:2511.22018, 2025. 2

[82] Chunzheng Zhu, Yangfang Lin, Jialin Shao, Jianxin Lin, and Yijun Wang. Pathology-aware prototype evolution via llm-driven semantic disambiguation for multicenter diabetic retinopathy diagnosis. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9196–9205, 2025. 3

[83] Siting Zhu, Guangming Wang, Hermann Blum, Zhong Wang, Ganlin Zhang, Daniel Cremers, Marc Pollefeys, and Hesheng Wang. Sni-slam++: Tightly-coupled semantic neural implicit slam. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2

[84] Yingshuang Zou, Yikang Ding, Chuanrui Zhang, Jiazhe Guo, Bohan Li, Xiaoyang Lyu, Feiyang Tan, Xiaojuan Qi, and Haoqian Wang. Mudg: Taming multi-modal diffusion with gaussian splatting for urban scene reconstruction. arXiv preprint arXiv:2503.10604, 2025. 2