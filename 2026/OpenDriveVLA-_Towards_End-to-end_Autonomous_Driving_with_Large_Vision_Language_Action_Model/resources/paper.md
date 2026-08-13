# OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model

Xingcheng Zhou<sup>1†</sup>, Xuyuan Han<sup>1</sup>, Feng Yang<sup>1</sup>, Yunpu Ma<sup>2</sup>, Volker Tresp<sup>2</sup>, Alois Knoll<sup>1</sup> <sup>1</sup>Technical University of Munich <sup>2</sup>Ludwig Maximilian University of Munich 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/f48c6138-d017-460b-851a-28755b186b96/d27b9fd0ee0dbb44527483b4eca49eee345bc36216f1359bf691b003b33d19d5.jpg)



Fig. 1: OpenDriveVLA leverages open-source pre-trained large vision-language-action models to generate driving actions conditioned on 3D environmental perception, ego-vehicle states, and driver commands. It achieves strong performance in both open-loop planning and driving-related question answering, demonstrating its proficiency in scene understanding and driving action tuning.


Abstract—We present OpenDriveVLA, a Vision-Language Action (VLA) model designed for end-toend autonomous driving, built upon open-source large language models. OpenDriveVLA generates spatiallygrounded driving actions by leveraging multimodal inputs, including both 2D and 3D instance-aware visual representations, ego vehicle states, and language commands. To bridge the modality gap between driving visual representations and language embeddings, we introduce a hierarchical vision-language alignment process, projecting both 2D and 3D structured visual tokens into a unified semantic space. Furthermore, we incorporate structured agent–environment–ego interaction modeling into the autoregressive decoding process, enabling the model to capture fine-grained spatial dependencies and behavior-aware dynamics critical for reliable trajectory planning. Extensive experiments on the nuScenes dataset demonstrate that OpenDriveVLA achieves state-of-the-art results across open-loop trajectory planning and drivingrelated question-answering tasks. Qualitative analyses further illustrate its superior capability to follow high-level driving commands and generate trajectories under challenging scenarios, highlighting its potential for next-generation end-to-end autonomous driving. 

## I. INTRODUCTION

End-to-end learning frameworks have emerged as a promising paradigm in autonomous driving, enabling perception, prediction, and planning to be jointly optimized within a unified neural network [1]. They learn policies directly from sensor inputs and generalize well across varied scenarios. Despite notable progress, existing approaches still face critical challenges, including limited longtail generalization, poor complex semantics understanding, and rigid task reasoning [2]. Meanwhile, large language models (LLMs) and vision-language models (VLMs) exhibit strong in-context reasoning, commonsense understanding, and zero-shot generalization abilities. These capabilities are promising for driving, where robust scene understanding is crucial [3], [4]. However, directly leveraging existing VLMs for autonomous driving poses fundamental challenges. Firstly, current VLMs are predominantly optimized for static, 2D image-language tasks, leading to poor spatial reasoning performance in dynamic 3D driving environments [5]. Besides, instance-agnostic VLMs [6] are prone to hallucinations, often yielding incorrect yet overconfi dent outputs, posing safety risks in autonomous driving. Motivated by these limitations, our work answers a central question: How can we harness the emergent capabilities of large VLMs to produce safe spatially-grounded driving actions in dynamic 3D environments, while balancing inference speed and planning effectiveness? 

To enhance spatial-awareness and safety in LLMbased vision-language action model, we introduce two key designs. First, we structure the driving environment using instance-aware, hierarchical 2D and 3D visual representations to reduce the risk of instance hallucinations. Second, we incorporate agent–environment–ego interaction modeling, which is originally explicitly modeled in traditional end-to-end driving systems, as an auxiliary objective into the autoregressive LLM training pipeline. It enables the model to internalize physical feasibility and dynamic multi-agent interactions, improving robustness in safety-critical scenarios. 

Built upon open-source large language models, OpenDriveVLA tightly integrates spatially grounded multimodal reasoning and driving trajectory generation within a unified autoregressive framework. Unlike prior VLM-based methods, OpenDriveVLA leverages structured 2D and 3D instance-aware representations, ego vehicle states, and high-level commands to directly produce reliable driving actions. Extensive experiments on nuScenes benchmark demonstrate that Open-DriveVLA achieves state-of-the-art performance in both open-loop planning and vision-language reasoning tasks. Our key contributions are: 

We present OpenDriveVLA, a 3D visionlanguage action model for end-to-end autonomous driving that generates reliable driving trajectories by integrating hierarchical visual input, ego state, and high-level language commands. 

We develop a multi-stage training strategy that aligns structured 2D and 3D visual features into a unified semantic space, enabling naive VLMs to generate spatially-grounded actions in complex driving scenarios. 

We introduce implicit agent–environment–ego interaction modeling into autoregressive LLMbased VLA training as an auxiliary task, enabling the model to learn behaviorally grounded and safety-aware driving actions. 

## II. RELATED WORK

## A. End-to-End Autonomous Driving

Autonomous driving (AD) evolves through two distinct stages. Traditional approaches rely on a modular design, decomposing the system into perception [7], prediction [8], and planning [9] components. While this structure ensures interpretability and allows for independent optimization, they suffer from cascading errors between stages and are not globally optimized for the final planning objective. In contrast, end-to-end autonomous driving frameworks [10] address this by jointly optimizing perception, prediction, and planning within a unified neural network. These models learn driving policies directly from raw sensor inputs, which improves the model’s adaptability to diverse driving conditions. More recent approaches introduce diffusion models [11] and unified scene representations [12] to further enhance the effectiveness and robostness. However, existing end-to-end methods still face semantic reasoning bottlenecks, as they struggle to fully comprehend high-level scene semantics, infer complex agent interactions, and adapt to dynamic task requirements. Moreover, their decision-making processes remain opaque, making it difficult to diagnose failure cases, especially in long-tail or unseen scenarios. 

## B. Large Vision Language Models

Large Language Models demonstrated strong emergent capabilities in in-context learning, instruction following, and reasoning [13], [14]. By training on vast amounts of Internet-scale data, these models acquire extensive world knowledge and exhibit strong adaptability across diverse tasks. Their success has also driven the rise of large VLMs, which extend these capabilities into cross-modal reasoning by integrating vision encoders with language models. State-of-the-art VLMs such as GPT-4V [15], LLaVA [16], and Qwen-VL [17] demonstrate strong visual understanding and multimodal reasoning in open-domain tasks. However, these models are primarily trained on static 2D images or videos and exhibit limited spatial reasoning in dynamic 3D driving environments. Moreover, VLMs are prone to hallucinations and generally over-confident but incorrect descriptions, which pose serious risks in safety-critical planning scenarios. Recently, Vision-Language Action models have emerged to directly predict actions from visual inputs, demonstrating strong performance in robotic manipulation tasks [18]. Currently, the application of such languageconditioned end-to-end action generation in autonomous driving remains underexplored. Yet, these methods are mostly limited to static setups and lack driving-specific 3D spatial design. 

## C. Vision Language Models in Autonomous Driving

VLMs have been applied to various autonomous driving tasks, including perception, scene description, synthetic data generation, and high-level decision-making [1]. These efforts aim to enhance interpretability, data efficiency, and instructionfollowing capabilities in driving models. We categorize recent works into 4 paradigms, as illustrated in Figure 2. One line of research in Fig.2 (a) integrates language heads, such as captioning or question-answering modules, into driving models to enhance the interpretability [19]. The second category in Fig.2 (b) employs vision language models to generate high-level driving instructions, such as directional commands or abstract maneuvers, which are subsequently interpreted by separate planning modules into low-level controls [20]–[22]. It’s also usually formed as a fast-slow dual system. This design allows VLMs to make independent semantic reasoning, but retains a separate module for endto-end driving planning, making joint optimization challenging. The third line in Fig.2 (c) applies native VLMs with 2D visual tokens to produce driving actions, and optionally scene captions or QA responses [23], [24]. These methods [25]–[27] process 2D images without explicit modeling of the instance, 3D spatial layout, and inter-agent interactions in the driving scene. It limits their spatial reasoning ability and understanding of agent dynamics in complex traffic environments. Recent studies [28] further indicate that such instance-agnostic approaches are more prone to hallucinate, often producing overconfident or semantically inconsistent text. In this work, we investigate how to extend 2D VLMs by explicitly modeling 3D instance-aware and spatial-aware scene representations into an endto-end autonomous driving framework, as shown in Fig.2(d). Notably, we focus on fully differentiable end-to-end models in this work, while LLM-based agentic driving systems, such as [29], [30], fall outside the scope of our study. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/f48c6138-d017-460b-851a-28755b186b96/6a15ce3c5bd4c6b759e2a05cb4ae7d0909e72a608786839b2617bd66c64a9276.jpg)



(b) VLM as high-level driving decision-maker.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/f48c6138-d017-460b-851a-28755b186b96/dd06969280afce08fcf7fce2b5c014395d9fd4fb34af1f9ffcd628d44fc52ed3.jpg)



(c) Native 2D VLM for end-to-end driving.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/f48c6138-d017-460b-851a-28755b186b96/d710d5741d73bda59b2aff7c635b9c885d5976d3495ca1591af3e5a92ffb5cfa.jpg)



(d) 3D spatial-aware driving VLA (ours).



Fig. 2: Taxonomy of vision-language model applications in endto-end autonomous driving.


## III. OPENDRIVEVLA

The overall architecture of OpenDriveVLA is shown in Figure 1, with its multi-stage training process further detailed in Figure 3. OpenDriveVLA uses a pre-trained vision encoder to extract tokenized environmental representations from multiview images. These visual tokens are then aligned into the textual domain through cross-modal learning. After alignment, it undergoes driving instruction tuning, followed by agent-ego-environment interaction modeling. Finally, OpenDriveVLA is trained end-to-end to predict the ego vehicle’s future trajectory, guided by the aligned visual-language tokens and driving instructions. 

## A. 3D Visual Environmental Perception

Recent VLM-based autonomous driving methods typically rely on pretrained 2D visual encoders [5], where visual token selection and attention are indirectly guided through language supervision. While effective in open-domain vision-language applications, this design lacks explicit 3D spatial grounding and structured instance-level attention, which can lead to severe hallucinations in safetycritical driving scenarios [31]. To mitigate this, OpenDriveVLA adopts a visual-centric query module, where the model first learns to focus on drivingrelevant objects and map tokens through 3D vision tasks, ensuring reliable visual token proposal. 

Specifically, given a set of multi-view images $I = \stackrel { \cdot } { \{ I ^ { i } \} } _ { i = 1 } ^ { N }$ , the visual module first extracts multiscale 2D features from each image using a shared 2D backbone, denoted as $f _ { 2 D }$ . These 2D features are then aggregated across views and lifted into BEV space, producing the BEV feature $f _ { b e v }$ . To obtain structured environmental representations, we adopt three visual query modules: Global Scene Sampler $\mathcal { Q } _ { \mathrm { s c e n e } } ,$ Agent QueryTransformer $\mathcal { Q } _ { \mathrm { a g e n t } } ,$ and Map QueryTransformer $\mathcal { Q } _ { \mathrm { m a p } }$ . Each module extracts tokens focusing on a specific semantic aspect of the driving environment. Global Scene Sampler encodes the surrounding driving scene context from multi-view 2D features, producing the scene token $v _ { s c e n e } = \mathcal { Q } _ { \mathrm { s c e n e } } ( f _ { 2 D } )$ . Agent Query-Transformer detects and tracks dynamic agents within the scene, extracting agent-centric tokens $\{ v _ { a g e n t } ^ { i } \} _ { i = 1 } ^ { N _ { a } } \ = \ Q _ { \mathrm { a g e n t } } ( f _ { b e v } )$ , where $N _ { a }$ denotes the number of detected agents. In parallel, Map QueryTransformer extracts static structural information, such as lane boundaries and drivable areas, forming the map token $v _ { m a p } = \mathcal { Q } _ { \mathrm { m a p } } ( f _ { b e v } )$ Through vision-centric perception tasks, including 3D detection, tracking, and segmentation, the visual encoder produces structured environmental tokens that capture both dynamic agent behaviors and static map structures in a spatially grounded manner. The output tokens, denoted as ${ \bf V } _ { e n v } \ =$ $\{ v _ { s c e n e } , v _ { a g e n t } , v _ { m a p } \}$ , serve as visual environment representation of the subsequent stages. 

## B. Stage 1 - Hierarchical Vision-Language Alignment

To bridge the modality gap between the extracted visual tokens and the word embedding space of a pre-trained LLM, we adopt a hierarchical visionlanguage feature alignment strategy. Given the visual tokens extracted from the 3D visual perception module, we introduce three token-specific projectors $\{ \Phi _ { \mathrm { s c e n e } } , \Phi _ { \mathrm { a g e n t } } , \Phi _ { \mathrm { m a p } } \}$ . During training, each active agent query from the 3D detection and tracking task denoted as $v _ { a g e n t } ^ { i } ,$ , is also matched to its corresponding ground-truth caption $\mathbf { X } _ { a g e n t } ^ { i } .$ These captions provide detailed descriptions, including 2D appearance descriptions and 3D spatial positions. For scene and map tokens, which encode holistic spatial context and static structural properties, a sample-wise alignment is applied, where each token is matched to a scene-level caption ${ \bf X } _ { s c e n e }$ or $\mathbf { X } _ { m a p } .$ The scene token $v _ { s c e n e }$ captures the global 2D environmental context, while the map token $v _ { m a p }$ encodes structural elements such as lane topology, road boundaries, and drivable areas. Each of these tokens is aligned to its corresponding caption, denoted as ${ \bf X } _ { s c e n e }$ and $\mathbf { X } _ { m a p } .$ During this stage, both the visual encoder and LLM remain frozen to preserve pretrained semantics, with only the token-specific projectors being trainable. The forward alignment step is formulated as follows: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/f48c6138-d017-460b-851a-28755b186b96/78f7296e6e669755c8d993577c60557e96bed3c9e50ab677664c52da8330298c.jpg)



Fig. 3: Illustration of main training stages on OpenDriveVLA. Stage 1: Hierarchical Feature Alignment. Stage 2: Driving Instruction Tuning. Stage 2.5: Agent-Env-Ego Interaction Modeling. Stage 3: Trajectory Planning Tuning.


$$
\hat {\mathbf {X}} _ {k} = \mathrm{LLM} \left(\Phi_ {k} (v _ {k})\right), \quad k \in \{\mathrm{scene}, \mathrm{map} \}\tag{1}
$$

$$
\hat {\mathbf {X}} _ {a g e n t} ^ {i} = \operatorname{LLM} \left(\Phi_ {\text { agent }} (v _ {a g e n t} ^ {i})\right), \quad i = 1, \dots , N _ {a}\tag{2}
$$

## C. Stage 2 - Driving Instruction Tuning

We distill high-level driving knowledge into the model via supervised instruction tuning, enabling it to internalize semantic reasoning patterns during training. This avoids costly chain-of-thought (CoT) reasoning at inference time and balances planning efficacy with runtime efficiency. 

During the tuning process, driving knowledge from the language domain is injected into the model using a curated driving instruction QA dataset. The dataset covers a wide range of driving-related reasoning, including perception understanding, motion prediction, attention allocation, action reasoning, and high-level decisionmaking. By training on this diverse set of driving queries, OpenDriveVLA learns to contextualize the driving scene, follow commands, and generate semantically and behaviorally grounded planning decisions. We formulate the tuning data as instruction-response pairs $\{ { \bf X } _ { i n p u t } , { \bf X } _ { a n s w e r } \}$ where $\begin{array} { r c l } { \mathbf { X } _ { i n p u t } } & { = } & { ( \mathbf { V } _ { e n v } , \mathbf { S } _ { e g o } , \dot { \mathbf { X } } _ { q u e r y } ) } \end{array}$ Here, ${ \bf X } _ { q u e r y }$ denotes the driving-related question, and $\mathbf { S } _ { e g o }$ encodes the textual ego vehicle state. Given this multimodal input, the LLM autoregressively learns to generate the target response. During instruction tuning, the visual encoder remains frozen while the token-specific projectors and the LLM are set to be trainable. The instruction prediction process is as: 

$$
\hat {\mathbf {X}} _ {a n s w e r} = \mathrm{LLM} \left(\mathbf {V} _ {e n v}, \mathbf {S} _ {e g o}, \mathbf {X} _ {q u e r y}\right)\tag{3}
$$

## D. Stage 2.5 - Agent Environment Ego Interaction

Reliable trajectory planning in autonomous driving necessitates a spatially grounded 3D representation of the environment. Beyond perception, it must also understand dynamic interactions between the ego vehicle and surrounding agents. Effective interaction modeling is essential to ensure that planned trajectories are both feasible and collisionfree under real-world driving constraints. However, existing pre-trained LLMs lack an inherent inductive bias for spatial reasoning in 3D driving scenes, as they are predominantly trained on 2D visionlanguage and text-based datasets. We introduce a conditional agent trajectory forecasting task as an auxiliary objective, encouraging the model to learn spatially grounded interaction priors. During this stage, OpenDriveVLA captures the underlying structure of multi-agent dynamics, enhancing its capability for scene-aware trajectory generation and improving decision-making in complex traffic scenarios. 

Given scene and map tokens, as well as the ego vehicle state ${ \bf { S } } _ { e g o } ,$ the LLM predicts the future motion of each detected agent based on its projected visual embedding $\Phi _ { \mathrm { a g e n t } } ( v _ { a g e n t } ^ { i } )$ . The future motion of agent $a _ { i }$ is represented as a sequence of waypoints ${ \mathcal W } _ { a } ^ { i }$ . The predicted trajectory is conditioned on the scene context, map structure, and ego vehicle state, enabling OpenDriveVLA to infer interaction-aware and spatially grounded motion sequences. The learning objective for the i-th agent is formulated as: 

$$
\max \prod_ {t = 1} ^ {T} p \left(w _ {t} ^ {i} \mid w _ {1: t - 1} ^ {i}, \mathbf {V} _ {\text { env }}, \mathbf {S} _ {\text { ego }}, \Phi_ {\text { agent }} (v _ {\text { agent }} ^ {i})\right) \tag {4}\tag{4}
$$

This provides OpenDriveVLA with essential spatial priors, enabling it to bridge the gap between high-level semantic reasoning and physically grounded motion planning. 

## E. Stage 3 - End-to-end Trajectory Planning Tuning

In this stage, OpenDriveVLA predicts ego trajectories as discrete waypoint sequences within a short horizon, denoted as ${ \mathcal W } _ { e g o } ~ = ~ \{ w _ { 1 } , w _ { 2 } , \ldots , w _ { T } \}$ Each waypoint $w _ { t }$ represents the 2D coordinates $( x _ { t } , y _ { t } )$ of the ego vehicle at time step t. The waypoints are tokenized into a sequence of discrete textual tokens for autoregressive generation in the $\mathbf { L L M } \colon \mathcal { T } t r a j = \mathrm { T o k e n i z e r } ( \mathcal { W } _ { e g o } )$ . The generation process is then cast as a causal sequence prediction task, where each token is predicted in a causal manner, conditioned on the visual perception tokens $\mathbf { V } _ { e n v } ,$ the ego state ${ \bf S } _ { e g o } ,$ and the driving command ${ \bf X } _ { d r i }$ 

$$
\hat {\mathcal {T}} _ {t r a j} = \operatorname{argmax} _ {\mathbf {T} _ {t r a j}} \prod_ {t = 1} ^ {T} p \left(w _ {t} \mid w _ {1: t - 1}, \mathbf {V} _ {e n v}, \mathbf {S} _ {e g o}, \mathbf {X} _ {d r i}\right)
$$

The entire pipeline, including the 3D visual encoder, cross-modality projectors, and LLM, is 

(5) 

jointly optimized end-to-end during training, with the 2D encoder kept frozen. At inference, the model autoregressively generates the tokenized trajectory $\hat { \mathcal { T } } _ { t r a j }$ , which is then decoded back into numerical waypoints: 

$$
\hat {\mathcal {W}} _ {e g o} = \mathrm{Decoder} (\hat {\mathcal {T}} _ {t r a j})\tag{6}
$$

## IV. EXPERIMENTS

## A. Training Datasets

We curate the training data of OpenDriveVLA based on its distinct training phases, drawing from: TOD3Cap [43], nuCaption [44], nuScenesQA [45], nuX [19], and GPT-Driver [38]. We conduct experiments on nuScenes [46], following standard data split into training and validation sets. Open-DriveVLA is trained using the training set paired with corresponding QA captions, while the validation set is exclusively used for performance evaluation to ensure fair comparisons with prior works. The details of training data can be found in supplementary materials. 

Hierarchical Vision-Language Alignment. For agent-level caption, we post-process data from [43], which provides the 2D visual description of individual objects. To further enhance spatial grounding, each object caption is augmented with its corresponding BEV coordinates, enabling the model to associate object attributes with precise spatial locations. For scene tokens, we process multi-view scene descriptions from [44], merging them into unified summaries that describe the driving environment across all camera views. For map tokens, structured language descriptions are derived from ground-truth annotations, translating map elements such as lane dividers, crosswalks, and road boundaries into descriptive text. 

Driving Instruction Tuning. We adopt multiple instruction-oriented datasets derived from nuScenes to inject driving-specific knowledge into Open-DriveVLA. We unify several datasets into a standardized instruction-based QA format, including driving-related question-answer pairs collected from nuCaption [44], nuScenesQA [45], and nuX [19] dataset. Each QA pair is conditioned on structured environmental visual tokens and the ego vehicle state, ensuring consistency across different data sources. This multimodal instruction tuning process allows OpenDriveVLA to effectively ground language understanding into both environmental perception and scene understanding, bridging perception, reasoning, and action within the language space. 

<table><tr><td rowspan="3">Method</td><td colspan="8">ST-P3 metrics</td><td colspan="8">UniAD metrics</td><td rowspan="3">LLM</td><td rowspan="3">Input</td></tr><tr><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td colspan="19">None-Autoregressive Methods</td></tr><tr><td>ST-P3 [32]</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Visual</td></tr><tr><td>VAD [33]</td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.07</td><td>0.10</td><td>0.24</td><td>0.14</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Visual</td></tr><tr><td>Ego-MLP [34]</td><td>0.46</td><td>0.76</td><td>1.12</td><td>0.78</td><td>0.21</td><td>0.35</td><td>0.58</td><td>0.38</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Ego</td></tr><tr><td>UniAD [10]</td><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td><td>-</td><td>Visual</td></tr><tr><td>InsightDrive [35]</td><td>0.23</td><td>0.41</td><td>0.68</td><td>0.44</td><td>0.09</td><td>0.10</td><td>0.27</td><td>0.15</td><td>0.30</td><td>0.72</td><td>1.41</td><td>0.81</td><td>0.08</td><td>0.15</td><td>0.84</td><td>0.36</td><td>-</td><td>Visual</td></tr><tr><td>FF [9]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.55</td><td>1.20</td><td>2.54</td><td>1.43</td><td>0.06</td><td>0.17</td><td>1.07</td><td>0.43</td><td>-</td><td>LiDAR</td></tr><tr><td>EO [36]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.67</td><td>1.36</td><td>2.78</td><td>1.60</td><td>0.04</td><td>0.09</td><td>0.88</td><td>0.33</td><td>-</td><td>LiDAR</td></tr><tr><td colspan="19">Autoregressive Methods</td></tr><tr><td>GPVL [37]</td><td>0.21</td><td>0.39</td><td>0.69</td><td>0.43</td><td>0.07</td><td>0.09</td><td>0.27</td><td>0.14</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>BERT</td><td>Textual</td></tr><tr><td>DriveVLM [21]</td><td>0.18</td><td>0.34</td><td>0.68</td><td>0.40</td><td>0.10</td><td>0.22</td><td>0.45</td><td>0.27</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Qwen-VL-7B</td><td>Visual</td></tr><tr><td>GPT-Driver [38]</td><td>0.20</td><td>0.40</td><td>0.70</td><td>0.44</td><td>0.04</td><td>0.12</td><td>0.36</td><td>0.17</td><td>0.27</td><td>0.74</td><td>1.52</td><td>0.84</td><td>0.07</td><td>0.15</td><td>1.10</td><td>0.44</td><td>GPT-3.5</td><td>Textual</td></tr><tr><td>RDA-Driver [39]</td><td>0.17</td><td>0.37</td><td>0.69</td><td>0.40</td><td>0.01</td><td>0.05</td><td>0.26</td><td>0.10</td><td>0.23</td><td>0.73</td><td>1.54</td><td>0.80</td><td>0.00</td><td>0.13</td><td>0.83</td><td>0.32</td><td>LLaVa-7B</td><td>Visual</td></tr><tr><td>OminiDrive [29]</td><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>LLaVA-7B</td><td>Visual</td></tr><tr><td>EMMA [40]</td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Gemini</td><td>Visual</td></tr><tr><td>OpenEMMA [41]</td><td>1.45</td><td>3.21</td><td>3.76</td><td>2.81</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Qwen-VL-7B</td><td>Visual</td></tr><tr><td>DME-Driver [42]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.45</td><td>0.91</td><td>1.58</td><td>0.98</td><td>0.05</td><td>0.28</td><td>0.55</td><td>0.29</td><td>LLaVa-7B</td><td>Visual</td></tr><tr><td>OpenDriveVLA-0.5B (Ours)</td><td>0.15</td><td>0.32</td><td>0.57</td><td>0.35</td><td>0.01</td><td>0.06</td><td>0.20</td><td>0.09</td><td>0.21</td><td>0.60</td><td>1.22</td><td>0.68</td><td>0.00</td><td>0.15</td><td>0.63</td><td>0.26</td><td>Qwen2.5-0.5B</td><td>Visual</td></tr><tr><td>OpenDriveVLA-3B (Ours)</td><td>0.14</td><td>0.30</td><td>0.55</td><td>0.33</td><td>0.02</td><td>0.07</td><td>0.22</td><td>0.10</td><td>0.19</td><td>0.58</td><td>1.24</td><td>0.67</td><td>0.02</td><td>0.18</td><td>0.70</td><td>0.30</td><td>Qwen2.5-3B</td><td>Visual</td></tr><tr><td>OpenDriveVLA-7B (Ours)</td><td>0.15</td><td>0.31</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.08</td><td>0.21</td><td>0.10</td><td>0.20</td><td>0.58</td><td>1.21</td><td>0.66</td><td>0.00</td><td>0.22</td><td>0.55</td><td>0.25</td><td>Qwen2.5-7B</td><td>Visual</td></tr></table>


TABLE I: Open-Loop planning performance comparison of different driving models, including both autoregressive methods and non-autoregressive methods. OpenDriveVLA shows powerful planning ability and achieves best-in-class results among open-source models, even with the 0.5B version. We refer to the result summary from [35], [37]–[39].


Motion Forecasting and Trajectory Prediction. We formulate both agent motion forecasting and ego trajectory planning in the ego system, where the model directly predicts future displacements within each entity’s local coordinate frame relative to the ego vehicle for planning and relative to each agent for forecasting. This formulation captures motion dynamics in a spatially consistent manner across all entities. Following [38], the ego vehicle state is encoded as textual input to ensure ego awareness throughout the training process. Both tasks predict 3-second future trajectories, sampled at 0.5-second intervals, resulting in 6 waypoints per trajectory. 

## B. Evaluations

We evaluate OpenDriveVLA on the open-loop planning task of nuScenes benchmark, where the model is reported under both ST-P3 [32] and 

UniAD [10] settings. The evaluation metrics include L2 displacement errors at 1, 2, and 3 seconds, along with the average collision rate over the prediction horizon. To further assess the scene understanding ability of OpenDriveVLA, we report its QA prediction performance on three driving visual question answering (VQA) datasets directly after the driving instruction tuning stage, i.e., [44], nuScenesQA [45], and nuX [19]. The VQA evaluation results adopt standard NLG metrics, including BLEU, ME-TEOR, CIDEr, BERT-Score, etc. 

## C. Implementation Details

The 3D visual perception module in Open-DriveVLA follows the vision-centric design from [10], using a ResNet-101 backbone for 2D feature extraction. The perception backbone is pre-trained via multi-task learning on 3D object detection, object tracking, and map segmentation. The resulting BEV feature map has a spatial resolution of 200 200. To construct a unified scene representation, the global SceneSampler applies 2D adaptive pooling to each camera view, subsequently concatenating the pooled multi-view features into a global scene token. Agent and map tokens are extracted from the final layer of their respective QueryTransformer modules. Each token type is then mapped into the language space using a separate two-layer MLP with GeLU activation. We adopt Qwen 2.5-Instruct [14] as the pre-trained LLM, which undergoes full parameter tuning during training. Training is performed on 4 NVIDIA H100 GPUs with a batch size of 1, completed in approximately two days. We freeze the 2D backbone during stage 3. During inference, we set the decoding temperature to 0 to ensure deterministic trajectory generation. See supplementary material for detailed training configurations. 

<table><tr><td rowspan="2">Method</td><td colspan="5">nu-Caption</td><td colspan="8">nuScenes-QA</td></tr><tr><td>BL-1</td><td>BL-2</td><td>BL-3</td><td>BL-4</td><td>BERT-S</td><td>Ext</td><td>Cnt</td><td>Obj</td><td>Sts</td><td>Cmp</td><td>H0</td><td>H1</td><td>Acc</td></tr><tr><td>Mini-GPT4 [47]</td><td>15.0</td><td>6.8</td><td>3.7</td><td>2.6</td><td>84.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Instruct-BLIP [48]</td><td>18.7</td><td>13.4</td><td>7.4</td><td>5.2</td><td>85.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLaMA-AdapV2 [49]</td><td>30.2</td><td>17.3</td><td>10.4</td><td>7.5</td><td>86.5</td><td>19.3</td><td>2.7</td><td>7.6</td><td>10.8</td><td>1.6</td><td>15.1</td><td>4.8</td><td>9.6</td></tr><tr><td>LLaVA1.5 [50]</td><td>20.0</td><td>12.1</td><td>8.6</td><td>5.4</td><td>85.0</td><td>45.8</td><td>7.7</td><td>7.8</td><td>9.0</td><td>52.1</td><td>25.7</td><td>41.5</td><td>26.2</td></tr><tr><td>LiDAR-LLM [44]</td><td>41.0</td><td>30.0</td><td>23.4</td><td>19.3</td><td>91.3</td><td>74.5</td><td>15.0</td><td>37.8</td><td>45.9</td><td>57.8</td><td>-</td><td>-</td><td>48.6</td></tr><tr><td>BEVDet+BUTD [45]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>83.7</td><td>20.9</td><td>48.8</td><td>52.0</td><td>67.7</td><td>-</td><td>-</td><td>57.0</td></tr><tr><td>OpenDriveVLA-0.5B (Ours)</td><td>47.2</td><td>35.8</td><td>29.4</td><td>25.2</td><td>91.9</td><td>83.9</td><td>22.0</td><td>50.2</td><td>57.0</td><td>68.4</td><td>62.3</td><td>56.5</td><td>58.4</td></tr><tr><td>OpenDriveVLA-3B (Ours)</td><td>48.3</td><td>36.9</td><td>30.3</td><td>26.1</td><td>92.0</td><td>84.0</td><td>22.3</td><td>50.3</td><td>56.9</td><td>68.5</td><td>62.6</td><td>56.5</td><td>58.5</td></tr><tr><td>OpenDriveVLA-7B (Ours)</td><td>49.6</td><td>38.3</td><td>31.9</td><td>27.6</td><td>92.2</td><td>84.2</td><td>22.7</td><td>49.6</td><td>54.5</td><td>68.8</td><td>62.4</td><td>56.1</td><td>58.2</td></tr></table>


TABLE II: Performance on nu-Caption [44] and nuScenes-QA [45]. BL-1/2/3/4: BLEU scores. QA metrics report accuracy on five question types: Existence, Counting, Object, Status, and Comparison.


## D. Main Results

Open Loop Trajectory Planning. We evaluate OpenDriveVLA on the open-loop trajectory planning task using both ST-P3 and UniAD metrics, ensuring comprehensive performance assessment across spatial accuracy and collision avoidance. As shown in Table I, OpenDriveVLA achieves stateof-the-art performance across both settings. Specifically, both 3B and 7B version models achieve an average L2 error of 0.33m under ST-P3 metrics, outperforming prior autoregressive language models [21], [38]. On the UniAD metrics, OpenDriveVLA-7B also achieves great performance with an average L2 error of 0.66m. Notably, despite significantly fewer parameters, the 0.5B version still outperforms prior models obviously. 

Driving Question Answering. We access Open-DriveVLA on the driving VQA task across three nuScenes-based datasets (Table II, Table III), reporting results after the second stage of training. Open-


TABLE III: Performance comparison of OpenDriveVLA on the Nu-X dataset [19].


<table><tr><td>Models</td><td>CIDER</td><td>BL-4</td><td>METEOR</td><td>ROUGE-L</td></tr><tr><td>Hint-UniAD [19]</td><td>21.7</td><td>4.2</td><td>12.7</td><td>27.0</td></tr><tr><td>Hint-VAD [19]</td><td>22.4</td><td>4.2</td><td>13.2</td><td>27.6</td></tr><tr><td>GPT-4o [24]</td><td>19.0</td><td>4.0</td><td>10.3</td><td>24.9</td></tr><tr><td>Gemini 1.5 [51]</td><td>17.6</td><td>3.4</td><td>9.3</td><td>23.4</td></tr><tr><td>Vote2CapDETR [52]</td><td>15.3</td><td>2.6</td><td>10.9</td><td>24.2</td></tr><tr><td>TOD<eq>^{3}</eq>Cap [43]</td><td>14.5</td><td>2.5</td><td>10.5</td><td>23.5</td></tr><tr><td colspan="5">OpenDriveVLA</td></tr><tr><td>0.5B (Ours)</td><td>32.3</td><td>5.4</td><td>12.5</td><td>27.9</td></tr><tr><td>3B (Ours)</td><td>25.5</td><td>4.3</td><td>12.8</td><td>27.8</td></tr><tr><td>7B (Ours)</td><td>26.2</td><td>4.5</td><td>12.8</td><td>27.4</td></tr></table>

DriveVLA reaches best-in-class performance across all three datasets, consistently outperforming previous language-enhanced driving models and generalpurpose multimodal baselines among most metrics. On nuCaption dataset, it achieves the best captioning performance among all evaluated models, outperforming both general VLMs LLaVA1.5 [50] and Mini-GPT4 [47], as well as autonomous drivingspecific models such as LiDAR-LLM [44]. For nuScenesQA dataset, OpenDriveVLA also achieves strong performance. Compared to models that directly fuse BEV features with language models such as BEVDet+BUTD [45], it demonstrates clear advantages in object and status-related questions, which highlights the benefit of its spatially grounded visual-language alignment. Notably, the 0.5B version outperforms even the larger 7B on the Nu-X dataset, which shows its powerful sceneunderstanding ability even with lightweight LLMs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/f48c6138-d017-460b-851a-28755b186b96/ad2027ebaad2d2f75495186687fd2d8ee687c84795450bf8fc77385c91a93d12.jpg)



Fig. 4: Visualization of OpenDriveVLA-7B planning actions under original dataset instruction to keep forward (left) and modified instruction to turn right (right). The QA prediction showcases (middle) are from results reported in Table II and Table III. The agent motion prediction results are visualized after the agent-env-ego interaction stage.


## E. Ablation Study

We conduct ablation studies to evaluate the impact of input modalities and our multi-stage training strategy on OpenDriveVLA’s performance. Additionally, we qualitatively assess the model’s ability to follow diverse driving commands. 

<table><tr><td rowspan="2">Visu</td><td rowspan="2">Ego</td><td rowspan="2">Hist</td><td rowspan="2">Cmd</td><td colspan="2">Avg. Collision (%) ↓</td><td colspan="2">Avg. L2 (m) ↓</td></tr><tr><td>UniAD</td><td>ST-P3</td><td>UniAD</td><td>ST-P3</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>0.77</td><td>0.24</td><td>1.34</td><td>0.75</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>1.14</td><td>0.49</td><td>1.30</td><td>0.75</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>0.29</td><td>0.10</td><td>0.77</td><td>0.39</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>0.33</td><td>0.13</td><td>0.80</td><td>0.40</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>0.26</td><td>0.09</td><td>0.68</td><td>0.35</td></tr></table>


TABLE IV: Ablation study on the effect of different input combinations on OpenDriveVLA-0.5B.


Effect of Input Modalities. We investigate how individual input components contribute to trajectory planning. Table IV presents the results of ablating visual perception, ego state, historical trajectory, and high-level language commands. The inclusion of visual inputs significantly boosts overall performance. Adding textual commands and historical information further improves the predictions, emphasizing the value of semantic intent and temporal context. Notably, ego-state features play a critical role in nuScenes open-loop benchmark, consistent with prior findings [53]. 

Effect of Multi-Stage Training Strategy. We evaluate the contribution of each training phase in our staged pipeline incrementally. As shown in Table V, each additional stage consistently improves performance, with the most notable reductions in collision rate observed after Hierarchical Vision-Language 

Alignment and Agent-Environment-Ego Interaction Modeling. These improvements highlight the effectiveness of cross-modal grounding and interactionaware reasoning in enhancing safety-critical planning behavior. 

<table><tr><td colspan="4">Training Stage</td><td colspan="2">Avg. Collision (%) ↓</td><td colspan="2">Avg. L2 (m) ↓</td></tr><tr><td>1</td><td>2</td><td>2.5</td><td>3</td><td>UniAD</td><td>ST-P3</td><td>UniAD</td><td>ST-P3</td></tr><tr><td></td><td></td><td></td><td>√</td><td>0.37</td><td>0.13</td><td>0.70</td><td>0.36</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td>0.32</td><td>0.12</td><td>0.69</td><td>0.35</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>0.31</td><td>0.11</td><td>0.68</td><td>0.35</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>0.26</td><td>0.09</td><td>0.68</td><td>0.35</td></tr></table>


TABLE V: Ablation study on the effect of multi-stage training of 0.5B model. Stage 1, 2, 2.5, and 3 correspond to hierarchical feature alignment, driving instruction tuning, Agent-Env-Ego modeling, and trajectory tuning, respectively.


Effect of Driving Command. Figure 4 presents the qualitative comparison at an intersection under two different driver instructions: keep forward and turn right, with the right turn as the ground truth. OpenDriveVLA accurately adapts its plan to the given command while maintaining context-aware and environment-consistent behavior, demonstrating robust command-following and generalization in complex scenes. In addition, we visualize the QA predictions for the same scene, showcasing the model’s ability to reason over decision-making and traffic scene understanding. 

## V. CONCLUSION

In this work, we present OpenDriveVLA, a scalable vision-language action model designed for endto-end autonomous driving. Built upon pre-trained large language models, OpenDriveVLA generates 

3D spatially grounded and semantically consistent driving actions from multimodal inputs. We introduce a hierarchical vision-language feature alignment module and realize agent-env-ego interaction in LLM to enable fine-grained spatial reasoning and dynamic scene understanding. Through multistage training paradigm, OpenDriveVLA achieves state-of-the-art performance in open-loop planning and driving-related question answering. Extensive evaluations on nuScenes dataset show its superior trajectory planning capability compared to existing approaches. Our work demonstrates the feasibility of a scalable vision-language-driven approach for autonomous driving and highlights the potential of large language models as a foundation for end-toend driving action systems. 

## REFERENCES



[1] X. Zhou, M. Liu, E. Yurtsever, B. L. Zagar, W. Zimmer, H. Cao, and A. C. Knoll, “Vision language models in autonomous driving: A survey and outlook,” IEEE Transactions on Intelligent Vehicles, pp. 1–20, 2024. 





[2] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-to-end autonomous driving: Challenges and frontiers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 





[3] M. Liu, E. Yurtsever, J. Fossaert, X. Zhou, W. Zimmer, Y. Cui, B. L. Zagar, and A. C. Knoll, “A survey on autonomous driving datasets: Statistics, annotation quality, and a future outlook,” IEEE Transactions on Intelligent Vehicles, pp. 1–29, 2024. 





[4] X. Zhou and A. C. Knoll, “Gpt-4v as traffic assistant: An in-depth look at vision language model on complex traffic events,” 2024. [Online]. Available: https://arxiv.org/ abs/2402.02205 





[5] X. Zhai, B. Mustafa, A. Kolesnikov, and L. B. et al., “Sigmoid loss for language image pre-training,” 2023. [Online]. Available: https://arxiv.org/abs/2303.15343 





[6] H. Liu, W. Xue, Y. Chen, D. Chen, X. Zhao, K. Wang, L. Hou, R. Li, and W. Peng, “A survey on hallucination in large vision-language models,” 2024. [Online]. Available: https://arxiv.org/abs/2402.00253 





[7] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai, “Bevformer: Learning bird’seye-view representation from nbsp;multi-camera images vinbsp;spatiotemporal transformers,” in Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IX. Berlin, Heidelberg: Springer-Verlag, 2022, p. 1–18. [Online]. Available: https://doi.org/10.1007/978-3-031-20077-9 1 





[8] S. Zhang, Y. Zhai, J. Mei, and Y. Hu, “Fusionocc: Multimodal fusion for 3d occupancy prediction,” in Proceedings of the 32nd ACM International Conference on Multimedia, ser. MM ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 787–796. [Online]. Available: https://doi.org/10.1145/3664647.3681293 





[9] P. Hu, A. Huang, J. Dolan, D. Held, and D. Ramanan, “Safe local motion planning with self-supervised freespace forecasting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 12 732–12 741. 





[10] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li, “Planning-oriented autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 





[11] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang, “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” arXiv preprint arXiv:2411.15139, 2024. 





[12] X. Jia, J. You, Z. Zhang, and J. Yan, “Drivetransformer: Unified transformer for scalable end-to-end autonomous driving,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=M42KR4W9P5 





[13] H. Touvron, T. Lavril, G. Izacard, and e. a. Xavier Martinet, “Llama: Open and efficient foundation language models,” 2023. [Online]. Available: https://arxiv.org/abs/2302.13971 





[14] A. Yang, B. Yang, and B. Z. et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024. 





[15] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, and J. A. et al., “Gpt-4 technical report,” 2024. [Online]. Available: https://arxiv.org/abs/2303.08774 





[16] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llava-next: Improved reasoning, ocr, and world knowledge,” January 2024. [Online]. Available: https://llava-vl.github.io/blog/2024-01-30-llava-next/ 





[17] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond,” arXiv preprint arXiv:2308.12966, 2023. 





[18] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, “Openvla: An open-source vision-language-action model,” arXiv preprin arXiv:2406.09246, 2024. 





[19] K. Ding, B. Chen, Y. Su, H.-a. Gao, B. Jin, C. Sima, X. Li, W. Zhang, P. Barsch, and H. e. a. Li, “Hint-ad: Holistically aligned interpretability in end-to-end autonomous driving,” in 8th Annual Conference on Robot Learning, 2024. 





[20] B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang, “Senna: Bridging large vision-language models and end-toend autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2410.22313 





[21] X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao, “DriveVLM: The convergence of autonomous driving and large vision-language models,” in 8th Annual Conference on Robot Learning, 2024. [Online]. Available: https: //openreview.net/forum?id=928V4Umlys 





[22] W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li et al., “Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving,” arXiv preprint arXiv:2312.09245, 2023. 





[23] B. Jin, X. Liu, Y. Zheng, P. Li, and H. Z. et al., “Adapt: Action-aware driving caption transformer,” 2023. [Online]. Available: https://arxiv.org/abs/2302.00673 





[24] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, “Drivegpt4: Interpretable endto-end autonomous driving via large language model,” IEEE Robotics and Automation Letters, vol. 9, no. 10, pp. 8186–8193, 2024. 





[25] J. Mei, Y. Ma, X. Yang, L. Wen, X. Cai, X. Li, D. Fu, B. Zhang, P. Cai, M. Dou, B. Shi, L. He, Y. Liu, and Y. Qiao, “Continuously learning, adapting, and improving: A dual-process approach to autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2405.15324 





[26] J. Zhang, Z. Huang, A. Ray, and E. Ohn-Bar, “Feedbackguided autonomous driving,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 15 000–15 011. 





[27] H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang, H. Xie, B. Wang, and X. Bai, “Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation,” 2025. [Online]. Available: https://arxiv.org/abs/2503.19755 





[28] A. Favero, L. Zancato, M. Trager, S. Choudhary, P. Perera, A. Achille, A. Swaminathan, and S. Soatto, “Multi-moda hallucination control by visual information grounding,” 2024. [Online]. Available: https://arxiv.org/abs/2403.14003 





[29] S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez, “OmniDrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning,” arXiv:2405.01533, 2024. 





[30] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li, “Drivelm: Driving with graph visual question answering,” arXiv preprint arXiv:2312.14150, 2023. 





[31] S. Xie, L. Kong, Y. Dong, C. Sima, and W. Z. et al., “Are vlms ready for autonomous driving? an empirical study from the reliability, data, and metric perspectives,” 2025. [Online]. Available: https://arxiv.org/abs/2501.04003 





[32] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao, “St-p3: End-to-end vision-based autonomous driving via spatialtemporal feature learning,” in European Conference on Computer Vision (ECCV), 2022. 





[33] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “Vad: Vectorized scene representation for efficient autonomous driving,” ICCV, 2023. 





[34] J.-T. Zhai, Z. Feng, J. Du, Y. Mao, J.-J. Liu, Z. Tan, Y. Zhang, X. Ye, and J. Wang, “Rethinking the open-loop evaluation of end-to-end autonomous driving in nuscenes,” 2023. [Online]. Available: https://arxiv.org/abs/2305.10430 





[35] R. Song, X. Guo, H. Wu, Q. Wei, and L. Chen, “Insightdrive: Insight scene representation for end-toend autonomous driving,” 2025. [Online]. Available: https://arxiv.org/abs/2503.13047 





[36] T. Khurana, P. Hu, A. Dave, J. Ziglar, D. Held, and D. Ramanan, “Differentiable raycasting for selfsupervised occupancy forecasting,” in Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVIII. Berlin, Heidelberg: Springer-Verlag, 2022, p. 353–369. [Online]. Available: https://doi.org/10.1007/978-3-031-19839-7 21 





[37] T. Li, H. Wang, X. Li, W. Liao, T. He, and P. Peng, “Generative planning with 3d-vision language pre-training for end-to-end autonomous driving,” 2025. [Online]. Available: https://arxiv.org/abs/2501.08861 





[38] J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang, “Gpt-driver: Learning to drive with gpt,” 2023. [Online]. Available: https://arxiv.org/abs/2310.01415 





[39] Z. Huang, T. Tang, S. Chen, S. Lin, Z. Jie, L. Ma, G. Wang, and X. Liang, “Making large language models better planners with reasoning-decision alignment,” 2024. [Online]. Available: https://arxiv.org/abs/2408.13890 





[40] J.-J. Hwang, R. Xu, H. Lin, W.-C. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, Y. Zhou, J. Guo, D. Anguelov, and M. Tan, “Emma: End-to-end multimodal model for autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2410.23262 





[41] S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu, “Openemma: Open-source multimodal model for end-to-end autonomous driving,” 2025. [Online]. Available: https://arxiv.org/abs/2412.15208 





[42] W. Han, D. Guo, C.-Z. Xu, and J. Shen, “Dme-driver: Integrating human decision logic and 3d scene perception in autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2401.03641 





[43] B. Jin, Y. Zheng, P. Li, W. Li, Y. Zheng, and S. e. a. Hu, “Tod3cap: Towards 3d dense captioning,” in Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29 – October 4, 2024, Proceedings, Part XVIII. Berlin, Heidelberg: Springer-Verlag, 2024, p. 367–384. [Online]. Available: https://doi.org/10.1007/978-3-031-72649-1 21 





[44] S. Yang, J. Liu, R. Zhang, M. Pan, Z. Guo, X. Li, Z. Chen, P. Gao, Y. Guo, and S. Zhang, “Lidarllm: Exploring the potential of large language models for 3d lidar understanding,” 2023. [Online]. Available: https://arxiv.org/abs/2312.14074 





[45] T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y.-G. Jiang, “Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario,” arXiv preprint arXiv:2305.14836, 2023. 





[46] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 





[47] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “MiniGPT-4: Enhancing vision-language understanding with advanced large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum? id=1tZbq88f27 





[48] W. Dai and J. L. et al., “InstructBLIP: Towards generalpurpose vision-language models with instruction tuning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https: //openreview.net/forum?id=vvoWPYqZJA 





[49] P. Gao, J. Han, R. Zhang, Z. Lin, S. Geng, A. Zhou, W. Zhang, and P. e. a. Lu, “Llama-adapter v2: Parameterefficient visual instruction model,” arXiv preprint arXiv:2304.15010, 2023. 





[50] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2024. [Online]. Available: https://arxiv.org/abs/2310.03744 





[51] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang, S. Mariooryad, Y. Ding, X. Geng, F. Alcober, R. Frostig, M. Omernick, and L. W. et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” 2024. [Online]. Available: https://arxiv.org/abs/ 2403.05530 





[52] S. Chen, H. Zhu, X. Chen, Y. Lei, G. Yu, and T. Chen, “End-to-end 3d dense captioning with vote2cap-detr,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 11 124–11 133. 





[53] Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez, “Is ego status all you need for open-loop end-toend autonomous driving?” in 2024 IEEE/CVF Conference 





on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 864–14 873. 





[54] L. Xiao, X. Yang, X. Lan, Y. Wang, and C. Xu, “Towards visual grounding: A survey,” 2024. [Online]. Available: https://arxiv.org/abs/2412.20206 





[55] S. Wang, D. Kim, A. Taalimi, C. Sun, and W. Kuo, “Learning visual grounding from generative vision and language model,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025, pp. 8057–8067. 





[56] X. Zhou, K. Larintzakis, H. Guo, W. Zimmer, M. Liu, H. Cao, J. Zhang, V. Lakshminarasimhan, L. Strand, and A. Knoll, “TUMTraf videoQA: Dataset and benchmark for unified spatio-temporal video understanding in traffic scenes,” in Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025. [Online] Available: https://openreview.net/forum?id=Yfoi5O68rf 





[57] U. contributors, “Planning-oriented autonomous driving,” https://github.com/OpenDriveLab/UniAD, 2023. 





[58] Z. Li, W. Wang, E. Xie, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo, and T. Lu, “Panoptic segformer: Delving deeper into panoptic segmentation with transformers,” 2022. [Online]. Available: https://arxiv.org/abs/2109.03814 





[59] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari, “Nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles,” 2022. [Online]. Available: https://arxiv.org/abs/2106.11810 





[60] X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan, “Bench2drive: Towards multi-ability benchmarking of closed-loop end-toend autonomous driving,” in NeurIPS 2024 Datasets and Benchmarks Track, 2024. 





[61] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta, “Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking,” 2024. [Online]. Available: https://arxiv.org/abs/2406.15349 

