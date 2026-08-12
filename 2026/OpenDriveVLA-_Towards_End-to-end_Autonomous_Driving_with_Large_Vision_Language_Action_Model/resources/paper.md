                                               OpenDriveVLA: Towards End-to-end Autonomous
                                               Driving with Large Vision Language Action Model

                                           Xingcheng Zhou1† , Xuyuan Han1 , Feng Yang1 , Yunpu Ma2 , Volker Tresp2 , Alois Knoll1
                                               1
                                                 Technical University of Munich 2 Ludwig Maximilian University of Munich




arXiv:2503.23463v2 [cs.CV] 21 Nov 2025
                                                                             https://drivevla.github.io




                                         Fig. 1: OpenDriveVLA leverages open-source pre-trained large vision-language-action models to generate driving
                                         actions conditioned on 3D environmental perception, ego-vehicle states, and driver commands. It achieves strong
                                         performance in both open-loop planning and driving-related question answering, demonstrating its proficiency in
                                         scene understanding and driving action tuning.


                                            Abstract—We present OpenDriveVLA, a Vision-             we incorporate structured agent–environment–ego in-
                                         Language Action (VLA) model designed for end-to-           teraction modeling into the autoregressive decoding
                                         end autonomous driving, built upon open-source large       process, enabling the model to capture fine-grained
                                         language models. OpenDriveVLA generates spatially-         spatial dependencies and behavior-aware dynamics
                                         grounded driving actions by leveraging multimodal          critical for reliable trajectory planning. Extensive
                                         inputs, including both 2D and 3D instance-aware            experiments on the nuScenes dataset demonstrate
                                         visual representations, ego vehicle states, and language   that OpenDriveVLA achieves state-of-the-art results
                                         commands. To bridge the modality gap between driv-         across open-loop trajectory planning and driving-
                                         ing visual representations and language embeddings,        related question-answering tasks. Qualitative analyses
                                         we introduce a hierarchical vision-language alignment      further illustrate its superior capability to follow
                                         process, projecting both 2D and 3D structured visual       high-level driving commands and generate trajectories
                                         tokens into a unified semantic space. Furthermore,         under challenging scenarios, highlighting its potential
                                                                                                    for next-generation end-to-end autonomous driving.
                                           † Corresponding author: xingcheng.zhou@tum.de
                I. I NTRODUCTION                         grounded multimodal reasoning and driving tra-
   End-to-end learning frameworks have emerged           jectory generation within a unified autoregres-
as a promising paradigm in autonomous driving,           sive framework. Unlike prior VLM-based meth-
enabling perception, prediction, and planning to be      ods, OpenDriveVLA leverages structured 2D and
jointly optimized within a unified neural network        3D instance-aware representations, ego vehicle
[1]. They learn policies directly from sensor in-        states, and high-level commands to directly pro-
puts and generalize well across varied scenarios.        duce reliable driving actions. Extensive experiments
Despite notable progress, existing approaches still      on nuScenes benchmark demonstrate that Open-
face critical challenges, including limited long-        DriveVLA achieves state-of-the-art performance in
tail generalization, poor complex semantics under-       both open-loop planning and vision-language rea-
standing, and rigid task reasoning [2]. Meanwhile,       soning tasks. Our key contributions are:
large language models (LLMs) and vision-language            • We present OpenDriveVLA, a 3D vision-
models (VLMs) exhibit strong in-context reasoning,            language action model for end-to-end au-
commonsense understanding, and zero-shot gener-               tonomous driving that generates reliable driv-
alization abilities. These capabilities are promising         ing trajectories by integrating hierarchical vi-
for driving, where robust scene understanding is              sual input, ego state, and high-level language
crucial [3], [4]. However, directly leveraging ex-            commands.
isting VLMs for autonomous driving poses funda-             • We develop a multi-stage training strategy that
mental challenges. Firstly, current VLMs are pre-             aligns structured 2D and 3D visual features
dominantly optimized for static, 2D image-language            into a unified semantic space, enabling naive
tasks, leading to poor spatial reasoning performance          VLMs to generate spatially-grounded actions
in dynamic 3D driving environments [5]. Besides,              in complex driving scenarios.
instance-agnostic VLMs [6] are prone to hallu-              • We introduce implicit agent–environment–ego
cinations, often yielding incorrect yet overconfi-            interaction modeling into autoregressive LLM-
dent outputs, posing safety risks in autonomous               based VLA training as an auxiliary task,
driving. Motivated by these limitations, our work             enabling the model to learn behaviorally
answers a central question: How can we harness                grounded and safety-aware driving actions.
the emergent capabilities of large VLMs to
                                                                        II. R ELATED W ORK
produce safe spatially-grounded driving actions
in dynamic 3D environments, while balancing              A. End-to-End Autonomous Driving
inference speed and planning effectiveness?                 Autonomous driving (AD) evolves through two
   To enhance spatial-awareness and safety in LLM-       distinct stages. Traditional approaches rely on a
based vision-language action model, we introduce         modular design, decomposing the system into per-
two key designs. First, we structure the driving         ception [7], prediction [8], and planning [9] com-
environment using instance-aware, hierarchical 2D        ponents. While this structure ensures interpretability
and 3D visual representations to reduce the risk         and allows for independent optimization, they suffer
of instance hallucinations. Second, we incorpo-          from cascading errors between stages and are not
rate agent–environment–ego interaction modeling,         globally optimized for the final planning objec-
which is originally explicitly modeled in traditional    tive. In contrast, end-to-end autonomous driving
end-to-end driving systems, as an auxiliary objec-       frameworks [10] address this by jointly optimizing
tive into the autoregressive LLM training pipeline. It   perception, prediction, and planning within a unified
enables the model to internalize physical feasibility    neural network. These models learn driving policies
and dynamic multi-agent interactions, improving          directly from raw sensor inputs, which improves the
robustness in safety-critical scenarios.                 model’s adaptability to diverse driving conditions.
   Built upon open-source large language mod-            More recent approaches introduce diffusion mod-
els, OpenDriveVLA tightly integrates spatially-          els [11] and unified scene representations [12] to
further enhance the effectiveness and robostness.
However, existing end-to-end methods still face
semantic reasoning bottlenecks, as they struggle to
fully comprehend high-level scene semantics, infer
complex agent interactions, and adapt to dynamic
task requirements. Moreover, their decision-making
processes remain opaque, making it difficult to                   (a) VLM as additional Caption or QA Head.
diagnose failure cases, especially in long-tail or
unseen scenarios.
B. Large Vision Language Models
   Large Language Models demonstrated strong
emergent capabilities in in-context learning, instruc-
tion following, and reasoning [13], [14]. By training            (b) VLM as high-level driving decision-maker.
on vast amounts of Internet-scale data, these mod-
els acquire extensive world knowledge and exhibit
strong adaptability across diverse tasks. Their suc-
cess has also driven the rise of large VLMs, which
extend these capabilities into cross-modal reasoning
by integrating vision encoders with language mod-
                                                                  (c) Native 2D VLM for end-to-end driving.
els. State-of-the-art VLMs such as GPT-4V [15],
LLaVA [16], and Qwen-VL [17] demonstrate strong
visual understanding and multimodal reasoning in
open-domain tasks. However, these models are pri-
marily trained on static 2D images or videos and
exhibit limited spatial reasoning in dynamic 3D
driving environments. Moreover, VLMs are prone                     (d) 3D spatial-aware driving VLA (ours).
to hallucinations and generally over-confident but
                                                         Fig. 2: Taxonomy of vision-language model applications in end-
incorrect descriptions, which pose serious risks in      to-end autonomous driving.
safety-critical planning scenarios. Recently, Vision-
Language Action models have emerged to directly
predict actions from visual inputs, demonstrating
strong performance in robotic manipulation tasks         in Figure 2. One line of research in Fig.2 (a)
[18]. Currently, the application of such language-       integrates language heads, such as captioning or
conditioned end-to-end action generation in au-          question-answering modules, into driving models to
tonomous driving remains underexplored. Yet, these       enhance the interpretability [19]. The second cate-
methods are mostly limited to static setups and lack     gory in Fig.2 (b) employs vision language models
driving-specific 3D spatial design.                      to generate high-level driving instructions, such as
                                                         directional commands or abstract maneuvers, which
C. Vision Language Models in Autonomous Driving          are subsequently interpreted by separate planning
   VLMs have been applied to various autonomous          modules into low-level controls [20]–[22]. It’s also
driving tasks, including perception, scene descrip-      usually formed as a fast-slow dual system. This
tion, synthetic data generation, and high-level          design allows VLMs to make independent semantic
decision-making [1]. These efforts aim to enhance        reasoning, but retains a separate module for end-
interpretability, data efficiency, and instruction-      to-end driving planning, making joint optimization
following capabilities in driving models. We cate-       challenging. The third line in Fig.2 (c) applies
gorize recent works into 4 paradigms, as illustrated     native VLMs with 2D visual tokens to produce
driving actions, and optionally scene captions or          relevant objects and map tokens through 3D vision
QA responses [23], [24]. These methods [25]–[27]           tasks, ensuring reliable visual token proposal.
process 2D images without explicit modeling of the            Specifically, given a set of multi-view images
instance, 3D spatial layout, and inter-agent interac-      I = {I i }Ni=1 , the visual module first extracts multi-
tions in the driving scene. It limits their spatial rea-   scale 2D features from each image using a shared
soning ability and understanding of agent dynamics         2D backbone, denoted as f2D . These 2D features
in complex traffic environments. Recent studies            are then aggregated across views and lifted into
[28] further indicate that such instance-agnostic          BEV space, producing the BEV feature fbev . To
approaches are more prone to hallucinate, often pro-       obtain structured environmental representations, we
ducing overconfident or semantically inconsistent          adopt three visual query modules: Global Scene
text. In this work, we investigate how to extend 2D        Sampler Qscene , Agent QueryTransformer Qagent ,
VLMs by explicitly modeling 3D instance-aware              and Map QueryTransformer Qmap . Each module
and spatial-aware scene representations into an end-       extracts tokens focusing on a specific semantic
to-end autonomous driving framework, as shown in           aspect of the driving environment. Global Scene
Fig.2(d). Notably, we focus on fully differentiable        Sampler encodes the surrounding driving scene
end-to-end models in this work, while LLM-based            context from multi-view 2D features, producing the
agentic driving systems, such as [29], [30], fall          scene token vscene = Qscene (f2D ). Agent Query-
outside the scope of our study.                            Transformer detects and tracks dynamic agents
                                                           within the scene, extracting agent-centric tokens
               III. O PEN D RIVE VLA                          i
                                                                   }N
                                                           {vagent   i=1 = Qagent (fbev ), where Na denotes
                                                                       a


   The overall architecture of OpenDriveVLA is             the number of detected agents. In parallel, Map
shown in Figure 1, with its multi-stage training           QueryTransformer extracts static structural infor-
process further detailed in Figure 3. OpenDriveVLA         mation, such as lane boundaries and drivable ar-
uses a pre-trained vision encoder to extract tok-          eas, forming the map token vmap = Qmap (fbev ).
enized environmental representations from multi-           Through vision-centric perception tasks, including
view images. These visual tokens are then aligned          3D detection, tracking, and segmentation, the vi-
into the textual domain through cross-modal learn-         sual encoder produces structured environmental to-
ing. After alignment, it undergoes driving instruc-        kens that capture both dynamic agent behaviors
tion tuning, followed by agent-ego-environment             and static map structures in a spatially grounded
interaction modeling. Finally, OpenDriveVLA is             manner. The output tokens, denoted as Venv =
trained end-to-end to predict the ego vehicle’s future     {vscene , vagent , vmap }, serve as visual environment
trajectory, guided by the aligned visual-language          representation of the subsequent stages.
tokens and driving instructions.
                                                           B. Stage 1 - Hierarchical Vision-Language Align-
A. 3D Visual Environmental Perception                      ment
   Recent VLM-based autonomous driving methods                To bridge the modality gap between the extracted
typically rely on pretrained 2D visual encoders            visual tokens and the word embedding space of a
[5], where visual token selection and attention            pre-trained LLM, we adopt a hierarchical vision-
are indirectly guided through language supervision.        language feature alignment strategy. Given the vi-
While effective in open-domain vision-language             sual tokens extracted from the 3D visual perception
applications, this design lacks explicit 3D spatial        module, we introduce three token-specific projec-
grounding and structured instance-level attention,         tors {Φscene , Φagent , Φmap }. During training, each
which can lead to severe hallucinations in safety-         active agent query from the 3D detection and track-
                                                                                    i
critical driving scenarios [31]. To mitigate this,         ing task denoted as vagent    , is also matched to its
OpenDriveVLA adopts a visual-centric query mod-            corresponding ground-truth caption Xiagent . These
ule, where the model first learns to focus on driving-     captions provide detailed descriptions, including 2D
Fig. 3: Illustration of main training stages on OpenDriveVLA. Stage 1: Hierarchical Feature Alignment. Stage 2: Driving Instruction
Tuning. Stage 2.5: Agent-Env-Ego Interaction Modeling. Stage 3: Trajectory Planning Tuning.



appearance descriptions and 3D spatial positions.                  making. By training on this diverse set of driv-
For scene and map tokens, which encode holis-                      ing queries, OpenDriveVLA learns to contextu-
tic spatial context and static structural properties,              alize the driving scene, follow commands, and
a sample-wise alignment is applied, where each                     generate semantically and behaviorally grounded
token is matched to a scene-level caption Xscene                   planning decisions. We formulate the tuning data
or Xmap . The scene token vscene captures the                      as instruction-response pairs {Xinput , Xanswer },
global 2D environmental context, while the map                     where Xinput = (Venv , Sego , Xquery ). Here,
token vmap encodes structural elements such as                     Xquery denotes the driving-related question, and
lane topology, road boundaries, and drivable areas.                Sego encodes the textual ego vehicle state. Given
Each of these tokens is aligned to its corresponding               this multimodal input, the LLM autoregressively
caption, denoted as Xscene and Xmap . During this                  learns to generate the target response. During in-
stage, both the visual encoder and LLM remain                      struction tuning, the visual encoder remains frozen
frozen to preserve pretrained semantics, with only                 while the token-specific projectors and the LLM
the token-specific projectors being trainable. The                 are set to be trainable. The instruction prediction
forward alignment step is formulated as follows:                   process is as:
  X̂k = LLM (Φk (vk )) ,  k ∈ {scene, map} (1)
X̂iagent = LLM Φagent (vagent
                        i
                               
                              ) , i = 1, . . . , Na                        X̂answer = LLM (Venv , Sego , Xquery )              (3)
                                                 (2)               D. Stage 2.5 - Agent Environment Ego Interaction
C. Stage 2 - Driving Instruction Tuning                               Reliable trajectory planning in autonomous driv-
   We distill high-level driving knowledge into the                ing necessitates a spatially grounded 3D represen-
model via supervised instruction tuning, enabling                  tation of the environment. Beyond perception, it
it to internalize semantic reasoning patterns during               must also understand dynamic interactions between
training. This avoids costly chain-of-thought (CoT)                the ego vehicle and surrounding agents. Effective
reasoning at inference time and balances planning                  interaction modeling is essential to ensure that
efficacy with runtime efficiency.                                  planned trajectories are both feasible and collision-
   During the tuning process, driving knowledge                    free under real-world driving constraints. However,
from the language domain is injected into the                      existing pre-trained LLMs lack an inherent induc-
model using a curated driving instruction QA                       tive bias for spatial reasoning in 3D driving scenes,
dataset. The dataset covers a wide range of                        as they are predominantly trained on 2D vision-
driving-related reasoning, including perception un-                language and text-based datasets. We introduce
derstanding, motion prediction, attention alloca-                  a conditional agent trajectory forecasting task as
tion, action reasoning, and high-level decision-                   an auxiliary objective, encouraging the model to
learn spatially grounded interaction priors. During                  jointly optimized end-to-end during training, with
this stage, OpenDriveVLA captures the underly-                       the 2D encoder kept frozen. At inference, the model
ing structure of multi-agent dynamics, enhancing                     autoregressively generates the tokenized trajectory
its capability for scene-aware trajectory generation                 T̂traj , which is then decoded back into numerical
and improving decision-making in complex traffic                     waypoints:
scenarios.
   Given scene and map tokens, as well as the ego                                 Ŵego = Decoder(T̂traj )            (6)
vehicle state Sego , the LLM predicts the future
motion of each detected agent based on its pro-                                      IV. E XPERIMENTS
                                   i
jected visual embedding Φagent (vagent  ). The future                A. Training Datasets
motion of agent ai is represented as a sequence                         We curate the training data of OpenDriveVLA
of waypoints Wai . The predicted trajectory is con-                  based on its distinct training phases, drawing from:
ditioned on the scene context, map structure, and                    TOD3Cap [43], nuCaption [44], nuScenesQA [45],
ego vehicle state, enabling OpenDriveVLA to infer                    nuX [19], and GPT-Driver [38]. We conduct ex-
interaction-aware and spatially grounded motion                      periments on nuScenes [46], following standard
sequences. The learning objective for the i-th agent                 data split into training and validation sets. Open-
is formulated as:                                                    DriveVLA is trained using the training set paired
                                                                     with corresponding QA captions, while the val-
      T
      Y                                                              idation set is exclusively used for performance
            p wti | w1:t−1
                     i                              i
                                                                
max                        , Venv , Sego , Φagent (vagent )          evaluation to ensure fair comparisons with prior
      t=1                                                            works. The details of training data can be found
                                               (4)
                                                                     in supplementary materials.
   This provides OpenDriveVLA with essential spa-
                                                                     Hierarchical Vision-Language Alignment. For
tial priors, enabling it to bridge the gap be-
                                                                     agent-level caption, we post-process data from [43],
tween high-level semantic reasoning and physically
                                                                     which provides the 2D visual description of individ-
grounded motion planning.
                                                                     ual objects. To further enhance spatial grounding,
E. Stage 3 - End-to-end Trajectory Planning Tuning                   each object caption is augmented with its cor-
   In this stage, OpenDriveVLA predicts ego trajec-                  responding BEV coordinates, enabling the model
tories as discrete waypoint sequences within a short                 to associate object attributes with precise spatial
horizon, denoted as Wego = {w1 , w2 , . . . , wT }.                  locations. For scene tokens, we process multi-view
Each waypoint wt represents the 2D coordinates                       scene descriptions from [44], merging them into
(xt , yt ) of the ego vehicle at time step t. The                    unified summaries that describe the driving envi-
waypoints are tokenized into a sequence of discrete                  ronment across all camera views. For map tokens,
textual tokens for autoregressive generation in the                  structured language descriptions are derived from
LLM: T traj = Tokenizer(Wego ). The generation                       ground-truth annotations, translating map elements
process is then cast as a causal sequence prediction                 such as lane dividers, crosswalks, and road bound-
task, where each token is predicted in a causal                      aries into descriptive text.
manner, conditioned on the visual perception tokens                  Driving Instruction Tuning. We adopt multiple
Venv , the ego state Sego , and the driving command                  instruction-oriented datasets derived from nuScenes
Xdri .                                                               to inject driving-specific knowledge into Open-
                                                                     DriveVLA. We unify several datasets into a stan-
                                                                     dardized instruction-based QA format, including
                       QT
T̂traj = argmaxTtraj     t=1 p (wt | w1:t−1 , Venv , Sego , Xdri )   driving-related question-answer pairs collected from
                                            (5)                      nuCaption [44], nuScenesQA [45], and nuX [19]
  The entire pipeline, including the 3D visual                       dataset. Each QA pair is conditioned on structured
encoder, cross-modality projectors, and LLM, is                      environmental visual tokens and the ego vehicle
                                                  ST-P3 metrics                                                  UniAD metrics
  Method                                                                                                                                                    LLM         Input
                                     L2 (m) ↓                     Collision (%) ↓                    L2 (m) ↓                   Collision (%) ↓
                             1s     2s     3s     Avg.    1s        2s       3s     Avg.    1s    2s       3s     Avg.   1s       2s     3s       Avg.

                                                                          None-Autoregressive Methods

  ST-P3 [32]                 1.33   2.11   2.90   2.11    0.23     0.62     1.27    0.71     -      -       -       -      -       -      -         -         -         Visual
  VAD [33]                   0.17   0.34   0.60   0.37    0.07     0.10     0.24    0.14     -      -       -       -      -       -      -         -         -         Visual
  Ego-MLP [34]               0.46   0.76   1.12   0.78    0.21     0.35     0.58    0.38     -      -       -       -      -       -      -         -         -          Ego
  UniAD [10]                 0.44   0.67   0.96   0.69    0.04     0.08     0.23    0.12   0.48   0.96    1.65    1.03   0.05    0.17   0.71      0.31        -         Visual
  InsightDrive [35]          0.23   0.41   0.68   0.44    0.09     0.10     0.27    0.15   0.30   0.72    1.41    0.81   0.08    0.15   0.84      0.36        -         Visual
  FF [9]                       -      -      -      -       -        -        -       -    0.55   1.20    2.54    1.43   0.06    0.17   1.07      0.43        -         LiDAR
  EO [36]                      -      -      -      -       -        -        -       -    0.67   1.36    2.78    1.60   0.04    0.09   0.88      0.33        -         LiDAR

                                                                            Autoregressive Methods

  GPVL [37]                  0.21   0.39   0.69   0.43    0.07     0.09     0.27    0.14     -      -       -       -      -       -      -         -       BERT        Textual
  DriveVLM [21]              0.18   0.34   0.68   0.40    0.10     0.22     0.45    0.27     -      -       -       -      -       -      -         -    Qwen-VL-7B     Visual
  GPT-Driver [38]            0.20   0.40   0.70   0.44    0.04     0.12     0.36    0.17   0.27   0.74    1.52    0.84   0.07    0.15   1.10      0.44     GPT-3.5      Textual
  RDA-Driver [39]            0.17   0.37   0.69   0.40    0.01     0.05     0.26    0.10   0.23   0.73    1.54    0.80   0.00    0.13   0.83      0.32    LLaVa-7B      Visual
  OminiDrive [29]            0.14   0.29   0.55   0.33    0.00     0.13     0.78    0.30     -      -       -       -      -       -      -         -     LLaVA-7B      Visual
  EMMA [40]                  0.14   0.29   0.54   0.32      -        -        -       -      -      -       -       -      -       -      -         -      Gemini       Visual
  OpenEMMA [41]              1.45   3.21   3.76   2.81      -        -        -       -      -      -       -       -      -       -      -         -    Qwen-VL-7B     Visual
  DME-Driver [42]              -      -      -      -       -        -        -       -    0.45   0.91    1.58    0.98   0.05    0.28   0.55      0.29    LLaVa-7B      Visual

  OpenDriveVLA-0.5B (Ours)   0.15   0.32   0.57   0.35    0.01     0.06     0.20    0.09   0.21   0.60    1.22    0.68   0.00    0.15   0.63      0.26   Qwen2.5-0.5B   Visual
  OpenDriveVLA-3B (Ours)     0.14   0.30   0.55   0.33    0.02     0.07     0.22    0.10   0.19   0.58    1.24    0.67   0.02    0.18   0.70      0.30    Qwen2.5-3B    Visual
  OpenDriveVLA-7B (Ours)     0.15   0.31   0.55   0.33    0.01     0.08     0.21    0.10   0.20   0.58    1.21    0.66   0.00    0.22   0.55      0.25    Qwen2.5-7B    Visual

TABLE I: Open-Loop planning performance comparison of different driving models, including both autoregressive methods and
non-autoregressive methods. OpenDriveVLA shows powerful planning ability and achieves best-in-class results among open-source
models, even with the 0.5B version. We refer to the result summary from [35], [37]–[39].



state, ensuring consistency across different data                                          UniAD [10] settings. The evaluation metrics include
sources. This multimodal instruction tuning pro-                                           L2 displacement errors at 1, 2, and 3 seconds, along
cess allows OpenDriveVLA to effectively ground                                             with the average collision rate over the prediction
language understanding into both environmental                                             horizon. To further assess the scene understanding
perception and scene understanding, bridging per-                                          ability of OpenDriveVLA, we report its QA predic-
ception, reasoning, and action within the language                                         tion performance on three driving visual question
space.                                                                                     answering (VQA) datasets directly after the driving
Motion Forecasting and Trajectory Prediction.                                              instruction tuning stage, i.e., [44], nuScenesQA
We formulate both agent motion forecasting and ego                                         [45], and nuX [19]. The VQA evaluation results
trajectory planning in the ego system, where the                                           adopt standard NLG metrics, including BLEU, ME-
model directly predicts future displacements within                                        TEOR, CIDEr, BERT-Score, etc.
each entity’s local coordinate frame relative to the
ego vehicle for planning and relative to each agent                                        C. Implementation Details
for forecasting. This formulation captures motion                                             The 3D visual perception module in Open-
dynamics in a spatially consistent manner across                                           DriveVLA follows the vision-centric design from
all entities. Following [38], the ego vehicle state is                                     [10], using a ResNet-101 backbone for 2D feature
encoded as textual input to ensure ego awareness                                           extraction. The perception backbone is pre-trained
throughout the training process. Both tasks predict                                        via multi-task learning on 3D object detection, ob-
3-second future trajectories, sampled at 0.5-second                                        ject tracking, and map segmentation. The resulting
intervals, resulting in 6 waypoints per trajectory.                                        BEV feature map has a spatial resolution of 200 ×
                                                                                           200. To construct a unified scene representation, the
B. Evaluations
                                                                                           global SceneSampler applies 2D adaptive pooling
   We evaluate OpenDriveVLA on the open-loop                                               to each camera view, subsequently concatenating
planning task of nuScenes benchmark, where the                                             the pooled multi-view features into a global scene
model is reported under both ST-P3 [32] and                                                token. Agent and map tokens are extracted from
                                            nu-Caption                                             nuScenes-QA
  Method
                             BL-1   BL-2     BL-3   BL-4    BERT-S       Ext      Cnt    Obj       Sts      Cmp       H0     H1       Acc
  Mini-GPT4 [47]             15.0     6.8     3.7     2.6    84.4         -         -      -         -         -       -       -        -
  Instruct-BLIP [48]         18.7    13.4     7.4     5.2    85.9         -         -      -         -         -       -       -        -
  LLaMA-AdapV2 [49]          30.2    17.3    10.4     7.5    86.5        19.3      2.7    7.6      10.8       1.6     15.1    4.8      9.6
  LLaVA1.5 [50]              20.0    12.1     8.6     5.4    85.0        45.8      7.7    7.8       9.0      52.1     25.7   41.5     26.2
  LiDAR-LLM [44]             41.0    30.0    23.4    19.3    91.3        74.5     15.0   37.8      45.9      57.8      -       -      48.6
  BEVDet+BUTD [45]            -        -       -       -      -          83.7     20.9   48.8      52.0      67.7      -       -      57.0
  OpenDriveVLA-0.5B (Ours)   47.2    35.8    29.4    25.2    91.9        83.9     22.0   50.2      57.0      68.4     62.3   56.5     58.4
  OpenDriveVLA-3B (Ours)     48.3    36.9    30.3    26.1    92.0        84.0     22.3   50.3      56.9      68.5     62.6   56.5     58.5
  OpenDriveVLA-7B (Ours)     49.6    38.3    31.9    27.6    92.2        84.2     22.7   49.6      54.5      68.8     62.4   56.1     58.2

TABLE II: Performance on nu-Caption [44] and nuScenes-QA [45]. BL-1/2/3/4: BLEU scores. QA metrics report accuracy on five
question types: Existence, Counting, Object, Status, and Comparison.


                                                               TABLE III: Performance comparison of OpenDriveVLA on the
the final layer of their respective QueryTransformer           Nu-X dataset [19].
modules. Each token type is then mapped into the
                                                                    Models                CIDER           BL-4      METEOR    ROUGE-L
language space using a separate two-layer MLP
                                                                    Hint-UniAD [19]         21.7          4.2        12.7           27.0
with GeLU activation. We adopt Qwen 2.5-Instruct                    Hint-VAD [19]           22.4          4.2        13.2           27.6
[14] as the pre-trained LLM, which undergoes full                   GPT-4o [24]             19.0          4.0        10.3           24.9
                                                                    Gemini 1.5 [51]         17.6          3.4         9.3           23.4
parameter tuning during training. Training is per-                  Vote2CapDETR [52]       15.3          2.6        10.9           24.2
formed on 4 NVIDIA H100 GPUs with a batch                           TOD3 Cap [43]           14.5          2.5        10.5           23.5

size of 1, completed in approximately two days.                                             OpenDriveVLA

We freeze the 2D backbone during stage 3. During                    0.5B (Ours)             32.3          5.4        12.5           27.9
                                                                    3B (Ours)               25.5          4.3        12.8           27.8
inference, we set the decoding temperature to 0                     7B (Ours)               26.2          4.5        12.8           27.4
to ensure deterministic trajectory generation. See
supplementary material for detailed training config-
urations.
D. Main Results
Open Loop Trajectory Planning. We evaluate                     DriveVLA reaches best-in-class performance across
OpenDriveVLA on the open-loop trajectory plan-                 all three datasets, consistently outperforming previ-
ning task using both ST-P3 and UniAD metrics,                  ous language-enhanced driving models and general-
ensuring comprehensive performance assessment                  purpose multimodal baselines among most metrics.
across spatial accuracy and collision avoidance. As            On nuCaption dataset, it achieves the best caption-
shown in Table I, OpenDriveVLA achieves state-                 ing performance among all evaluated models, out-
of-the-art performance across both settings. Specif-           performing both general VLMs LLaVA1.5 [50] and
ically, both 3B and 7B version models achieve an               Mini-GPT4 [47], as well as autonomous driving-
average L2 error of 0.33m under ST-P3 metrics,                 specific models such as LiDAR-LLM [44]. For
outperforming prior autoregressive language models             nuScenesQA dataset, OpenDriveVLA also achieves
[21], [38]. On the UniAD metrics, OpenDriveVLA-                strong performance. Compared to models that di-
7B also achieves great performance with an average             rectly fuse BEV features with language mod-
L2 error of 0.66m. Notably, despite significantly              els such as BEVDet+BUTD [45], it demonstrates
fewer parameters, the 0.5B version still outperforms           clear advantages in object and status-related ques-
prior models obviously.                                        tions, which highlights the benefit of its spatially
Driving Question Answering. We access Open-                    grounded visual-language alignment. Notably, the
DriveVLA on the driving VQA task across three                  0.5B version outperforms even the larger 7B on
nuScenes-based datasets (Table II, Table III), report-         the Nu-X dataset, which shows its powerful scene-
ing results after the second stage of training. Open-          understanding ability even with lightweight LLMs.
Fig. 4: Visualization of OpenDriveVLA-7B planning actions under original dataset instruction to keep forward (left) and modified
instruction to turn right (right). The QA prediction showcases (middle) are from results reported in Table II and Table III. The agent
motion prediction results are visualized after the agent-env-ego interaction stage.



E. Ablation Study                                                    Alignment and Agent-Environment-Ego Interaction
   We conduct ablation studies to evaluate the im-                   Modeling. These improvements highlight the effec-
pact of input modalities and our multi-stage training                tiveness of cross-modal grounding and interaction-
strategy on OpenDriveVLA’s performance. Addi-                        aware reasoning in enhancing safety-critical plan-
tionally, we qualitatively assess the model’s ability                ning behavior.
to follow diverse driving commands.
                                                                            Training Stage     Avg. Collision (%) ↓    Avg. L2 (m) ↓
                                                                        1       2    2.5   3   UniAD        ST-P3     UniAD     ST-P3
                            Avg. Collision (%) ↓    Avg. L2 (m) ↓
  Visu   Ego   Hist   Cmd                                                                 ✓     0.37        0.13       0.70     0.36
                            UniAD        ST-P3     UniAD     ST-P3
                                                                        ✓                 ✓     0.32        0.12       0.69     0.35
   ✓            ✓      ✓     0.77        0.24       1.34     0.75       ✓      ✓          ✓     0.31        0.11       0.68     0.35
   ✓      ✓            ✓     1.14        0.49       1.30     0.75       ✓      ✓     ✓    ✓     0.26        0.09       0.68     0.35
          ✓     ✓      ✓     0.29        0.10       0.77     0.39
   ✓      ✓     ✓            0.33        0.13       0.80     0.40
   ✓      ✓     ✓      ✓     0.26        0.09       0.68     0.35    TABLE V: Ablation study on the effect of multi-stage training
                                                                     of 0.5B model. Stage 1, 2, 2.5, and 3 correspond to hierarchical
TABLE IV: Ablation study on the effect of different input            feature alignment, driving instruction tuning, Agent-Env-Ego
combinations on OpenDriveVLA-0.5B.                                   modeling, and trajectory tuning, respectively.


Effect of Input Modalities. We investigate how                       Effect of Driving Command. Figure 4 presents
individual input components contribute to trajectory                 the qualitative comparison at an intersection under
planning. Table IV presents the results of ablating                  two different driver instructions: keep forward and
visual perception, ego state, historical trajectory,                 turn right, with the right turn as the ground truth.
and high-level language commands. The inclusion                      OpenDriveVLA accurately adapts its plan to the
of visual inputs significantly boosts overall perfor-                given command while maintaining context-aware
mance. Adding textual commands and historical                        and environment-consistent behavior, demonstrating
information further improves the predictions, em-                    robust command-following and generalization in
phasizing the value of semantic intent and temporal                  complex scenes. In addition, we visualize the QA
context. Notably, ego-state features play a critical                 predictions for the same scene, showcasing the
role in nuScenes open-loop benchmark, consistent                     model’s ability to reason over decision-making and
with prior findings [53].                                            traffic scene understanding.
Effect of Multi-Stage Training Strategy. We eval-
uate the contribution of each training phase in our                                        V. C ONCLUSION
staged pipeline incrementally. As shown in Table V,                     In this work, we present OpenDriveVLA, a scal-
each additional stage consistently improves perfor-                  able vision-language action model designed for end-
mance, with the most notable reductions in collision                 to-end autonomous driving. Built upon pre-trained
rate observed after Hierarchical Vision-Language                     large language models, OpenDriveVLA generates
3D spatially grounded and semantically consistent
driving actions from multimodal inputs. We intro-
duce a hierarchical vision-language feature align-
ment module and realize agent-env-ego interaction
in LLM to enable fine-grained spatial reasoning
and dynamic scene understanding. Through multi-
stage training paradigm, OpenDriveVLA achieves
state-of-the-art performance in open-loop planning
and driving-related question answering. Extensive
evaluations on nuScenes dataset show its superior
trajectory planning capability compared to existing
approaches. Our work demonstrates the feasibility
of a scalable vision-language-driven approach for
autonomous driving and highlights the potential of
large language models as a foundation for end-to-
end driving action systems.
                        R EFERENCES                                   [13] H. Touvron, T. Lavril, G. Izacard, and e. a. Xavier Martinet,
                                                                           “Llama: Open and efficient foundation language models,”
                                                                           2023. [Online]. Available: https://arxiv.org/abs/2302.13971
 [1] X. Zhou, M. Liu, E. Yurtsever, B. L. Zagar, W. Zimmer,
                                                                      [14] A. Yang, B. Yang, and B. Z. et al., “Qwen2.5 technical
     H. Cao, and A. C. Knoll, “Vision language models in
                                                                           report,” arXiv preprint arXiv:2412.15115, 2024.
     autonomous driving: A survey and outlook,” IEEE Trans-
                                                                      [15] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad,
     actions on Intelligent Vehicles, pp. 1–20, 2024.
                                                                           I. Akkaya, F. L. Aleman, D. Almeida, and J. A.
 [2] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and                  et al., “Gpt-4 technical report,” 2024. [Online]. Available:
     H. Li, “End-to-end autonomous driving: Challenges and                 https://arxiv.org/abs/2303.08774
     frontiers,” IEEE Transactions on Pattern Analysis and
                                                                      [16] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and
     Machine Intelligence, 2024.
                                                                           Y. J. Lee, “Llava-next: Improved reasoning, ocr, and
 [3] M. Liu, E. Yurtsever, J. Fossaert, X. Zhou, W. Zimmer,                world knowledge,” January 2024. [Online]. Available:
     Y. Cui, B. L. Zagar, and A. C. Knoll, “A survey on                    https://llava-vl.github.io/blog/2024-01-30-llava-next/
     autonomous driving datasets: Statistics, annotation quality,     [17] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang,
     and a future outlook,” IEEE Transactions on Intelligent               J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A versatile
     Vehicles, pp. 1–29, 2024.                                             vision-language model for understanding, localization, text
 [4] X. Zhou and A. C. Knoll, “Gpt-4v as traffic assistant:                reading, and beyond,” arXiv preprint arXiv:2308.12966,
     An in-depth look at vision language model on complex                  2023.
     traffic events,” 2024. [Online]. Available: https://arxiv.org/   [18] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakr-
     abs/2402.02205                                                        ishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. San-
 [5] X. Zhai, B. Mustafa, A. Kolesnikov, and L. B. et al.,                 keti, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake,
     “Sigmoid loss for language image pre-training,” 2023.                 D. Sadigh, S. Levine, P. Liang, and C. Finn, “Openvla: An
     [Online]. Available: https://arxiv.org/abs/2303.15343                 open-source vision-language-action model,” arXiv preprint
 [6] H. Liu, W. Xue, Y. Chen, D. Chen, X. Zhao, K. Wang,                   arXiv:2406.09246, 2024.
     L. Hou, R. Li, and W. Peng, “A survey on hallucination in        [19] K. Ding, B. Chen, Y. Su, H.-a. Gao, B. Jin, C. Sima, X. Li,
     large vision-language models,” 2024. [Online]. Available:             W. Zhang, P. Barsch, and H. e. a. Li, “Hint-ad: Holistically
     https://arxiv.org/abs/2402.00253                                      aligned interpretability in end-to-end autonomous driving,”
 [7] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu,                        in 8th Annual Conference on Robot Learning, 2024.
     Y. Qiao, and J. Dai, “Bevformer: Learning bird’s-                [20] B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin,
     eye-view representation from nbsp;multi-camera images                 Q. Zhang, C. Huang, W. Liu, and X. Wang, “Senna:
     vinbsp;spatiotemporal transformers,” in Computer Vision –             Bridging large vision-language models and end-to-
     ECCV 2022: 17th European Conference, Tel Aviv, Israel,                end autonomous driving,” 2024. [Online]. Available:
     October 23–27, 2022, Proceedings, Part IX. Berlin,                    https://arxiv.org/abs/2410.22313
     Heidelberg: Springer-Verlag, 2022, p. 1–18. [Online].            [21] X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao,
     Available: https://doi.org/10.1007/978-3-031-20077-9 1                K. Zhan, P. Jia, X. Lang, and H. Zhao, “DriveVLM:
 [8] S. Zhang, Y. Zhai, J. Mei, and Y. Hu, “Fusionocc: Multi-              The convergence of autonomous driving and large
     modal fusion for 3d occupancy prediction,” in Proceedings             vision-language models,” in 8th Annual Conference
     of the 32nd ACM International Conference on Multimedia,               on Robot Learning, 2024. [Online]. Available: https:
     ser. MM ’24. New York, NY, USA: Association                           //openreview.net/forum?id=928V4Umlys
     for Computing Machinery, 2024, p. 787–796. [Online].             [22] W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong,
     Available: https://doi.org/10.1145/3664647.3681293                    Y. Wen, S. Wu, H. Deng, Z. Li et al., “Drivemlm: Align-
 [9] P. Hu, A. Huang, J. Dolan, D. Held, and D. Ramanan,                   ing multi-modal large language models with behavioral
     “Safe local motion planning with self-supervised freespace            planning states for autonomous driving,” arXiv preprint
     forecasting,” in Proceedings of the IEEE/CVF Conference               arXiv:2312.09245, 2023.
     on Computer Vision and Pattern Recognition (CVPR), June          [23] B. Jin, X. Liu, Y. Zheng, P. Li, and H. Z. et al., “Adapt:
     2021, pp. 12 732–12 741.                                              Action-aware driving caption transformer,” 2023. [Online].
[10] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu,                      Available: https://arxiv.org/abs/2302.00673
     S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu,          [24] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K.
     J. Dai, Y. Qiao, and H. Li, “Planning-oriented autonomous             Wong, Z. Li, and H. Zhao, “Drivegpt4: Interpretable end-
     driving,” in Proceedings of the IEEE/CVF Conference on                to-end autonomous driving via large language model,”
     Computer Vision and Pattern Recognition, 2023.                        IEEE Robotics and Automation Letters, vol. 9, no. 10, pp.
[11] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan,                  8186–8193, 2024.
     X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang,                [25] J. Mei, Y. Ma, X. Yang, L. Wen, X. Cai, X. Li, D. Fu,
     “Diffusiondrive: Truncated diffusion model for end-to-end             B. Zhang, P. Cai, M. Dou, B. Shi, L. He, Y. Liu, and
     autonomous driving,” arXiv preprint arXiv:2411.15139,                 Y. Qiao, “Continuously learning, adapting, and improving:
     2024.                                                                 A dual-process approach to autonomous driving,” 2024.
[12] X. Jia, J. You, Z. Zhang, and J. Yan, “Drivetransformer:              [Online]. Available: https://arxiv.org/abs/2405.15324
     Unified transformer for scalable end-to-end autonomous           [26] J. Zhang, Z. Huang, A. Ray, and E. Ohn-Bar, “Feedback-
     driving,” in The Thirteenth International Conference                  guided autonomous driving,” in 2024 IEEE/CVF Confer-
     on Learning Representations, 2025. [Online]. Available:               ence on Computer Vision and Pattern Recognition (CVPR),
     https://openreview.net/forum?id=M42KR4W9P5                            2024, pp. 15 000–15 011.
[27] H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang,          [41] S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou,
     D. Zhang, H. Xie, B. Wang, and X. Bai, “Orion:                      and Z. Tu, “Openemma: Open-source multimodal model
     A holistic end-to-end autonomous driving framework                  for end-to-end autonomous driving,” 2025. [Online].
     by vision-language instructed action generation,” 2025.             Available: https://arxiv.org/abs/2412.15208
     [Online]. Available: https://arxiv.org/abs/2503.19755          [42] W. Han, D. Guo, C.-Z. Xu, and J. Shen, “Dme-driver:
[28] A. Favero, L. Zancato, M. Trager, S. Choudhary, P. Perera,          Integrating human decision logic and 3d scene perception
     A. Achille, A. Swaminathan, and S. Soatto, “Multi-modal             in autonomous driving,” 2024. [Online]. Available:
     hallucination control by visual information grounding,”             https://arxiv.org/abs/2401.03641
     2024. [Online]. Available: https://arxiv.org/abs/2403.14003    [43] B. Jin, Y. Zheng, P. Li, W. Li, Y. Zheng, and
[29] S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang,                 S. e. a. Hu, “Tod3cap: Towards 3d dense captioning,”
     J. Kautz, Y. Li, and J. M. Alvarez, “OmniDrive: A holistic          in Computer Vision – ECCV 2024: 18th European
     llm-agent framework for autonomous driving with 3d per-             Conference, Milan, Italy, September 29 – October 4,
     ception, reasoning and planning,” arXiv:2405.01533, 2024.           2024, Proceedings, Part XVIII. Berlin, Heidelberg:
[30] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang,                     Springer-Verlag, 2024, p. 367–384. [Online]. Available:
     C. Xie, P. Luo, A. Geiger, and H. Li, “Drivelm: Driving             https://doi.org/10.1007/978-3-031-72649-1 21
     with graph visual question answering,” arXiv preprint          [44] S. Yang, J. Liu, R. Zhang, M. Pan, Z. Guo, X. Li,
     arXiv:2312.14150, 2023.                                             Z. Chen, P. Gao, Y. Guo, and S. Zhang, “Lidar-
[31] S. Xie, L. Kong, Y. Dong, C. Sima, and W. Z. et al., “Are           llm: Exploring the potential of large language models
     vlms ready for autonomous driving? an empirical study               for 3d lidar understanding,” 2023. [Online]. Available:
     from the reliability, data, and metric perspectives,” 2025.         https://arxiv.org/abs/2312.14074
     [Online]. Available: https://arxiv.org/abs/2501.04003          [45] T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y.-G. Jiang,
[32] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao, “St-p3:           “Nuscenes-qa: A multi-modal visual question answer-
     End-to-end vision-based autonomous driving via spatial-             ing benchmark for autonomous driving scenario,” arXiv
     temporal feature learning,” in European Conference on               preprint arXiv:2305.14836, 2023.
     Computer Vision (ECCV), 2022.                                  [46] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong,
                                                                         Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom,
[33] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou,
                                                                         “nuscenes: A multimodal dataset for autonomous driving,”
     Q. Zhang, W. Liu, C. Huang, and X. Wang, “Vad: Vector-
                                                                         in Proceedings of the IEEE/CVF Conference on Computer
     ized scene representation for efficient autonomous driving,”
                                                                         Vision and Pattern Recognition (CVPR), June 2020.
     ICCV, 2023.
                                                                    [47] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny,
[34] J.-T. Zhai, Z. Feng, J. Du, Y. Mao, J.-J. Liu, Z. Tan,              “MiniGPT-4: Enhancing vision-language understanding
     Y. Zhang, X. Ye, and J. Wang, “Rethinking the open-loop             with advanced large language models,” in The Twelfth
     evaluation of end-to-end autonomous driving in nuscenes,”           International Conference on Learning Representations,
     2023. [Online]. Available: https://arxiv.org/abs/2305.10430         2024. [Online]. Available: https://openreview.net/forum?
[35] R. Song, X. Guo, H. Wu, Q. Wei, and L. Chen,                        id=1tZbq88f27
     “Insightdrive: Insight scene representation for end-to-        [48] W. Dai and J. L. et al., “InstructBLIP: Towards general-
     end autonomous driving,” 2025. [Online]. Available:                 purpose vision-language models with instruction tuning,”
     https://arxiv.org/abs/2503.13047                                    in Thirty-seventh Conference on Neural Information
[36] T. Khurana, P. Hu, A. Dave, J. Ziglar, D. Held,                     Processing Systems, 2023. [Online]. Available: https:
     and D. Ramanan, “Differentiable raycasting for self-                //openreview.net/forum?id=vvoWPYqZJA
     supervised occupancy forecasting,” in Computer Vision –        [49] P. Gao, J. Han, R. Zhang, Z. Lin, S. Geng, A. Zhou,
     ECCV 2022: 17th European Conference, Tel Aviv, Israel,              W. Zhang, and P. e. a. Lu, “Llama-adapter v2: Parameter-
     October 23–27, 2022, Proceedings, Part XXXVIII. Berlin,             efficient visual instruction model,” arXiv preprint
     Heidelberg: Springer-Verlag, 2022, p. 353–369. [Online].            arXiv:2304.15010, 2023.
     Available: https://doi.org/10.1007/978-3-031-19839-7 21        [50] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines
[37] T. Li, H. Wang, X. Li, W. Liao, T. He, and P. Peng,                 with visual instruction tuning,” 2024. [Online]. Available:
     “Generative planning with 3d-vision language pre-training           https://arxiv.org/abs/2310.03744
     for end-to-end autonomous driving,” 2025. [Online].            [51] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai,
     Available: https://arxiv.org/abs/2501.08861                         A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang,
[38] J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang, “Gpt-driver:          S. Mariooryad, Y. Ding, X. Geng, F. Alcober, R. Frostig,
     Learning to drive with gpt,” 2023. [Online]. Available:             M. Omernick, and L. W. et al., “Gemini 1.5: Unlocking
     https://arxiv.org/abs/2310.01415                                    multimodal understanding across millions of tokens of
[39] Z. Huang, T. Tang, S. Chen, S. Lin, Z. Jie, L. Ma,                  context,” 2024. [Online]. Available: https://arxiv.org/abs/
     G. Wang, and X. Liang, “Making large language models                2403.05530
     better planners with reasoning-decision alignment,” 2024.      [52] S. Chen, H. Zhu, X. Chen, Y. Lei, G. Yu, and T. Chen,
     [Online]. Available: https://arxiv.org/abs/2408.13890               “End-to-end 3d dense captioning with vote2cap-detr,” in
[40] J.-J. Hwang, R. Xu, H. Lin, W.-C. Hung, J. Ji, K. Choi,             Proceedings of the IEEE/CVF Conference on Computer
     D. Huang, T. He, P. Covington, B. Sapp, Y. Zhou, J. Guo,            Vision and Pattern Recognition, 2023, pp. 11 124–11 133.
     D. Anguelov, and M. Tan, “Emma: End-to-end multimodal          [53] Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M.
     model for autonomous driving,” 2024. [Online]. Available:           Alvarez, “Is ego status all you need for open-loop end-to-
     https://arxiv.org/abs/2410.23262                                    end autonomous driving?” in 2024 IEEE/CVF Conference
     on Computer Vision and Pattern Recognition (CVPR),
     2024, pp. 14 864–14 873.
[54] L. Xiao, X. Yang, X. Lan, Y. Wang, and C. Xu, “Towards
     visual grounding: A survey,” 2024. [Online]. Available:
     https://arxiv.org/abs/2412.20206
[55] S. Wang, D. Kim, A. Taalimi, C. Sun, and W. Kuo,
     “Learning visual grounding from generative vision and
     language model,” in 2025 IEEE/CVF Winter Conference
     on Applications of Computer Vision (WACV), 2025, pp.
     8057–8067.
[56] X. Zhou, K. Larintzakis, H. Guo, W. Zimmer, M. Liu,
     H. Cao, J. Zhang, V. Lakshminarasimhan, L. Strand, and
     A. Knoll, “TUMTraf videoQA: Dataset and benchmark
     for unified spatio-temporal video understanding in traffic
     scenes,” in Proceedings of the 42nd International
     Conference on Machine Learning (ICML), 2025. [Online].
     Available: https://openreview.net/forum?id=Yfoi5O68rf
[57] U. contributors, “Planning-oriented autonomous driving,”
     https://github.com/OpenDriveLab/UniAD, 2023.
[58] Z. Li, W. Wang, E. Xie, Z. Yu, A. Anandkumar, J. M.
     Alvarez, P. Luo, and T. Lu, “Panoptic segformer: Delving
     deeper into panoptic segmentation with transformers,”
     2022. [Online]. Available: https://arxiv.org/abs/2109.03814
[59] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff,
     A. Lang, L. Fletcher, O. Beijbom, and S. Omari,
     “Nuplan: A closed-loop ml-based planning benchmark
     for autonomous vehicles,” 2022. [Online]. Available:
     https://arxiv.org/abs/2106.11810
[60] X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan, “Bench2drive:
     Towards multi-ability benchmarking of closed-loop end-to-
     end autonomous driving,” in NeurIPS 2024 Datasets and
     Benchmarks Track, 2024.
[61] D. Dauner, M. Hallgarten, T. Li, X. Weng,
     Z. Huang, Z. Yang, H. Li, I. Gilitschenski,
     B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta,
     “Navsim: Data-driven non-reactive autonomous vehicle
     simulation and benchmarking,” 2024. [Online]. Available:
     https://arxiv.org/abs/2406.15349
Appendix-OpenDriveVLA: Towards End-to-End Autonomous
    Driving with Large Vision Language Action Model

                                          VI. I MPLEMENTATION D ETAILS
A. Model Details
   1) Vision Encoder: To obtain accurate instance-level token representations, one option is to adopt
language-guided visual grounding tasks [54], [55], where visual regions are aligned with textual
descriptions with cross-modality supervision. However, such supervision is often ambiguous and imprecise,
especially in complex traffic environments where spatial accuracy is essential. This ambiguity arises from
the inherent subjectivity of textual annotations and the weak spatial constraints in general vision-language
datasets [56]. Moreover, they typically lack consistent object definitions and fail to capture structured
scene semantics, making them suboptimal for autonomous driving applications that demand precise object
localization and spatial understanding.
   Motivated by these, we adopt a vision-centric pretraining visual encoder and instance token decoder
based on high-quality autonomous driving datasets. The overall architecture of the vision encoder and
token extraction modules is illustrated in Figure 5.




Fig. 5: Overview of the visual encoder pretraining stage. The 3D vision-centric pretraining tasks include 3D object detection,
tracking, and BEV panoptic segmentation. The extracted features are projected into scene, agent, and map tokens for downstream
vision-language alignment and planning.

   We adopt the perception training stage introduced in UniAD [10], [57] to produce a robust visual
encoder for the traffic environment. It is trained via vision-centric 3D perception tasks, including 3D object
detection, tracking, and BEV panoptic segmentation. Compared to grounding via general vision-language
grounding tasks, vision-centric tasks offer more structured and semantically instance-level supervision,
which is critical for autonomous driving.
   The visual encoder follows a multi-view, query-based architecture, where multi-view images are
processed by a shared ResNet-101 backbone with FPN to extract multi-scale 2D features, which are
aggregated into BEV representation using BEVFormer [7]. A detailed summary of the visual module
setting in OpenDriveVLA is provided in Table VI.
   The vision-centric task is supervised using a combination of detection, tracking and segmentation losses
following original setting:

                                                  Lvis = Ltrack + Lmap                                                   (13)
 Module                    Input               Output               Architecture                     Configuration
 2D Backbone               Multi-view images   2D features          ResNet-101 + FPN                 Output strides: {1/8, 1/16, 1/32}
 BEV Encoder               2D features         BEV feature map      BEVFormer (6-layer encoder)      Hidden dim: 256; BEV: 200 × 200
 Scene Sampler             2D features         Global scene token   Adaptive max pooling             Output size: (6, 3, 5)
 Agent Query Transformer   BEV features        Agent tokens         TrackQFormer (6-layer decoder)   900 queries
 Map Query Transformer     BEV features        Map tokens           MapQFormer (6-layer decoder)     300 queries (3 thing + 1 stuff)

                           TABLE VI: Summary of components in the 3D visual perception module.



   The tracking loss Ltrack combines focal classification loss and L1 bounding box regression, optimized via
a Hungarian matching algorithm. The map segmentation loss Lmap includes classification, bounding box,
mask, and IoU losses over both thing and stuff classes. Map supervision follows Panoptic SegFormer [58].
This vision-centric supervision provides dense and structured grounding signals, enabling the encoder to
learn spatially precise and semantically aligned features.
   2) Structural Token Extraction: Based on multi-scale 2D features F2D from multi-view images and
aggregated BEV representation Fbev , OpenDriveVLA retains the structured perception architecture of
UniAD and extracts semantically meaningful tokens through three query-based modules. These modules
encode complementary aspects of the driving scene and serve as a compact interface for downstream
alignment and planning.
   Agent QueryTransformer: TrackQFormer decodes dynamic object-level semantics from the BEV
feature map fbev ∈ R200×200×D using learnable queries. Each token encodes an individual agent’s spatial
location, category, and motion trajectory. We retain the top-Na tokens filtered by confidence threshold:
   i
{vagent }i = 1Na = Qagent (fbev ). To improve efficiency and robustness, we filter out low-confidence agent
predictions based on detection scores. This selective mechanism reduces the number of input visual tokens,
leading to faster downstream processing and a more focused representation. Moreover, it helps mitigate
hallucination by suppressing uncertain or noisy detections from the vision input source.
   Map QueryTransformer: MapQFormer focuses on extracting static structural elements from the BEV
features via separate decoder heads for thing and stuff categories. It produces up to Nm map tokens,
each encoding elements such as lane dividers, road boundaries, and drivable areas: {vmap      j
                                                                                                 }j = 1Nm =
Qmap (fbev ).
   Global Scene Sampler: To capture holistic scene-level context, we apply adaptive max pooling over
the 2D features F2D ∈ R6×256×H×W from six camera views, compressing them into (3 × 5) spatial grids
per view. These tokens encode global contextual information, including weather, lighting, scene layout,
and traffic flow, which are difficult to infer solely from BEV-based modules. By abstracting high-level
semantics from raw visual features, the scene tokens serve as a complementary source of information to
the agent and map tokens, providing redundancy and enhancing robustness for downstream alignment and
planning: vscene = Qscene (f2D ) ∈ R90×D .
   Together, three token extractors provide a compact and structural encoding of the traffic environment,
covering dynamic agents, static infrastructure, and global context: Venv = {Vscene , Vagent , Vmap }.
   3) Large Language Action Module: We adopt Qwen2.5-Instruct [14] as the pre-trained LLM backbone
for generating structured driving actions. To balance model capacity and computational efficiency, we
evaluate three variants with 0.5B, 3B, and 7B parameters. Our implementation builds upon the LLaVA
NeXT framework [16], which enables the language model to perform structured cross-modality reasoning
and autoregressive action generation.
   To support structured multimodal OpenDriveVLA input, we extend the Qwen2.5 tokenizer with a set of
special tokens and token indices, as summarized in Table VII. These tokens formulate inputs into discrete
semantic segments, allowing the LLM to differentiate between scene-level context, dynamic agents, static
                  Token / Index                               Description
                  Placeholder Token Indices
                  IMAGE_TOKEN_INDEX                           Placeholder for global image token
                  SCENE_TOKEN_INDEX                           Placeholder for scene-level visual token
                  TRACK_TOKEN_INDEX                           Placeholder for agent-level track token
                  MAP_TOKEN_INDEX                             Placeholder for map-level token
                  OBJECT_TOKEN_INDEX                          Reserved placeholder for object token
                  High-Level Markers
                  <SCENE>, <TRACK>, <MAP>                     Denote visual token segments
                  <EGO>                                       Textualized ego-vehicle state
                  <COMMAND>                                   Driving command or query string
                  <trajectory>                                Start of autoregressive trajectory output
                  Token Delimiters (Start / End Wrappers)
                  <scene_start>, <scene_end>                  Scene token span delimiters
                  <track_start>, <track_end>                  Track token span delimiters
                  <map_start>, <map_end>                      Map token span delimiters
                  <ego_start>, <ego_end>                      Ego state string delimiters
                  <command_start>, <command_end>              Driving command delimiters
                  <traj_start>, <traj_end>                    Generated trajectory delimiters
                  Optional QA Format Tokens
                  <question_start>, <question_end>            Input question delimiters
                  <answer_start>, <answer_end>                Output answer delimiters

          TABLE VII: Extended special tokens and placeholder indices used in OpenDriveVLA’s LLM tokenizer.



maps, ego vehicle state, and high-level driving instructions. The input sequence passed to the LLM is
structured as:
                       <SYSTEM><SCENE><TRACK><MAP><EGO><COMMAND>
  During input construction, <SCENE>, <TRACK>, and <MAP> are replaced with projected visual tokens,
while <EGO> and <COMMAND> are filled with formatted textual strings.
B. Prompting Techniques


   1) System Prompt: The system prompt follows the BEV coordinate conventions introduced in GPT
Driver [38], while specifying the model role, driving objectives, and output format. It guides the LLM to
perform perception, reasoning, and trajectory generation in a unified manner, with optional user interaction.
The system prompt is prepended to all inputs during both training and inference to ensure consistent task
framing and instruction following.


   System Prompt

        <|im_start|>system
        You are Open-DriveVLA, an advanced vision-language driving model. Your core
            responsibilities include safe trajectory planning and interpretable
            decision-making. You generate collision-free driving plans while providing
             clear, logical explanations for user queries.

        Context:
         - Coordinates: X-axis is pointing to the right, and Y-axis is pointing to the
             front. You’re at point (0,0). All coordinates are in meters.
         - Objective: Generate a 3-second safe driving plan consisting of 6 waypoints,
             one every 0.5 seconds. Provide logical responses to user queries.

        Task:
        - Perception & Prediction: Analyze the driving environment using visual data.
            Identify road users and hazards and predict their motion.
        - Thought Process: Determine critical objects and assess potential hazards.
            Consider road constraints and traffic rules.
        - Trajectory Planning: Define the driving objective. Generate a safe, feasible
            3-second route consisting of 6 waypoints.
        - Explainability & User Interaction: If the user asks a question, provide a
            clear and logical response.

        Output Format:
        1. Trajectory (MOST IMPORTANT):
          - Format: <traj_start>[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]<
              traj_end>
        2. User Question Response (OPTIONAL):
          - Format: <answer_start> Answer to the user’s question <answer_end>




  2) Prompts for Hierarchical Feature Alignment: Each type of visual token is associated with a
separate captioning prompt. These prompts instruct the LLM to generate textual descriptions based on the
corresponding visual input segment.


   Instance Caption

        Please provide a caption and the BEV coordinate for the following object: <
            track_start><OBJECT><track_end>
   Map Caption

        Please provide a caption for the following map: <map_start><MAP><map_end>




   Scene Caption

        Please provide a caption for the following scene: <scene_start><SCENE><
            scene_end>




   3) Prompts for Driving Question Answering: All VQA datasets adopt a consistent structured prompt
format, integrating scene, agent, map, ego state, and historical trajectory components. Question Placehold-
ers are replaced with dataset-specific content during training and evaluation.


   Driving Question Answering Prompt

           Scene information: <scene_start><SCENE><scene_end>\nObject-wise tracking
               information: <track_start><TRACK><track_end>\nMap information: <
               map_start><MAP><map_end>\nEgo states: - Velocity (vx,vy): <Velocity
               Placeholder> - Heading Angular Velocity (v_yaw): <Angular Velocity
               Placeholder> - Acceleration (ax,ay): <Acceleration Placeholder> - Can
               Bus: <Can Bus Placeholder> - Heading Speed: <Speed Placeholder> -
               Steering: <Steering Placeholder>\nHistorical trajectory (last 2 seconds)
               : <Trajectory Placeholder> \nPlease answer the following question: <
               Question Placeholder>




  4) Prompts for Agent-Env-Ego Interaction: This prompt is used in Stage 2.5-modeling agent-
environment-ego interactions. The input includes structured visual context, historical ego trajectory, and
a queried target object, and the model is instructed to predict the future motion of the agent.


   Driving Question Answering Prompt

           <scene_start><SCENE><scene_end>\nObject-wise tracking information: <
               track_start><TRACK><track_end>\nMap information: <map_start><MAP><
               map_end>\nEgo Vehicle Token: <trajectory>\nPlease predict relative
               motion trajectory for the following object: <track_start><OBJECT><
               track_end>




   5) Prompts for Trajectory Planning Tuning: This prompt is used in the training stage 3 for trajectory
planning tuning, where the model is supervised to generate a 3-second driving plan based on structured
multi-modal context.
   Driving Question Answering Prompt

           Scene information: <scene_start><SCENE><scene_end>\nObject-wise tracking
               information: <track_start><TRACK><track_end>\nMap information: <
               map_start><MAP><map_end>\nEgo states: - Velocity (vx,vy): <Velocity
               Placeholder> - Heading Angular Velocity (v_yaw): <Angular Velocity
               Placeholder> - Acceleration (ax,ay): <Acceleration Placeholder> - Can
               Bus: <Can Bus Placeholder> - Heading Speed: <Speed Placeholder> -
               Steering: <Steering Placeholder>\nHistorical trajectory (last 2 seconds)
               : <Trajectory Placeholder>\nMission goal: <Command Placeholder>\
               nPlanning trajectory: <trajectory>



C. Training and Inference Details
   1) Training Configuration.: OpenDriveVLA is trained on 4 NVIDIA H100 GPUs with a per-GPU
batch size of 1. The full training process takes approximately two days for the 0.5B variant. All training
stages use mixed-precision (bf16) and gradient checkpointing to improve memory efficiency and speed.
The 2D vision backbone is frozen during the final end-to-end stage. LLM parameters are fully tuned
unless specified otherwise.

                                        Stage 1         Stage 2         Stage 2.5                     Stage 3
                    Tunable parts      projector      projector,LLM   projector,LLM        Full model (except 2D encoder)
               Trainable Params (MB)      3.1              496.9           496.9                        552.6
                 Per-GPU batch size        1                 1               1                            1
       Train
                        GPUs               4                 4               4                            4
                     LR (ψvision )         –                 -               -                       1 × 10−5
                  LR ({θproj/LLM })    1 × 10−4         1 × 10−5        1 × 10−5                     1 × 10−5
                       Epochs              1                 1               1                            1

                       TABLE VIII: Multi-stage training hyperparameters of OpenDriveVLA-0.5B.

  2) Inference Efficiency: Table IX reports the inference performance of OpenDriveVLA across three
LLM scales (0.5B, 3B, 7B) under bf16 precision on a single NVIDIA A100 GPU. Evaluation is conducted
on the NuScenes trajectory validation set with 6019 samples.

           Model            LLM              GPUs        Speed (Sample/s)    Latency (s)        Max VRAM (GB)
           0.5B     Qwen2.5-0.5B-Instruct         1            0.74                 1.36                1.56
           3B        Qwen2.5-3B-Instruct          1            0.54                 1.85                7.35
           7B        Qwen2.5-7B-Instruct          1            0.57                 1.74               17.15

          TABLE IX: Inference details of OpenDriveVLA under BF16 precision on a single NVIDIA A100 GPU.
                                               VII. DATASETS D ETAILS
A. Dataset Overview
  We utilize a curated set of multimodal driving datasets derived from nuScenes [46] to support the
multi-stage training of OpenDriveVLA, covering object-level captioning, visual question answering,
scene description, and decision reasoning. An overview of the datasets used in Stage 1 and Stage 2
of OpenDriveVLA training is provided in Table X.

    Dataset          #Train      #Val                    Annotation Types                                    Type
    TOD3Cap          1.89M      410K     Object Caption: appearance, motion, relationships           Dense Captioning
    nuScenes-QA      376K        83K      existence, counting, object, status, comparison        Visual Question Answering
    nuCaption        348K        72K          Scene Caption: layout, agents, hazards                 Scene Description
    nuX               28K         6K              Driving Decision Justification                   Reasoning & Narration

          TABLE X: VQA Datasets in OpenDriveVLA Stage 1/2 training, with sample counts, annotation, and types.

   TOD3Cap [43]: TOD3Cap introduces the task of object-centric dense captioning in 3D driving scenes.
It provides 2.3M human-verified natural language descriptions for over 64K objects across 850 nuScenes
[46] scenes, covering appearance, motion, context, and inter-object relations. Each caption captures fine-
grained semantics including what, where, how, and why, facilitating object-level alignment between 3D
perception and language.
   nuScenes-QA [45]: nuScenes-QA is a large-scale visual question answering benchmark tailored for
autonomous driving. It contains 460K question–answer pairs over 34K multimodal driving scenes with
synchronized images and LiDAR. Questions are generated using scene graphs and structured templates,
spanning reasoning types such as existence, counting, attribute queries, spatial relations, and comparisons.
   nuCaption [44]: nuCaption is a 3D scene captioning dataset constructed from nuScenes. It comprises
both image-text and LiDAR-text pairs, with both global and viewpoint-specific captions describing traffic
layout, object interactions, and potential risks. By aligning 3D spatial representations with language,
nuCaption allows 3D captioning and scene-level reasoning.
   nuX [19]: nuX is a human-annotated explanation dataset designed for interpretable autonomous driving.
For each keyframe, it provides natural language explanations combining factual narration (what is
happening) and causal reasoning (why it is happening), grounded in the outputs of perception, prediction,
and planning modules. It supports the development of driving models with aligned and transparent
decision-making with textual interpretability.
B. Dataset Sample Visualization

                              Stage 1                        Stage 2                    Stage 2.5                   Stage 3
      Annos      2D/3D Caption, Scene Description          Driving QA                   Trajectory                Trajectory
       Task       Hierarchical Feature Alignment    Driving Instruction Tuning   Agent-Env-Ego Interaction   Trajectory Planning
      Source      TOD3Cap, nuScenes, nuCaption      nuCaption, nuS-QA, nuX              nuScenes                  nuScenes
     #Samples                  536k                            566k                       459k                       28k

TABLE XI: Overview of the Dataset for multi-stage training pipeline of OpenDriveVLA, detailing the specific tasks, annotation
types, data sources, and number of training samples.

   An overview of the multi-stage training data is provided in Table XI, covering task types, annotations,
data sources, and sample counts. We provide visualization examples of gannotations we used during the
hierarchical feature alignment training process of OpenDriveVLA, as shown in Figure 6 and Figure 7.
Fig. 6: Example 1 of Stage 1 dataset visualization from nuScenes Sample 90. Green boxes show instance token captions; Yellow
boxes indicate map token captions; Red boxes represent scene token captions.
Fig. 7: Example 2 of Stage 1 dataset visualization from nuScenes Sample 3287. Green boxes show instance token captions; Yellow
boxes indicate map token captions; Red boxes represent scene token captions.
                                        VIII. R ESULTS AND D ISCUSSIONS
A. Results of Driving Question Answering
   Figure 8 and Figure 9 show representative examples of OpenDriveVLA’s responses to diverse driving-
related questions drawn from three datasets: nuScenes-QA, nuCaption, and nuX. These qualitative results
highlight the model’s multi-level reasoning capabilities across perception, commonsense understanding,
and contextual decision-making.
   In Figure 8, the model provides a narration of the ego vehicle’s behavior while approaching and
passing over a speed bump, accompanied by a causal explanation grounded in the surrounding scene. This
demonstrates the model’s ability to generate coherent natural language outputs that reflect its interpretation
of driving decisions. However, in the scene-level description, the model hallucinates that there are no
pedestrians visible in the right-front view, despite the presence of one pedestrian in the image.




Fig. 8: Example 1 of Stage 2 dataset visualization from nuScenes Sample 6801. Green boxes show nuX dataset predictions; blue
boxes are from nuScenes-QA dataset, while yellow boxes are from nuCaption.
   Figure 9 depicts a more complex urban scenario involving multiple interacting agents near an
intersection. The model identifies the ego vehicle’s maneuver and provides a causal explanation grounded
in its spatial context, correctly localizing the vehicle as it turns at the junction. In the scene-level
description, the model can ground key static and dynamic elements, including the presence of a traffic
light, pedestrians near a crosswalk, and parked vehicles along the roadside. However, the caption exhibits
positional inaccuracy when referencing the camera view, particularly in the identification of the front view.
This may be attributed to the model being trained with scene-level captions in a camera-agnostic fashion,
without explicit supervision for distinguishing between different camera perspectives.




Fig. 9: Example 2 of Stage 2 dataset visualization from nuScenes Sample 6801. Green boxes show nuX dataset predictions; blue
boxes are from nuScenes-QA dataset, while yellow boxes are from nuCaption.
B. Results of Agent Motion Prediction
   Figure 10 and Figure 11 present qualitative results of OpenDriveVLA on the agent motion prediction
task of Stage 2.5, where the model jointly reasons over agent trajectories, environment, and ego vehicle
state.




Fig. 10: Example 1 of agent motion prediction result of stage 2.5 Agent-Environment-Ego interaction modeling on nuScenes
validation set Sample 8587.


   Figure 10 depicts a night-time urban intersection with multiple static and dynamic agents under low-
light conditions. The figure highlights two turning vehicles. The predicted motion for the white sedan
on the right side of the scene is consistent with the lane orientation and road semantics. In contrast, the
white vehicle ahead of the ego car, the model predicts a trajectory curving toward the right. Yet, given
the surrounding road geometry and lane configuration, a left turn would be the more plausible maneuver.




Fig. 11: Example 2 of agent motion prediction result after stage 2.5 Agent-Environment-Ego interaction modeling on nuScenes
validation set Sample 32496.

  In Figure 11, the model predicts motion trajectories for multiple agents in a curved road scenario
during daytime. Notably, the visualization highlights two predicted trajectories: one for a gray sedan in
the front view, and another for a vehicle observed in the back view. Both predictions reflect distinct motion
patterns consistent with the road layout and surrounding context. This example demonstrates the model’s
capacity to capture plausible motion uncertainty and agent-specific intention under partially observable
environments.
C. Planning Results
  1) Comparison with Prior Methods.: Figure 12 presents a qualitative comparison between the open-
loop planning results of OpenDriveVLA and UniAD [10] in a challenging narrow-road scenario with
multiple parked vehicles.




Fig. 12: Qualitative comparison of open-loop planning. Top: Planning results of UniAD [10]. Bottom: Planning results of
OpenDriveVLA-7B (Ours). The agent motion prediction results are visualized after the agent-environment-ego interaction stage.

   In this scenario, UniAD exhibits overly sensitive reactions to the parked vehicles on the right side of
the road, resulting in unstable and zigzagging planned trajectories. In contrast, OpenDriveVLA generates
smoother and more consistent motion plans that better follow the intended driving path while maintaining
safe clearance from surrounding objects. This demonstrates a stronger ability to reason about scene
semantics and generate robust and spatially grounded trajectories.
   2) Qualitative Results of Driving Instruction Following.: Figure 13 and Figure 14 illustrate the trajectory
planning behavior of OpenDriveVLA under different conditional driving instructions, including left turn,
right turn, and keep forward.
   Figure 13 illustrates a complex urban intersection where the ground-truth instruction in nuScenes dataset
is going forward. The predicted paths remain safe and well-aligned with the scene context under new
driving instructions, demonstrating the model’s capacity to flexibly interpret high-level commands while
maintaining collision-free behavior. Similarly, Figure 14 shows another intersection scenario in a suburban
environment. Despite differences in layout and visual context, OpenDriveVLA consistently adapts its
planned motion to match the given instruction safely. These results demonstrate the instruction following
ability of OpenDriveVLA across various environments and its robustness in instruction-conditioned
planning.




                              (a) OpenDriveVLA planning under instruction turning left




                             (b) OpenDriveVLA planning under instruction turning right




                     (c) OpenDriveVLA planning under ground-truth instruction: keep forward
     Fig. 13: Example 1 of trajectory planning results of OpenDriveVLA after stage 3 with different driving instructions.
                         (a) OpenDriveVLA planning under instruction turning left




                        (b) OpenDriveVLA planning under instruction turning right




                (c) OpenDriveVLA planning under ground-truth instruction: keep forward
Fig. 14: Example 2 of trajectory planning results of OpenDriveVLA after stage 3 with different driving instructions.
D. Discussions
   1) Results Regarding Model Size.: The results in Main Content (Table 1,2,3) show that, while increasing
model size generally leads to improved performance, the larger OpenDriveVLA-7B model does not
consistently outperform the smaller 0.5B and 3B variants across all benchmarks. Specifically, on the Nu-
X dataset, the 0.5B model achieves obviously higher CIDEr and ROUGE-L scores than the 7B version.
In open-loop trajectory planning, the differences in average L2 error and collision rate among the three
models are relatively small. Although the 7B model remains competitive overall, its performance gains
are not always significant and, in some cases, fall behind those of the smaller models.
   There are several possible reasons based on our experimental observations. First, the performance gains
associated with scaling large language models typically depend on access to extensive and diverse training
data. As detailed in Table X, the instruction tuning data and domain-specific driving annotations used in
our current setup may not be sufficient to fully leverage the representation capacity of the 7B model,
especially in the autonomous driving domain, where large-scale open-source vision language datasets are
still very few. Hence, the expected improvements from model scaling are not consistently realized across
all tasks.
   Second, larger autoregressive models tend to rely more heavily on language priors during generation. In
structured multimodal reasoning tasks, this reliance may weaken the model’s ability to maintain accurate
visual grounding, particularly in scenarios requiring fine-grained spatial understanding or factual precision.
Smaller models, though with limited capacity, can retain stronger coupling between vision and language
inputs, which can lead to more stable behavior under constrained supervision. This also suggests that
additional high-quality visual-textual data is necessary to facilitate further development.
   Third, the optimization dynamics of larger models can be more sensitive to hyperparameters and
experimental settings in domain-specific tasks. Without sufficient diversity in the training dataset, the
larger model may overfit to dominant patterns in the training set, resulting in reduced robustness and
generalization during inference.
   In summary, model size alone is not a robust indicator of performance in this domain-specific
autonomous driving task with limited data sources. Instead, data quantity and quality, and multi-stage
training strategies play a critical role in achieving robust and scalable performance.
   2) Current Limitations: While OpenDriveVLA achieves promising results, several limitations remain.
First, the model relies on implicit reasoning patterns acquired through instruction tuning, without explicit
step-by-step chain-of-thought deduction during inference. Although this design helps maintain inference
efficiency, it may eteriorate the reasoning performance and limit the model’s ability to handle complex
scenarios. Second, the autoregressive nature of LLM introduces inference latency. Even with careful
control of token lengths, the sequential decoding process remains a bottleneck for deployment in high-
speed driving scenarios. Further improvements in model quantization and improved decoding strategies
will be beneficial to meet real-time requirements. Finally, our current planning evaluation is limited to an
open-loop setting. This does not capture the interactive feedback dynamics of realistic driving and may
lead to overestimated robustness, underscoring the importance of extending OpenDriveVLA to closed-loop,
language-aware simulation environments.
   3) Extension to Closed-Loop Planning: As mentioned, the current planning evaluation of Open-
DriveVLA is conducted in an open-loop setting on the nuScenes dataset. While this setup allows for
controlled and reproducible benchmarking, it does not account for the feedback dynamics of ego-agent
interactions in real-world traffic. Prior studies also show that nuscences open-loop benchmark may lead
to overly optimistic conclusions [34]. Several established closed-loop benchmarks, such as nuPlan [59],
Bench2Drive [60], and NaviSim [61], have been introduced for more reliable planning assessment.
   Unfortunately, these benchmarks currently lack vision-language annotations that are central to the
training of domain-specific LLM-based Autonomous Driving models. While nuScenes-derived datasets
such as TOD3Cap, nuScenes QA, nuCaption, and nuX provide rich multimodal supervision in the form
of instance-level and scene-level captions and question–answer pairs. These additional datasets are the
result of extensive annotation efforts from multiple prior works and serve as a foundation for both training
and evaluating LLM-based autonomous driving models. To extend OpenDriveVLA toward closed-loop
evaluation, future work involves building an automated data generation pipeline that (semi-)automatically
generates instance-level, map-level, and scene-level descriptions, as well as question–answer pairs linked
to driving behaviors.
   Despite the limitations, OpenDriveVLA still demonstrates strong performance across both open-loop
planning and multiple driving VQA benchmarks, including nuScenes-QA, nuCaption, and nuX. The model
consistently exhibits robust scene understanding, accurate instruction following, and semantically grounded
action generation. These results validate the effectiveness of the proposed model and training techniques,
and provide a solid experimental foundation for future extensions into closed-loop evaluation frameworks.

<!-- MinerU conversion could not be completed: the configured MINERU_TOKEN was rejected with API error A0202 (user authenticate failed). This file is a local pdftotext fallback extracted from the source PDF. Re-run mineru_doc2md after replacing the token to obtain the MinerU-formatted Markdown. -->
