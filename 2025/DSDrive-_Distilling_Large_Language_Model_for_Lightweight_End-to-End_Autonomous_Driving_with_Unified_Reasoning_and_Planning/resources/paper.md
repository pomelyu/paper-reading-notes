                                                                                                                                                                                  1




                                           DSDrive: Distilling Large Language Model for
                                         Lightweight End-to-End Autonomous Driving with
                                                 Unified Reasoning and Planning
                                                                            Wenru Liu, Pei Liu, and Jun Ma, Senior Member, IEEE



                                           Abstract—In conventional end-to-end frameworks for Au-                 development of AVs [7]. In this context, the application of




arXiv:2505.05360v1 [cs.RO] 8 May 2025
                                        tonomous driving (AD), the underlying cognitive processes are             LLMs to AD, particularly concerning their integration with
                                        inadequately addressed. While large language models (LLMs)                the E2E AD framework, has garnered substantial research
                                        offer improved understanding and reasoning capabilities, inte-
                                        grating them into AD systems poses two major challenges. First,           interest [8]. Essentially, LLMs effectively address the limi-
                                        there is an invariably significant discrepancy between the high           tations of the traditional E2E AD framework, which lacks a
                                        computational demands of LLMs and the high efficiency required            genuine understanding of the driving task and is deficient in
                                        for autonomous vehicles. Second, although LLMs can generate               explainability [9]. This is because LLMs provide a compelling
                                        high-quality semantic reasoning, it remains an open challenge to          solution through their cognitive processing capabilities and
                                        map high-level textual reasoning to low-level trajectory planning
                                        for autonomous vehicles. To deal with these issues, we present            language generation abilities. The former enables nuanced
                                        DSDrive, a streamlined end-to-end paradigm tailored for in-               contextual reasoning for the driving task [10], while the lat-
                                        tegrating the reasoning and planning of autonomous vehicles               ter facilitates a natural human-computer interaction interface.
                                        into a unified framework. DSDrive leverages a compact LLM                 However, deploying an LLM-based AD framework remains
                                        that employs a distillation method to preserve the enhanced               challenging. Intrinsically, the computational demands of large-
                                        reasoning capabilities of a larger-sized vision language model
                                        (VLM). To effectively align the reasoning and planning tasks,             scale language models create substantial bottlenecks in real-
                                        a waypoint-driven dual-head coordination module is further                time decision-making, and this imposes substantial constraints
                                        developed, which synchronizes dataset structures, optimization            on the memory and processing latency of critical planning
                                        objectives, and the learning process. By integrating these tasks          functions [11].
                                        into a unified framework, DSDrive anchors on the planning                    While downsizing the models provides an underpinning
                                        results while incorporating detailed reasoning insights, thereby
                                        enhancing the interpretability and reliability of the end-to-end          basis, the current research efforts demonstrate that the capabil-
                                        pipeline. DSDrive has been thoroughly tested in closed-loop               ities of LLMs are generally related to model size [12], which
                                        simulations, where it performs on par with benchmark models               implies that directly employing compact-sized architectures
                                        and even outperforms in many key metrics, all while being                 may compromise essential cognitive functions required for
                                        more compact in size. Additionally, the computational efficiency          safe autonomous operations. Recent advances in model op-
                                        of DSDrive (as reflected in its time and memory requirements
                                        during inference) has been significantly enhanced. Evidently thus,        timization offer promising pathways forward. Notably, knowl-
                                        this work brings promising aspects and underscores the potential          edge distillation techniques exemplified by DeepSeek-R1 [13]
                                        of lightweight systems in delivering interpretable and efficient          demonstrate that sophisticated reasoning capabilities can be
                                        solutions for AD.                                                         effectively transferred from large-sized teacher models to
                                          Index Terms—End-to-end autonomous driving, vision language              compact-sized student architectures. This approach enables
                                        model, large language model, chain-of-thought.                            the creation of compact yet capable LLM implementations
                                          Video of the experiments: https://youtu.be/op8PzQurugY                  that maintain sufficient decision-making competence while
                                                                                                                  addressing computational constraints. This implication holds
                                                                                                                  significant relevance for the application of LLMs in the context
                                                                I. I NTRODUCTION
                                                                                                                  of of E2E AD. The core challenge is to design efficient knowl-
                                           Autonomous vehicles (AVs) have demonstrated pervasive                  edge transfer mechanisms that can endow compact models
                                        power in modern intelligent transportation systems, which                 with advanced cognitive capabilities beyond their inherent
                                        significantly enhances on-road driving safety and travel ef-              performance limitations; and this can be achieved through
                                        ficiency [1]. The end-to-end (E2E) autonomous driving (AD)                strategic architectural design and targeted training paradigms.
                                        framework [2], leveraging large-scale driving datasets [3] and            Essentially, knowledge distillation has proven effective in rec-
                                        advanced learning methodologies [4], plays an important role              onciling discrepancies in model size and capability. However,
                                        in advancing the forefront of AVs. This fully differentiable              it does not resolve the inherent discrepancy between the
                                        framework processes raw sensor inputs to output low-level                 high-level textual reasoning offered by LLMs and the low-
                                        control through joint task optimization. In recent years, large           level trajectory planning required by AVs [14]. While LLMs
                                        language models (LLMs) [5], [6] present new frontiers for the             excel at semantic reasoning, they are not optimized for the
                                                                                                                  numerical computation and spatial reasoning tasks that are
                                           Wenru Liu, Pei Liu, and Jun Ma are with The Hong Kong University of
                                        Science and Technology, China (e-mail: wliu354@connect.hkust-gz.edu.cn;   essential for trajectory planning in AD [15]. This mismatch
                                        pliu061@connect.hkust-gz.edu.cn; jun.ma@ust.hk).                          has led most existing LLM-based AD research to rely on
                                                                                                                                  2



open-loop evaluations [16]. The transition of LLM-based E2E         II delves into the related work, offering essential background
AD frameworks to closed-loop planning is imperative for             on E2E AD and the application of LLMs in AD. Section
practical AD deployment [17]. The E2E AD framework serves           III outlines the methodology, detailing the framework com-
an effective approach to coordinate different tasks toward          ponents and the distillation-based training scheme. In Section
closed-loop planning. By focusing on the ultimate driving           IV, we present the experiments and pertinent results, which
objective, the planning result aligns various tasks within a        vividly demonstrate the effectiveness and performance of our
unified framework, which ensures that they can collectively         proposed approach. Finally, Section V provides concluding
advance toward the shared goal [18]. Motivated by this, it          remarks and discusses potential directions for future research.
leaves an interesting problem to integrate LLMs into the E2E
AD framework from a planning-oriented perspective [19].
   In this work, we propose a novel lightweight E2E AD                                  II. R ELATED W ORK
framework to tackle the challenges of computational efficiency
and the disconnection between reasoning and planning. This          A. End-to-End Autonomous Driving
framework leverages compact LLMs to achieve high reasoning
and planning performance comparable to large-sized systems.            One of the motivations behind the E2E AD framework
Our approach hinges on two key strategies: Firstly, we utilize a    is to develop a unified approach that integrates perception,
distillation method to enhance the compact LLM’s capabilities       decision-making, and planning tasks, optimizing the entire
as the core of our AD system. This entails externalizing the        pipeline for the ultimate driving performance. This goal has
reasoning power of a large-sized Vision Language Model              been pursued through innovative architectural approaches. ST-
(VLM) via a chain-of-thought (CoT) prompting process [20].          P3 [21] advances spatial-temporal feature learning to unify
We generate structured datasets that explicitly capture think-      scene understanding across multiple driving tasks. Building
and-answer reasoning, enabling the transfer of reasoning skills     on this foundation, UniAD [22] establishes an integrated
from foundational models to specialized systems. We propose         framework that synergizes perception and prediction modules
a waypoint-driven dual-head coordination module that facil-         to enhance planning capability. Architectural innovations con-
itates the appropriate alignment of reasoning and planning          tinue to reshape system design, aiming to advance the E2E
tasks. This module entails the co-design of the training dataset,   AD framework with more resourceful skills. DriveTransformer
ensuring that both tasks are grounded in homogeneous data.          [23] achieves unified task interaction through parallel pro-
It innovatively incorporates the planning outcome as the final      cessing architectures with sparse representations, improving
answer for the reasoning process, thereby establishing a unified    operational stability. Inspired by cognitive mechanisms, the
optimization objective for the two tasks. The two tasks are         cascaded decision-making framework incorporates multi-stage
trained jointly within the E2E framework, simultaneously            safety verification for action refinement [24]. Complementary
optimizing for the quality of the reasoning answers and             approaches include a planning-centric coordination system
the accuracy of waypoint predictions. By interlinking these         [25] and a vision-based multi-agent collaboration model [26],
tasks, our framework supports closed-loop AD with explicit          which enhances emergency response through cross-vehicle
reasoning, thereby enhancing the explainability and reliability     perception. ReasonNet [27] implements hierarchical reasoning
of autonomous systems. The contributions of our work are            architectures for improved scenario understanding and traffic
summarized as follows:                                              participant behavior prediction. The research community has
                                                                    concurrently explored alternative paradigms to pushing the
  • We present DSDrive, a lightweight E2E AD framework              limit of the E2E AD framework. VAD [28] investigates
    that employs a compact LLM to process multi-modal               vectorized scene representations to reduce map dependency in
    inputs for explicit reasoning and closed-loop planning.         motion planning, while its successor VADv2 [29] introduces
    Specifically, we utilize knowledge distillation to empower      probabilistic planning strategies and environmental tokeniza-
    the compact LLM to undertake the reasoning and plan-            tion for enhanced uncertainty handling.
    ning tasks, thereby improving its overall performance.
                                                                       Despite these innovations, a fundamental limitation of the
  • We present a novel waypoint-driven dual-head coordina-
                                                                    E2E approach is its lack of structured reasoning mechanisms to
    tion module that bridges high-level reasoning and low-
                                                                    decompose complex scenarios into logical decision pathways.
    level trajectory planning. By integrating waypoints into
                                                                    This shortcoming can be ascribed to two perspectives. Firstly,
    the explicit reasoning process, we establish a unified
                                                                    the traditional deep learning network does not have intrinsic
    objective for both tasks and facilitate their collaborative
                                                                    capabilities to perform sophisticated reasoning or interpreta-
    progression toward the overarching driving goal.
                                                                    tion. Secondly, the reasoning patterns necessary for the system
  • We perform extensive closed-loop simulations in CARLA
                                                                    to learn a comprehensive understanding are typically absent in
    to showcase the efficacy of knowledge distillation and the
                                                                    the training data. As discussed in many previous works [30],
    dual-head coordination module. The proposed DSDrive
                                                                    [31], [32], the acquisition of reasoning ability cannot rely on
    achieves driving performance comparable to larger-sized
                                                                    merely scaling up the size of neural network models and data.
    benchmark models, and even outperforms them in several
                                                                    Therefore, it necessitates a paradigm shift towards integrating
    key metrics, particularly in terms of enhanced computa-
                                                                    structured reasoning mechanisms in both model architecture
    tional efficiency.
                                                                    and training data to address the limitations of the current E2E
  The remainder of this paper is organized as follows: Section      pipeline for AD.
                                                                                                                                 3



                           TABLE I                               leverages a large-sized VLM, while the driving model utilizes
    C OMPARISON OF REPRESENTATIVE WORKS IN LLM- BASED AD.        a compact-sized LLM. Our method employs distillation as
                                                                 the learning mechanism, enabling the compact-sized driving
         Method         E2E   Closed-loop   Explicit reasoning
                                                                 model to acquire reasoning capabilities from the VLM, as
      DriveLM [37]       ✗        ✗                 ✓
        VLP [38]         ✗        ✗                 ✗
                                                                 elaborated in Section III-C.
      WiseAD [39]        ✓        ✗                 ✓
     Sce2DriveX [40]     ✓        ✗                 ✓
        Senna [42]       ✓        ✗                 ✓
                                                                 A. Reasoning Model
      LMDrive [41]       ✓        ✓                 ✗               Our method utilizes two types of inputs: image I and
          Ours           ✓        ✓                 ✓
                                                                 text X. The image input is a sequence represented as I ∈
                                                                 RT ×H×W ×C , where T is the sequence length, H, W , and
                                                                 C represent the height, width, and number of channels, re-
B. LLMs in Autonomous Driving                                    spectively. The textual inputs are subdivided into navigation
   It is readily reported in the recent works that the inte-     instructions and questions.
gration of LLMs in the E2E AD framework has demon-                  At a high level, the process of utilizing the VLM in the
strated advancements in environmental reasoning and decision-    reasoning model can be defined as:
making. To address interpretability challenges pertinent to
E2E AD, RAG-Driver [33] employs retrieval-augmented mul-                        Avlm = V (I, Xnavi , Xprompt ),               (1)
timodal LLMs to generate human-understandable explanations       where V (·) represents the VLM, which processes visual and
for driving actions while maintaining precise control sig-       textual inputs to generate answers Avlm . To enable the distil-
nal prediction, demonstrating notable zero-shot generalization   lation of reasoning capabilities from the VLM to the driving
across unfamiliar environments. In planning optimization, Al-    model, we ensure the VLM processes driving image sequences
phaDrive [34] combines GRPO-based reinforcement learning         and produces multi-faceted analyses aligned with the driving
with reasoning strategies to enhance training efficiency and     model’s input structure.
multimodal planning capability. PRIMEDrive-CoT [35] ad-             Specifically, we employ Qwen2.5-VL-max [43], a state-of-
vances safety-critical decision-making through Bayesian graph    the-art open-sourced VLM, to generate a structured reasoning
neural networks integrated with CoT reasoning, employing         dataset regarding scene understanding, key object identifica-
visual attention mechanisms to improve transparency in risk      tion, and informed driving decision-making. As Qwen2.5-VL-
assessment. GPT-Driver [36] explores human-AI interaction        max is a general-purpose VLM designed to handle a wide
paradigms using ChatGPT-3.5 for natural language command         range of tasks, we implement a structured CoT strategy to
interpretation in trajectory planning scenarios.                 refine its specialization for AD, with the following considera-
   Recent advancements in integrating LLMs into AD systems       tions:
have yielded several notable frameworks. DriveLM [37] pi-          1) Scenario Comprehension: Grasp the driving environ-
oneers a multimodal fusion architecture that aligns linguis-           ment by analyzing elements such as weather, time of day,
tic reasoning with sensor inputs through CoT prompting.                road type, and road conditions.
VLP [38] employs vision-language pre-training to enhance           2) Key Item Description: Detail the key item by its name,
scene understanding. WiseAD [39] introduces knowledge                  position, features, and the potential risks it poses to the
distillation to compress LLM knowledge into lightweight                ego vehicle’s operation.
policy networks. Sce2DriveX [40] proposes a scenario-              3) Strategic Driving Plan Formulation: Develop a well-
aware encoder-decoder framework that improves generaliza-              thought-out driving plan to navigate the scenario effec-
tion across driving conditions. LMDrive [41] develops an               tively.
E2E architecture with spatiotemporal tokenization for trajec-      4) Human-Interpretable Explanation Generation: Create
tory prediction, but it lacks an explicit reasoning mechanism          understandable explanations for the driving decisions and
within its design. As summarized by Table I, although these            actions taken.
approaches have shown progress in open-loop evaluations by
                                                                    Thus far, VLM has primarily provided insights centered
improving multimodality alignment in perception and reason-
                                                                 on high-level reasoning. However, planning tasks in AD
ing, they collectively fail to adequately address closed-loop
                                                                 generally involve predicting waypoints for trajectory planning
driving scenarios. The closed-loop driving demands alignment
                                                                 and subsequent vehicle control. Accurately forecasting target
between reasoning and planning tasks, and also places stricter
                                                                 waypoints for precise maneuvers poses a significant challenge
requirements on computational resources. This limitation un-
                                                                 for general-purpose VLMs. This is mainly because spatial
derscores the critical need for developing an E2E AD system
                                                                 positioning has long been a recognized weakness in these mod-
that harmonizes lightweight LLM deployment with robust
                                                                 els. Consequently, additional strategies are needed to bridge
closed-loop planning capabilities.
                                                                 the gap between abstract reasoning and concrete planning tasks
                                                                 in AD. We have innovatively utilized ground-truth waypoints
                       III. M ETHODOLOGY                         to connect the reasoning process to the planning task. Our
   As an overview, our method consists of two main compo-        inspiration comes from the training template in [13], which
nents: a reasoning model detailed in Section III-A, and an E2E   emphasizes a step-by-step thinking process followed by a final
driving model outlined in Section III-B. The reasoning model     answer. This dataset design aims to prevent shortcuts to the
                                                                                                                                                4



                                                                                across the framework. While the questions for the reason-
                                                                                ing model, Xvlm , consist of a series of questions regarding
                                                                                subtasks in AD, the question text Xqn in the driving model
                                                                                is a straightforward commanding sentence like “think and
                                                                                answer”.
                                                                                   As illustrated in Fig. 2, the driving model integrates several
                                                                                specialized components to enable comprehensive reasoning
                                                                                and planning: (1) Visual encoder and tokenizer for processing
                                                                                the inputs, (2) Q-former for aligning visual features and textual
                                                                                tokens, (3) LLM backbone, and (4) Multi-tasks outputs with
                                                                                a dual-head design of the reasoning and planning tasks.
                                                                                   1) Inputs Encoding: Both the image and text inputs are
                                                                                processed to prepare tokens for use by the LLM. We adopt
                                                                                a pre-trained vision encoder from [41], which effectively
                                                                                transforms multi-view, multi-modality sensor data into em-
                                                                                bedded features. The image sequence I ∈ RT ×H×W ×C is
                                                                                processed through the visual encoder to extract visual features
Fig. 1. The dataset for distillation is prepared with an explicit think-and-    Fv ∈ RT ×Nv ×Dv , where Nv is the number of visual tokens
answer process. This design enables the model to learn complex reasoning        and Dv is the hidden dimension. Alongside the visual encoder,
steps before generating a final answer, enhancing its learning of nuanced
driving tasks. The dataset is diverse and realistic, covering various driving   we apply the LLaMA tokenizer [44] to convert the textual
scenarios to improve model robustness and generalization.                       inputs, including the navigation instruction and questions, into
                                                                                textual tokens Hl ∈ RNl ×Dl , where Nl denotes the length of
                                                                                the textual sequence and Dl is the hidden dimension of the
answer by requiring the explicit expression of reasoning steps.                 language model embeddings. To ensure temporal alignment
In our work, this approach serves a more critical purpose.                      with the image sequence, text tokens are repeated T times,
By embedding ground-truth waypoints within the think-and-                       corresponding to each frame in the sequence.
answer process, we have seamlessly integrated the planning                         2) Aligning Visual and Textual Modalities: A critical aspect
task as a natural outcome of the reasoning process.                             of processing visual and textual inputs is ensuring effective
   We summarize our approach in leveraging the reason-                          cross-modal alignment. To achieve this, we employ the Q-
ing model to prepare the training dataset in Fig. 1. The                        Former architecture from BLIP-2 [45]. The Q-Former strate-
complete prompts for the VLM are defined as Xprompt =                           gically queries encoder outputs to align visual features with
[S, Q1 , Q2 , Q3 ], where S represents the system prompt, which                 textual tokens, bridging the visual and text modalities. To
outlines the task’s user-assistant interactive setting along with               enhance efficiency, we generate a set of learnable queries
the input-output structure. This is followed by a sequence                      Q ∈ RK×Dl , where K is the number of query tokens. These
of task-specific prompts, Q1 , Q2 , Q3 , which guide the VLM                    tokens reduce the dimensionality of visual tokens, helping the
through coherent progression from perception to prediction                      model focus on relevant visual input aspects. Next, we apply
and decision-making. The generated answer by the VLM is                         a cross-attention mechanism with attention masks on query
then concatenated with the navigation instruction and ground-                   tokens, text tokens, and image embeddings. The Q-Former’s
truth waypoints to form a structured think-and-answer reason-                   output, specifically the last hidden states, passes through a
ing process, which constitutes the training dataset. Remark-                    linear layer. This projects the visual features into the language
ably, the construction of our training dataset emphasizes the                   model’s embedding space, ensuring compatibility with the
externalization of reasoning mechanisms that are inherent to                    LLM backbone.
the VLM but are typically absent in conventional driving mod-                      3) LLM Backbone: We employ LLaMA-1B as the LLM
els. Furthermore, it focuses on establishing a direct linkage                   backbone, as compared to LLaMA-7B utilized in LMDrive.
between the reasoning process and the planning outcome.                         The LLM backbone processes the integrated embeddings
                                                                                derived from images, navigation instructions, and questions.
B. End-to-End Driving Model                                                     To facilitate the alignment of reasoning and planning tasks
                                                                                through a multi-task design, we opt to decode the outputs using
   The driving model takes an E2E approach, defined as                          the hidden states H, which contain contextual information
follows:                                                                        from multiple modalities, rather than using the direct outputs
            Wpred , Apred , Cpred = E(I, Xnavi , Xqn ),                  (2)    from the LLM.
                                                                                   4) Multi-Task Outputs: Based on the hidden states H
The driving model processes multimodal inputs, including                        from the LLM, the driving model is designed with a multi-
visual data and textual information, to output predicted way-                   task output structure, enabling the simultaneous execution of
points (Wpred ), predicted answers (Apred ), and an indicator                   several functions. It generates planned trajectories that can
of whether the current navigation command has completed                         be transformed into precise vehicle maneuvers via waypoints
(Cpred ). The image inputs and navigation instructions are                      prediction, provides natural language explanations of the step-
homogeneous to the reasoning model, ensuring consistency                        by-step reasoning behind driving decisions, and determines
                                                                                                                                                         5




Fig. 2. The proposed E2E driving model boasts advanced reasoning and planning capabilities. Its student model ensures inference efficiency and is enhanced
via distillation from a teacher model, which substantially improves reasoning skills. By integrating reasoning and planning, it achieves robust performance
and interpretability, making it highly reliable and accurate for AD tasks.



whether the driving task under the current navigation is                         End State Prediction Similar to the waypoint predictor,
completed.                                                                     the end state predictor fend uses two MLPs with a ReLU
   Waypoint Prediction The waypoint predictor fwp projects                     activation in between to predict the probability of the ending
the hidden states to predict future vehicle positions:                         position:
                                                                                                    Cpred = fend (H).
                            W = fwp (H),
where fwp is a simple feed-forward neural network with two                     C. Reasoning Distillation
MLPs and a ReLU activation in between.                                            Within our proposed framework, the large-sized VLM is
   CoT Reasoning Prediction The CoT reasoning predictor,                       characterized as the reasoning model, as it comes with ad-
denoted as fcot , is implemented using customized Qwen2-                       vanced reasoning capability and is expected to enhance the
style decoder layers. The reasoning predictor is connected                     AD system by providing an explicit reasoning process. On
with the waypoint predictor such that both components share                    the other hand, the driving model is responsible for the actual
the same objective of planning for waypoints, and they both                    vehicle planning but possesses limited reasoning ability due
require supervision from ground-truth waypoints. As outlined                   to its compact size. A critical aspect of our framework is the
in Algorithm 1 and Algorithm 2, we employ two strategies for                   distillation of reasoning capability from the reasoning model
reasoning generation during training and inference. In training,               (i.e., a large-sized VLM) to the driving model (i.e., a compact-
we leverage the teacher forcing strategy that is discussed in                  sized LLM).
detail later in Section III-C. When deployed in closed-loop                       To achieve this, a teacher-forcing approach (Algorithm 1) is
simulations, the model uses an auto-regressive approach to                     employed. The distillation exposes the compact-sized driving
generate answers for reasoning. This involves generating one                   model to the same input data as the large-sized reasoning
token at a time, where each token is predicted based on the                    model. The driving model is guided to produce reasoning
context provided by the previously generated tokens and the                    answers that closely match those of the reasoning model.
initial hidden states from the LLM.                                            Based on a sequence of hidden states by the LLM backbone,
   Remark 1: As reasoning and planning are two closely                         the hidden states from the last few key frames are extracted
correlated tasks in AD, we strategically develop a waypoint-                   and concatenated with the embedded tokens of the ground-
driven dual-head coordination module for the alignment of the                  truth answers to facilitate the teacher-forcing training. This
two tasks. This module embeds ground-truth waypoints as the                    combined sequence is then passed through the answer predic-
final output of the reasoning process, ensuring that both the                  tor, which generates predictions for each token in the sequence.
reasoning and driving predictors share a common objective.                     This training process aims to minimize the divergence between
In this sense, the reasoning task also acts as a translator                    the answers predicted by the driving model and those provided
for the planning task by explicitly referencing the predicted                  in the training template, which consists of answers generated
waypoints, thereby enhancing the explainability of the E2E                     by the reasoning model alongside ground-truth data. The
AD framework.                                                                  objective is to enable the driving model to internalize the
                                                                                                                                   6



Algorithm 1 Reasoning Training with Teacher Forcing              Algorithm 2 Reasoning Inference with Autoregression
 1: Input: Output context H from the LLM, ground truth            1: Input: Output context H from the LLM, maximum num-
    answer A, maximum sequence length L                              ber of tokens N
 2: Return: Training loss L                                       2: Return: Generated answer Â
 3: Tokenize A to get cot answer token                            3: Initialization
 4: if cot answer token[0, −1] ̸= EOS then                        4:       generated tokens ← []
 5:      Append EOS token to cot answer token                     5:       current token ← BOS
 6: end if                                                        6:       Embed current token to get current embed
 7: target labels ← cot answer token[:, 1 :]                      7: for step = 1 to N do                         ▷ Autoregression
 8: cot answer token ← cot answer token[:, : −1]                  8:     Combine H and current embed as full sequence
 9: Embed cot answer token to get cot answer embeds               9:     Create attention mask M and position IDs P
10: Concatenate H and cot answer embeds to form sequence         10:     outputs = fcot predictor (full sequence, M, P )
11: Create attention mask M and position IDs P for sequence      11:     last hidden state = outputs[:, −1, :]
12: outputs = fcot predictor (sequence, M, P )       ▷ Answer    12:     logits = fproj (last hidden state)
    prediction using Qwen2style decoder layers                   13:     Apply repeat penalty to logits
13: Extract hidden states: hidden states = outputs[0]            14:     Greedy sampling: next token = arg max(logits)
14: context length ← len(H)                                      15:     Append next token to generated tokens
15: dec output ← hidden states[:, context length :, :]           16:     if next token = EOS then
16: if len(dec output) ̸= len(target labels) then                17:         break
17:      min len ← min(len(dec output), len(target labels))      18:     end if
18:      dec output ← dec output[:, : min len, :]                19: end for
19:      target labels ← target labels[:, : min len]             20: Decode generated tokens into a readable answer Â
20: end if                                                       21: return Â
21: predicted logits ← fproj (dec output) ▷ Project output to
    logits
22: L = CrossEntropy(predicted logits, target labels)            essential reasoning capability through distillation, thereby en-
23: return L                                                     hancing overall closed-loop driving performance while avoid-
                                                                 ing substantial computational overhead. Centered around this
                                                                 research motivation, the experiments are designed to provide a
reasoning strategies demonstrated by the large-sized model,      comprehensive evaluation of the proposed method along three
thereby enhancing its reasoning capability.                      key aspects: its quantitative and qualitative performance, the
   Only the driving model is involved in the training process.   effectiveness of the waypoint-driven dual-head coordination
Cross-entropy loss between the predicted logits and the target   module, and its computational efficiency.
tokens is computed and used to optimize the answer predictor.
The teacher-forcing approach ensures that the model learns
to generate coherent and precise explanations by directly        A. Implementation Details
conditioning on the ground-truth sequence during training.          1) Training: DSDrive is trained on the dataset generated
   In total, three loss terms are defined based on the outputs   in Section III-A, containing 11067 clips. The model is trained
from the driving model.                                          for 50 epochs using a batch size of 1. A linear warm-up
   • losswp : The waypoint loss is an l1 loss of the predicted   followed by a cosine learning rate decay schedule is employed
     waypoint and the ground-truth waypoint.                     to facilitate stable and efficient optimization. During the initial
   • lossans : The answer loss is computed using cross-entropy   2000 iteration steps, a warm-up phase is used to gradually
     loss between predicted and ground-truth answers.            increase the learning rate from 1 × 10−6 to the base learning
   • lossend : This loss checks whether the input instruction    rate of 1×10−4 , thereby stabilizing the early stages of training.
     is completed, and we employ cross-entropy loss for this     After the warm-up phase, the learning rate decays following
     classification task.                                        a cosine schedule, reaching a minimum value of 1 × 10−5 by
   Therefore, the training objectives are defined through the    the end of training. To mitigate overfitting, a weight decay of
following loss function:                                         0.06 is applied. The training is conducted on 4 NVIDIA RTX
                                                                 3090 with 24 GB of memory.
         loss = λ1 losswp + λ2 lossans + λ3 lossend ,      (3)      2) Closed-Loop Simulations: The closed-loop evaluation
                                                                 is implemented using the CARLA simulator [46]. The per-
where λ1 , λ2 , and λ3 are weighting coefficients in the range
                                                                 formance of DSDrive is compared with LMDrive [41], the
of zero to one that control the importance of each component
                                                                 benchmark method dedicated to closed-loop E2E AD. The
to the joint loss.
                                                                 efficacy of our proposed approach is also assessed utilizing
                                                                 the LangAuto benchmark provided by [41]. LangAuto is a
              IV. E XPERIMENT AND R ESULT                        collection of designated routes in CARLA that covers all
   We hypothesize that by focusing on key reasoning patterns,    eight publicly available towns with various kinds of environ-
the E2E driving model using compact LLM can acquire              mental and weather conditions. In the CARLA simulator, the
                                                                                                                                       7



                                                            TABLE II
    C LOSED - LOOP DRIVING PERFORMANCE OF DSD RIVE IN CARLA SIMULATIONS , WITH COMPARISON TO BASELINE AND ABLATION METHODS .

                                                                                    Normalized number of critical incidents
                                    Model           DS ↑    RC ↑     IS ↑
                                                                            Collision ↓   Red light violation ↓    Outside lanes ↓
                              Vanilla (LLaMA-1B)    21.27   23.03    0.93      0.00               0.03                  0.16
                             LMDrive (LLaMA-1B)     21.59   27.52    0.87      0.19               0.06                  0.13
            LangAuto-Long    LMDrive (LLaVA-7B)     28.51   37.87    0.80      0.13               0.22                  0.25
                             DSDrive (LLaMA-1B)     29.57   39.31    0.77      0.12               0.24                  0.32
                              Vanilla (LLaMA-1B)    38.75   42.88    0.87      0.07               0.14                  0.21
                             LMDrive (LLaMA-1B)     41.86   48.71    0.88      0.07               0.14                  0.21
            LangAuto-Short
                             LMDrive (LLaVA-7B)     45.03   52.90    0.84      0.07               0.20                  1.00
                             DSDrive (LLaMA-1B)     62.05   76.18    0.81      0.07               0.27                  1.00
                              Vanilla (LLaMA-1B)    47.76   53.70    0.84      0.06               0.00                  0.19
                             LMDrive (LLaMA-1B)     49.97   53.57    0.88      0.25               0.00                  0.06
             LangAuto-Tiny   LMDrive (LLaVA-7B)     61.91   73.19    0.84      0.19               0.06                  0.13
                             DSDrive (LLaMA-1B)     60.67   72.53    0.84      0.06               0.00                  0.25



closed-loop evaluation operates at a frequency of 10 Hz. The           3) Vanilla (LLaMA-1B): This vanilla version applies
proposed E2E AD model, DSDrive, processes sensor data,                    LLaMA-1B directly within the E2E AD framework with-
including images captured from the front, left, right, and rear           out any fine-tuning. It is intended to elucidate the intrinsic
cameras, as well as LiDAR point clouds. It then generates                 capability of the original compact-sized LLM in driving
the reasoning process and predicted waypoints. Subsequently,              tasks.
these predicted waypoints are translated into control variables         Our method, DSDrive, along with the comparative models,
of steering, throttle, and brake inputs within the CARLA             is evaluated using the LangAuto benchmark, which encom-
simulator.                                                           passes three distinct types of routes: LangAuto-Long, featuring
   3) Metrics: We consider the following metrics for the             routes longer than 500 meters; LangAuto-Short, comprising
evaluation of driving performance in closed-loop: Route Com-         routes between 150 and 500 meters in length; and LangAuto-
pletion (RC), Infraction Score (IS), and Driving Score (DS).         Tiny, consisting of routes shorter than 150 meters. The testing
Note that RC is the percentage of the route distance completed.      results for DSDrive, along with the comparison methods,
A higher RC indicates a better completion of the driving task.       across the three LangAuto benchmarks, are presented in Ta-
IS quantifies infractions such as collisions, lane violations,       ble II. The comparative study is centered on one theme: a
traffic signal breaches, and route deviations. Starting at a base    compact-sized LLM is inherently limited in its capabilities,
of 1.0, the IS is reduced by each incident according to its          and a conventional training strategy can only improve its
severity. A higher IS suggests better adherence to traffic rules     performance to a certain extent, while our method is able to
and safer driving practices. DS serves as a composite metric,        boost its performance to surpass that of a system with larger
capturing both driving progress and safety. It represents a          size.
balanced view of route navigation and regulations compliance.           Starting with Vanilla (LLaMA-1B) and LMDrive (LLaMA-
A higher DS is desirable. Additionally, we count the number          1B), which are based on the same LLM backbone as DSDrive.
of critical incidents, such as collisions with vehicles, red light   Firstly, it is essential to recognize that LLaMA-1B, due to its
violations, and driving outside of lanes. These counted values       compact size, inherently exhibits limitations when adapting to
are normalized to the number of simulated scenarios.                 the complicated closed-loop driving task. The untrained model
                                                                     (Vanilla) records the lowest performance scores across all three
B. Benchmark Comparison                                              benchmarks. The results are consistent with our hypothesis
                                                                     that the direct utilization of a compact-sized LLM imposes a
   We compare the performance of the proposed method,                substantial constraint on the overall performance of the system.
DSDrive, in closed-loop simulations with the following al-           Secondly, using the same LLaMA-1B backbone, LMDrive
ternatives:                                                          applies its training strategy, but only marginally enhances the
 1) LMDrive (LLaVA-7B): We directly employ the check-                capabilities of LLaMA-1B. It fails to surpass the performance
    point provided by LMDrive [41], which is the SOTA                of large-sized systems, as demonstrated by LMDrive (LLaVA-
    method in LLM-based closed-loop E2E AD.                          7B). Finally, our method endows the compact-sized LLM with
 2) LMDrive (LLaMA-1B): This tiny version of LMDrive is              reasoning capability distilled from a large-sized VLM. This
    based on a backbone LLM of LLaMA-1B, and is trained              enhancement effectively complements the planning task and
    under the same strategy and dataset as LMDrive. It is            results in a significant improvement in the overall driving
    designed to demonstrate the capability of a lightweight          performance. This underscores the efficacy of integrating
    LLM like LLaMA-1B when integrated into E2E AD sce-               reasoning abilities into compact-sized LLMs as a means to
    narios. It acts as an ablation study of DSDrive, validating      augment AD performance.
    that the integrated reasoning capability is effective in            Specifically, the performance of DSDrive compared to the
    enhancing closed-loop driving performance.                       baseline method, LMDrive (LLaVA-7B), demonstrates the
                                                                                                                                                         8




Fig. 3. The explicit think-and-answer reasoning process for representative driving scenarios from the CARLA simulator. This includes urban road and highway
settings, as well as daytime and nighttime conditions.



effectiveness of DSDrive in E2E AD, and highlights its                         CARLA in Fig. 3. The top panel displays selected keyframes
reliability and consistency across varying route lengths. In                   for each scenario, providing a visual context of the driving
the aspects of DS and RC, DSDrive closely trails LMDrive                       environment. The bottom panel includes detailed reasoning
(LLaVA-7B) in the LangAuto-Tiny scenario. It shows marginal                    content. Scenario (a) depicts a nighttime urban setting with
improvements over the baseline in the LangAuto-Long sce-                       clear weather conditions. In this low-light condition, DSDrive
nario and exhibits remarkable proficiency in the LangAuto-                     manages to identify the traffic signal and execute the required
Short scenario. The IS of DSDrive is slightly lower than the                   action of stopping. Scenario (b) requires the vehicle to make
baseline, but still maintains at a closely comparable level.                   a right turn at a T-intersection, where the vehicle needs to
The performance in IS could be further broken down into                        accurately identify the correct traffic signal among multiple
undesirable driving behaviors. DSDrive demonstrates compet-                    visible signals, each displaying different indications for its
itive performance in minimizing collision rates and red light                  turning options. DSDrive effectively discerns the appropri-
violations. However, it records higher lane deviations, which                  ate traffic light to execute the right turn, demonstrating its
suggests a need for targeted enhancements to improve lateral                   adeptness in parsing complex visual information and adhering
stability and adherence to lane discipline.                                    to traffic rules. Scenario (c) features a suburban environment
   In summary, DSDrive demonstrates notable strengths in DS                    with incoming vehicles on the closely adjacent lane. DSDrive
and RC across a variety of scenarios, showing its potential                    effectively identifies the black sedan approaching from the
as an effective E2E AD agent. Its performance in colli-                        left side. Scenario (d) takes place in a highway environment
sion avoidance and traffic rule compliance is commendable,                     with relatively heavy traffic. Despite the congestion, DSDrive
although the higher frequency of lane deviations suggests                      effectively identifies that none of the surrounding vehicles are
an area for improvement. Importantly, the ablative studies                     directly impacting its path, allowing it to maintain a steady
comparing DSDrive with LMDrive (LLaMA-1B) and Vanilla                          and stable course. In the aforementioned scenarios, DSDrive
(LLaMA-1B) substantiate the efficacy of our approach: by                       demonstrates robust performance in recognizing traffic signals,
integrating advanced reasoning capability into compact-sized                   executing turns, and navigating a wide range of environments,
LLMs through distillation, we can enhance their performance                    from urban streets to highway settings. The explicit think-
in E2E closed-loop AD systems. Furthermore, the comparison                     and-answer reasoning process shows that DSDrive manages to
with the baseline model, LMDrive (LLaVA-7B), highlights                        learn advanced reasoning capability from the large-sized VLM
that, even with a compact-sized backbone model, our method                     and applies it within the E2E AD framework. This consistent
has the potential to elevate the performance of compact-sized                  performance across diverse scenarios underscores the effec-
LLMs to the level of well-trained systems utilizing large-sized                tiveness and reliability of DSDrive in interpreting visual cues
models.                                                                        and making informed decisions. More importantly, through
                                                                               the waypoint-driven dual-head coordination module, the LLM
C. Qualitative Analysis                                                        within DSDrive exhibits a critical ability to accurately predict
   1) Think-and-Answer Reasoning Process: We demonstrate                       waypoints that adapts well to dynamic traffic situations. This
the reasoning process with representative scenarios from                       capability is essential for delivering the final planning phase
                                                                                                                                                                     9




Fig. 4. Performance at a typical urban intersection in response to traffic light
signals. The AV first stops at a red light and subsequently starts driving when
the light turns green (left). The visualizations of the throttle and brake control   Fig. 6. Performance of the AV in consecutively yielding to approaching
inputs of the AV (right) demonstrate its prompt deceleration and acceleration        vehicles on a curved roadway (left). The visualizations of the steering, throttle,
in accordance with the traffic signal phase.                                         and brake control inputs (right) demonstrate the AV’s capability to identify
                                                                                     objects with potential collision risks and react with informed planning and
                                                                                     control strategies.




Fig. 5. Performance when the AV approaches an intersection and makes a left
turn. The image (top left) shows the predicted waypoints from the prediction
heads in planning (red dots) and reasoning (cyan dots). The visualizations of
the steering (bottom left) and BEV trajectory (right) are included to validate
the efficacy of closed-loop planning and control.



of AD systems.
   2) Representative Scenarios: Representative scenarios are
included to provide insights to the ability of DSDrive in
closed-loop driving, as this is the goal of this study. The
ability to manage the traffic light is discussed in Fig. 4. Firstly,                 Fig. 7. Comparison of reasoning capabilities of our method to Qwen2.5-VL-
                                                                                     7B. Our method demonstrates superior accuracy and context enrichment in
DSDrive is able to detect the red light and stop at a safe                           answers, including enhanced waypoint prediction abilities.
distance from the vehicle ahead. Once the traffic light changes
to green, DSDrive seamlessly transitions from a stationary
position to active driving. Subsequently, as indicated by Fig. 5,                    demonstrate DSDrive’s ability to integrate situational aware-
the vehicle executes a left turn as directed by the navigation                       ness with precise control.
instruction. This sequence underscores DSDrive’s capability                             3) Reasoning Capability: We further assess the quality of
to accurately interpret and execute navigation commands,                             answers to provide insights into the reasoning capability of
reflecting its effective integration of reasoning and operational                    DSDrive. We compare the answers generated by DSDrive
execution in complex urban environments. In the suburban                             and Qwen2.5-VL-7B across the dimensions of scenario un-
area with approaching vehicles nearby in Fig. 6, DSDrive                             derstanding, object detection, driving actions, and waypoints
yields appropriately to the incoming traffic, ensuring safety                        prediction in Fig. 7. Despite its smaller model size (1B) as
while smoothly navigating through the curved road. Visual-                           compared to Qwen2.5-VL-7B, DSDrive does not fall short in
izations of the key control variables pertinent to the planning                      the designated reasoning abilities for AD. In certain aspects,
process are included for the aforementioned scenarios, which                         it outperforms Qwen2.5-VL-7B. For example, in scenario
                                                                                                                                                            10




       t = 0.85 s, v = 2.29 m/s               t = 5.00 s, v = 4.41 m/s                t = 7.95 s, v = 6.16 m/s               t = 13.4 s, v = 3.98 m/s
                                                         (a) Highway with moderately heavy traffic.




      t = 11.45 s, v = 4.03 m/s               t = 12.2 s, v = 3.37 m/s               t = 15.25 s, v = 0.56 m/s              t = 17.20 s, v = 4.21 m/s
                                                                       (b) Roundabout.




       t = 3.65 s, v = 3.79 m/s               t = 14.45 s, v = 0.97 m/s              t = 16.05 s, v = 2.59 m/s              t = 18.80 s, v = 4.13 m/s
                                                           (c) Urban driving with an intersection.


Fig. 8. Visualization of the reasoning and planning alignment in CARLA. As illustrated across three distinct operational scenarios: (a) Highway with moderately
heavy traffic; (b) Roundabout; (c) Urban driving with an intersection. The black vehicle represents the AV. The waypoints generated by the reasoning head
are shown in cyan, and those predicted by the planning head are shown in red. t denotes time and v represents velocity.



understanding, DSDrive accurately recognizes the wet road                        reasoning capability within the proposed framework, creating a
condition with a reflective surface, whereas Qwen2.5-VL-7B                       shared goal for the two tasks. Secondly, the E2E driving model
fails to identify this detail. In object detection, DSDrive attends              is built with interlinked prediction heads for reasoning and
to the cyclist on the left side and considers its potential impact               planning. The two closely related tasks thereby complement
on the vehicle’s path. In driving actions, DSDrive considers                     each other in that the reasoning outputs provide explanation
the further maneuver to proceed when safe, in addition to                        for the planning outcome, and the planning head can generate
stopping at the stop sign, while Qwen2.5-VL-7B terminates                        trajectory prediction akin to the reasoning head, but with faster
the answer at stopping at the stop sign. Notably, DSDrive is                     inference for efficiency in deployment.
capable of predicting waypoints through a text-based reasoning                      This investigation raises the following questions: (1) Does
process, aligning with the purpose of the waypoints predictor                    the dual-head coordination module acts effectively for the
in the E2E AD framework, while Qwen2.5-VL-7B is not                              alignment of reasoning and planning tasks as intended? (2)
explicitly tuned to support this functionality. This difference                  Does this design enhance the performance of DSDrive as an
underscores the effectiveness of our alignment between the                       AD agent in closed-loop driving scenarios?
semantic reasoning task and the numerical trajectory planning
task, achieved through the purposeful design of both the                            Focusing on the first question, Fig. 8 provides a qualitative
training dataset and the model architecture.                                     demonstration through a direct visualization of the alignment
                                                                                 between waypoints predicted by the reasoning and planning
                                                                                 heads. Scenario (a) depicts a highway with moderately heavy
D. Dual-Head Coordination                                                        traffic. The predicted waypoints generated by both the reason-
   The innovations of the waypoint-driven dual-head coordi-                      ing and planning heads display similar movement trajectories.
nation module in DSDrive are twofold. Firstly, the waypoints                     This enables the AV to follow the vehicle ahead, navigate
are embedded in the training dataset for the distillation of                     through curved roads, and maintain smooth cruising within
                                                                                                                                        11



                          TABLE III                                                            TABLE IV
  A BLATIONS ON THE WAYPOINT- DRIVEN DUAL - HEAD COORDINATION           C OMPUTATIONAL TIME AND MEMORY REQUIRED FOR INFERENCE .
                       MODULE DESIGN .
                                                                                                Inference time (s)   Peak memory (MB)
         Benchmark      Train Method   DS ↑    RC ↑    IS ↑
                                                                        LMDrive (LLaVA-7B)            0.06                14263
                          CoT only     23.56   27.08   0.88             LMDrive (LLaMA-1B)            0.05                 6682
       LangAuto-Long      GT only      21.91   26.51   0.85             DSDrive (LLaMA-1B)            0.05                8082
                          Dual-task    27.35   29.52   0.89
                          CoT only     38.42   44.99   0.83
       LangAuto-Short     GT only      40.90   49.72   0.86
                          Dual-task    42.91   51.76   0.85          LMDrive implemented with LLaVA-7B and LLaMA-1B, to
                                                                     understand the inference resource requirement of DSDrive.
                          CoT only     46.14   54.46   0.87
                          GT only      48.50   55.83   0.88             The results highlight the comparative computational effi-
       LangAuto-Tiny
                          Dual-task    49.35   54.50   0.89          ciency of DSDrive. In terms of inference time, the performance
                                                                     of DSDrive is at a comparable level with the LMDrive model
                                                                     utilizing LLaMA-1B and is slightly faster than the LLaVA-7B.
the traffic flow. Scenario (b) depicts a roundabout scenario         In terms of memory usage, DSDrive exhibits a peak memory
where the AV exhibits the following behaviors, as evidenced          requirement of 8082 MB. While this represents an increase
by the predicted waypoints: entering the roundabout, main-           compared to the LMDrive with LLaMA-1B (6682 MB), it
taining normal driving within the roundabout, and yielding           remains significantly lower than the LLaVA-7B model’s con-
to a cyclist crossing the path. In this yielding instance, the       sumption of 14263 MB. The modest increase in memory usage
waypoints predicted by both the reasoning and planning heads         relative to LLaMA-1B is justified by the enhanced capability
suggest a stopping tendency to ensure safe interaction with          and robustness that DSDrive offers, suggesting a balanced
the cyclist. Subsequently, the AV accelerates back to normal         trade-off between memory efficiency and model performance.
speed once the path is clear of obstacles. Scenario (c) depicts         The implications of these results are twofold. Firstly, the
a typical urban driving scenario with an intersection. Initially,    comparable inference time between DSDrive and the LMDrive
the predicted waypoints reflect a deceleration phase as the AV       series suggests that the integration of reasoning capability
approaches a red traffic light. Subsequently, as the traffic light   within DSDrive’s architecture does not significantly compro-
transitions to green, the waypoints indicate an acceleration         mise the processing speed of the E2E AD system. Secondly,
phase, enabling the AV to proceed through the intersection.          the memory efficiency of DSDrive, especially when contrasted
Following this, the predicted waypoints generated by both            with LLaVA-7B, underscores its suitability for deployment
the reasoning and planning heads correspond to the task              in environments where hardware constraints are a critical
of executing a left turn at the intersection. Finally, these         consideration. It is important to note that the modest memory
waypoints align with the subsequent cruising phase.                  requirements of DSDrive make it feasible for deployment on
   Regarding the second question, we further provide an              advanced embedded infrastructures such as the NVIDIA Orin
ablation study for evaluating the impact of the dual-head            platform, facilitating broader adoption of AD technology.
coordination module on driving performance. We implement
closed-loop simulations in the LangAuto benchmarks across                                    V. C ONCLUSION
tiny, short, and long routes, and the experiment results are
                                                                        In this paper, we present DSDrive, an E2E AD frame-
summarized in Table III. Dual-task refers to our method. Two
                                                                     work leveraging a compact LLM. This framework features a
ablations on the training dataset are considered: CoT only
                                                                     knowledge distillation process to facilitate the deployment of
uses the answer generated from the large-sized VLM, without
                                                                     small-sized LLMs, which lead to efficient computation while
embedding of ground-truth data such as navigation commands
                                                                     enhancing their capabilities. Furthermore, it manages to bridge
or waypoints. GT only, on the other hand, is built with only
                                                                     the gap between high-level reasoning and low-level planning
ground-truth data, but with no inputs from the large-sized
                                                                     through a waypoint-driven dual-head coordination module.
VLM. It appears that our method beats the two ablatives
                                                                     We conduct extensive closed-loop simulation experiments to
with higher DS, RC, and IS, supporting the superiority of
                                                                     evaluate the performance of DSDrive. The results demonstrate
the waypoint-driven dual-head coordination module design in
                                                                     that DSDrive achieves driving performance comparable to that
facilitating E2E closed-loop AD.
                                                                     of larger systems and even outperforms them in several key
                                                                     metrics, despite its compact nature. Also, DSDrive success-
                                                                     fully navigates representative scenarios, including traffic lights,
E. Computation Efficiency
                                                                     intersections, and heavy traffic conditions. The efficacy of the
  Designed as a compact LLM specifically tailored for E2E            dual-head coordination module is also thoroughly investigated.
AD systems, the primary objective of DSDrive is to optimize          The findings indicate that this module effectively aligns rea-
computational efficiency by reducing both inference time and         soning and planning tasks by utilizing waypoints as a shared
memory consumption, thereby enhancing the feasibility of             objective, thereby enhancing overall driving performance. Ad-
deploying such systems in real-world applications with limited       ditionally, DSDrive reduces inference time and memory usage
computational resources. As shown in Table IV, we conduct            during deployment, offering a practical solution for integration
a comparative analysis against existing models, specifically         into AD systems with constrained computational resources.
                                                                                                                                                            12



   Future research could evolve along two interesting direc-                     [19] H. Lu, X. Jia, Y. Xie, W. Liao, X. Yang, and J. Yan, “ActiveAD:
tions. Firstly, we can focus on enhancing the capability of                           Planning-Oriented Active Learning for End-to-End Autonomous Driv-
                                                                                      ing,” arXiv preprint arXiv:2403.02877, 2024.
compact-sized LLMs using methods such as reinforcement                           [20] R. Yao, Y. Wang, H. Liu, R. Yang, Z. Peng, L. Zhu, and J. Ma,
learning or contrastive learning. Secondly, efforts should be                         “CALMM-Drive: Confidence-aware autonomous driving with large mul-
directed toward further reducing the size of compact-sized                            timodal model,” arXiv preprint arXiv:2412.04209, 2024.
                                                                                 [21] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao, “ST-P3: End-to-
LLMs to enable the deployment of LLM-based E2E AD                                     End Vision-Based Autonomous Driving via Spatial-Temporal Feature
systems in real vehicles as well as other edge infrastructures.                       Learning,” in European Conference on Computer Vision, 2022, pp. 533–
                                                                                      549.
                              R EFERENCES                                        [22] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du,
                                                                                      T. Lin, W. Wang et al., “Planning-Oriented Autonomous Driving,” in
 [1] S. Mozaffari, O. Y. Al-Jarrah, M. Dianati, P. Jennings, and A. Mouzaki-          Proceedings of the IEEE/CVF Conference on Computer Vision and
     tis, “Deep Learning-Based Vehicle Behavior Prediction for Autonomous             Pattern Recognition, 2023, pp. 17 853–17 862.
     Driving Applications: A Review,” IEEE Transactions on Intelligent           [23] X. Jia, J. You, Z. Zhang, and J. Yan, “DriveTransformer: Unified Trans-
     Transportation Systems, vol. 23, no. 1, pp. 33–47, 2020.                         former for Scalable End-to-End Autonomous Driving,” arXiv preprint
 [2] L. Chen, Y. Li, C. Huang, B. Li, Y. Xing, D. Tian, L. Li, Z. Hu, X. Na,          arXiv:2503.07656, 2025.
     Z. Li et al., “Milestones in Autonomous Driving and Intelligent Vehicles:   [24] X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li, “Think
     Survey of Surveys,” IEEE Transactions on Intelligent Vehicles, vol. 8,           Twice Before Driving: Towards Scalable Decoders for End-to-End
     no. 2, pp. 1046–1056, 2022.                                                      Autonomous Driving,” in Proceedings of the IEEE/CVF Conference on
 [3] J. Guo, U. Kurup, and M. Shah, “Is it safe to drive? an overview of              Computer Vision and Pattern Recognition, 2023, pp. 21 983–21 994.
     factors, metrics, and datasets for driveability assessment in autonomous    [25] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du,
     driving,” IEEE Transactions on Intelligent Transportation Systems,               T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li,
     vol. 21, no. 8, pp. 3135–3151, 2019.                                             “Goal-oriented Autonomous Driving,” 2022.
 [4] K. Muhammad, A. Ullah, J. Lloret, J. Del Ser, and V. H. C. De Al-           [26] J. Cui, H. Qiu, D. Chen, P. Stone, and Y. Zhu, “COOPERNAUT: End-
     buquerque, “Deep Learning for Safe Autonomous Driving: Current                   to-End Driving With Cooperative Perception for Networked Vehicles,”
     Challenges and Future Directions,” IEEE Transactions on Intelligent              in Proceedings of the IEEE/CVF Conference on Computer Vision and
     Transportation Systems, vol. 22, no. 7, pp. 4316–4336, 2020.                     Pattern Recognition, 2022, pp. 17 252–17 262.
 [5] Anthropic, “Claude 3.5 Sonnet,” https://www.anthropic.com/news/             [27] H. Shao, L. Wang, R. Chen, S. L. Waslander, H. Li, and Y. Liu, “Rea-
     claude-3.5-sonnet, Jul. 2024, [Computer software].                               sonNet: End-to-End Driving With Temporal and Global Reasoning,”
 [6] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,              in Proceedings of the IEEE/CVF Conference on Computer Vision and
     D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “GPT-4                Pattern Recognition, 2023, pp. 13 723–13 733.
     Technical Report,” arXiv preprint arXiv:2303.08774, 2023.                   [28] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu,
 [7] H. Liu, R. Yao, W. Liu, Z. Huang, S. Shen, and J. Ma, “CoDriveVLM:               C. Huang, and X. Wang, “VAD: Vectorized Scene Representation
     VLM-Enhanced Urban Cooperative Dispatching and Motion Planning                   for Efficient Autonomous Driving,” in Proceedings of the IEEE/CVF
     for Future Autonomous Mobility on Demand Systems,” arXiv preprint                International Conference on Computer Vision, 2023, pp. 8340–8350.
     arXiv:2501.06132, 2025.
                                                                                 [29] S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu,
 [8] P. Liu, H. Liu, H. Liu, X. Liu, J. Ni, and J. Ma, “VLM-E2E: Enhancing
                                                                                      and X. Wang, “VADv2: End-to-End Vectorized Autonomous Driving via
     End-to-End Autonomous Driving with Multimodal Driver Attention
                                                                                      Probabilistic Planning,” arXiv preprint arXiv:2402.13243, 2024.
     Fusion,” arXiv preprint arXiv:2502.18042, 2025.
                                                                                 [30] M. Bansal, A. Krizhevsky, and A. Ogale, “ChauffeurNet: Learning to
 [9] Y. Xiao, F. Codevilla, A. Gurram, O. Urfalioglu, and A. M. López,
                                                                                      Drive by Imitating the Best and Synthesizing the Worst,” arXiv preprint
     “Multimodal End-to-End Autonomous Driving,” IEEE Transactions on
                                                                                      arXiv:1812.03079, 2018.
     Intelligent Transportation Systems, vol. 23, no. 1, pp. 537–547, 2020.
[10] M. Nie, R. Peng, C. Wang, X. Cai, J. Han, H. Xu, and L. Zhang,              [31] J. Cheng, Y. Chen, and Q. Chen, “PLUTO: Pushing the Limit of
     “Reason2Drive: Towards Interpretable and Chain-Based Reasoning for               Imitation Learning-based Planning for Autonomous Driving,” arXiv
     Autonomous Driving,” in European Conference on Computer Vision,                  preprint arXiv:2404.14327, 2024.
     2024, pp. 292–308.                                                          [32] Y. Lu, J. Fu, G. Tucker, X. Pan, E. Bronstein, R. Roelofs, B. Sapp,
[11] T. Gong, L. Zhu, F. R. Yu, and T. Tang, “Edge intelligence in intelligent        B. White, A. Faust, S. Whiteson et al., “Imitation Is Not Enough:
     transportation systems: A survey,” IEEE Transactions on Intelligent              Robustifying Imitation with Reinforcement Learning for Challenging
     Transportation Systems, vol. 24, no. 9, pp. 8919–8944, 2023.                     Driving Scenarios,” in 2023 IEEE/RSJ International Conference on
[12] X. Zhou, M. Liu, E. Yurtsever, B. L. Zagar, W. Zimmer, H. Cao, and               Intelligent Robots and Systems, 2023, pp. 7553–7560.
     A. C. Knoll, “Vision Language Models in Autonomous Driving: A               [33] J. Yuan, S. Sun, D. Omeiza, B. Zhao, P. Newman, L. Kunze,
     Survey and Outlook,” IEEE Transactions on Intelligent Vehicles, 2024.            and M. Gadd, “RAG-Driver: Generalisable Driving Explanations with
[13] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu,                     Retrieval-Augmented In-Context Learning in Multi-Modal Large Lan-
     S. Ma, P. Wang, X. Bi et al., “DeepSeek-R1: Incentivizing Reason-                guage Model,” arXiv preprint arXiv:2402.10828, 2024.
     ing Capability in LLMs via Reinforcement Learning,” arXiv preprint          [34] B. Jiang, S. Chen, Q. Zhang, W. Liu, and X. Wang, “AlphaDrive: Un-
     arXiv:2501.12948, 2025.                                                          leashing the Power of VLMs in Autonomous Driving via Reinforcement
[14] Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayara-                 Learning and Reasoning,” arXiv preprint arXiv:2503.07608, 2025.
     man, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-Level                [35] S. Mandalika, A. Nambiar et al., “PRIMEDrive-CoT: A Precognitive
     Reward Design via Coding Large Language Models,” arXiv preprint                  Chain-of-Thought Framework for Uncertainty-Aware Object Interaction
     arXiv:2310.12931, 2023.                                                          in Driving Scene Scenario,” arXiv preprint arXiv:2504.05908, 2025.
[15] L. Chen, O. Sinavski, J. Hünermann, A. Karnsund, A. J. Willmott,           [36] J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang, “GPT-Driver: Learning
     D. Birch, D. Maund, and J. Shotton, “Driving with LLMs: Fusing                   to Drive with GPT,” arXiv preprint arXiv:2310.01415, 2023.
     Object-Level Vector Modality for Explainable Autonomous Driving,”           [37] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger,
     in 2024 IEEE International Conference on Robotics and Automation,                P. Luo, A. Geiger, and H. Li, “DriveLM: Driving with Graph Visual
     2024, pp. 14 093–14 100.                                                         Question Answering,” in European Conference on Computer Vision,
[16] Z. Huang, C. Feng, F. Yan, B. Xiao, Z. Jie, Y. Zhong, X. Liang,                  2024, pp. 256–274.
     and L. Ma, “DriveMM: All-in-One Large Multimodal Model for Au-              [38] C. Pan, B. Yaman, T. Nesti, A. Mallik, A. G. Allievi, S. Velipasalar, and
     tonomous Driving,” arXiv preprint arXiv:2412.07689, 2024.                        L. Ren, “VLP: Vision Language Planning for Autonomous Driving,”
[17] C. Zhang, R. Guo, W. Zeng, Y. Xiong, B. Dai, R. Hu, M. Ren,                      in Proceedings of the IEEE/CVF Conference on Computer Vision and
     and R. Urtasun, “Rethinking Closed-Loop Training for Autonomous                  Pattern Recognition, 2024, pp. 14 760–14 769.
     Driving,” in European Conference on Computer Vision, 2022, pp. 264–         [39] S. Zhang, W. Huang, Z. Gao, H. Chen, and C. Lv, “WiseAD: Knowledge
     282.                                                                             Augmented End-to-End Autonomous Driving with Vision-Language
[18] W. Liu, Y. Song, C. Meng, Z. Huang, H. Liu, C. Lv, and J. Ma,                    Model,” arXiv preprint arXiv:2412.09951, 2024.
     “Integrating Decision-Making Into Differentiable Optimization Guided        [40] R. Zhao, Q. Yuan, J. Li, H. Hu, Y. Li, C. Zheng, and F. Gao,
     Learning for End-to-End Planning of Autonomous Vehicles,” arXiv                  “Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive
     preprint arXiv:2412.01234, 2024.                                                 Learning,” arXiv preprint arXiv:2502.14917, 2025.
                                                                               13



[41] H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li,
     “LMDrive: Closed-Loop End-to-End Driving with Large Language
     Models,” in Proceedings of the IEEE/CVF Conference on Computer
     Vision and Pattern Recognition, 2024, pp. 15 120–15 130.
[42] B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang,
     C. Huang, W. Liu, and X. Wang, “Senna: Bridging Large Vision-
     Language Models and End-to-End Autonomous Driving,” arXiv preprint
     arXiv:2410.22313, 2024.
[43] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang,
     S. Wang, J. Tang et al., “Qwen2.5-VL Technical Report,” arXiv preprint
     arXiv:2502.13923, 2025.
[44] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle,
     A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The Llama 3
     Herd of Models,” arXiv preprint arXiv:2407.21783, 2024.
[45] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping Language-
     Image Pre-training with Frozen Image Encoders and Large Language
     Models,” in International Conference on Machine Learning, 2023, pp.
     19 730–19 742.
[46] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun,
     “CARLA: An Open Urban Driving Simulator,” in Conference on Robot
     Learning, 2017, pp. 1–16.
