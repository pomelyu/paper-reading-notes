                                                        4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large
                                                                               Language Models

                                                    Wanhua Li1,∗ ,
                                                               Renping Zhou1,2,∗ , Jiawei Zhou3 , Yingwei Song1,4 , Johannes Herter1,5 ,
                                                                  Minghan Qin2 , Gao Huang2, , Hanspeter Pfister1,
                                           1
                                             Harvard University Tsinghua University 3 Stony Brook University 4 Brown University 5 ETH Zürich
                                                               2




arXiv:2503.10437v2 [cs.CV] 1 Apr 2025
                                                                       Project page:                https://4d-langsplat.github.io/




                                        RGB




                                        4D Field




                                        RGB




                                        4D Field


                                        Figure 1. Visualization of the learned language features of our 4D LangSplat. We observe that 4D LangSplat effectively learns dynamic
                                        semantic features that change over time, such as the gradual diffusion of coffee shown in the first two rows, and the “chicken” toggling
                                        between open and closed states in the latter two rows. Additionally, our semantic field captures consistent features for semantics that
                                        remain unchanged over time, with the clear object boundaries in the visualization demonstrating the precision of our semantic field.

                                                                             Abstract                          designed for static image-text tasks, cannot capture tempo-
                                                                                                               ral dynamics in videos. Real-world environments are in-
                                        Learning 4D language fields to enable time-sensitive, open-            herently dynamic, with object semantics evolving over time.
                                        ended language queries in dynamic scenes is essential for              Building a precise 4D language field necessitates obtain-
                                        many real-world applications. While LangSplat success-                 ing pixel-aligned, object-wise video features, which cur-
                                        fully grounds CLIP features into 3D Gaussian representa-               rent vision models struggle to achieve. To address these
                                        tions, achieving precision and efficiency in 3D static scenes,         challenges, we propose 4D LangSplat, which learns 4D
                                        it lacks the ability to handle dynamic 4D fields as CLIP,              language fields to handle time-agnostic or time-sensitive
                                                                                                               open-vocabulary queries in dynamic scenes efficiently. 4D
                                                   ∗ Equal contribution.   Corresponding authors.
LangSplat bypasses learning the language field from vi-          text matching [11, 30], struggles to capture temporal in-
sion features and instead learns directly from text gener-       formation such as state changes, actions, and object condi-
ated from object-wise video captions via Multimodal Large        tions [50, 51]. Learning a precise 4D language field would
Language Models (MLLMs). Specifically, we propose a              require pixel-aligned, object-level video features as the 2D
multimodal object-wise video prompting method, consist-          supervision to capture the spatiotemporal semantics of each
ing of visual and text prompts that guide MLLMs to gen-          object in a scene, yet current vision models [60, 62] pre-
erate detailed, temporally consistent, high-quality captions     dominantly extract global, video-level features. One could
for objects throughout a video. These captions are encoded       extract features by cropping interested objects and then ob-
using a Large Language Model into high-quality sentence          tain patch features. It inevitably includes background in-
embeddings, which then serve as pixel-aligned, object-           formation, leading to imprecise semantic features [40]. Re-
specific feature supervision, facilitating open-vocabulary       moving the background and extracting vision features only
text queries through shared embedding spaces. Recognizing        from the foreground object with accurate object masks leads
that objects in 4D scenes exhibit smooth transitions across      to ambiguity in distinguishing between object and camera
states, we further propose a status deformable network to        motion, since only the precise foreground objects are visi-
model these continuous changes over time effectively. Our        ble without a reference to the background context. These
results across multiple benchmarks demonstrate that 4D           pose significant challenges for building an accurate and ef-
LangSplat attains precise and efficient results for both time-   ficient 4D language field.
sensitive and time-agnostic open-vocabulary queries.                To address these challenges, we propose 4D LangSplat,
                                                                 which constructs a precise and efficient 4D Language Gaus-
                                                                 sian field to support time-agnostic and time-sensitive open-
1. Introduction                                                  vocabulary queries. We first train a 4D Gaussian Splatting
The ability to construct a language field [21, 40] that sup-     (4D-GS) [61] model to reconstruct the RGB scene, which is
ports open vocabulary queries holds significant promise          represented by a group of Gaussian points and a deformable
for various applications such as robotic navigation [16],        decoder defining how the Gaussian point changes its loca-
3D scene editing [23], and interactive virtual environ-          tion and shape over time. Our 4D LangSplat then enhances
ments [35]. Due to the scarcity of large-scale 3D datasets       each Gaussian in 4D-GS with two language fields, where
with rich language annotations, current methods [21, 35,         one learns time-invariant semantic fields with CLIP features
43] leverage pre-trained models like CLIP [41] to extract        as did in LangSplat, and the other learns time-varying se-
pixel-wise features, which are then mapped to 3D spaces.         mantic field to capture the dynamic semantics. The time-
Among them, LangSplat [40] received increasing atten-            invariant semantic field encodes semantic information that
tion due to its efficiency and accuracy, which grounds the       does not change over time such as “human”, “cup”, and
precise masks generated by the Segment Anything Model            “dog”. They are learned with CLIP features on three SAM-
(SAM) [22] with CLIP features into 3D Gaussians, achiev-         defined semantic levels.
ing an accurate and efficient 3D language field by leverag-          For the time-varying semantic field, instead of learning
ing 3D Gaussian Splatting (3D-GS) [20]. LangSplat sup-           from vision features, we propose to directly learn from tex-
ports open-vocabulary queries in various semantic levels by      tual features to capture temporally dynamic semantics. Re-
learning three SAM-defined semantic levels.                      cent years have witnessed huge progress [37, 48] of Mul-
    Nothing endures but change. Real-world 3D scenes are         timodal Large Language Models (MLLMs), which take
rarely static, and they continuously change and evolve. To       multimodal input, including image, video, and text, and
enable open-vocabulary queries in dynamic 4D scenes, it is       generate coherent responses. Encouraged by the success
crucial to consider that target objects may be in motion or      of MLLMs, we propose a multimodal object-wise video
transformation. For instance, querying a scene for “dog”         prompting method that combines visual and text prompts
in a dynamic environment may involve the dog running,            to guide MLLMs in generating detailed, temporally con-
jumping, or interacting with other elements. Beyond spa-         sistent, high-quality captions for each object throughout a
tial changes, users may also want time-related queries, such     video. We then encode these captions using a large lan-
as “running dog”, which should only respond during the           guage model (LLM) to extract sentence embeddings, cre-
time segments when the dog is indeed running. Therefore,         ating pixel-aligned, object-level features that serve as su-
supporting time-agnostic and time-sensitive queries within       pervision for the 4D Language field. Recognizing the
a 4D language field is essential for realistic applications.     smooth transitions exhibited by objects across states in 4D
    A straightforward approach to extend LangSplat to a 4D       scenes, we further introduce a status deformable network
scene is to learn a deformable Gaussian field [32, 61, 64]       to model these continuous state changes effectively over
with CLIP features. However, it cannot model the dynamic,        time. Our network captures the gradual transitions across
time-evolving semantics as CLIP, designed for static image-      object states, enhancing the model’s temporal consistency
and improving its handling of dynamic scenes. Figure 1          Multimodal Large Language Models. The remarkable
visualizes the learned time-varying semantic field. Our         success of LLMs [3, 10, 52, 53] has shown their ability
experiments across multiple benchmarks validate that 4D         to perform new tasks [31] following human instructions.
LangSplat achieves precise and efficient results, support-      Based on LLMs, the research on MLLMs [4, 34, 38] ex-
ing both time-agnostic and time-sensitive open-vocabulary       plores the possibility of multimodal chat ability [17], which
queries in dynamic, real-world environments.                    represents a significant step forward in integrating visual
   In summary, our contributions are threefold:                 and textual modalities for complex scene understanding.
• We introduce 4D LangSplat for open-vocabulary 4D              MLLMs usually employ a vision encoder to extract visual
  spatial-temporal queries. To the best of our knowledge,       features and learn a connector to align visual features with
  we are the first to construct 4D language fields with ob-     LLMs. The recent models [9, 27, 59] demonstrate remark-
  ject textual captions generated by MLLMs.                     able capabilities in generating coherent captions from mul-
• To model the smooth transitions across states for objects     timodal inputs, including images and videos. In this paper,
  in 4D scenes, we further propose a status deformable net-     we propose to utilize the powerful multimodal process abil-
  work to capture continuous temporal changes.                  ity of MLLMs to convert video data into object-level cap-
• Experiential results show that our method attains state-      tions, which are then used to train a 4D language field.
  of-the-art performance for both time-agnostic and time-
  sensitive open-vocabulary queries.                            3. Method
2. Related Work                                                 3.1. Preliminaries
                                                                3D Gaussian Splatting. In 3D-GS [20], a scene is repre-
3D Gaussian Splatting. 3D-GS [20] is a powerful volu-
                                                                sented as a set of 3D Gaussian points. Each pixel in 2D im-
metric rendering technique that has gained attention for its
                                                                ages is computed by blending N sorted 3D Gaussian points
real-time, high-quality rendering ability. It represents com-
                                                                that overlap the pixel:
plex surfaces and scenes by projecting 3D Gaussian distri-
butions into 2D image space. It has been widely used for
many applications such as human reconstruction [28, 42],                                        C = \sum _{i=1}^{N} c_i \alpha _i \prod _{j=1}^{i-1} (1 - \alpha _j), \label {eq:rendering_3dgs}                     (1)
3D editing [7, 56], mesh extraction [15, 55], autonomous
driving [68, 70]. Recent work [2, 32, 33, 64] including
4D Gaussian Splatting (4D-GS) [61] has extended Gaussian        where ci and αi are the color and density of i-th Gaussian.
Splatting to 4D by introducing deformable fields, allowing      LangSplat. Building upon 3D-GS, LangSplat [40] grounds
for dynamic scenes where Gaussian parameters evolve over        2D CLIP features into 3D Gaussians. To obtain a precise
time to capture both spatial and temporal transformations.      field, SAM is used to obtain accurate object masks and then
However, 4D-GS primarily focuses on visual fidelity rather      CLIP features are extracted with masked objects. LangSplat
than semantic understanding, which limits its applicability     adopts feature splatting to train the 3D language field:
in open-vocabulary language queries.
3D Language Field. Some early work [23, 54] usually
                                                                                               \bm {F} = \sum _{i =1}^{N} \bm {f}_i \alpha _i \prod _{j=1}^{i-1} (1 - \alpha _j), \label {eq:rendering_langsplat}    (2)
ground 2D foundation model features [6, 26, 41] into a
neural radiance field (NeRF) [36]. For example, Distilled
Feature Fields (DFFs) propose to distill CLIP-LSeg [26]         where fi represents the language feature of the i-th Gaus-
into NeRF for semantic scene editing. LERF [21] pro-            sians and F is the rendered embedding in 2D images.
poses to distill CLIP [41] features into NeRF to support        4D Gaussian Splatting. 4D-GS [61] extends the 3D-GS
open-vocabulary 3D querying. With the emergence of 3D-          for dynamic scenes by introducing a deformable Gaussian
GS, many methods [44, 65, 66, 69] adopt 3D-GS as the 3D         field. Here, Gaussian parameters, including position, rota-
scene representation and lift 2D foundation model features      tion, and scaling factor, are allowed to vary over time:
into 3D Gaussians. Among them, LangSpalt [40] attains
precise and efficient language fields due to the introduc-                (\mathcal {X}', r', s')= (\mathcal {X} + \Delta \mathcal {X}, r + \Delta r, s + \Delta s), \label {eq:rendering_4dgs}                      (3)
tion of SAM masks. By incorporating multiple levels of
semantic granularity, LangSplat effectively supports open-      where X , r, and s represent the position, rotation, and
vocabulary queries across whole objects, parts, and sub-        scaling parameters, respectively. ∆X , ∆r, and ∆s denote
parts. Recently, several methods have attempted to embed        the corresponding deformable networks, which are imple-
semantic fields in 4D scenes and have achieved promising        mented by lightweight MLPs. The HexPlane [5, 14] repre-
progress, such as DGD [25] and 4-LEGS [13]. However,            sentation is used to obtain rich 3D Gaussian features.
these approaches have not leveraged the powerful genera-           A straightforward approach to adapting LangSplat for
tive capabilities of Multimodal Large Language Models.          4D scenes is to extend its static 3D language Gaussian field
with a deformable Gaussian field, as done in 4D-GS. How-            three levels of semantic granularity provided by SAM. Al-
ever, this approach faces significant limitations due to the        though each Gaussian point’s position and shape dynami-
nature of CLIP features. CLIP [41] is designed primarily for        cally change over time, its semantic feature remains static.
static image-text alignment, making it ill-suited for captur-       These static embeddings ensure spatial accuracy while fo-
ing dynamic and time-evolving semantics in video. Recent            cusing on stable semantic information derived from CLIP
research [50, 51, 58] further confirms that it struggles with       features. On the other hand, to learn the time-varying se-
understanding state changes, actions, object conditions, and        mantic field, we propose a novel approach that bypasses the
temporal context. For a precise and accurate 4D language            limitations of vision-based feature supervision. Instead, vi-
field, it is essential to obtain pixel-aligned, object-level fea-   sual data is converted into object-level captions by lever-
tures that track temporal semantics with fine-grained detail        aging MLLMs. These captions are then encoded using an
for each object in a scene. However, existing vision mod-           LLM to extract sentence embeddings, which are used as
els [60, 62] primarily offer global, video-level features that      pixel-aligned, object-level features for training the semantic
overlook specific object-level information, making it dif-          field. To effectively model the smooth, continuous transi-
ficult to represent spatiotemporal semantics at the object          tions of Gaussian points between a limited set of states, we
level. While cropping objects and obtaining patch-based             further introduce a status deformable network to enhance
features is possible, this includes background information,         reconstruction quality. The framework of training time-
leading to inaccurate language fields. Further cropping ob-         varying 4D fields is illustrated in Figure 2.
jects with accurate masks makes it difficult for vision mod-
els to distinguish between object movement and camera mo-           3.3. Multimodal Object-Wise Video Prompting
tion, as there is no background reference.                          Constructing a high-quality, dynamic 4D semantic field re-
                                                                    quires detailed, pixel-aligned object-level features that cap-
3.2. 4D LangSplat Framework                                         ture time-evolving semantics in video data. However, ob-
To address these challenges, we introduce 4D LangSplat,             taining these fine-grained visual features is challenging due
which constructs accurate and efficient 4D language fields          to the limitations of current vision models in distinguishing
to support both time-sensitive and time-agnostic open-              object-level details over time. To overcome this, we pro-
vocabulary queries in dynamic scenes. We first reconstruct          pose converting video segments into object-wise captions
the 4D dynamic RGB scene using 4D-GS [61]. In this stage,           and extracting sentence embeddings from these captions to
the RGB scene is represented by a set of deformable Gaus-           serve as precise, temporally consistent features.
sian points, each with parameters that adjust over time to                    Advances in MLLMs like GPT-4o [38], LLaVA-
capture object movement and shape transformations within            OneVision [27], and Qwen2-VL [59] enable high-quality
the scene. Building on the learned 4D-GS model, we extend           language generation from multimodal inputs. These mod-
each Gaussian point with language embeddings to learn               els process video, image, and text inputs to generate tem-
4D language fields. To further capture temporal and spa-            porally consistent responses. Leveraging these capabili-
tial details, and to handle both time-sensitive and time-           ties, we propose a multimodal object-wise video prompt-
agnostic queries effectively, we simultaneously construct           ing method, which combines visual and textual prompts
two types of semantic fields: a time-agnostic semantic field        to guide the MLLM in generating temporally consistent,
and a time-varying semantic field. The time-agnostic se-            object-specific, high-quality captions across video frames,
mantic field focuses on capturing semantic information that         encapsulating both spatial and temporal details.
does not change over time. Although objects in the scene                      Formally, let V = \{I_1, I_2, \dots , I_T\} be a video segment
are dynamic, they still exhibit attributes that remain con-         of T frames. For each frame, we apply SAM [22] in con-
stant across time, such as static properties of entities like       junction with DEVA tracking [8] to segment objects and
“dog”, “human”, and other objects within the environment.           maintain consistent object identities over time. This process
This semantic field emphasizes spatial details of these time-       yields temporally consistent masks for n objects present in
agnostic semantics. Conversely, the time-varying seman-             the video, denoted as \ifmmode \lbrace \else \textbraceleft \fi M_1, M_2, \dots , M_n\}, where each mask
tic field captures temporally dynamic semantics, such as “a         M_i represents a specific object tracked across frames. Each
running dog” , emphasizing semantic transitions over time.          frame I_t is segmented with the object masks at time step t
    For the time-agnostic semantic field, we still use CLIP         \ifmmode \lbrace \else \textbraceleft \fi M_{1,t}, M_{2,t}, \dots , M_{n,t}\}.
features and lift them to 4D space, as they are sufficient for                To effectively generate instance-wise, object-specific
capturing time-agnostic semantics. Specifically, we learn           captions while preserving the broader scene context, we
a static language embedding for each deformable Gaussian            need to guide the MLLM through precise prompting. Our
point in the 4D-GS model. Similar to LangSplat, we uti-             goal is for the MLLM to generate captions focused solely on
lize SAM’s hierarchical segmentation masks, learning three          the target object without introducing details of other objects.
distinct time-agnostic semantic fields corresponding to the         However, the presence of other objects as background ref-
                                                                                                                                                                                                                               Multimodal Object-Wise Video Prompting
                                                                                                                                                                                                                                                                                                                                  …
                                                                                                                                      Object 1
                                                                                                                                                                                                                                                                                                                                  …

                                                                                                                                                                                                                         Captions 1                                                                                               …
                                                     SAM                                                                                     …                       …                    MLLM                                    …                                             LLM                                               …

                                                                                                                                                                                                                                                                                                                              …
                                                                                                                                      Object 𝑛
                                                                                                                                                                                                                                                                                                                                  …
                                                                                                                                                                                                                                                                                                                                       t
                                                                                                                                                                                                                         Captions 𝑛                                                                                   Features
                                                                                                                                                                                                                                                                                                       Supervise
                                      (𝑥, 𝑦, 𝑧)                                                                                                                                             𝑤"                                           𝑆"
                                                                                𝑥𝑦                         𝑥𝑧                         𝑦𝑧
                                                                                                                                                                                            𝑤#                                           𝑆#
                                                                                                                                                                           MLP                  …
                                                                                                                                                                                                           ×              …
                                                                                𝑥𝑡                         𝑦𝑡                          𝑧𝑡                                                                                                                  Splatting
                                                                                                                                                                                            𝑤!                                           𝑆!
                             𝑡                                                                    HexPlane                                                       Status Deformable Network                         (𝑥, 𝑦, 𝑧)


Figure 2. The framework of constructing a time-varying semantic field in 4D LangSplat. We first use multimodal object-wise prompting
to convert a video into pixel-aligned object-level caption features. Then, we learn a 4D language field with a status deformable network.


erence remains essential; without this context, the MLLM                                                                                                                     as context for generating frame-specific captions. For each
may lose track of spatial relationships and environmental                                                                                                                    frame I_t , we combine \protect \mathcal  {D}_i with the visual prompt \protect \mathcal  {P}_{i,t} to gen-
context, which are critical for understanding the action and                                                                                                                 erate a time-specific caption C_{i,t} , capturing both the tempo-
status of the target object. Thus, our approach employs                                                                                                                      ral and contextual details for object i in frame I_t :
prompting techniques to direct the MLLM’s attention to
each object, enabling region-specific captioning that main-                                                                                                                                       C_{i,t} = \operatorname {MLLM}(\mathcal {D}_i, \mathcal {P}_{i,t}, \mathcal {T}_{frame}, V_t ), \label {eq:framecaption}            (6)
tains overall scene awareness. Inspired by the recent visual
prompting progress [45, 47, 63], we first use visual prompts
                                                                                                                                                                             where Tframe denotes the textual prompt that instructs the
to highlight the object of interest. Specifically, we build a
                                                                                                                                                                             MLLM to generate an object caption describing the object’s
visual prompt \protect \mathcal  {P}_{i,t} for each object i in frame I_t :
                                                                                                                                                                             current action and status at a specific time step.
    \mathcal {P}_{i,t} = \operatorname {Contour}(M_{i,t}) \cup \operatorname {Gray}( M_{i,t}) \cup \operatorname {Blur}( M_{i,t}), \label {eq:visualprompt}  (4)                 Each caption C_{i,t} provides semantic information for an
                                                                                                                                                                             object i at time t. To encode this semantic data into fea-
where \protect \operatorname  {Contour}(M_{i,t}) highlights M_{i,t} with a red contour,                                                                                      tures for training the 4D language field, we extract sentence
\protect \operatorname  {Gray}(M_{i,t}) converts the non-object area to grayscale, and                                                                                       embeddings \protect \bm  {e}_{i,t} for each caption C_{i,t} . As LLMs exhibit
\protect \operatorname  {Blur}( M_{i,t}) applies a Gaussian blur to the background pix-                                                                                      strong processing ability for free-form text [49, 52], we fur-
els. This prompt preserves essential background informa-                                                                                                                     ther propose to utilize them to extract sentence embeddings.
tion while ensuring focus on the object of interest, improv-                                                                                                                 Specifically, a fined-tuned LLM [57] for sentence embed-
ing the MLLM’s attention to the relevant target.                                                                                                                             ding tasks is used to extract features. This design choice
           For temporal coherence, we first generate a high-level                                                                                                            allows our model to respond effectively to open-vocabulary
video-level motion description for object i, noted as \protect \mathcal  {D}_i ,                                                                                             queries as the embeddings are generated within a shared
which summarizes the motion dynamics over T frames.                                                                                                                          language space that aligns with natural language queries.
This description is derived by prompting the MLLM with                                                                                                                       Thus, for every pixel (x, y) \in M_{i,t} within object i’s mask
the entire video sequence V to capture object motion and                                                                                                                     in frame I_t , the feature \protect \bm  {F}_{x,y,t} is given by:
interactions, defined as:
                                                                                                                                                                                                                                         \bm {F}_{x,y,t} = \bm {e}_{i,t}, \label {eq:featurelabels}                                   (7)
                     \mathcal {D}_i = \operatorname {MLLM}(\{\mathcal {P}_{i,1},..., \mathcal {P}_{i,T}\},\mathcal {T}_{video}, V ), \label {eq:videocaption}       (5)

where Tvideo denotes the textual prompt that instructs the                                                                                                                   where the embeddings \protect \bm  {F}_{x,y,t} serve as 2D supervision for
MLLM to generate video-level motion descriptions based                                                                                                                       the time-variable semantic field, providing pixel-aligned,
on the visual prompts. This description \protect \mathcal  {D}_i is then used                                                                                                object-wise features across frames.
3.4. Status Deformable Network                                                                                                                                  For time-sensitive queries, we combine both the time-
                                                                                                                                                            agnostic and time-sensitive semantic fields. First, the time-
With the 2D semantic feature supervision information avail-
                                                                                                                                                            agnostic semantic field is used to derive an initial mask for
able, we use it to train a 4D field. A straightforward ap-
                                                                                                                                                            each frame, following the same procedure described above.
proach, analogous to the method used 4D-GS, would be
                                                                                                                                                            This mask identifies where the queried object or entity ex-
to directly learn a deformation field ∆f for the seman-
                                                                                                                                                            ists, irrespective of time. To refine the query to specific time
tic features of deformable Gaussian points. However, this
                                                                                                                                                            segments where the queried term is active (e.g., an action
straightforward approach allows the semantic features of
                                                                                                                                                            occurring within a particular timeframe), we calculate the
each Gaussian point to change to any arbitrary semantic
                                                                                                                                                            cosine similarity between the time-sensitive semantic field
state, potentially increasing the learning complexity and
                                                                                                                                                            on the initial mask region and the query text. This similar-
compromising the temporal consistency of the features. In
                                                                                                                                                            ity is computed across each frame within the masked region
real-world dynamic scenes, each Gaussian point typically
                                                                                                                                                            to determine when the time-sensitive characteristics of the
exhibits a gradual transition between a limited set of se-
                                                                                                                                                            query term are most strongly represented. Using the mean
mantic states. For instance, an object like a person may
                                                                                                                                                            cosine similarity value across the entire video as a threshold,
transition smoothly among a finite set of actions (e.g., stand-
                                                                                                                                                            we identify the frames that exceed this threshold, indicating
ing, walking, running), rather than shifting to entirely unre-
                                                                                                                                                            relevant time segments. The spatial mask obtained with the
lated semantic states. To model these smooth transitions
                                                                                                                                                            time-agnostic field is retained as the final mask prediction
and maintain a stable 4D semantic field, we propose a sta-
                                                                                                                                                            for the identified time segments.
tus deformable network that restricts the Gaussian point’s
semantic features to evolve within a predefined set of states.                                                                                                  This combination of time-agnostic and time-sensitive se-
                                                                                                                                                            mantic fields enables accurate and efficient spatiotemporal
    Specifically, we represent the semantic feature of a Gaus-
                                                                                                                                                            querying, allowing 4D LangSplat to capture both the persis-
sian point i at any time t as a linear combination of K state
                                                                                                                                                            tent and dynamic characteristics of objects in the scene.
prototype features, \ifmmode \lbrace \else \textbraceleft \fi \bm {S}_{i,1}, \bm {S}_{i,2}, \dots , \bm {S}_{i,K}\}, where each state
captures a specific, distinct semantic meaning. The seman-
tic feature fi,t of a Gaussian point i at time t is:                                                                                                        4. Experiment
                                                                                                                                                            4.1. Setup
                                               \bm {f}_{i, t} = \sum _{k=1}^{K} w_{i,t,k} \bm {S}_{i,k}, \label {eq:status_deformable_combination}    (8)   Datasets. We conduct evaluations using two widely
                                                                                                                                                            adopted datasets: HyperNeRF [39] and Neu3D [29] . Given
                                                                                                                                                            the absence of semantic segmentation annotations for dy-
where w_{i,t,k} denotes the weighting coefficient for each state                                                                                            namic scenes in these datasets, we perform manual anno-
                                                                       \DOTSB \sum@ \slimits@ _{k=1}^{K} w_{i,t,k} = 1
k at time t, with                                                                                                      . This linear combination            tations to facilitate evaluation. More details regarding this
ensures that each Gaussian point’s semantic features transi-                                                                                                process are provided in the Appendix A.
tion gradually between predefined states.                                                                                                                   Implementation Details. All experiments are conducted
             To determine the appropriate weighting coefficients w_{k,t}                                                                                    on a single Nvidia A100 GPU. For extracting CLIP features,
for each Gaussian point over time, we employ an MLP                                                                                                         we use the OpenCLIP ViT-B/16 model . For dynamic se-
decoder ϕ. This MLP takes as input the spatial-temporal                                                                                                     mantics, we leverage the Qwen2-VL-7B model as the back-
features from Hexplane [5] and predicts weighting coef-                                                                                                     bone MLLM to generate time-varying captions, and use e5-
ficients that reflect the temporal progression of semantic                                                                                                  mistral-7b [57] to encode them into embeddings. Following
states. The MLP decoder ϕ and the per-Gaussian states                                                                                                       LangSplat [40], we also train an autoencoder to compress
\ifmmode \lbrace \else \textbraceleft \fi \bm {S}_{i,1}, \bm {S}_{i,2}, \dots , \bm {S}_{i,K}\} are jointly trained. This design en-                        the feature dimension. The CLIP and the text features are
sures that the status deformable network adapts to both the                                                                                                 compressed into 3 and 6 dimensions, respectively.
spatial and temporal context, enabling smooth, consistent                                                                                                   Baselines. Due to the absence of publicly available mod-
transitions among semantic states.                                                                                                                          els for 4D language feature rendering, we use several 3D
                                                                                                                                                            language feature rendering methods as baselines for evalu-
3.5. Open-vocabulary 4D Querying                                                                                                                            ating time-agnostic querying, including LangSplat [40] and
After training, 4D LangSplat enables both time-agnostic                                                                                                     Feature-3DGS [69] . We also incorporate segmentation-
and time-sensitive open-vocabulary queries. For time-                                                                                                       based techniques, such as Gaussian Grouping [65], to assess
agnostic queries, we utilize only the time-agnostic semantic                                                                                                semantic mask generation quality in our approach. Inspired
field. We first render a feature image and then compute the                                                                                                 by Segment Any 4D Gaussians [18], we enhance Gaussian
relevance score [21] between this rendered feature image                                                                                                    Grouping to adapt to dynamic scenes.
and the query. Following the post-processing strategy in                                                                                                        Given the lack of dynamic language field rendering
LangSplat [40], we obtain the segmentation mask for each                                                                                                    methods, we consider two additional baselines besides
frame from the relevance score maps.                                                                                                                        LangSplat for time-sensitive querying: Deformable CLIP
                                       americano                chickchicken         split-cookie               espresso                    Average
  Method
                                   Acc(%)     vIoU(%)         Acc(%)    vIoU(%)   Acc(%)   vIoU(%)       Acc(%)        vIoU(%)      Acc(%)      vIoU(%)
  LangSplat [40]                    45.19       23.16         53.26      18.20    73.58      33.08        44.03          16.15          54.01    22.65
  Deformable CLIP                   60.57       39.96         52.17      42.77    89.62      75.28        44.85          20.86          61.80    44.72
  Non-Status Field                  83.65       59.59         94.56      86.28    91.50      78.46        78.60          47.95          87.58    68.57
  Ours                              89.42       66.07         96.73      90.62    95.28      83.14        81.89          49.20          90.83    72.26
                                Table 1. Quantitative comparisons of time-sensitive querying on the HyperNeRF [39] dataset.
                            Query: Closed Chicken Container                                    Query: liquid become darker in Glasses




 Ours D-CLIP RGB




 Similarity Score


                                                                                                                  Frame Index
                                       Frame Index                                                                 Frame Index


Figure 3. Visualization of time-sensitive querying results between Deformable CLIP and ours. The bottom row depicts the cosine similarity
across frames, rescaled to (0,1) for direct comparison, while the horizontal bars indicate frames identified as relevant time segments. We
observed that the CLIP-based method cannot understand dynamic semantics correctly, while our method recognizes them.


                                            HyperNeRF               Neu3D          then learns static CLIP fields on these pre-trained RGB
   Method                                                                          fields. The Non-Status Field method utilizes both the time-
                                            mIoU mAcc mIoU mAcc
                                                                                   agnostic semantic field and the time-sensitive semantic field
   Feature-3DGS [69]      36.63 74.02 34.96 87.12                                  of our method while removing the status deformable net-
   Gaussian Grouping [65] 50.49 80.92 49.93 95.05                                  work. Instead, it directly learns a deformation field ∆f .
   LangSplat [40]         74.92 97.72 61.49 91.89                                  Metrics. For time-agnostic querying, we evaluate per-
   Ours                                     82.48 98.01 85.11 98.32                formance using mean accuracy (mAcc) and mean inter-
                                                                                   section over union (mIoU), calculated across all frames
Table 2. Quantitative comparisons of time-agnostic querying on                     in the test set. For time-sensitive querying, we evaluate
the HyperNeRF [39] and Neu3D [29] datasets (Numbers in %).                         temporal performance using an accuracy metric, defined
  Blur              Gray   Contour      ∆sim          Video    Image     ∆sim      as Acc = ncorrect /nall , where ncorrect and nall repre-
                                                                                   sent the number of correctly predicted frames and the to-
            ✓                           0.33                             0.14      tal frames in the test set, respectively. To assess segmen-
            ✓        ✓                  2.15             ✓               1.01      tation quality, Pwe adopt the metric from [67] and define
            ✓        ✓          ✓       3.32             ✓          ✓    3.32      vIoU = |S1u | t∈Si IoU(ŝt , st ), where ŝt and st are the
Table 3. Comparisons of                              Table 4. Comparisons of       predicted and ground truth masks at time t, and Su and Si
Visual prompts.                                      Text prompts.                 are the sets of frames in the union and intersection.
     K                      2           3            4          5        6
                                                                                   4.2. Main Results
     Acc (%)               94.56      97.82      95.65        94.56     94.56
     vIoU (%)              88.05      91.93      89.11        88.98     86.28      Time-Agnostic Querying. Table 2 shows our results on
                                                                                   two datasets. Our approach achieves the highest mIoU
        Table 5. Results for different state numbers on chick chicken.             and mAcc scores, demonstrating strong segmentation per-
and Non-Status Field. Deformable CLIP only utilizes the                            formance across both datasets. In contrast, other methods
time-agnostic semantic fields of our method, which first                           struggle to capture object movement and shape changes,
trains a 4D-GS model to learn dynamic RGB fields, and                              leading to worse performance on dynamic objects.
                        Query: complete cookie                                       Query: empty glass cup



 RGB




 D-CLIP




 Ours




 GT



Figure 4. Comparison of time-sensitive query mask. We compare time-sensitive query masks between Deformable CLIP and ours. The
CLIP-based method fails to identify time segments accurately, especially at the demarcation points during state transitions.


Time-Sensitive Querying. We perform dynamic query-                time segments within dynamic video semantics, whereas
ing on the HyperNeRF dataset, with Acc and vIoU re-               our method successfully identifies these segments. In Fig-
sults presented in Table 1. Our approach outperforms              ure 4, we present specific query masks. We observe that
not only the LangSplat method but also the Deformable             the CLIP-based approach fails to accurately capture time
CLIP and Non-Status Field approaches. Specifically, our           segments, especially at transition points in object states.
method achieves accuracy improvements of 29.03% and               For example, CLIP cannot reliably detect subtle transitions,
3.25% and vIoU gains of 28.04% and 4.19%, respectively.           such as when a cookie has just cracked or when a glass cup
Our approach introduces a multimodal object-wise video            has started dripping coffee. In contrast, our method effec-
prompting method that surpasses traditional CLIP-based            tively identifies these nuanced changes, demonstrating its
techniques. In comparison to Deformable CLIP, our time-           capability to handle dynamic state transitions accurately.
varying semantic field effectively integrates spatial and tem-
poral information. This ensures fluidity and coherence in         4.3. Ablation Studies
semantic state transitions, underscoring the importance of
                                                                  Multimodal Prompting. We evaluate the quality of gen-
MLLM video prompting (Section 3.3). Additionally, when
                                                                  erated captions using different combinations of textual and
compared to the Non-Status Field method, our approach
                                                                  visual prompting methods. To quantify this, we defined a
highlights the significance of status modelling by introduc-
                                                                  metric, ∆sim = scorepos − scoreneg , where scorepos and
ing a status deformable network (Section 3.4), which en-
                                                                  scoreneg represent the average cosine similarity scores be-
hances the model’s capability to handle complex, evolving
                                                                  tween query and caption features, encoded by the e5 model,
states and further solidifies the robustness and versatility of
                                                                  for positive and negative samples, respectively. A higher
our method in capturing nuanced dynamics.
                                                                  ∆sim indicates a stronger distinction between positive and
Visualization. To demonstrate our learned time-sensitive          negative examples, suggesting that the generated caption
language field, we applied PCA to reduce the dimension-           more effectively captures the spatiotemporal dynamics and
ality of the learned semantic features, producing a 3D vi-        semantic features of objects in the scene. Table 3 shows that
sualization as shown in Figure 1. Our method better cap-          utilizing all three visual prompting strategies maximizes the
tures the dynamic semantic features of objects and renders        MLLM’s focus on target objects. As shown in Table 4, in-
consistent features accurately. In Figure 3, we illustrate the    corporating pre-generated video-level motion descriptions
change in query-frame similarity scores over time for time-       resulting in a 0.87% improvement. Furthermore, adding im-
sensitive queries, comparing our approach to a CLIP-based         age prompts enables a more accurate description.
method. As shown, CLIP, which is optimized for static             State Numbers. Table 5 shows the ablation results of the
image-text alignment, struggles to capture the most relevant      status number K. We observe that an appropriate increase
in K led to better results, with K = 3 achieving the optimal           [7] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
performance, which was adopted in our experiments.                         aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
                                                                           Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
5. Conclusion                                                              lable 3d editing with gaussian splatting. In Proceedings of
                                                                           the IEEE/CVF Conference on Computer Vision and Pattern
We present 4D LangSplat, a novel approach to construct-                    Recognition, pages 21476–21485, 2024. 3
ing a dynamic 4D language field that supports both time-               [8] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexan-
agnostic and time-sensitive open-vocabulary queries within                 der Schwing, and Joon-Young Lee. Tracking anything
evolving scenes. Our method leverages MLLMs to produce                     with decoupled video segmentation. In Proceedings of the
high-quality, object-specific captions that capture tempo-                 IEEE/CVF International Conference on Computer Vision,
rally consistent semantics across video frames. This enables               pages 1316–1326, 2023. 4
4D LangSplat to overcome the limitations of traditional vi-            [9] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin
                                                                           Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang
sion feature-based approaches, which struggle to generate
                                                                           Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-
precise, object-level features in dynamic contexts. By incor-              temporal modeling and audio understanding in video-llms.
porating multimodal object-wise video prompting, we ob-                    arXiv preprint arXiv:2406.07476, 2024. 3
tain pixel-aligned language embeddings as training super-             [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao
vision. Furthermore, we introduce a status deformable net-                 Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao
work, which enforces smooth, structured transitions across                 Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source
limited object states. Our experimental results across mul-                chatbot impressing gpt-4 with 90%* chatgpt quality. See
tiple benchmarks demonstrate that 4D LangSplat achieves                    https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6,
state-of-the-art performance in dynamic scenarios.                         2023. 3
                                                                      [11] Tong Ding, Wanhua Li, Zhongqi Miao, and Hanspeter Pfis-
Acknowledgements                                                           ter. Tree of attributes prompt learning for vision-language
                                                                           models. arXiv preprint arXiv:2410.11201, 2024. 2
The work is supported in part by the National Key R&D                 [12] Gueter Josmy Faure, Min-Hung Chen, and Shang-Hong Lai.
Program of China under Grant 2024YFB4708200 and Na-                        Holistic interaction transformer network for action detection.
tional Natural Science Foundation of China under Grant                     In WACV, 2023. 4
U24B20173, and in part by US NIH grant R01HD104969.                   [13] Gal Fiebelman, Tamir Cohen, Ayellet Morgenstern, Peter
                                                                           Hedman, and Hadar Averbuch-Elor. 4-legs: 4d language em-
References                                                                 bedded gaussian splatting. arXiv preprint arXiv:2410.10719,
                                                                           2024. 3
 [1] Hassan Akbari, Dan Kondratyuk, Yin Cui, Rachel Hornung,          [14] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
     Huisheng Wang, and Hartwig Adam. Alternating gradi-                   Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
     ent descent and mixture-of-experts for integrated multimodal          Explicit radiance fields in space, time, and appearance. In
     perception. NeurIPS, 2023. 4                                          Proceedings of the IEEE/CVF Conference on Computer Vi-
 [2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun                sion and Pattern Recognition, pages 12479–12488, 2023. 3
     Bang, and Youngjung Uh. Per-gaussian embedding-based             [15] Lin Gao, Jie Yang, Bo-Tao Zhang, Jia-Mu Sun, Yu-Jie Yuan,
     deformation for deformable 3d gaussian splatting. arXiv               Hongbo Fu, and Yu-Kun Lai. Mesh-based gaussian splat-
     preprint arXiv:2404.03613, 2024. 3                                    ting for real-time large-scale deformation. arXiv preprint
 [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,                 arXiv:2402.04796, 2024. 3
     Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei                  [16] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram
     Huang, et al. Qwen technical report. arXiv preprint                   Burgard. Visual language maps for robot navigation. In 2023
     arXiv:2309.16609, 2023. 3                                             IEEE International Conference on Robotics and Automation
 [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan               (ICRA), pages 10608–10615. IEEE, 2023. 2
     Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren             [17] Gao Huang. Dynamic neural networks: advantages and chal-
     Zhou. Qwen-vl: A versatile vision-language model for un-              lenges. National Science Review, 11(8):nwae088, 2024. 3
     derstanding, localization, text reading, and beyond. arXiv       [18] Shengxiang Ji, Guanjun Wu, Jiemin Fang, Jiazhong Cen,
     preprint arXiv:2308.12966, 1(2):3, 2023. 3                            Taoran Yi, Wenyu Liu, Qi Tian, and Xinggang Wang. Seg-
 [5] Ang Cao and Justin Johnson. Hexplane: A fast representa-              ment any 4d gaussians. arXiv preprint arXiv:2407.04504,
     tion for dynamic scenes. In Proceedings of the IEEE/CVF               2024. 6, 2
     Conference on Computer Vision and Pattern Recognition,           [19] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,
     pages 130–141, 2023. 3, 6                                             Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,
 [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou,             Tim Green, Trevor Back, Paul Natsev, et al. The kinetics hu-
     Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-            man action video dataset. arXiv preprint arXiv:1705.06950,
     ing properties in self-supervised vision transformers. In Pro-        2017. 4
     ceedings of the IEEE/CVF international conference on com-        [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler,
     puter vision, pages 9650–9660, 2021. 3                                and George Drettakis. 3d gaussian splatting for real-time
     radiance field rendering. ACM Trans. Graph., 42(4):139–1,       [33] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao.
     2023. 2, 3                                                           Gaussian-flow: 4d reconstruction with dynamic 3d gaus-
[21] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo                     sian particle. In Proceedings of the IEEE/CVF Conference
     Kanazawa, and Matthew Tancik. Lerf: Language embedded                on Computer Vision and Pattern Recognition, pages 21136–
     radiance fields. In Proceedings of the IEEE/CVF Interna-             21145, 2024. 3
     tional Conference on Computer Vision, pages 19729–19739,        [34] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
     2023. 2, 3, 6                                                        Improved baselines with visual instruction tuning. In Pro-
[22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,            ceedings of the IEEE/CVF Conference on Computer Vision
     Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-            and Pattern Recognition, pages 26296–26306, 2024. 3
     head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-         [35] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu,
     thing. In Proceedings of the IEEE/CVF International Con-             Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt,
     ference on Computer Vision, pages 4015–4026, 2023. 2, 4              Eric Xing, and Shijian Lu. Weakly supervised 3d open-
[23] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitz-                vocabulary segmentation. Advances in Neural Information
     mann. Decomposing nerf for editing via feature field distil-         Processing Systems, 36:53433–53456, 2023. 2
     lation. Advances in Neural Information Processing Systems,      [36] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
     35:23311–23330, 2022. 2, 3                                           Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
[24] Hildegard Kuehne, Hueihan Jhuang, Estı́baliz Garrote,                Representing scenes as neural radiance fields for view syn-
     Tomaso Poggio, and Thomas Serre. Hmdb: a large video                 thesis. Communications of the ACM, 65(1):99–106, 2021.
     database for human motion recognition. In 2011 Inter-                3
     national conference on computer vision, pages 2556–2563.        [37] OpenAI. Gpt-4v. https://openai.com/index/
     IEEE, 2011. 4                                                        gpt-4v-system-card/, 2023. 2
                                                                     [38] OpenAI. Hello gpt-4o. https : / / openai . com /
[25] Isaac Labe, Noam Issachar, Itai Lang, and Sagie Benaim.
                                                                          index/hello-gpt-4o/, 2024. 3, 4
     Dgd: Dynamic 3d gaussians distillation. In European Con-
     ference on Computer Vision, pages 361–378. Springer, 2024.      [39] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
     3                                                                    Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
                                                                          Brualla, and Steven M Seitz.         Hypernerf: A higher-
[26] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen
                                                                          dimensional representation for topologically varying neural
     Koltun, and René Ranftl. Language-driven semantic seg-
                                                                          radiance fields. arXiv preprint arXiv:2106.13228, 2021. 6,
     mentation. arXiv preprint arXiv:2201.03546, 2022. 3
                                                                          7, 2, 3, 4
[27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng
                                                                     [40] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
     Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and
                                                                          Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
     Chunyuan Li. Llava-onevision: Easy visual task transfer.
                                                                          In Proceedings of the IEEE/CVF Conference on Computer
     arXiv preprint arXiv:2408.03326, 2024. 3, 4
                                                                          Vision and Pattern Recognition, pages 20051–20060, 2024.
[28] Mengtian Li, Shengxiang Yao, Zhifeng Xie, and Keyu Chen.             2, 3, 6, 7
     Gaussianbody: Clothed human reconstruction via 3d gaus-         [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
     sian splatting. arXiv preprint arXiv:2401.09720, 2024. 3             Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
[29] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon                 Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
     Green, Christoph Lassner, Changil Kim, Tanner Schmidt,               transferable visual models from natural language supervi-
     Steven Lovegrove, Michael Goesele, Richard Newcombe,                 sion. In International conference on machine learning, pages
     et al. Neural 3d video synthesis from multi-view video. In           8748–8763. PMLR, 2021. 2, 3, 4
     Proceedings of the IEEE/CVF Conference on Computer Vi-          [42] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang,
     sion and Pattern Recognition, pages 5521–5531, 2022. 6, 7,           Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang.
     2, 3, 4                                                              Splattingavatar: Realistic real-time human avatars with
[30] Wanhua Li, Xiaoke Huang, Zheng Zhu, Yansong Tang, Xiu                mesh-embedded gaussian splatting. In Proceedings of the
     Li, Jie Zhou, and Jiwen Lu. Ordinalclip: Learning rank               IEEE/CVF Conference on Computer Vision and Pattern
     prompts for language-guided ordinal regression. Advances             Recognition, pages 1606–1616, 2024. 3
     in Neural Information Processing Systems, 35:35313–35325,       [43] William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack
     2022. 2                                                              Kaelbling, and Phillip Isola. Distilled feature fields enable
[31] Wanhua Li, Zibin Meng, Jiawei Zhou, Donglai Wei, Chuang              few-shot language-guided manipulation. In Conference on
     Gan, and Hanspeter Pfister. Socialgpt: Prompting llms for            Robot Learning, pages 405–424. PMLR, 2023. 2
     social relation reasoning via greedy segment optimization. In   [44] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
     Advances in Neural Information Processing Systems, pages             Hua Guan. Language embedded 3d gaussians for open-
     2267–2291, 2024. 3                                                   vocabulary scene understanding. In Proceedings of the
[32] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-            IEEE/CVF Conference on Computer Vision and Pattern
     sian feature splatting for real-time dynamic view synthesis.         Recognition, pages 5333–5343, 2024. 3
     In Proceedings of the IEEE/CVF Conference on Computer           [45] Aleksandar Shtedritski, Christian Rupprecht, and Andrea
     Vision and Pattern Recognition, pages 8508–8520, 2024. 2,            Vedaldi. What does clip know about a red circle? vi-
     3                                                                    sual prompt engineering for vlms. In Proceedings of the
     IEEE/CVF International Conference on Computer Vision,           [58] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip:
     pages 11987–11997, 2023. 5                                           A new paradigm for video action recognition. arXiv preprint
[46] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah.                 arXiv:2109.08472, 2021. 4
     Ucf101: A dataset of 101 human actions classes from videos      [59] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
     in the wild. arXiv preprint arXiv:1212.0402, 2012. 4                 Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
[47] Sanjay Subramanian, William Merrill, Trevor Darrell, Matt            Ge, et al. Qwen2-vl: Enhancing vision-language model’s
     Gardner, Sameer Singh, and Anna Rohrbach. Reclip: A                  perception of the world at any resolution. arXiv preprint
     strong zero-shot baseline for referring expression compre-           arXiv:2409.12191, 2024. 3, 4
     hension. arXiv preprint arXiv:2204.05991, 2022. 5               [60] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu,
[48] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell,             Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui
     Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent,             Wang, et al. Internvid: A large-scale video-text dataset for
     Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking                multimodal understanding and generation. arXiv preprint
     multimodal understanding across millions of tokens of con-           arXiv:2307.06942, 2023. 2, 4
     text. arXiv preprint arXiv:2403.05530, 2024. 2                  [61] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
[49] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert                   Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
     Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre,            4d gaussian splatting for real-time dynamic scene rendering.
     Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al.           In Proceedings of the IEEE/CVF Conference on Computer
     Gemma: Open models based on gemini research and tech-                Vision and Pattern Recognition, pages 20310–20320, 2024.
     nology. arXiv preprint arXiv:2403.08295, 2024. 5                     2, 3, 4
[50] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun               [62] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko,
     Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang,                Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and
     Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-            Christoph Feichtenhofer. Videoclip: Contrastive pre-training
     1: A fully open, vision-centric exploration of multimodal            for zero-shot video-text understanding. arXiv preprint
     llms. arXiv preprint arXiv:2406.16860, 2024. 2, 4                    arXiv:2109.14084, 2021. 2, 4
[51] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann          [63] Lingfeng Yang, Yueze Wang, Xiang Li, Xinlong Wang, and
     LeCun, and Saining Xie. Eyes wide shut? exploring the                Jian Yang. Fine-grained visual prompting. Advances in Neu-
     visual shortcomings of multimodal llms. In Proceedings of            ral Information Processing Systems, 36, 2024. 5
     the IEEE/CVF Conference on Computer Vision and Pattern          [64] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
     Recognition, pages 9568–9578, 2024. 2, 4                             Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
[52] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier                fidelity monocular dynamic scene reconstruction. In Pro-
     Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste            ceedings of the IEEE/CVF Conference on Computer Vision
     Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.             and Pattern Recognition, pages 20331–20341, 2024. 2, 3
     Llama: Open and efficient foundation language models.           [65] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
     arXiv preprint arXiv:2302.13971, 2023. 3, 5                          Gaussian grouping: Segment and edit anything in 3d scenes.
[53] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert,               In European Conference on Computer Vision, pages 162–
     Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,                  179. Springer, 2025. 3, 6, 7, 2
     Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.         [66] Justin Yu, Kush Hari, Kishore Srinivas, Karim El-Refai,
     Llama 2: Open foundation and fine-tuned chat models. arXiv           Adam Rashid, Chung Min Kim, Justin Kerr, Richard Cheng,
     preprint arXiv:2307.09288, 2023. 3                                   Muhammad Zubair Irshad, Ashwin Balakrishna, et al.
[54] Vadim Tschernezki, Iro Laina, Diane Larlus, and Andrea               Language-embedded gaussian splats (legs): Incrementally
     Vedaldi. Neural feature fusion fields: 3d distillation of            building room-scale representations with a mobile robot.
     self-supervised 2d image representations. In 2022 Inter-             arXiv preprint arXiv:2409.18108, 2024. 3
     national Conference on 3D Vision (3DV), pages 443–453.          [67] Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng
     IEEE, 2022. 3                                                        Liu, and Lianli Gao. Where does it exist: Spatio-temporal
[55] Joanna Waczyńska, Piotr Borycki, Sławomir Tadeja, Jacek             video grounding for multi-form sentences. In Proceedings
     Tabor, and Przemysław Spurek. Games: Mesh-based adapt-               of the IEEE/CVF Conference on Computer Vision and Pat-
     ing and modification of gaussian splatting. arXiv preprint           tern Recognition, pages 10668–10677, 2020. 7
     arXiv:2402.01459, 2024. 3                                       [68] Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao
[56] Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and            Qiu, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi
     Qi Tian. Gaussianeditor: Editing 3d gaussians delicately             Liao. Hugs: Holistic urban 3d scene understanding via gaus-
     with text instructions. In Proceedings of the IEEE/CVF Con-          sian splatting. In Proceedings of the IEEE/CVF Conference
     ference on Computer Vision and Pattern Recognition, pages            on Computer Vision and Pattern Recognition, pages 21336–
     20902–20911, 2024. 3                                                 21345, 2024. 3
[57] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,              [69] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
     Rangan Majumder, and Furu Wei. Improving text em-                    hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
     beddings with large language models.           arXiv preprint        Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
     arXiv:2401.00368, 2023. 5, 6                                         3d gaussian splatting to enable distilled feature fields. In
     Proceedings of the IEEE/CVF Conference on Computer Vi-
     sion and Pattern Recognition (CVPR), pages 21676–21685,
     2024. 3, 6, 7
[70] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang,
     Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian:
     Composite gaussian splatting for surrounding dynamic au-
     tonomous driving scenes. In Proceedings of the IEEE/CVF
     Conference on Computer Vision and Pattern Recognition,
     pages 21634–21643, 2024. 3
        4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large
                               Language Models
                                              Supplementary Material
A. Datasets                                                         Video            Image prompts
                                                                    prompts
Since there are no publicly available ground truth segmen-
tation mask labels for the HyperNeRF [39] and Neu3D [29]            I       high-    You have an understanding of the
datasets, nor annotations tailored for time-sensitive query-        lighted the      overall transformation process of the
ing, we adopt the annotation pipeline outlined in Segment           objects      I   object: {video prompt}. Now, I have
Any 4D Gaussians [18] and manually annotate the mask la-            want you to      provided you with images extracted
bels ourselves. Specifically, we leverage the Roboflow plat-        describe in      from this process. Please describe
form alongside the SAM (Segment Anything Model) frame-              red outline      the specific state of the object(s) in
work for interactive annotation.                                    and blurred      the given image, without referring to
    For the HyperNeRF dataset, where data is captured with          the objects      the entire video process. Avoid de-
a monocular camera, we select one frame every four frames           that don’t       scribing states that you can’t infer di-
as the training set. From the remaining data, we annotate           need     you     rectly from the picture. Avoid repeat-
a subset as the test set to ensure no overlap between the           to describe.     ing descriptions in context. For ex-
two sets. For the Neu3D dataset with 21 camera views, one           First please     ample, if the context suggests the ob-
is reserved for testing, and the remaining 20 are used for          determine        ject is moving up and down but the
training, aligning with the 4D-GS [61] setting. To evaluate         the object       image shows it is just moving down,
on the Neu3D dataset, we annotate every 20 frames from              highlighted      explicitly only state that the object is
the test views.                                                     in red line      in a moving down state. If the con-
                                                                    in the video.    text suggests the object is breaking
B. Implementation Details                                           Then briefly     but the image shows it is complete
                                                                    summarize        right now, explicitly only state that
Multimodal Object-Wise Video Prompting. For Multi-                  the trans-       the object appears to be complete. If
modal Object-Wise Video Prompting, we utilize the largest           formation        context tells you something changes
SAM-defined semantic levels as mask inputs for the Multi-           process of       from green to blue, but it’s blue in
modal Large Language Models (MLLMs). The prompting                  this object.     this image, just state that the object
process is outlined in Table 6, which provides the specific                          is blue.
prompts used for MLLM prompting. For visual prompt-
                                                                                 Table 6. Details of Text prompts
ing, we employ a red contour line with a radius of 2 to
delineate object boundaries. Additionally, we apply Gaus-
sian blur with a radius of 10 and convert the images to                       Method                           FPS
grayscale mode to achieve gray-level augmentation. These                      Gaussian Grouping [65]           1.47
techniques enhance the effectiveness of the visual input dur-                 Ours-agnostic                    5.24
ing the prompting process.                                                    Ours-sensitive                   4.05
Autoencoder. Following LangSplat [40], we employ two
autoencoders to compress the high-dimensional CLIP fea-                     Table 7. Query Performance Comparison.
ture (512-dimension) and LLM feature (4096-dimension)
separately. Specifically, two MLPs are used to compress          scene. 2) Next, we incorporate semantic information into
512-dimensional CLIP features and 4096-dimensional               the static Gaussian field without introducing deformable
video features to 3 and 6 dimensions, respectively. The au-      networks. Semantic features are embedded into the scene
toencoders are optimized with L2 loss. To enhance stability,     by minimizing an L1 loss, ensuring accurate representations
a cosine similarity loss is also included as a regularization.   of the static scene’s semantics. 3) In the third stage, we
Training Details. Our training pipeline is structured into       extend the model to dynamic RGB scenes by introducing
four stages, progressively refining the model for robust per-    non-semantic deformation fields. Leveraging the approach
formance in dynamic 4D language field construction. 1)           of 4D-GS [61], we employ deformable networks to learn
In the initial stage, we train a static Gaussian field to re-    temporal and motion-based deformations that capture spa-
construct the RGB channel of static scenes. This provides        tial and temporal dynamics for RGB scenes. 4) For time-
a foundation for modeling the visual appearance of the           agnostic semantic rendering, we refine the semantic features
                                                        americano                   chickchicken                split-cookie
                      Method
                                               mIoU(%)           mAcc(%)        mIoU(%)      mAcc(%)      mIoU(%)       mAcc(%)
             Feature-3DGS [69]                    34.65             62.96        47.21         87.22        47.03         68.25
           Gaussian Grouping [65]                 61.77             71.31        34.65         75.52        72.71         96.56
               LangSplat [40]                     72.08             97.61        75.98         97.86        76.54         97.32
                       Ours                       83.48             98.77        86.50         98.81        90.04         98.67
                                                         espresso                     keyboard                  torchocolate
                      Method
                                               mIoU(%)           mAcc(%)        mIoU(%)      mAcc(%)      mIoU(%)       mAcc(%)
             Feature-3DGS [69]                    24.04             80.13        42.14         80.98        24.71         64.58
           Gaussian Grouping [65]                 32.45             82.46        42.44         74.15        58.95         85.52
               LangSplat [40]                     82.93             98.66        72.42         96.75        69.55         98.09
                       Ours                       83.52             97.95        79.53         95.71        71.79         98.10
                 Table 8. Comparison of mean IoU and mean Accuracy for various methods on the HyperNeRF [39] datasets.

                                                     coffee martini                 cook spinach              cut roasted beef
                      Method
                                               mIoU(%)           mAcc(%)        mIoU(%)      mAcc(%)      mIoU(%)       mAcc(%)
             Feature-3DGS [69]                    30.23             84.74        41.50         95.59        31.66         91.07
           Gaussian Grouping [65]                 71.37             97.34        46.45         93.79        54.70         93.25
               LangSplat [40]                     67.97             98.47        78.29         98.60        36.53         97.04
                       Ours                       85.16             99.23        85.09         99.38        85.32         99.28
                                                      flame salmon                   flame steak                 sear steak
                      Method
                                               mIoU(%)           mAcc(%)        mIoU(%)      mAcc(%)      mIoU(%)       mAcc(%)
             Feature-3DGS [69]                    54.33             77.13        27.27         88.23        24.78         85.94
           Gaussian Grouping [65]                 35.72             94.69        36.92         95.96        54.44         95.27
               LangSplat [40]                     66.01             82.16        64.05         97.77        78.29         98.60
                       Ours                       89.88             94.35        88.44         98.27        76.78         99.38
                    Table 9. Comparison of mean IoU and mean Accuracy for various methods on the Neu3D [29] dataset.

from the second stage while keeping the deformable net-                         segmentation accuracy and reliability compared to existing
work parameters fixed. For time-sensitive semantic render-                      methods, even in dynamic scenes.
ing, we jointly train the status deformable network and the                        Table 7 further compares the runtime efficiency of our
state prototype features to refine and model dynamic seman-                     method with the baseline on the HyperNeRF dataset. The
tics effectively. For all datasets, the iterations for four stages              comparison encompasses the total time required for ren-
are 3000, 1000, 10000, and 10000. The learning rates for                        dering semantic features and conducting open-vocabulary
the deformable network and the state prototype features are                     queries. Our method demonstrates significant advantages
set to 1.6 \times 10^{-4} and 2.5 \times 10^{-3} , respectively. Other train-   over the Gaussian Grouping approach, achieving faster
ing parameters remain consistent with those used in 4D-GS.                      runtime for both time-agnostic and time-sensitive queries.
                                                                                These findings validate our method as an efficient and scal-
C. More Quantitative Results                                                    able solution for handling open-vocabulary queries in dy-
                                                                                namic 4D scenes.
In Table 8 and Table 9, we present a detailed evaluation
of time-agnostic querying performance on the HyperNeRF                          D. More Visualization Results
and Neu3D datasets, respectively. Our method achieves a
mean IoU exceeding 85% across all scenarios, outperform-                        Figure 5 illustrates visualization results for time-agnostic
ing the baseline methods in most scenes for both mean IoU                       querying. As depicted, our method demonstrates superior
and mean accuracy. These results underscore the robustness                      accuracy in capturing objects that correspond to seman-
of our approach, demonstrating its ability to deliver superior                  tic descriptions, compared to other methods. Furthermore,
               split-cookie: cookie                                 flame steak: chef wearing apron



   RGB




 Feature-

  3DGS



 Gaussian

 Grouping




   LangSplat




   Ours




   GT



                Figure 5. Visualization of time-agnostic querying results on HyperNeRF [39] and Neu3D [29] datasets.


it effectively tracks the spatial dynamics of these objects          Method HMDB51 [24] UCF101 [46] Kinetics400 [19]
across different temporal steps, showcasing its effectiveness        MLLM     58.34       78.97          55.14
in handling dynamic scenarios.                                       IMP [1]   59.1        91.5           77.0

                                                                     Table 10. Accuracy Results (%) on the Video Classification task.
E. MLLM-based Embeddings
Since our method utilizes MLLMs to generate captions,                  Method       VmAP@0.1        VmAP@0.2           VmAP@0.5
the feature representation capability of the obtained embed-           MLLM           78.13           75.78              64.38
dings is inherently limited by the capacity of the MLLMs,              HIT [12]        86.1            88.8               74.3
which constitutes a limitation of our approach. To verify
                                                                     Table 11. Spatial-Temporal Action Localization Results (%) on
that our MLLM-based embeddings indeed encode spatial-
                                                                     UCF101 [46].
temporal information, we directly apply the MLLM-based
embeddings, without any fine-tuning, to video classifica-
tion and spatial-temporal action localization tasks using            beddings inherently capture some spatial-temporal informa-
2D videos. As shown in Tables 10 and 11, our results                 tion. However, we also acknowledge that the performance
demonstrate that, even in a zero-shot setting, the MLLM-             of our approach is ultimately constrained by the representa-
based embeddings achieve competitive performance com-                tional capacity of the MLLMs.
pared to state-of-the-art (SOTA) methods specifically de-
signed for these tasks. This indicates that MLLM-based em-
