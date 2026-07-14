# YOLOv10: Real-Time End-to-End Object Detection

Ao Wang Hui Chen<sup>∗</sup> Lihao Liu Kai Chen Zijia Lin Jungong Han Guiguang Ding<sup>∗</sup> Tsinghua University

![](images/10c251debfcf41f2e15d2d53c8a8bc9d05bc94ac7b3afdd97ee0e08226c8f6ad.jpg)

![](images/59712bc6f03cab5d265255c6410b312105cf911b9da64a6abc1536e5feb8869a.jpg)  
Figure 1: Comparisons with others in terms of latency-accuracy (left) and size-accuracy (right) trade-offs. We measure the end-to-end latency using the official pre-trained models.

## Abstract

Over the past years, YOLOs have emerged as the predominant paradigm in the field of real-time object detection owing to their effective balance between computational cost and detection performance. Researchers have explored the architectural designs, optimization objectives, data augmentation strategies, and others for YO-LOs, achieving notable progress. However, the reliance on the non-maximum suppression (NMS) for post-processing hampers the end-to-end deployment of YOLOs and adversely impacts the inference latency. Besides, the design of various components in YOLOs lacks the comprehensive and thorough inspection, resulting in noticeable computational redundancy and limiting the model’s capability. It renders the suboptimal efficiency, along with considerable potential for performance improvements. In this work, we aim to further advance the performance-efficiency boundary of YOLOs from both the post-processing and the model architecture. To this end, we first present the consistent dual assignments for NMS-free training of YOLOs, which brings the competitive performance and low inference latency simultaneously. Moreover, we introduce the holistic efficiency-accuracy driven model design strategy for YOLOs. We comprehensively optimize various components of YOLOs from both the efficiency and accuracy perspectives, which greatly reduces the computational overhead and enhances the capability. The outcome of our effort is a new generation of YOLO series for real-time end-to-end object detection, dubbed YOLOv10. Extensive experiments show that YOLOv10 achieves the stateof-the-art performance and efficiency across various model scales. For example, our YOLOv10-S is 1.8× faster than RT-DETR-R18 under the similar AP on COCO, meanwhile enjoying 2.8× smaller number of parameters and FLOPs. Compared with YOLOv9-C, YOLOv10-B has 46% less latency and 25% fewer parameters for the same performance. Code: https://github.com/THU-MIG/yolov10.

## 1 Introduction

Real-time object detection has always been a focal point of research in the area of computer vision, which aims to accurately predict the categories and positions of objects in an image under low latency. It is widely adopted in various practical applications, including autonomous driving [3], robot navigation [11], and object tracking [66], etc. In recent years, researchers have concentrated on devising CNN-based object detectors to achieve real-time detection [18, 22, 43, 44, 45, 51, 12]. Among them, YOLOs have gained increasing popularity due to their adept balance between performance and efficiency [2, 19, 27, 19, 20, 59, 54, 64, 7, 65, 16, 27]. The detection pipeline of YOLOs consists of two parts: the model forward process and the NMS post-processing. However, both of them still have deficiencies, resulting in suboptimal accuracy-latency boundaries.

Specifically, YOLOs usually employ one-to-many label assignment strategy during training, whereby one ground-truth object corresponds to multiple positive samples. Despite yielding superior performance, this approach necessitates NMS to select the best positive prediction during inference. This slows down the inference speed and renders the performance sensitive to the hyperparameters of NMS, thereby preventing YOLOs from achieving optimal end-to-end deployment [71]. One line to tackle this issue is to adopt the recently introduced end-to-end DETR architectures [4, 74, 67, 28, 34, 40, 61]. For example, RT-DETR [71] presents an efficient hybrid encoder and uncertainty-minimal query selection, propelling DETRs into the realm of real-time applications. Nevertheless, the inherent complexity in deploying DETRs impedes its ability to attain the optimal balance between accuracy and speed. Another line is to explore end-to-end detection for CNN-based detectors, which typically leverages one-to-one assignment strategies to suppress the redundant predictions [5, 49, 60, 73, 16]. However, they usually introduce additional inference overhead or achieve suboptimal performance.

Furthermore, the model architecture design remains a fundamental challenge for YOLOs, which exhibits an important impact on the accuracy and speed [45, 16, 65, 7]. To achieve more efficient and effective model architectures, researchers have explored different design strategies. Various primary computational units are presented for the backbone to enhance the feature extraction ability, including DarkNet [43, 44, 45], CSPNet [2], EfficientRep [27] and ELAN [56, 58], etc. For the neck, PAN [35], BiC [27], GD [54] and RepGFPN [65], etc., are explored to enhance the multi-scale feature fusion. Besides, model scaling strategies [56, 55] and re-parameterization [10, 27] techniques are also investigated. While these efforts have achieved notable advancements, a comprehensive inspection for various components in YOLOs from both the efficiency and accuracy perspectives is still lacking. As a result, there still exists considerable computational redundancy within YOLOs, leading to inefficient parameter utilization and suboptimal efficiency. Besides, the resulting constrained model capability also leads to inferior performance, leaving ample room for accuracy improvements.

In this work, we aim to address these issues and further advance the accuracy-speed boundaries of YOLOs. We target both the post-processing and the model architecture throughout the detection pipeline. To this end, we first tackle the problem of redundant predictions in the post-processing by presenting a consistent dual assignments strategy for NMS-free YOLOs with the dual label assignments and consistent matching metric. It allows the model to enjoy rich and harmonious supervision during training while eliminating the need for NMS during inference, leading to competitive performance with high efficiency. Secondly, we propose the holistic efficiency-accuracy driven model design strategy for the model architecture by performing the comprehensive inspection for various components in YOLOs. For efficiency, we propose the lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design, to reduce the manifested computational redundancy and achieve more efficient architecture. For accuracy, we explore the large-kernel convolution and present the effective partial self-attention module to enhance the model capability, harnessing the potential for performance improvements under low cost.

Based on these approaches, we succeed in achieving a new family of real-time end-to-end detectors with different model scales, i.e., YOLOv10-N / S / M / B / L / X. Extensive experiments on standard benchmarks for object detection, i.e., COCO [33], demonstrate that our YOLOv10 can significantly outperform previous state-of-the-art models in terms of computation-accuracy trade-offs across various model scales. As shown in Fig. 1, our YOLOv10-S / X are 1.8× / 1.3× faster than RT-DETR-R18 / R101, respectively, under the similar performance. Compared with YOLOv9-C, YOLOv10-B achieves a 46% reduction in latency with the same performance. Moreover, YOLOv10 exhibits highly efficient parameter utilization. Our YOLOv10-L / X outperforms YOLOv8-L / X by 0.3 AP and 0.5 AP, with 1.8× and 2.3× smaller number of parameters, respectively. YOLOv10-M achieves the

similar AP compared with YOLOv9-M / YOLO-MS, with 23% / 31% fewer parameters, respectively.   
We hope that our work can inspire further studies and advancements in the field.

## 2 Related Work

Real-time object detectors. Real-time object detection aims to classify and locate objects under low latency, which is crucial for real-world applications. Over the past years, substantial efforts have been directed towards developing efficient detectors [18, 51, 43, 32, 72, 69, 30, 29, 39]. Particularly, the YOLO series [43, 44, 45, 2, 19, 27, 56, 20, 59] stand out as the mainstream ones. YOLOv1, YOLOv2, and YOLOv3 identify the typical detection architecture consisting of three parts, i.e., backbone, neck, and head [43, 44, 45]. YOLOv4 [2] and YOLOv5 [19] introduce the CSPNet [57] design to replace DarkNet [42], coupled with data augmentation strategies, enhanced PAN, and a greater variety of model scales, etc. YOLOv6 [27] presents BiC and SimCSPSPPF for neck and backbone, respectively, with anchor-aided training and self-distillation strategy. YOLOv7 [56] introduces E-ELAN for rich gradient flow path and explores several trainable bag-of-freebies methods. YOLOv8 [20] presents C2f building block for effective feature extraction and fusion. Gold-YOLO [54] provides the advanced GD mechanism to boost the multi-scale feature fusion capability. YOLOv9 [59] proposes GELAN to improve the architecture and PGI to augment the training process.

End-to-end object detectors. End-to-end object detection has emerged as a paradigm shift from traditional pipelines, offering streamlined architectures [48]. DETR [4] introduces the transformer architecture and adopts Hungarian loss to achieve one-to-one matching prediction, thereby eliminating hand-crafted components and post-processing. Since then, various DETR variants have been proposed to enhance its performance and efficiency [40, 61, 50, 28, 34]. Deformable-DETR [74] leverages multi-scale deformable attention module to accelerate the convergence speed. DINO [67] integrates contrastive denoising, mix query selection, and look forward twice scheme into DETRs. RT-DETR [71] further designs the efficient hybrid encoder and proposes the uncertainty-minimal query selection to improve both the accuracy and latency. Another line to achieve end-to-end object detection is based CNN detectors. Learnable NMS [23] and relation networks [25] present another network to remove duplicated predictions for detectors. OneNet [49] and DeFCN [60] propose oneto-one matching strategies to enable end-to-end object detection with fully convolutional networks. $\mathrm { F C O S } _ { \mathrm { p s s } }$ [73] introduces a positive sample selector to choose the optimal sample for prediction.

## 3 Methodology

## 3.1 Consistent Dual Assignments for NMS-free Training

During training, YOLOs [20, 59, 27, 64] usually leverage TAL [14] to allocate multiple positive samples for each instance. The adoption of one-to-many assignment yields plentiful supervisory signals, facilitating the optimization and achieving superior performance. However, it necessitates YOLOs to rely on the NMS post-processing, which causes the suboptimal inference efficiency for deployment. While previous works [49, 60, 73, 5] explore one-to-one matching to suppress the redundant predictions, they usually introduce additional inference overhead or yield suboptimal performance. In this work, we present a NMS-free training strategy for YOLOs with dual label assignments and consistent matching metric, achieving both high efficiency and competitive performance.

Dual label assignments. Unlike one-to-many assignment, one-to-one matching assigns only one prediction to each ground truth, avoiding the NMS post-processing. However, it leads to weak supervision, which causes suboptimal accuracy and convergence speed [75]. Fortunately, this deficiency can be compensated for by the one-to-many assignment [5]. To achieve this, we introduce dual label assignments for YOLOs to combine the best of both strategies. Specifically, as shown in Fig. 2.(a), we incorporate another one-to-one head for YOLOs. It retains the identical structure and adopts the same optimization objectives as the original one-to-many branch but leverages the one-to-one matching to obtain label assignments. During training, two heads are jointly optimized with the model, allowing the backbone and neck to enjoy the rich supervision provided by the oneto-many assignment. During inference, we discard the one-to-many head and utilize the one-to-one head to make predictions. This enables YOLOs for the end-to-end deployment without incurring any additional inference cost. Besides, in the one-to-one matching, we adopt the top one selection, which achieves the same performance as Hungarian matching [4] with less extra training time.

![](images/bc13e64958106f0c1ff5dfc1a5278097c163aa45b63ccbae6dd5da02b5367e96.jpg)

![](images/b566dd73655b30d1a0b31bc0533eac9ce1be96493e2e06063c072e5df943f9f2.jpg)  
Figure 2: (a) Consistent dual assignments for NMS-free training. (b) Frequency of one-to-one assignments in Top-1/5/10 of one-to-many results for YOLOv8-S which employs $\alpha _ { o 2 m } . = 0 . 5$ and $\beta _ { o 2 m } { = } 6$ by default [20]. For consistency, $\alpha _ { o 2 o } { = } 0 . 5 ; \beta _ { o 2 o } { = } 6$ . For inconsistency, $\alpha _ { o 2 o } { = } 0 . 5 ; \beta _ { o 2 o } { = } 2$

Consistent matching metric. During assignments, both one-to-one and one-to-many approaches leverage a metric to quantitatively assess the level of concordance between predictions and instances. To achieve prediction aware matching for both branches, we employ a uniform matching metric, $i . e .$

$$
m ( \alpha , \beta ) = s \cdot p ^ { \alpha } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { \beta } ,\tag{1}
$$

where $p$ is the classification score, $\hat { b }$ and $b$ denote the bounding box of prediction and instance, respectively. s represents the spatial prior indicating whether the anchor point of prediction is within the instance [20, 59, 27, 64]. α and $\beta$ are two important hyperparameters that balance the impact of the semantic prediction task and the location regression task. We denote the one-to-many and one-to-one metrics as $m _ { o 2 m } { = } m ( \alpha _ { o 2 m } , \beta _ { o 2 m } )$ and $m _ { o 2 o } { = } m ( \alpha _ { o 2 o } , \beta _ { o 2 o } )$ , respectively. These metrics influence the label assignments and supervision information for the two heads.

In dual label assignments, the one-to-many branch provides much richer supervisory signals than one-to-one branch. Intuitively, if we can harmonize the supervision of the one-to-one head with that of one-to-many head, we can optimize the one-to-one head towards the direction of one-to-many head’s optimization. As a result, the one-to-one head can provide improved quality of samples during inference, leading to better performance. To this end, we first analyze the supervision gap between the two heads. Due to the randomness during training, we initiate our examination in the beginning with two heads initialized with the same values and producing the same predictions, $i . e .$ , one-to-one head and one-to-many head generate the same $p$ and IoU for each prediction-instance pair. We note that the regression targets of two branches do not conflict, as matched predictions share the same targets and unmatched predictions are ignored. The supervision gap thus lies in the different classification targets. Given an instance, we denote its largest IoU with predictions as $u ^ { * }$ , and the largest one-to-many and one-to-one matching scores as $m _ { o 2 m } ^ { * }$ and $m _ { o 2 o } ^ { * }$ , respectively. Suppose that one-to-many branch yields the positive samples Ω and one-to-one branch selects i-th prediction with the metric $m _ { o 2 o , i } { = } m _ { o 2 o } ^ { * } ,$ , we can then derive the classification target $\begin{array} { r } { { t _ { o 2 m , j } } \mathrm { { = } } u ^ { \ast } \cdot \frac { { m _ { o 2 m , \ j } } } { { m _ { o 2 m } ^ { \ast } } } \le u ^ { \ast } } \end{array}$ for $j \in \Omega$ and $\begin{array} { r } { t _ { o 2 o , i } { = } u ^ { * } \cdot \frac { m _ { o 2 o , i } } { m _ { o 2 o } ^ { * } } { = } u ^ { * } } \end{array}$ for task aligned loss as in [20, 59, 27, 64, 14]. The supervision gap between two branches can thus be derived by the 1-Wasserstein distance [41] of different classification objectives, $i . e .$

$$
A = t _ { o 2 o , i } - \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } + \sum _ { k \in \Omega \backslash \{ i \} } t _ { o 2 m , k } ,\tag{2}
$$

We can observe that the gap decreases as $t _ { o 2 m , i }$ increases, $i . e . , i$ ranks higher within Ω. It reaches the minimum when $t _ { o 2 m , i } { = } u ^ { * } , i . e . , i$ i is the best positive sample in $\Omega$ , as shown in Fig. 2.(a). To achieve this, we present the consistent matching metric, $i . e . , \alpha _ { o 2 o } { = } r \cdot \alpha _ { o 2 m }$ and $\beta _ { o 2 o } = r \cdot \beta _ { o 2 m }$ , which implies $\scriptstyle m _ { o 2 o } = m _ { o 2 m } ^ { r }$ . Therefore, the best positive sample for one-to-many head is also the best for one-to-one head. Consequently, both heads can be optimized consistently and harmoniously. For simplicity, we take $r { = } 1$ , by default, $i . e . , \alpha _ { o 2 o } = \alpha _ { o 2 m }$ and $\beta _ { o 2 o } = \beta _ { o 2 m }$ . To verify the improved supervision alignment, we count the number of one-to-one matching pairs within the top-1 $/ 5 /$ 10 of the one-to-many results after training. As shown in Fig. 2.(b), the alignment is improved under the consistent matching metric. For a more comprehensive understanding of the mathematical proof, please refer to the appendix.

## 3.2 Holistic Efficiency-Accuracy Driven Model Design

In addition to the post-processing, the model architectures of YOLOs also pose great challenges to the efficiency-accuracy trade-offs [45, 7, 27]. Although previous works explore various design strategies, the comprehensive inspection for various components in YOLOs is still lacking. Consequently, the model architecture exhibits non-negligible computational redundancy and constrained capability, which impedes its potential for achieving high efficiency and performance. Here, we aim to holistically perform model designs for YOLOs from both efficiency and accuracy perspectives.

![](images/4485cbe312cff1ae88405fccb2247403437eec89a11e7feab6b50a188656e6d5.jpg)

![](images/6e3fb8886ff9c25242be5db425d7542f054ee18fbbf4577f1451ae6e4cd4bd43.jpg)  
Figure 3: (a) The intrinsic ranks across stages and models in YOLOv8. The stage in the backbone and neck is numbered in the order of model forward process. The numerical rank r is normalized to $r / C _ { o }$ for y-axis and its threshold is set to $\lambda _ { m a x } / 2$ , by default, where $C _ { o }$ denotes the number of output channels and $\lambda _ { m a x }$ is the largest singular value. It can be observed that deep stages and large models exhibit lower intrinsic rank values. (b) The compact inverted block (CIB). (c) The partial self-attention module (PSA).

Efficiency driven model design. The components in YOLO consist of the stem, downsampling layers, stages with basic building blocks, and the head. The stem incurs few computational cost and we thus perform efficiency driven model design for other three parts.

(1) Lightweight classification head. The classification and regression heads usually share the same architecture in YOLOs. However, they exhibit notable disparities in computational overhead. For example, the FLOPs and parameter count of the classification head (5.95G/1.51M) are 2.5× and 2.4× those of the regression head (2.34G/0.64M) in YOLOv8-S, respectively. However, after analyzing the impact of classification error and the regression error (seeing Tab. 6), we find that the regression head undertakes more significance for the performance of YOLOs. Consequently, we can reduce the overhead of classification head without worrying about hurting the performance greatly. Therefore, we simply adopt a lightweight architecture for the classification head, which consists of two depthwise separable convolutions [24, 8] with the kernel size of 3×3 followed by a 1×1 convolution.

(2) Spatial-channel decoupled downsampling. YOLOs typically leverage regular 3×3 standard convolutions with stride of 2, achieving spatial downsampling (from $\begin{array} { r } { H \times W \mathrm { t o } \frac { \breve { H } } { 2 } \times \frac { W } { 2 } ) } \end{array}$ and channel transformation (from C to 2C) simultaneously. This introduces non-negligible computational cost of $O ( { \textstyle { \frac { 9 } { 2 } } } H W C ^ { 2 } )$ and parameter count of $\mathcal { O } ( 1 8 C ^ { \tilde { 2 } } )$ ). Instead, we propose to decouple the spatial reduction and channel increase operations, enabling more efficient downsampling. Specifically, we firstly leverage the pointwise convolution to modulate the channel dimension and then utilize the depthwise convolution to perform spatial downsampling. This reduces the computational cost to $\mathcal { O } ( 2 H \bar { W } C ^ { 2 }$ + ${ } _ { \overline { { { 2 } } } } H W C )$ and the parameter count to $\mathcal { O } ( \bar { 2 } C ^ { 2 ^ { * } } { + } 1 8 C )$ . Meanwhile, it maximizes information retention during downsampling, leading to competitive performance with latency reduction.

(3) Rank-guided block design. YOLOs usually employ the same basic building block for all stages [27, 59], e.g., the bottleneck block in YOLOv8 [20]. To thoroughly examine such homogeneous design for YOLOs, we utilize the intrinsic rank [31, 15] to analyze the redundancy<sup>2</sup> of each stage. Specifically, we calculate the numerical rank of the last convolution in the last basic block in each stage, which counts the number of singular values larger than a threshold. Fig. 3.(a) presents the results of YOLOv8, indicating that deep stages and large models are prone to exhibit more redundancy. This observation suggests that simply applying the same block design for all stages is suboptimal for the best capacity-efficiency trade-off. To tackle this, we propose a rank-guided block design scheme which aims to decrease the complexity of stages that are shown to be redundant using compact architecture design. We first present a compact inverted block (CIB) structure, which adopts the cheap depthwise convolutions for spatial mixing and cost-effective pointwise convolutions for channel mixing, as shown in Fig. 3.(b). It can serve as the efficient basic building block, e.g., embedded in the ELAN structure [58, 20] (Fig. 3.(b)). Then, we advocate a rank-guided block allocation strategy to achieve the best efficiency while maintaining competitive capacity. Specifically, given a model, we sort its all stages based on their intrinsic ranks in ascending order. We further inspect the performance variation of replacing the basic block in the leading stage with CIB. If there is no performance degradation compared with the given model, we proceed with the replacement of the next stage and halt the process otherwise. Consequently, we can implement adaptive compact block designs across stages and model scales, achieving higher efficiency without compromising performance. Due to the page limit, we provide the details of the algorithm in the appendix.

Accuracy driven model design. We further explore the large-kernel convolution and self-attention for accuracy driven design, aiming to boost the performance under minimal cost.

(1) Large-kernel convolution. Employing large-kernel depthwise convolution is an effective way to enlarge the receptive field and enhance the model’s capability [9, 38, 37]. However, simply leveraging them in all stages may introduce contamination in shallow features used for detecting small objects, while also introducing significant I/O overhead and latency in high-resolution stages [7]. Therefore, we propose to leverage the large-kernel depthwise convolutions in CIB within the deep stages. Specifically, we increase the kernel size of the second 3×3 depthwise convolution in the CIB to 7×7, following [37]. Additionally, we employ the structural reparameterization technique [10, 9, 53] to bring another 3×3 depthwise convolution branch to alleviate the optimization issue without inference overhead. Furthermore, as the model size increases, its receptive field naturally expands, with the benefit of using large-kernel convolutions diminishing. Therefore, we only adopt large-kernel convolution for small model scales.

(2) Partial self-attention (PSA). Self-attention [52] is widely employed in various visual tasks due to its remarkable global modeling capability [36, 13, 70]. However, it exhibits high computational complexity and memory footprint. To address this, in light of the prevalent attention head redundancy [63], we present an efficient partial self-attention (PSA) module design, as shown in Fig. 3.(c). Specifically, we evenly partition the features across channels into two parts after the 1×1 convolution. We only feed one part into the $N _ { \mathrm { P S A } }$ blocks comprised of multi-head self-attention module (MHSA) and feed-forward network (FFN). Two parts are then concatenated and fused by a 1×1 convolution. Besides, we follow [21] to assign the dimensions of the query and key to half of that of the value in MHSA and replace the LayerNorm [1] with BatchNorm [26] for fast inference. Furthermore, PSA is only placed after the Stage 4 with the lowest resolution, avoiding the excessive overhead from the quadratic computational complexity of self-attention. In this way, the global representation learning ability can be incorporated into YOLOs with low computational costs, which well enhances the model’s capability and leads to improved performance.

## 4 Experiments

## 4.1 Implementation Details

We select YOLOv8 [20] as our baseline model, due to its commendable latency-accuracy balance and its availability in various model sizes. We employ the consistent dual assignments for NMS-free training and perform holistic efficiency-accuracy driven model design based on it, which brings our YOLOv10 models. YOLOv10 has the same variants as YOLOv8, i.e., N / S / M / L / X. Besides, we derive a new variant YOLOv10-B, by simply increasing the width scale factor of YOLOv10-M. We verify the proposed detector on COCO [33] under the same training-from-scratch setting [20, 59, 56]. Moreover, the latencies of all models are tested on T4 GPU with TensorRT FP16, following [71].

## 4.2 Comparison with state-of-the-arts

As shown in Tab. 1, our YOLOv10 achieves the state-of-the-art performance and end-to-end latency across various model scales. We first compare YOLOv10 with our baseline models, i.e., YOLOv8. On N / S / M / L / X five variants, our YOLOv10 achieves 1.2% / 1.4% / 0.5% / 0.3% / 0.5% AP improvements, with 28% / 36% / 41% / 44% / 57% fewer parameters, 23% / 24% / 25% / 27% / 38% less calculations, and 70% / 65% / 50% / 41% / 37% lower latencies. Compared with other YOLOs, YOLOv10 also exhibits superior trade-offs between accuracy and computational cost. Specifically, for lightweight and small models, YOLOv10-N / S outperforms YOLOv6-3.0-N / S by 1.5 AP and 2.0

Table 1: Comparisons with state-of-the-arts. Latency is measured using official pre-trained models. Latency<sup>f</sup> denotes the latency in the forward process of model without post-processing. † means the results of YOLOv10 with the original one-to-many training using NMS. All results below are without the additional advanced training techniques like knowledge distillation or PGI for fair comparisons.
<table><tr><td>Model</td><td>#Param.(M)</td><td>FLOPs(G)</td><td> $\mathbf { A P } ^ { v a l } ( \% )$ </td><td>Latency(ms)</td><td> $\mathrm { L a t e n c y } ^ { f } ( \mathrm { m s } )$ </td></tr><tr><td>YOLOv6-3.0-N [27]</td><td>4.7</td><td>11.4</td><td>37.0</td><td>2.69</td><td>1.76</td></tr><tr><td>Gold-YOLO-N [54]</td><td>5.6</td><td>12.1</td><td>39.6</td><td>2.92</td><td>1.82</td></tr><tr><td>YOLOv8-N [20]</td><td>3.2</td><td>8.7</td><td>37.3</td><td>6.16</td><td>1.77</td></tr><tr><td>YOLOv10-N (Ours)</td><td>2.3</td><td>6.7</td><td>38.5 / 39.5†</td><td>1.84</td><td>1.79</td></tr><tr><td>YOLOv6-3.0-S [27]</td><td>18.5</td><td>45.3</td><td>44.3</td><td>3.42</td><td>2.35</td></tr><tr><td>Gold-YOLO-S [54]</td><td>21.5</td><td>46.0</td><td>45.4</td><td>3.82</td><td>2.73</td></tr><tr><td>YOLO-MS-XS [7]</td><td>4.5</td><td>17.4</td><td>43.4</td><td>8.23</td><td>2.80</td></tr><tr><td>YOLO-MS-S [7]</td><td>8.1</td><td>31.2</td><td>46.2</td><td>10.12</td><td>4.83</td></tr><tr><td>YOLOv8-S [20]</td><td>11.2</td><td>28.6</td><td>44.9</td><td>7.07</td><td>2.33</td></tr><tr><td>YOLOv9-S [59]</td><td>7.1</td><td>26.4</td><td>46.7</td><td></td><td></td></tr><tr><td>RT-DETR-R18 [71]</td><td>20.0</td><td>60.0</td><td>46.5</td><td>4.58</td><td>4.49</td></tr><tr><td>YOLOv10-S (Ours)</td><td>7.2</td><td>21.6</td><td>46.3 / 46.8†</td><td>2.49</td><td>2.39</td></tr><tr><td>YOLOv6-3.0-M [27]</td><td>34.9</td><td>85.8</td><td>49.1</td><td>5.63</td><td>4.56</td></tr><tr><td>Gold-YOLO-M [54]</td><td>41.3</td><td>87.5</td><td>49.8</td><td>6.38</td><td>5.45</td></tr><tr><td>YOLO-MS [7]</td><td>22.2</td><td>80.2</td><td>51.0</td><td>12.41</td><td>7.30</td></tr><tr><td>YOLOv8-M [20]</td><td>25.9</td><td>78.9</td><td>50.6</td><td>9.50</td><td>5.09</td></tr><tr><td>YOLOv9-M [59]</td><td>20.0</td><td>76.3</td><td>51.1</td><td></td><td></td></tr><tr><td>RT-DETR-R34 [71]</td><td>31.0</td><td>92.0</td><td>48.9</td><td>6.32</td><td>6.21</td></tr><tr><td>RT-DETR-R50m [71]</td><td>36.0</td><td>100.0</td><td>51.3</td><td>6.90</td><td>6.84</td></tr><tr><td>YOLOv10-M (Ours)</td><td>15.4</td><td>59.1</td><td>51.1 / 51.3†</td><td>4.74</td><td>4.63</td></tr><tr><td>YOLOv6-3.0-L [27]</td><td>59.6</td><td>150.7</td><td>51.8</td><td>9.02</td><td>7.90</td></tr><tr><td>Gold-YOLO-L [54]</td><td>75.1</td><td>151.7</td><td>51.8</td><td>10.65</td><td>9.78</td></tr><tr><td>YOLOv9-C [59]</td><td>25.3</td><td>102.1</td><td>52.5</td><td>10.57</td><td>6.13</td></tr><tr><td>YOLOv10-B (Ours)</td><td>19.1</td><td>92.0</td><td>52.5 / 52.7†</td><td>5.74</td><td>5.67</td></tr><tr><td>YOLOv8-L [20]</td><td>43.7</td><td>165.2</td><td>52.9</td><td>12.39</td><td>8.06</td></tr><tr><td>RT-DETR-R50 [71]</td><td>42.0</td><td>136.0</td><td>53.1</td><td>9.20</td><td>9.07</td></tr><tr><td>YOLOv10-L (Ours)</td><td>24.4</td><td>120.3</td><td>53.2 / 53.4†</td><td>7.28</td><td>7.21</td></tr><tr><td>YOLOv8-X [20]</td><td>68.2</td><td>257.8</td><td>53.9</td><td>16.86</td><td>12.83</td></tr><tr><td>RT-DETR-R101 [71]</td><td>76.0</td><td>259.0</td><td>54.3</td><td>13.71</td><td>13.58</td></tr><tr><td>YOLOv10-X (Ours)</td><td>29.5</td><td>160.4</td><td>54.4 / 54.4†</td><td>10.70</td><td>10.60</td></tr></table>

AP, with 51% / 61% fewer parameters and 41% / 52% less computations, respectively. For medium models, compared with YOLOv9-C / YOLO-MS, YOLOv10-B / M enjoys the 46% / 62% latency reduction under the same or better performance, respectively. For large models, compared with Gold-YOLO-L, our YOLOv10-L shows 68% fewer parameters and 32% lower latency, along with a significant improvement of 1.4% AP. Furthermore, compared with RT-DETR, YOLOv10 obtains significant performance and latency improvements. Notably, YOLOv10-S / X achieves 1.8× and 1.3× faster inference speed than RT-DETR-R18 / R101, respectively, under the similar performance. These results well demonstrate the superiority of YOLOv10 as the real-time end-to-end detector.

We also compare YOLOv10 with other YOLOs using the original one-to-many training approach. We consider the performance and the latency of model forward process (Latency<sup>f</sup> ) in this situation, following [56, 20, 54]. As shown in Tab. 1, YOLOv10 also exhibits the state-of-the-art performance and efficiency across different model scales, indicating the effectiveness of our architectural designs.

## 4.3 Model Analyses

Ablation study. We present the ablation results based on YOLOv10-S and YOLOv10-M in Tab. 2. It can be observed that our NMS-free training with consistent dual assignments significantly reduces the end-to-end latency of YOLOv10-S by 4.63ms, while maintaining competitive performance of 44.3% AP. Moreover, our efficiency driven model design leads to the reduction of 11.8 M parameters and 20.8 GFlOPs, with a considerable latency reduction of 0.65ms for YOLOv10-M, well showing its effectiveness. Furthermore, our accuracy driven model design achieves the notable improvements of 1.8 AP and 0.7 AP for YOLOv10-S and YOLOv10-M, alone with only 0.18ms and 0.17ms latency overhead, respectively, which well demonstrates its superiority.

Table 2: Ablation study with YOLOv10-S and YOLOv10-M on COCO.
<table><tr><td># Model</td><td></td><td>NMS-free. Efficiency. Accuracy. #Param.(M) FLOPs(G)</td><td></td><td></td><td></td><td></td><td> $\mathbf { A P } ^ { v a l } ( \% )$ </td><td>Latency(ms)</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td>11.2</td><td>28.6</td><td>44.9</td><td>7.07</td></tr><tr><td>2</td><td>YOLOv10-S</td><td>√</td><td></td><td></td><td>11.2</td><td>28.6</td><td>44.3</td><td>2.44</td></tr><tr><td>3</td><td></td><td>√</td><td>√</td><td></td><td>6.2</td><td>20.8</td><td>44.5</td><td>2.31</td></tr><tr><td>4</td><td></td><td>√</td><td>√</td><td>√</td><td>7.2</td><td>21.6</td><td>46.3</td><td>2.49</td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td>25.9</td><td>78.9</td><td>50.6</td><td>9.50</td></tr><tr><td>6</td><td>YOLOv10-M</td><td>√</td><td></td><td></td><td>25.9</td><td>78.9</td><td>50.3</td><td>5.22</td></tr><tr><td>7</td><td></td><td>√</td><td>√</td><td></td><td>14.1</td><td>58.1</td><td>50.4</td><td>4.57</td></tr><tr><td>8</td><td></td><td>√</td><td>√</td><td>了</td><td>15.4</td><td>59.1</td><td>51.1</td><td>4.74</td></tr></table>

Table 3: Dual assign. Table 4: Matching metric.  
Table 5: Efficiency. for YOLOv10-S/M.
<table><tr><td colspan="4">02m o20 AP Latency</td><td colspan="2">α₀2o  $\beta _ { o 2 o }$   $\mathsf { A P } ^ { v a l }$ </td><td colspan="2"> $\alpha _ { o 2 o }$  βo20  $\mathsf { A P } ^ { v a l }$ </td></tr><tr><td>√</td><td></td><td>44.9</td><td>7.07</td><td>0.5 2.0</td><td>42.7</td><td>0.25 3.0 44.3</td><td></td></tr><tr><td></td><td></td><td>√43.4</td><td>2.44</td><td>0.5 4.0</td><td>44.2</td><td>0.256.0</td><td>43.5</td></tr><tr><td>V</td><td>√</td><td>44.3</td><td>2.44</td><td>0.5 6.0</td><td>44.3</td><td>1.0 6.0</td><td>43.9</td></tr><tr><td></td><td></td><td></td><td></td><td>0.5 8.0</td><td>44.0</td><td></td><td>1.012.044.3</td></tr></table>

<table><tr><td># Model</td><td>#Param</td><td>FLOPs  $\mathsf { A P } ^ { v a l }$ </td><td>Latency</td></tr><tr><td>1 base.</td><td>11.2/25.9</td><td>28.6/78.944.3/50.32.44/5.22</td><td></td></tr><tr><td>2 +cls.</td><td>9.9/23.2</td><td>23.5/67.744.2/50.22.39/5.07</td><td></td></tr><tr><td>3 +downs.</td><td>8.0/19.7</td><td>22.2/65.044.4/50.42.36/4.97</td><td></td></tr><tr><td>4 +block.</td><td>6.2/14.1</td><td>20.8/58.144.5/50.42.31/4.57</td><td></td></tr></table>

Analyses for NMS-free training.

• Dual label assignments. We present dual label assignments for NMS-free YOLOs, which can bring both rich supervision of one-to-many (o2m) branch during training and high efficiency of one-to-one (o2o) branch during inference. We verify its benefit based on YOLOv8-S, i.e., #1 in Tab. 2. Specifically, we introduce baselines for training with only o2m branch and only o2o branch, respectively. As shown in Tab. 3, our dual label assignments achieve the best AP-latency trade-off.

• Consistent matching metric. We introduce consistent matching metric to make the one-to-one head more harmonious with the one-to-many head. We verify its benefit based on YOLOv8-S, i.e., #1 in Tab. 2, under different $\alpha _ { o 2 o }$ and $\beta _ { o 2 o }$ . As shown in Tab. 4, the proposed consistent matching metric, $i . e . , \alpha _ { o 2 o } { = } r \cdot \alpha _ { o 2 m }$ and $\beta _ { o 2 o } = r \cdot \beta _ { o 2 m } ,$ , can achieve the optimal performance, where $\alpha _ { o 2 m } . = 0 . 5$ and $\beta _ { o 2 m } { = } 6 . 0$ in the one-to-many head [20]. Such an improvement can be attributed to the reduction of the supervision gap (Eq. (2)), which provides improved supervision alignment between two branches. Moreover, the proposed consistent matching metric eliminates the need for exhaustive hyper-parameter tuning, which is appealing in practical scenarios.

Analyses for efficiency driven model design. We conduct experiments to gradually incorporate the efficiency driven design elements based on YOLOv10-S/M. Our baseline is the YOLOv10-S/M model without efficiency-accuracy driven model design, i.e., #2/#6 in Tab. 2. As shown in Tab. 5, each design component, including lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design, contributes to the reduction of parameters count, FLOPs, and latency. Importantly, these improvements are achieved while maintaining competitive performance.

• Lightweight classification head. We analyze the impact of category and localization errors of predictions on the performance, based on the YOLOv10-S of #1 and #2 in Tab. 5, like [6]. Specifically, we match the predictions to the instances by the one-to-one assignment. Then, we substitute the predicted category score with instance labels, resulting in $\mathsf { A } \breve { \mathsf { P } } _ { w / o \mathrm { ~ } c } ^ { v a l }$ with no classification errors. Similarly, we replace the predicted locations with those of instances, yielding $\mathsf { A P } _ { w / o } ^ { v a l }$ with no regression errors. As shown in Tab. $6 , \mathrm { A P } _ { w / o } ^ { v a l } ,$ is much higher than $\mathsf { A P } _ { w / o \ c } ^ { v a l } ,$ revealing that eliminating the regression errors achieves greater improvement. The performance bottleneck thus lies more in the regression task. Therefore, adopting the lightweight classification head can allow higher efficiency without compromising the performance.

• Spatial-channel decoupled downsampling. We decouple the downsampling operations for efficiency, where the channel dimensions are first increased by pointwise convolution (PW) and the resolution is then reduced by depthwise convolution (DW) for maximal information retention. We compare it with the baseline way of spatial reduction by DW followed by channel modulation by PW, based on the YOLOv10-S of #3 in Tab. 5. As shown in Tab. 7, our downsampling strategy achieves the 0.7% AP improvement by enjoying less information loss during downsampling.

• Compact inverted block (CIB). We introduce CIB as the compact basic building block. We verify its effectiveness based on the YOLOv10-S of #4 in the Tab. 5. Specifically, we introduce the inverted residual block [46] (IRB) as the baseline, which achieves the suboptimal 43.7% AP, as shown in Tab. 8. We then append a 3×3 depthwise convolution (DW) after it, denoted as “IRB-DW”, which • Rank-guided block design. We introduce the rank-guided block design to adaptively integrate compact block design for improving the model efficiency. We verify its benefit based on the YOLOv10-S of #3 in the Tab. 5. The stages sorted in ascending order based on the intrinsic ranks are Stage 8-4-7-3-5-1-6-2, like in Fig. 3.(a). As shown in Tab. 9, when gradually replacing the bottleneck block in each stage with the efficient CIB, we observe the performance degradation starting from Stage 7. In the Stage 8 and 4 with lower intrinsic ranks and more redundancy, we can thus adopt the efficient block design without compromising the performance. These results indicate that rank-guided block design can serve as an effective strategy for higher model efficiency.

Table 6: cls. results.
<table><tr><td></td><td>base. +cls.</td><td></td></tr><tr><td> $\mathsf { A P } ^ { v a l }$ </td><td>44.3</td><td>44.2</td></tr><tr><td> $\mathsf { A P } _ { w / o \mathrm { ~ c ~ } } ^ { v a l }$ </td><td>59.9</td><td>59.9</td></tr><tr><td> $\mathsf { A P } _ { w / o \textit { r } } ^ { v a l }$ </td><td>64.5</td><td>64.2</td></tr></table>

Table 7: Results of d.s.
<table><tr><td>Model</td><td> $\mathsf { A P } ^ { v a l }$  Latency</td></tr><tr><td>base. 43.7</td><td>2.33</td></tr><tr><td>ours 44.4</td><td>2.36</td></tr></table>

Table 8: Results of CIB.
<table><tr><td>Model</td><td> $\mathsf { A P } ^ { v a l }$ </td><td>Latency</td></tr><tr><td>IRB</td><td>43.7</td><td>2.30</td></tr><tr><td>IRB-DW</td><td>44.2</td><td>2.30</td></tr><tr><td>ours</td><td>44.5</td><td>2.31</td></tr></table>

Table 9: Rank-guided.
<table><tr><td>Stages with CIB  $\mathsf { A P } ^ { v a l }$ </td></tr><tr><td>empty 44.4 8 44.5</td></tr><tr><td>8,4, 44.5</td></tr><tr><td>8,4,7 44.3</td></tr></table>

Table 10: Accuracy. for S/M. Table 11: L.k. results. Table 12: L.k. usage. Table 13: PSA results.
<table><tr><td># Model</td><td> $\mathsf { A P } ^ { v a l }$ </td><td>Latency</td></tr><tr><td>1 base.</td><td>44.5/50.42.31/4.57</td><td></td></tr><tr><td> $2 \ + \mathrm { L . k . }$ </td><td>44.9/-</td><td>2.34/-</td></tr><tr><td> $3 \ + \mathrm { P S A }$ </td><td>46.3/51.12.49/4.74</td><td></td></tr></table>

<table><tr><td>Model</td><td> $\mathsf { A P } ^ { v a l }$  Latency</td></tr><tr><td> $\mathrm { k . s . } { = } 5 $ </td><td>44.7 2.32</td></tr><tr><td> $\mathrm { k } . \mathrm { s } . { = } 7$ </td><td>44.9 2.34</td></tr><tr><td> $\mathrm { k . s . } { = } 9$ </td><td>44.9 2.37</td></tr><tr><td>w/o rep. 44.8</td><td>2.34</td></tr></table>

<table><tr><td>w/o L.k. w/ L.k.</td><td></td></tr><tr><td>N 36.3</td><td>36.6</td></tr><tr><td>S 44.5</td><td>44.9</td></tr><tr><td>M 50.4</td><td>50.4</td></tr></table>

<table><tr><td>Model</td><td> $\mathsf { A P } ^ { v a l }$ </td><td>Latency</td></tr><tr><td>PSA</td><td>46.3</td><td>2.49</td></tr><tr><td>Trans.</td><td>46.0</td><td>2.54</td></tr><tr><td> $N _ { \mathrm { P S A } } = 1$ </td><td>46.3</td><td>2.49</td></tr><tr><td> $N _ { \mathrm { P S A } } = 2$ </td><td>46.5</td><td>2.59</td></tr></table>

brings 0.5% AP improvement. Compared with $\mathrm { ^ { 6 4 } I R B { - } D W ^ { 3 } }$ , our CIB further achieves 0.3% AP improvement by prepending another DW with minimal overhead, indicating its superiority.

Analyses for accuracy driven model design. We present the results of gradually integrating the accuracy driven design elements based on YOLOv10-S/M. Our baseline is the YOLOv10-S/M model after incorporating efficiency driven design, i.e., #3/#7 in Tab. 2. As shown in Tab. 10, the adoption of large-kernel convolution and PSA module leads to the considerable performance improvements of 0.4% AP and 1.4% AP for YOLOv10-S under minimal latency increase of 0.03ms and 0.15ms, respectively. Note that large-kernel convolution is not employed for YOLOv10-M (see Tab. 12).

• Large-kernel convolution. We first investigate the effect of different kernel sizes based on the YOLOv10-S of #2 in Tab. 10. As shown in Tab. 11, the performance improves as the kernel size increases and stagnates around the kernel size of 7×7, indicating the benefit of large perception field. Besides, removing the reparameterization branch during training achieves 0.1% AP degradation, showing its effectiveness for optimization. Moreover, we inspect the benefit of large-kernel convolution across model scales based on YOLOv10-N / S / M. As shown in Tab. 12, it brings no improvements for large models, i.e., YOLOv10-M, due to its inherent extensive receptive field. We thus only adopt large-kernel convolutions for small models, $i . e . .$ , YOLOv10-N / S.

• Partial self-attention (PSA). We introduce PSA to enhance the performance by incorporating the global modeling ability under minimal cost. We first verify its effectiveness based on the YOLOv10- S of #3 in Tab. 10. Specifically, we introduce the transformer block, $i . e .$ , MHSA followed by FFN, as the baseline, denoted as “Trans.”. As shown in Tab. 13, compared with it, PSA brings 0.3% AP improvement with 0.05ms latency reduction. The performance enhancement may be attributed to the alleviation of optimization problem [62, 9] in self-attention, by mitigating the redundancy in attention heads. Moreover, we investigate the impact of different $N _ { \mathrm { P S A } }$ . As shown in Tab. 13, increasing $N _ { \mathrm { P S A } }$ to 2 obtains 0.2% AP improvement but with 0.1ms latency overhead. Therefore, we set $N _ { \mathrm { P S A } }$ to 1, by default, to enhance the model capability while maintaining high efficiency.

## 5 Conclusion

In this paper, we target both the post-processing and model architecture throughout the detection pipeline of YOLOs. For the post-processing, we propose the consistent dual assignments for NMSfree training, achieving efficient end-to-end detection. For the model architecture, we introduce the holistic efficiency-accuracy driven model design strategy, improving the performance-efficiency tradeoffs. These bring our YOLOv10, a new real-time end-to-end object detector. Extensive experiments show that YOLOv10 achieves the state-of-the-art performance and latency compared with other advanced detectors, well demonstrating its superiority.

## References

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection, 2020.

[3] Daniel Bogdoll, Maximilian Nitsche, and J Marius Zöllner. Anomaly detection in autonomous driving: A survey. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4488–4499, 2022.

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

[5] Yiqun Chen, Qiang Chen, Qinghao Hu, and Jian Cheng. Date: Dual assignment for end-to-end fully convolutional object detection. arXiv preprint arXiv:2211.13859, 2022.

[6] Yiqun Chen, Qiang Chen, Peize Sun, Shoufa Chen, Jingdong Wang, and Jian Cheng. Enhancing your trained detrs with box refinement. arXiv preprint arXiv:2307.11828, 2023.

[7] Yuming Chen, Xinbin Yuan, Ruiqi Wu, Jiabao Wang, Qibin Hou, and Ming-Ming Cheng. Yolo-ms: rethinking multi-scale representation learning for real-time object detection. arXiv preprint arXiv:2308.05480, 2023.

[8] François Chollet. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1251–1258, 2017.

[9] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11963–11975, 2022.

[10] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13733–13742, 2021.

[11] Douglas Henke Dos Reis, Daniel Welfer, Marco Antonio De Souza Leite Cuadros, and Daniel Fernando Tello Gamarra. Mobile robot navigation using an object recognition software with rgbd images and the yolo algorithm. Applied Artificial Intelligence, 33(14):1290–1305, 2019.

[12] Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian. Centernet: Keypoint triplets for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6569–6578, 2019.

[13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

[14] Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R Scott, and Weilin Huang. Tood: Task-aligned one-stage object detection. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3490–3499. IEEE Computer Society, 2021.

[15] Ruili Feng, Kecheng Zheng, Yukun Huang, Deli Zhao, Michael Jordan, and Zheng-Jun Zha. Rank diminishing in deep neural networks. Advances in Neural Information Processing Systems, 35:33054–33065, 2022.

[16] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021.

[17] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2918–2928, 2021.

[18] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015.

[19] Jocher Glenn. Yolov5 release v7.0. https: // github. com/ ultralytics/ yolov5/ tree/ v7. 0 , 2022.

[20] Jocher Glenn. Yolov8. https: // github. com/ ultralytics/ ultralytics/ tree/ main , 2023.

[21] Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, and Matthijs Douze. Levit: a vision transformer in convnet’s clothing for faster inference. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12259– 12269, 2021.

[22] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.

[23] Jan Hosang, Rodrigo Benenson, and Bernt Schiele. Learning non-maximum suppression. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4507–4515, 2017.

[24] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

[25] Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3588–3597, 2018.

[26] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015.

[27] Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng Cheng, Bo Zhang, Zaidan Ke, Xiaoming Xu, and Xiangxiang Chu. Yolov6 v3.0: A full-scale reloading. arXiv preprint arXiv:2301.05586, 2023.

[28] Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M Ni, and Lei Zhang. Dn-detr: Accelerate detr training by introducing query denoising. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13619–13627, 2022.

[29] Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss v2: Learning reliable localization quality estimation for dense object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11632–11641, 2021.

[30] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. Advances in Neural Information Processing Systems, 33:21002–21012, 2020.

[31] Ming Lin, Hesen Chen, Xiuyu Sun, Qi Qian, Hao Li, and Rong Jin. Neural architecture design for gpu-efficient networks. arXiv preprint arXiv:2006.14090, 2020.

[32] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

[33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

[34] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor boxes are better queries for detr. arXiv preprint arXiv:2201.12329, 2022.

[35] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8759–8768, 2018.

[36] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

[37] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

[38] Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel. Understanding the effective receptive field in deep convolutional neural networks. Advances in neural information processing systems, 29, 2016.

[39] Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang, Yanyi Liu, Shilong Zhang, and Kai Chen. Rtmdet: An empirical study of designing real-time object detectors. arXiv preprint arXiv:2212.07784, 2022.

[40] Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, and Jingdong Wang. Conditional detr for fast training convergence. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3651–3660, 2021.

[41] Victor M Panaretos and Yoav Zemel. Statistical aspects of wasserstein distances. Annual review of statistics and its application, 6:405–431, 2019.

[42] Joseph Redmon. Darknet: Open source neural networks in c. http://pjreddie.com/ darknet/, 2013–2016.

[43] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[44] Joseph Redmon and Ali Farhadi. Yolo9000: Better, faster, stronger. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[45] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement, 2018.

[46] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4510–4520, 2018.

[47] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019.

[48] Russell Stewart, Mykhaylo Andriluka, and Andrew Y Ng. End-to-end people detection in crowded scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2325–2333, 2016.

[49] Peize Sun, Yi Jiang, Enze Xie, Wenqi Shao, Zehuan Yuan, Changhu Wang, and Ping Luo. What makes for end-to-end object detection? In International Conference on Machine Learning, pages 9934–9944. PMLR, 2021.

[50] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14454–14463, 2021.

[51] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: A simple and strong anchor-free object detector. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(4):1922–1933, 2020.

[52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[53] Ao Wang, Hui Chen, Zijia Lin, Hengjun Pu, and Guiguang Ding. Repvit: Revisiting mobile cnn from vit perspective. arXiv preprint arXiv:2307.09283, 2023.

[54] Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, and Kai Han. Gold-yolo: Efficient object detector via gather-and-distribute mechanism. Advances in Neural Information Processing Systems, 36, 2024.

[55] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. Scaled-yolov4: Scaling cross stage partial network. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 13029–13038, 2021.

[56] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. Yolov7: Trainable bag-offreebies sets new state-of-the-art for real-time object detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7464–7475, 2023.

[57] Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. Cspnet: A new backbone that can enhance learning capability of cnn. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 390–391, 2020.

[58] Chien-Yao Wang, Hong-Yuan Mark Liao, and I-Hau Yeh. Designing network design strategies through gradient path analysis. arXiv preprint arXiv:2211.04800, 2022.

[59] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. Yolov9: Learning what you want to learn using programmable gradient information. arXiv preprint arXiv:2402.13616, 2024.

[60] Jianfeng Wang, Lin Song, Zeming Li, Hongbin Sun, Jian Sun, and Nanning Zheng. End-to-end object detection with fully convolutional network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15849–15858, 2021.

[61] Yingming Wang, Xiangyu Zhang, Tong Yang, and Jian Sun. Anchor detr: Query design for transformer-based detector. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 2567–2575, 2022.

[62] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22–31, 2021.

[63] Haiyang Xu, Zhichao Zhou, Dongliang He, Fu Li, and Jingdong Wang. Vision transformer with attention map hallucination and ffn compaction. arXiv preprint arXiv:2306.10875, 2023.

[64] Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang, Cheng Cui, Kaipeng Deng, Guanzhong Wang, Qingqing Dang, Shengyu Wei, Yuning Du, et al. Pp-yoloe: An evolved version of yolo. arXiv preprint arXiv:2203.16250, 2022.

[65] Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang, and Xiuyu Sun. Damo-yolo: A report on real-time object detection design. arXiv preprint arXiv:2211.15444, 2022.

[66] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. Motr: End-to-end multiple-object tracking with transformer. In European Conference on Computer Vision, pages 659–675. Springer, 2022.

[67] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.

[68] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

[69] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9759–9768, 2020.

[70] Wenqiang Zhang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu, and Chunhua Shen. Topformer: Token pyramid transformer for mobile semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12083–12093, 2022.

[71] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. Detrs beat yolos on real-time object detection. arXiv preprint arXiv:2304.08069, 2023.

[72] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-iou loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 12993–13000, 2020.

[73] Qiang Zhou and Chaohui Yu. Object detection made simpler by eliminating heuristic nms. IEEE Transactions on Multimedia, 2023.

[74] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.

[75] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6748–6758, 2023.

## A Appendix

## A.1 Implementation Details

Following [20, 56, 59], all YOLOv10 models are trained from scratch using the SGD optimizer for 500 epochs. The SGD momentum and weight decay are set to 0.937 and $5 \times 1 0 ^ { - 4 }$ , respectively. The initial learning rate is $1 \times 1 0 ^ { - 2 }$ and it decays linearly to $1 \times 1 0 ^ { - 4 } .$ . For data augmentation, we adopt the Mosaic [2, 19], Mixup [68] and copy-paste augmentation [17], etc., like [20, 59]. Tab. 14 presents the detailed hyper-parameters. All models are trained on 8 NVIDIA 3090 GPUs. Besides, we increase the width scale factor of YOLOv10-M to 1.0 to obtain YOLOv10-B. For PSA, we employ it after the SPPF module [20] and adopt the expansion factor of 2 for FFN. For CIB, we also adopt the expansion ratio of 2 for the inverted bottleneck block structure. Following [59, 56], we report the standard mean average precision (AP) across different object scales and IoU thresholds on the COCO dataset [33].

Moreover, we follow [71] to establish the end-to-end speed benchmark. Since the execution time of NMS is affected by the input, we thus measure the latency on the COCO val set, like [71]. We adopt the same NMS hyperparameters used by the detectors during their validation. The TensorRT efficientNMSPlugin is appended for post-processing and the I/O overhead is omitted. We report the average latency across all images.

Table 14: Hyper-parameters of YOLOv10.
<table><tr><td>hyper-parameter</td><td>YOLOv10-N/S/M/B/L/X</td></tr><tr><td>epochs</td><td>500</td></tr><tr><td>optimizer</td><td>SGD</td></tr><tr><td>momentum</td><td>0.937</td></tr><tr><td>weight decay</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>warm-up epochs</td><td>3</td></tr><tr><td>warm-up momentum</td><td>0.8</td></tr><tr><td>warm-up bias learning rate</td><td>0.1</td></tr><tr><td>initial learning rate</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>final learning rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>learning rate schedule</td><td>linear decay</td></tr><tr><td>box loss gain</td><td>7.5</td></tr><tr><td>class loss gain</td><td>0.5</td></tr><tr><td>DFL loss gain</td><td>1.5</td></tr><tr><td>HSV saturation augmentation</td><td>0.7</td></tr><tr><td>HSV value augmentation</td><td>0.4</td></tr><tr><td>HSV hue augmentation</td><td>0.015</td></tr><tr><td>translation augmentation</td><td>0.1</td></tr><tr><td>scale augmentation</td><td>0.5/0.5/0.9/0.9/0.9/0.9</td></tr><tr><td>mosaic augmentation</td><td>1.0</td></tr><tr><td>Mixup augmentation</td><td>0.0/0.0/0.1/0.1/0.15/0.15</td></tr><tr><td>copy-paste augmentation</td><td>0.0/0.0/0.1/0.1/0.3/0.3</td></tr><tr><td>close mosaic epochs</td><td>10</td></tr></table>

## A.2 Details of Consistent Matching Metric

We provide the detailed derivation of consistent matching metric here.

As mentioned in the paper, we suppose that the one-to-many positive samples is Ω and the one-toone branch selects i-th prediction. We can then leverage the normalized metric [14] to obtain the classification target for task alignment learning [20, 14, 59, 27, 64], i.e., $\begin{array} { r } { t _ { o 2 m , j } = \bar { u } ^ { * } \cdot \frac { m _ { o 2 m , j } } { m _ { o 2 m } ^ { * } } \leq u ^ { * } } \end{array}$ for $j \in \Omega$ and $\begin{array} { r } { t _ { o 2 o , i } = u ^ { * } \cdot \frac { m _ { o 2 o , i } } { m _ { o 2 o } ^ { * } } = u ^ { * } } \end{array}$ . We can thus derive the supervision gap between two branches by the 1-Wasserstein distance [41] of the different classification targets, $i . e .$

$$
\begin{array} { r l } & { A = | ( 1 - t _ { o 2 o , i } ) - ( 1 - \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } ) | + \sum _ { k \in \Omega \setminus \{ i \} } | 1 - ( 1 - t _ { o 2 m , k } ) | } \\ & { \quad = | t _ { o 2 o , i } - \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } | + \sum _ { k \in \Omega \setminus \{ i \} } t _ { o 2 m , k } } \\ & { \quad = t _ { o 2 o , i } - \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } + \sum _ { k \in \Omega \setminus \{ i \} } t _ { o 2 m , k } , } \end{array}\tag{3}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. We denote the classification targets of the predictions in Ω as $\{ \hat { t } _ { 1 } , \hat { t } _ { 2 } , . . . , \hat { t } _ { | \Omega | } \}$ in descending order, with $\hat { t } _ { 1 } \geq \hat { t } _ { 2 } \geq \dots \geq \hat { t } _ { | \Omega | }$ . We can then replace $t _ { o 2 o , i }$ with $u ^ { * }$ and obtain:

$$
\begin{array} { r l } & { A = u ^ { * } - \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } + \sum _ { k \in \Omega \backslash \{ i \} } t _ { o 2 m , k } } \\ & { \quad = u ^ { * } + \sum _ { k \in \Omega } t _ { o 2 m , k } - 2 \cdot \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } } \\ & { \quad = u ^ { * } + \sum _ { k = 1 } ^ { | \Omega | } \hat { t } _ { k } - 2 \cdot \mathbb { I } ( i \in \Omega ) t _ { o 2 m , i } } \end{array}\tag{4}
$$

We further discuss the supervision gap in two scenarios, i.e.,

1. Supposing $i \not \in \Omega .$ , we can obtain:

$$
A = u ^ { * } + et { } { ' } \sum _ { k = 1 } \hat { t } _ { k }\tag{5}
$$

2. Supposing $i \in \Omega .$ , we denote $t _ { o 2 m , i } = \hat { t } _ { n }$ and obtain:

$$
A = u ^ { * } + \sum _ { k = 1 } ^ { | \Omega | } \hat { t } _ { k } - 2 \cdot \hat { t } _ { n }\tag{6}
$$

Due to $\hat { t } _ { n } \geq 0$ , the second case can lead to smaller supervision gap. Besides, we can observe that A decreases as $\hat { t } _ { n }$ increases, indicating that n decreases and the ranking of i within Ω improves. Due to $\hat { t } _ { n } \leq \hat { t } _ { 1 } , A$ thus achieves the minimum when $\hat { t } _ { n } = \hat { t } _ { 1 } , i . e .$ , i is the best positive sample in Ω with $m _ { o 2 m , i } = m _ { o 2 m } ^ { * }$ and $\begin{array} { r } { t _ { o 2 m , i } = u ^ { * } \cdot \frac { m _ { o 2 m , i } } { m _ { o 2 m } ^ { * } } = u ^ { * } } \end{array}$

Furthermore, we prove that we can achieve the minimized supervision gap by the consistent matching metric. We suppose $\alpha _ { o 2 m } > 0$ and $\beta _ { o 2 m } > 0$ , which are common in [20, 59, 27, 14, 64]. Similarly, we assume $\alpha _ { o 2 o } > 0$ and $\beta _ { o 2 o } > 0$ . We can obtain $\begin{array} { r } { r _ { 1 } = \frac { \alpha _ { o 2 o } } { \alpha _ { o 2 m } } > 0 } \end{array}$ and $\begin{array} { r } { r _ { 2 } = \frac { \beta _ { o 2 o } } { \beta _ { o 2 m } } > 0 } \end{array}$ , and then derive $m _ { o 2 o }$ by

$$
\begin{array} { r l } & { m _ { o 2 o } = s \cdot p ^ { \alpha _ { o 2 o } } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { \beta _ { o 2 o } } } \\ & { \qquad = s \cdot p ^ { r _ { 1 } \cdot \alpha _ { o 2 m } } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { r _ { 2 } \cdot \beta _ { o 2 m } } } \\ & { \qquad = s \cdot ( p ^ { \alpha _ { o 2 m } } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { \beta _ { o 2 m } } ) ^ { r _ { 1 } } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { ( r _ { 2 } - r _ { 1 } ) \cdot \beta _ { o 2 m } } } \\ & { \qquad = m _ { o 2 m } ^ { r _ { 1 } } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { ( r _ { 2 } - r _ { 1 } ) \cdot \beta _ { o 2 m } } } \end{array}\tag{7}
$$

To achieve $m _ { o 2 m , i } = m _ { o 2 m } ^ { * }$ and $m _ { o 2 o , i } = m _ { o 2 o } ^ { * }$ , we can make $m _ { o 2 o }$ monotonically increase with $m _ { o 2 m }$ by assigning $( r _ { 2 } - r _ { 1 } ) = 0 , i . e .$

$$
\begin{array} { r l } { { m _ { o 2 o } } = { m _ { o 2 m } ^ { r _ { 1 } } } \cdot \mathrm { I o U } ( \hat { b } , b ) ^ { 0 \cdot \beta _ { o 2 m } } } \\ { ~ } & { { } = { m _ { o 2 m } ^ { r _ { 1 } } } } \end{array}\tag{8}
$$

Supposing $r _ { 1 } = r _ { 2 } = r$ , we can thus derive the consistent matching metric, $i . e . , \alpha _ { o 2 o } = r \cdot \alpha _ { o 2 m }$ and $\beta _ { o 2 o } = r \cdot \beta _ { o 2 m }$ . By simply taking $r = 1$ , we obtain $\alpha _ { o 2 o } = \alpha _ { o 2 m }$ and $\beta _ { o 2 o } = \beta _ { o 2 m }$

## A.3 Details of Rank-Guided Block Design

We present the details of the algorithm of rank-guided block design in Algo. 1. Besides, to calculate the numerical rank of the convolution, we reshape its weight to the shape of $( C _ { o } , K ^ { 2 } \times C _ { i } )$ , where $C _ { o }$ and $C _ { i }$ denote the number of output and input channels, and K means the kernel size, respectively.

## A.4 More Results on COCO

We report the detailed performance of YOLOv10 on COCO, including $\mathsf { A P } _ { 5 0 } ^ { v a l }$ and $\mathsf { A P } _ { 7 5 } ^ { v a l }$ at different IoU thresholds, as well as $\mathbf { A P } _ { s m a l l } ^ { v a l } , \mathbf { A P } _ { m e d i u m } ^ { v a l } ,$ and $\mathsf { A P } _ { l a r g e } ^ { v a l }$ across different scales, in Tab. 15.

## A.5 More Analyses for Holistic Efficiency-Accuracy Driven Model Design

We note that reducing the latency of YOLOv10-S (#2 in Tab. 2) is particularly challenging due to its small model scale. However, as shown in Tab. 2, our efficiency driven model design still achieves a 5.3% reduction in latency without compromising performance. This provides substantial support for the further accuracy driven model design. YOLOv10-S achieves a better latency-accuracy trade-off with our holistic efficiency-accuracy driven model design, showing a 2.0% AP improvement with only

Algorithm 1: Rank-guided block design   
Input: Intrinsic ranks R for all stages S; Original Network Θ; CIB $\theta _ { c i b } ;$   
Output: New network $\Theta ^ { * }$ with CIB for certain stages.   
1 t ← 0;   
2 $\Theta _ { 0 } \gets \Theta ; \Theta ^ { * } \gets \Theta _ { 0 } ;$   
3 $a p _ { 0 } \gets \Delta \mathbf { P } ( \mathrm { T } ( \Theta _ { 0 } ) )$ ; // T:training the network; AP:evaluating the AP performance.   
4 while $S \neq \emptyset$ do   
5 s<sub>t</sub> ← argmin $s \in S \mathrm { ~ } R ;$   
6 $\Theta _ { t + 1 } \gets ]$ Replace $\left( \Theta _ { t } , \theta _ { c i b } , \pmb { s } _ { t } \right)$ ; // Replace the block in Stage s<sub>t</sub> of $\Theta _ { t }$ with CIB $\theta _ { c i b }$   
7 $a p _ { t + 1 } \gets \mathrm { A P } ( \mathrm { T } ( \Theta _ { t + 1 } ) ) ;$   
8 if $a p _ { t + 1 } \geq$ ap<sub>0</sub> then   
9 $\Theta ^ { * }  \bar { \Theta _ { t + 1 } } ; S  S \setminus \{ s _ { t } \} .$   
10 else   
11 return $\Theta ^ { * } ;$   
12 end   
13 end   
14 return $\Theta ^ { * } ;$

Table 15: Detailed performance of YOLOv10 on COCO.
<table><tr><td>Model</td><td> $\mathbf { A P } ^ { v a l } ( \% )$ </td><td> $\mathrm { A P _ { 5 0 } ^ { \it v a l } } ( \% )$ </td><td> $\mathrm { A P } _ { 7 5 } ^ { v a l } ( \% )$ </td><td> $\mathrm { A P } _ { s m a l l } ^ { v a l } ( \% )$ </td><td> $\mathbf { A P } _ { m e d i u m } ^ { v a l } ( \% )$ </td><td> $\mathbf { A P } _ { l a r g e } ^ { v a l } ( \% )$ </td></tr><tr><td>YOLOv10-N</td><td>38.5</td><td>53.8</td><td>41.7</td><td>18.9</td><td>42.4</td><td>54.6</td></tr><tr><td>YOLOv10-S</td><td>46.3</td><td>63.0</td><td>50.4</td><td>26.8</td><td>51.0</td><td>63.8</td></tr><tr><td>YOLOv10-M</td><td>51.1</td><td>68.1</td><td>55.8</td><td>33.8</td><td>56.5</td><td>67.0</td></tr><tr><td>YOLOv10-B</td><td>52.5</td><td>69.6</td><td>57.2</td><td>35.1</td><td>57.8</td><td>68.5</td></tr><tr><td>YOLOv10-L</td><td>53.2</td><td>70.1</td><td>58.1</td><td>35.8</td><td>58.5</td><td>69.4</td></tr><tr><td>YOLOv10-X</td><td>54.4</td><td>71.3</td><td>59.3</td><td>37.0</td><td>59.8</td><td>70.9</td></tr></table>

0.05ms latency overhead. Besides, for YOLOv10-M (#6 in Tab. 2), which has a larger model scale and more redundancy, our efficiency driven model design results in a considerable 12.5% latency reduction, as shown in Tab. 2. When combined with accuracy driven model design, we observe a notable 0.8% AP improvement for YOLOv10-M, along with a favorable latency reduction of 0.48ms. These results well demonstrate the effectiveness of our design strategy across different model scales.

## A.6 Visualization Results

Fig. 4 presents the visualization results of our YOLOv10 in the complex and challenging scenarios. It can be observed that YOLOv10 can achieve precise detection under various difficult conditions, such as low light, rotation, etc. It also demonstrates a strong capability in detecting diverse and densely packed objects, such as bottle, cup, and person. These results indicate its superior performance.

## A.7 Contribution, Limitation, and Broader Impact

Contribution. In summary, our contributions are three folds as follows:

1. We present a novel consistent dual assignments strategy for NMS-free YOLOs. A dual label assignments way is designed to provide rich supervision by one-to-many branch during training and high efficiency by one-to-one branch during inference. Besides, to ensure the harmonious supervision between two branches, we innovatively propose the consistent matching metric, which can well reduce the theoretical supervision gap and lead to improved performance.

2. We propose a holistic efficiency-accuracy driven model design strategy for the model architecture of YOLOs. We present novel lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design, which greatly reduce the computational redundancy and achieve high efficiency. We further introduce the large-kernel convolution and innovative partial self-attention module, which effectively enhance the performance under low cost.

3. Based on the above approaches, we introduce YOLOv10, a new real-time end-to-end object detector. Extensive experiments demonstrate that our YOLOv10 achieves the state-of-the-art performance and efficiency trade-offs compared with other advanced detectors.

![](images/9c4e95c7ad7ba8ecc05acf5c18dfe2f8fa268e6b21aee54418f9aee728ef6ddd.jpg)  
Figure 4: Visualization results under complex and challenging scenarios.

Limitation. Due to the limited computational resources, we do not investigate the pretraining of YOLOv10 on large-scale datasets, e.g., Objects365 [47]. Besides, although we can achieve competitive end-to-end performance using the one-to-one head under NMS-free training, there still exists a performance gap compared with the original one-to-many training using NMS, especially noticeable in small models. For example, in YOLOv10-N and YOLOv10-S, the performance of one-to-many training with NMS outperforms that of NMS-free training by 1.0% AP and 0.5% AP, respectively. We will explore ways to further reduce the gap and achieve higher performance for YOLOv10 in the future work.

Broader impact. The YOLOs can be widely applied in various real-world applications, including medical image analyses and autonomous driving, etc. We hope that our YOLOv10 can assist in these fields and improve the efficiency. However, we acknowledge the potential for malicious use of our models. We will make every effort to prevent this.