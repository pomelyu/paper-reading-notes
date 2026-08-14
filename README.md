# Paper Notes

### Common
- [Technical Terms](common/terms/README.md)

### 2026
<ul>
  <li><details><summary><a href="2026/OpenDriveVLA-_Towards_End-to-end_Autonomous_Driving_with_Large_Vision_Language_Action_Model/">(OpenDriveVLA) OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model</a></summary>
    <ul>
      <li>receive multi-view images and ego state as input and predict car trajector.</li>
      <li>multi-stage training to align spatial tokens with LLM inputs and specialize the LLM for autonomous driving.</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2026/VGGT-Omega/">(VGGT-Ω) VGGT-Ω</a></summary>
    <ul>
      <li>VGGT improvement with lower GPU usage and better camera accuracy.</li>
      <li>predict depth map and camera parameters from a group of images or <b>a video</b></li>
      <li>propose a data annotation pipeline to use VLM to filter video data</li>
      <li>showing the scaling laws for 3D reconstruction in both model and data size</li>
    </ul>
  </details></li>
</ul>

### 2025
<ul>
  <li><a href="2025/GaussianDWM-_3D_Gaussian_Driving_World_Model_for_Unified_Scene_Understanding_and_Multi-Modal_Generation/">(GaussianDWM) 3D Gaussian Driving World Model for Unified Scene Understanding and Multi-Modal Generation</a></li>
  <li><a href="2025/LangSplatV2-_High-dimensional_3D_language_Gaussian_Splatting_with_450+_FPS/">(LangSplatV2) LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS</a></li>
  <li><a href="2025/4D_LangSplat-_4D_Language_Gaussian_Splatting_via_Multimodal_Large_Language_Models/">(4D LangSplat) 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models</a></li>
  <li><a href="2025/ObjectGS-_Object-aware_scene_reconstruction_and_scene_understanding_via_Gaussian_Splatting/">(ObjectGS) ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting</a></li>
  <li><a href="2025/RAD-_Training_an_End-to-End_Driving_Policy_via_Large-Scale_3DGS-based_Reinforcement_Learning/">(RAD) RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning</a></li>
  <li><a href="2025/Vision_Language_Models-_A_Survey_of_26K_Papers_(CVPR,_ICLR,_NeurIPS_2023-2025)/">(VLM Survey) Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025)</a></li>
  <li><a href="2025/DINOv3/">(DINOv3) DINOv3</a></li>
  <li><details><summary><a href="2025/VGGT-_Visual_Geometry_Grounded_Transformer/">(VGGT) VGGT: Visual Geometry Grounded Transformer</a></summary>
    <ul>
      <li>1.2B FFN transformer to predict depth map, camera parameters and etc. from a group of images.</li>
      <li>Backbone for downstream tasks, such as feed-forward NVS and dynamic point tracking</li>
    </ul>
  </details></li>
</ul>

### 2024
<ul>
  <li><details><summary><a href="2024/Medusa-_Simple_LLM_Inference_Acceleration_Framework_with_Multiple_Decoding_Heads/">(Medusa) Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a></summary>
    <ul>
      <li>Draft-model-free speculative decoding: predict K extra tokens from the last hidden state and verify all of them at once in a single forward pass</li>
      <li>Provide different training receipts: <b>Medusa-1</b> (frozen backbone, lossless, trainable on 1 GPU in hours) vs <b>Medusa-2</b> (joint training, faster); plus <b>self-distillation</b> (no data needed)</li>
      <li>2.2× (Medusa-1) to 2.3–2.8× (Medusa-2) wall-clock speedup at batch size 1 on Vicuna-7B/13B/33B, Zephyr-7B with ~unchanged quality; superseded on acceptance by EAGLE-family</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2024/DriveVLM-_The_Convergence_of_Autonomous_Driving_and_Large_Vision-Language_Models/">(DriveVLM) DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models</a></summary>
    <ul>
      <li>One of the papers that opened the VLM/VLA era of AD: a large VLM (Qwen-VL) predicts coarse waypoints via a driving-specific chain-of-thought — scene description → scene analysis (critical objects) → hierarchical planning</li>
      <li>The VLM handles long-tail / ambiguous scenes and produces interpretable decisions that classical perception→prediction→planning stacks miss</li>
      <li><b>DriveVLM-Dual</b> = slow-fast hybrid: the slow VLM (semantics + coarse plan) paired with a fast traditional 3D-detector/planner (VAD) for spatial grounding + waypoint refinement; deployed on a production car (~410 ms on dual OrinX), for which the paper details the choice of LLM and ViT (a smaller &lt;4B LLM is used onboard)</li>
    </ul>
  </details></li>
  <li><a href="2024/SAM_2-_Segment_Anything_in_Images_and_Videos/">(SAM2) SAM 2: Segment Anything in Images and Videos</a></li>
  <li><a href="2024/DUSt3R-_Geometric_3D_Vision_Made_Easy/">(DUSt3R) DUSt3R: Geometric 3D Vision Made Easy</a></li>
  <li><a href="2024/2D_Gaussian_Splatting_for_geometrically_accurate_radiance_fields">(2DGS) 2D Gaussian Splatting for Geometrically Accurate Radiance Fields</a></li>
  <li><a href="2024/4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering/">(4DGS) 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering</a></li>
  <li><a href="2024/Street_Gaussians-_Modeling_Dynamic_Urban_Scenes_with_Gaussian_Splatting/">(Street Gaussians) Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting</a></li>
  <li><a href="2024/HUGSIM-_A_Real-Time,_Photo-Realistic_and_Closed-Loop_Simulator_for_Autonomous_Driving/">(HUGSIM) HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving</a></li>
  <li><a href="2024/Gaussian_Splatting_SLAM/">(MonoGS) Gaussian Splatting SLAM</a></li>
  <li><details><summary><a href="2024/YOLOv10-_Real-Time_End-to-End_Object_Detection/">(YOLOv10) YOLOv10: Real-Time End-to-End Object Detection</a></summary>
    <ul>
      <li>Improved from YOLOv8 and removes the NMS post-processing via a dual-head training setup, hugely speeding up inference</li>
      <li>Optimizes the model structure design for better speed and accuracy</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2024/Extreme_Compression_of_Large_Language_Models_via_Additive_Quantization/">(AQLM) Extreme Compression of Large Language Models via Additive Quantization</a></summary>
    <ul>
      <li>Represent the weights as the sum of per-layer trained codebook elements. (weight-only)</li>
      <li>Large inference overhead due to codebook lookups and long training time(days), large calibration dataset to train codebook</li>
      <li>Beat by FP8 inference and MoE VRAM offload.</li>
    </ul>
  </details></li>
</ul>

### 2023
<ul>
  <li><details><summary><a href="2023/Planning-oriented_Autonomous_Driving/">(UniAD) Planning-oriented Autonomous Driving</a></summary>
    <ul>
      <li>CVPR 2023 best paper: unifies the full driving stack (detection, tracking, mapping, motion forecast, occupancy, planning) into one end-to-end network, connected by task queries</li>
      <li>"Planning-oriented" philosophy: choose and order perception/prediction tasks so each feeds better info to the final planner (vs. modular pipelines or naive multi-task learning)</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2023/FlashOcc-_Fast_and_Memory-Efficient_Occupancy_Prediction_via_Channel-to-Height_Plugin/">(FlashOcc) FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin</a></summary>
    <ul>
      <li>Plug-and-play efficiency trick to replace highly cost 3D convs by 2D conv in BEV and later use <b>Channel-to-Height</b> transform to recover height information</li>
      <li>Matches or beats voxel baselines (+1.3 mIoU on BEVDetOcc) at ~2× speed and ~69% less inference memory; a standard efficient occupancy head</li>
    </ul>
  </details></li>
  <li><a href="2023/Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis/">(Live 3D Portrait) Real-Time Radiance Fields for Single-Image Portrait View Synthesis</a></li>
  <li><a href="2023/Scaffold-GS-_Structured_3D_Gaussians_for_View-Adaptive_Rendering">(Scaffold-GS) Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering</a></li>
  <li><a href="2023/Dynamic_3D_Gaussians-_Tracking_by_Persistent_Dynamic_View_Synthesis/">(Dynamic 3D Gaussians) Tracking by Persistent Dynamic View Synthesis</a></li>
  <li><a href="2023/Segment_Anything/">(SAM) Segment Anything</a></li>
  <li><a href="2023/LangSplat-_3D_Language_Gaussian_Splatting/">(LangSplat) LangSplat: 3D Language Gaussian Splatting</a></li>
  <li><a href="2023/Tracking_Anything_with_Decoupled_Video_Segmentation/">(DEVA) Tracking Anything with Decoupled Video Segmentation</a></li>
  <li><a href="2023/Gaussian_Grouping-_Segment_and_Edit_Anything_in_3D_Scenes/">(Gaussian Grouping) Gaussian Grouping: Segment and Edit Anything in 3D Scenes</a></li>
  <li><a href="2023/Visual_Instruction_Tuning/">(LLaVA) Visual Instruction Tuning</a></li>
  <li><details><summary><a href="2023/Efficient_Memory_Management_for_Large_Language_Model_Serving_with_PagedAttention/">(vLLM)(PageAttention) Efficient Memory Management for Large Language Model Serving with PagedAttention</a></summary>
    <ul>
      <li>Complete inference system (vLLM) built on top of PagedAttention</li>
      <li>Identifies the KV-cache memory fragmentation in prior LLM serving systems that limits inference throughput, and solves it with paged memory management</li>
      <li>Allows block swapping (offloading to CPU RAM) or recomputation when under memory pressure</li>
      <li>Allows KV-cache sharing for a shared prompt-prefix (system prompt) or advanced decoding strategies (beam search, parallel sampling)</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2023/AWQ-_Activation-aware_Weight_Quantization_for_LLM_Compression_and_Acceleration/">(AWQ) AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration</a></summary>
    <ul>
      <li>Find the most salient weights by analyzing the input statistics.</li>
      <li>Use per-channel scaling to preserve the accuracy of salient weights (<code>X*Q(W) -> X/s*Q(sW)</code>)</li>
      <li>The input scaling can be fused into the previous layer, typically LayerNorm, making the operation free.</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2023/QLoRA-_Efficient_Finetuning_of_Quantized_LLMs/">(QLoRA) QLoRA: Efficient Finetuning of Quantized LLMs</a></summary>
    <ul>
      <li>Fine-tunes LoRA adapters over a frozen 4-bit NF4 base model while retaining bfloat16 computation.</li>
      <li>Reduces 65B fine-tuning memory from more than 780 GB to less than 48 GB using double quantization and paged optimizers.</li>
    </ul>
  </details></li>
</ul>

### 2022
<ul>
  <li><details><summary><a href="2022/BEVFormer-_Learning_Bird's-Eye-View_Representation_from_Multi-Camera_Images_via_Spatiotemporal_Transformers/">(BEVFormer) BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers</a></summary>
    <ul>
      <li>Attention-based ("pull") BEV encoder: grid of learnable BEV queries gather multi-camera features via deformable spatial cross-attention, no explicit depth needed (vs. LSS's depth-based "push")</li>
      <li>Recurrent temporal self-attention fuses the previous frame's BEV (RNN-style), greatly improving velocity estimation and occluded-object recall</li>
      <li>Unified BEV feature serves both 3D detection and map segmentation; 56.9% NDS on nuScenes test, on par with some LiDAR baselines</li>
      <li>Backbone for end-to-end driving stacks (UniAD) and occupancy models</li>
    </ul>
  </details></li>
</ul>

### 2021
<ul>
  <li><details><summary><a href="2021/DETR3D-_3D_Object_Detection_from_Multi-view_Images_via_3D-to-2D_Queries/">(DETR3D) DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries</a></summary>
    <ul>
      <li>Predict 3D object bbox, class label, heading and velocity from multi-view images</li>
      <li>Use concept similar with DETR and calculate set-to-set loss</li>
      <li>learnable object query (900) → MLP to predict draft 3D point centers → [project to 2D points in image space → get image features which are encoded by ResNet + FPN → use attention to refine center points] x6 → predict class label, box and heading, velocity</li>
    </ul>
  </details></li>
</ul>

### 2020
<ul>
  <li><details><summary><a href="2020/Lift,_Splat,_Shoot-_Encoding_Images_from_Arbitrary_Camera_Rigs_by_Implicitly_Unprojecting_to_3D/">(LSS) Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D</a></summary>
    <ul>
      <li>Fused images from multi-view cameras to a feature map in BEV view for downstream tasks.</li>
      <li>Turns each image into point cloude frustrum and then project to BEV view, use sum-pooling along each verical voxel(related to a pixel in feature map) to create feature map</li>
      <li>Foundational template for the BEV-perception wave (BEVDet, BEVDepth, BEVFusion, BEVFormer)</li>
    </ul>
  </details></li>
  <li><details><summary><a href="2020/End-to-End_Object_Detection_with_Transformers/">(DETR) End-to-End Object Detection with Transformers</a></summary>
    <ul>
      <li>Reframes detection as direct set prediction: CNN + transformer encoder-decoder with learned object queries, trained via bipartite (Hungarian) matching</li>
      <li>Removes anchors and NMS entirely; strong on large objects, weak on small ones, and needs very long training</li>
    </ul>
  </details></li>
</ul>
