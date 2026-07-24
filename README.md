# Paper Notes

### Common
- [Technical Terms](common/terms/README.md)

### 2026
- [x] [VGGT-Ω](2026/VGGT-Omega/)
  - VGGT improvement with lower GPU usage and better camera accuracy.
  - predict depth map and camera parameters from a group of images or **a video**
  - propose a data annotation pipeline to use VLM to filter video data
  - showing the scaling laws for 3D reconstruction in both model and data size

### 2025
- [x] [(GaussianDWM) 3D Gaussian Driving World Model for Unified Scene Understanding and Multi-Modal Generation](2025/GaussianDWM-_3D_Gaussian_Driving_World_Model_for_Unified_Scene_Understanding_and_Multi-Modal_Generation/)
- [x] [(LangSplatV2) LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](2025/LangSplatV2-_High-dimensional_3D_language_Gaussian_Splatting_with_450+_FPS/)
- [x] [(4D LangSplat) 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](2025/4D_LangSplat-_4D_Language_Gaussian_Splatting_via_Multimodal_Large_Language_Models/)
- [x] [(ObjectGS) ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting](2025/ObjectGS-_Object-aware_scene_reconstruction_and_scene_understanding_via_Gaussian_Splatting/)
- [ ] [(RAD) RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning](2025/RAD-_Training_an_End-to-End_Driving_Policy_via_Large-Scale_3DGS-based_Reinforcement_Learning/)
- [x] [Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025)](2025/Vision_Language_Models-_A_Survey_of_26K_Papers_(CVPR,_ICLR,_NeurIPS_2023-2025)/)
- [x] [DINOv3](2025/DINOv3/)

- [x] [(VGGT) VGGT: Visual Geometry Grounded Transformer](2025/VGGT-_Visual_Geometry_Grounded_Transformer/)
  - 1.2B FFN transformer to predict depth map, camera parameters and etc. from a group of images.
  - Backbone for downstream tasks, such as feed-forward NVS and dynamic point tracking

### 2024
- [x] [(DriveVLM) DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](2024/DriveVLM-_The_Convergence_of_Autonomous_Driving_and_Large_Vision-Language_Models/)
  - One of the papers that opened the VLM/VLA era of AD: a large VLM (Qwen-VL) predicts coarse waypoints via a driving-specific chain-of-thought — scene description → scene analysis (critical objects) → hierarchical planning
  - The VLM handles long-tail / ambiguous scenes and produces interpretable decisions that classical perception→prediction→planning stacks miss
  - **DriveVLM-Dual** = slow-fast hybrid: the slow VLM (semantics + coarse plan) paired with a fast traditional 3D-detector/planner (VAD) for spatial grounding + waypoint refinement; deployed on a production car (~410 ms on dual OrinX), for which the paper details the choice of LLM and ViT (a smaller <4B LLM is used onboard)
- [x] [(SAM2) SAM 2: Segment Anything in Images and Videos](2024/SAM_2-_Segment_Anything_in_Images_and_Videos/)
- [x] [(DUSt3R) DUSt3R: Geometric 3D Vision Made Easy](2024/DUSt3R-_Geometric_3D_Vision_Made_Easy/)
- [x] [(2DGS) 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](2024/2D_Gaussian_Splatting_for_geometrically_accurate_radiance_fields)
- [x] [(4DGS) 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](2024/4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering/)
- [x] [(Street Gaussians) Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting](2024/Street_Gaussians-_Modeling_Dynamic_Urban_Scenes_with_Gaussian_Splatting/)
- [x] [(HUGSIM) HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving](2024/HUGSIM-_A_Real-Time,_Photo-Realistic_and_Closed-Loop_Simulator_for_Autonomous_Driving/)
- [x] [(MonoGS) Gaussian Splatting SLAM](2024/Gaussian_Splatting_SLAM/)
- [x] [(YOLOv10) YOLOv10: Real-Time End-to-End Object Detection](2024/YOLOv10-_Real-Time_End-to-End_Object_Detection/)
  - Improved from YOLOv8 and removes the NMS post-processing via a dual-head training setup, hugely speeding up inference
  - Optimizes the model structure design for better speed and accuracy
- [ ] [(AQLM) Extreme Compression of Large Language Models via Additive Quantization](2024/Extreme_Compression_of_Large_Language_Models_via_Additive_Quantization/)
  - Represent the weights as the sum of per-layer trained codebook elements. (weight-only)
  - Large inference overhead due to codebook lookups and long training time(days), large calibration dataset to train codebook
  - Beat by FP8 inference and MoE VRAM offload.

### 2023
- [x] [(UniAD) Planning-oriented Autonomous Driving](2023/Planning-oriented_Autonomous_Driving/)
  - CVPR 2023 best paper: unifies the full driving stack (detection, tracking, mapping, motion forecast, occupancy, planning) into one end-to-end network, connected by task queries
  - "Planning-oriented" philosophy: choose and order perception/prediction tasks so each feeds better info to the final planner (vs. modular pipelines or naive multi-task learning)
- [x] [(FlashOcc) FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin](2023/FlashOcc-_Fast_and_Memory-Efficient_Occupancy_Prediction_via_Channel-to-Height_Plugin/)
  - Plug-and-play efficiency trick to replace highly cost 3D convs by 2D conv in BEV and later use **Channel-to-Height** transform to recover height information
  - Matches or beats voxel baselines (+1.3 mIoU on BEVDetOcc) at ~2× speed and ~69% less inference memory; a standard efficient occupancy head
- [x] [(Lp3D) Real-Time Radiance Fields for Single-Image Portrait View Synthesis](2023/Real-Time_Radiance_Fields_for_Single-Image_Portrait_View_Synthesis/)
- [x] [(Scaffold-GS) Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering](2023/Scaffold-GS-_Structured_3D_Gaussians_for_View-Adaptive_Rendering)
- [x] [(Dynamic 3D Gaussians) Tracking by Persistent Dynamic View Synthesis](2023/Dynamic_3D_Gaussians-_Tracking_by_Persistent_Dynamic_View_Synthesis/)
- [x] [(SAM) Segment Anything](2023/Segment_Anything/)
- [x] [(LangSplat) LangSplat: 3D Language Gaussian Splatting](2023/LangSplat-_3D_Language_Gaussian_Splatting/)
- [ ] [(DEVA) Tracking Anything with Decoupled Video Segmentation](2023/Tracking_Anything_with_Decoupled_Video_Segmentation/)
- [x] [(Gaussian Grouping) Gaussian Grouping: Segment and Edit Anything in 3D Scenes](2023/Gaussian_Grouping-_Segment_and_Edit_Anything_in_3D_Scenes/)
- [x] [(LLaVA) Visual Instruction Tuning](2023/Visual_Instruction_Tuning/)
- [x] [(vLLM)(PageAttention) Efficient Memory Management for Large Language Model Serving with PagedAttention](2023/Efficient_Memory_Management_for_Large_Language_Model_Serving_with_PagedAttention/)
  - Complete inference system (vLLM) built on top of PagedAttention
  - Identifies the KV-cache memory fragmentation in prior LLM serving systems that limits inference throughput, and solves it with paged memory management
  - Allows block swapping (offloading to CPU RAM) or recomputation when under memory pressure
  - Allows KV-cache sharing for a shared prompt-prefix (system prompt) or advanced decoding strategies (beam search, parallel sampling)

### 2022
- [x] [(BEVFormer) BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](2022/BEVFormer-_Learning_Bird's-Eye-View_Representation_from_Multi-Camera_Images_via_Spatiotemporal_Transformers/)
  - Attention-based ("pull") BEV encoder: grid of learnable BEV queries gather multi-camera features via deformable spatial cross-attention, no explicit depth needed (vs. LSS's depth-based "push")
  - Recurrent temporal self-attention fuses the previous frame's BEV (RNN-style), greatly improving velocity estimation and occluded-object recall
  - Unified BEV feature serves both 3D detection and map segmentation; 56.9% NDS on nuScenes test, on par with some LiDAR baselines
  - Backbone for end-to-end driving stacks (UniAD) and occupancy models

### 2021
- [x] [(DETR3D) DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](2021/DETR3D-_3D_Object_Detection_from_Multi-view_Images_via_3D-to-2D_Queries/)
  - Predict 3D object bbox, class label, heading and velocity from multi-view images
  - Use concept similar with DETR and calculate set-to-set loss
  - learnable object query (900) → MLP to predict draft 3D point centers → [project to 2D points in image space → get image features which are encoded by ResNet + FPN → use attention to refine center points] x6 → predict class label, box and heading, velocity

### 2020
- [x] [(LSS) Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](2020/Lift,_Splat,_Shoot-_Encoding_Images_from_Arbitrary_Camera_Rigs_by_Implicitly_Unprojecting_to_3D/)
  - Fused images from multi-view cameras to a feature map in BEV view for downstream tasks.
  - Turns each image into point cloude frustrum and then project to BEV view, use sum-pooling along each verical voxel(related to a pixel in feature map) to create feature map
  - Foundational template for the BEV-perception wave (BEVDet, BEVDepth, BEVFusion, BEVFormer)
- [x] [(DETR) End-to-End Object Detection with Transformers](2020/End-to-End_Object_Detection_with_Transformers/)
  - Reframes detection as direct set prediction: CNN + transformer encoder-decoder with learned object queries, trained via bipartite (Hungarian) matching
  - Removes anchors and NMS entirely; strong on large objects, weak on small ones, and needs very long training
