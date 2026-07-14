# Glossary

Shared abbreviation reference for all paper notes. See individual notes for paper-specific context.

## Terms needs footnote
| Term | Expansion | Brief Definition |
|------|-----------|-----------------|
| AP | Average Precision | Area under the precision-recall curve; COCO reports mean AP averaged over IoU thresholds 0.5:0.95 and object scales. |
| BA | Bundle Adjustment | Joint non-linear optimization of camera poses and 3D point positions to minimize reprojection error. |
| DETR | DEtection TRansformer | End-to-end object detector that treats detection as set prediction, using a transformer and bipartite (Hungarian) matching to remove hand-crafted components like NMS. |
| DPT | Dense Prediction Transformer | Decoder head that fuses multi-scale ViT features through progressive convolutional upsampling to full resolution. |
| FFN | Feed-Forward Network | Two-layer MLP sublayer inside each transformer block; applied per-token after attention. |
| InfoNCE | Information Noise-Contrastive Estimation | Contrastive loss that maximises mutual information between matched pairs by pulling them together and pushing unmatched pairs apart. |
| IoU | Intersection over Union | Overlap ratio between predicted and ground-truth boxes (area of intersection / area of union); the standard localization metric and matching criterion in detection. |
| MVS | Multi-View Stereo | Dense 3D reconstruction from multiple calibrated images by finding per-pixel depth via multi-view photo-consistency. |
| NMS | Non-Maximum Suppression | Post-processing step that removes duplicate detections by greedily keeping the highest-scoring box and suppressing overlapping boxes above an IoU threshold. |
| NVS | Novel View Synthesis | Task of rendering a scene from a viewpoint not present in the input images. |
| SLAM | Simultaneous Localization and Mapping | Problem of building a map of an unknown environment while tracking an agent's position within it in real time. |
| VLA | Vision-Language-Action model | Robot policy that takes image observations + natural-language instructions as input and outputs low-level actions (e.g., RT-2, OpenVLA). |
| VLM | Vision-Language Model | Model trained on image-text pairs to align visual and language representations (e.g., CLIP, LLaVA, GPT-4V). |

## Commonly-used Terms
| Term | Expansion | Brief Definition |
|------|-----------|-----------------|
| 3DGS | 3D Gaussian Splatting | Scene representation using millions of anisotropic 3D Gaussians optimized for novel-view synthesis. |
| CLIP | Contrastive Language-Image Pre-training | OpenAI model trained to align image and text embeddings via contrastive loss on 400M image-text pairs. |
| COLMAP | — | Widely-used open-source SfM + MVS pipeline (Schönberger & Frahm, CVPR 2016). |
| EMA | Exponential Moving Average | Weight update rule for teacher networks: θ_T ← m·θ_T + (1−m)·θ_S; keeps teacher a slow-moving ensemble of the student. |
| FLOPs | Floating Point Operations | Hardware-agnostic measure of computation cost (multiply-adds). |
| FoV | Field of View | Angular extent of the scene visible to a camera; determined by focal length and sensor size. |
| LLM | Large Language Model | Transformer-based model trained on large text corpora for general language understanding and generation (e.g., GPT-4, LLaMA). |
| MLP | Multi-Layer Perceptron | Fully-connected feedforward neural network. |
| NeRF | Neural Radiance Fields | Implicit scene representation as a continuous function (MLP or hash grid) mapping 3D position + view direction to color and density. |
| SfM | Structure from Motion | Pipeline that estimates 3D scene structure and camera poses from unordered 2D images via feature matching and bundle adjustment. |
| ViT | Vision Transformer | Transformer architecture applied to sequences of flattened image patches (Dosovitskiy et al., ICLR 2021). |
