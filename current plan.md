Project: Multimodal Deepfake Detection with Faithfulness Evaluation

Goal:
Build a model that detects deepfakes and evaluates explanation faithfulness using CMAS.

Architecture:
- Visual: EfficientNet-B4 → feature vector F_v
- Audio: wav2vec 2.0 → feature vector F_a
- Fusion: Cross-modal transformer (bidirectional attention)
- Classifier: 2-layer MLP (GELU + dropout)

Outputs:
- Prediction: real/fake
- Visual explanation: GradCAM
- Audio explanation: Integrated Gradients

Key Metric:
CMAS (Cross-Modal Attribution Score)
- Compare attribution vector [v, a] with ground truth [1,0] or [0,1]

Dataset:
FakeAVCeleb (FARV, RAFV, FAFV)

Current Goal:
Implement Phase 1–4 (model only, no XAI yet)