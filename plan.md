# Project Plan

## Phase 1: Data Preprocessing
- [ ] Dataset loading
- [ ] Train/val/test split
- [ ] Face extraction (MediaPipe)
- [ ] Audio extraction (16kHz)
- [ ] DataLoader creation

## Phase 2: Feature Extraction
- [ ] EfficientNet-B4 integration
- [ ] wav2vec 2.0 integration

## Phase 3: Fusion
- [ ] Cross-modal transformer

## Phase 4: Classification
- [ ] MLP classifier


Read README.md and idea.md.

We are implementing a multimodal deepfake detection system using:
- EfficientNet-B4 (visual)
- wav2vec 2.0 (audio)
- Cross-modal transformer
- MLP classifier

Current goal:
Implement dataset loader + train/val/test split
Constraints:
- Keep code modular
- Use PyTorch
- Keep it scalable for later XAI integration

Start step-by-step and explain briefly.