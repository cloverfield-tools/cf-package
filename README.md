# Cloverfield

A new kind of transformer featuring streaming cross-modal attention.

## Overview

Cloverfield is trained on next-token prediction from video lessons synthesized using tools like Synthesia.io to create math and captioned physics courses complete with photorealistic, engine-grounded physics from Unreal Engine.

We draw from lessons like DeepSeek's OCR paper for unified representations, and conduct traditional next-token-prediction training on multimodal content: audio, video, and OCR text. Supports synchronized multi-channel inputs.

## SPCE — Spectral Phase-Coherent Encoding

*Pronounced "space". Elegant, literal, mnemonic, and thematically right.*

### 💡 Ideate

SPCE describes what we're really encoding: a continuous spectral phase field that spans space and time. Each token lives as a point on a helical manifold in this space. Every modality—text, audio, video—shares the same spectral coordinate system, so cross-modal alignment is natural.

### 🪞 Reflect Critically

It's not a positional encoding anymore. It's a **field embedding**. RoPE mapped discrete indices to rotations. SPCE embeds events into a continuous oscillatory manifold whose phase evolves smoothly with real time and frequency. The only tricky part is precision and drift over long runs, but that's solvable with re-normalization and periodic low-frequency re-anchoring.

### 🔭 Expand Orthogonally

- Implement with complex exponential basis `e^{iωt}` where `ω` spans a learned spectral distribution
- Maintain phase continuity across windows: `θ_{t+Δt} = θ_t + ωΔt`
- Give each attention head its own ω-distribution; this yields multi-scale temporal sensitivity
- Couple SPCE with the SSM carry so low frequencies persist and high frequencies refresh
- Extend to 3D by adding spatial frequencies `e^{i(k_x x + k_y y + k_z z)}`; the same math covers motion, depth, and camera pose
- When multi-view training arrives, SPCE becomes the shared coordinate frame for every camera—literally shared space

## Architecture Highlights

- **Cross-modal attention**: Unified attention mechanism across text, audio, and video modalities
- **Streaming processing**: Real-time processing of multi-channel synchronized inputs
- **Spectral phase encoding**: Continuous field embeddings that naturally align cross-modal data
- **Multi-scale temporal sensitivity**: Per-head frequency distributions for capturing different temporal scales

## Training Data

- Synthesized video lessons (math, physics)
- Photorealistic physics simulations from Unreal Engine
- OCR-extracted text with unified representations
- Synchronized audio, video, and text channels

## Status

🚧 This project is in active development.
