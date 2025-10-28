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

## Implementation: Continuous Field Representation

### Philosophy

Treat SPCE as a continuous field with discrete evaluation. Keep one global tick. Encode phase as analytic functions with small parameter sets. **Never integrate noisy increments if you can compute phase from absolute time.** Bind all modalities to the same clock. Use a tiny SSM carry for slow harmonics. Fuse everything with next-token prediction.

### Core Principles

**The traps:**
- Numerical drift from cumulative integration
- Poor ω priors leading to unstable training
- Gradient fragility through phase computations
- Kernel cost from naive complex operations

**The solutions:**
- Absolute time evaluation: `θ = ωt + φ₀`
- Spectral priors with small mixtures
- Complex-safe autograd
- Fused CUDA kernels
- Measure and clamp everything

### Dual-View Phase Representation

Use two views of phase simultaneously:

1. **Absolute phase** from `ω × t` for stability
2. **Local incremental phase** for fine alignment inside the window

Blend them with a learned gate. Use keyframes to re-anchor slow terms. Learn ω as a small mixture per head. Share a tiny pool across heads and let heads pick.

### ⚖️ Best Tradeoff Design

- Absolute time phase computation
- Mixture of log-normal ω atoms per head
- Shared atom pool with per-head gates
- Single global tick with fractional offsets for each modality
- Fused complex rotary kernels

This is practical and fast.

---

## Technical Implementation

### 1. Continuous Parameterization

**Absolute time representation:**
```
t = 64-bit tick + float32 residual
θ = ωt + φ₀  // Direct computation, no cumulative sum, no drift
```

**Phase encoding:**
- Store phase as complex pair `(cos θ, sin θ)` to avoid unwrap in hot path
- Keep auxiliary unwrapped `θ̂` for long-range SSM only
- Use mixed precision safely:
  - Store `ω` in `float32`
  - Accumulate `t` in `float64`
  - Compute `cos θ, sin θ` in `float32` with range reduction

### 2. Phase Unwrapping at Scale

Keep two channels:

1. **Unit circle** `(cos θ, sin θ)` for attention
2. **Slow unwrapped** `θ̂` tracked only in the carry for low-frequency anchors

**Epoch-residual representation:**
```
θ̂ = E·π + r
```
- Update `E` only at keyframes
- Use Kahan-style compensated updates for `r`
- Optional Kalman corrector nudges `θ̂` toward `ωt` at keyframes to kill drift

### 3. Learnable Spectral Distributions

**Basis as small mixture per head:**
```
ωₕ(f) = Σₖ αₕₖ · ωₖ
```

where `ωₖ` are shared atoms.

**Configuration:**
- `K ∈ [8, 16]` atoms per pool
- Two pools:
  - **Low:** log-uniform from `10⁻⁴` to `1`
  - **High:** log-uniform from `1` to `10³` (in tick units)
- Heads learn gates `α` (simplex-constrained via softmax)
- Optional chirp term `dω/dt` for acceleration (tiny linear head, clamped)

### 4. Cross-Modal Synchronization

**One global tick** (e.g., 1/960 millisecond)

**Modality-specific fractional offsets:**
- Audio @ 16kHz → ticks with zero offset
- Video @ 30fps → ticks with fixed frame offset
- Text → event timestamps from narrator (may repeat last offset when idle)

**Token packing:**
- Pack tokens in tick order
- `[TICK]` tokens hard-anchor phase
- Backpressure tokens let encoders slow down to maintain sync

### 5. Training Dynamics

**Gradient flow through phase:**

Use complex rotary apply with Wirtinger-safe autograd:
```
z · e^(iθ) = [x cos θ - y sin θ, x sin θ + y cos θ]
```

**Regularization:**
- Clip `∂θ/∂ω` within sane range
- Spectral total variation on ω gates across layers (prevents jitter)
- Tiny auxiliary loss: predict next keyframe phase from carry (encourages stable slow harmonics)

**Initialization of ω:**
- Sample shared atoms log-uniform over modality Nyquist range (dictated by global tick)
- Seed per-head gates to broad low-frequency bias + small high-frequency bump
- **Warm-start rule:**
  1. Early epochs: freeze ω atoms, only train gates α
  2. Then unfreeze ω atoms with small learning rate

**Backprop with differential forms:**
- Treat SPCE as `θ = ωt` with optional chirp (no ODE integration needed)
- For SSM carry: use diagonal state space with closed-form update:
  ```
  x_next = exp(-aΔt) ⊙ (b·x + gain·u)
  ```
- Backprop through `exp` with stable Padé or series approximations
- Use precomputed lookup for common Δt

### 6. Computational Efficiency

**Phase field updates:**
- `θ(t + Δt)` from absolute `ωt` is one fused op per head
- Precompute `ωt` per block of tokens
- Cache `(cos, sin)` for small tile of `t`, reuse across heads with different ω
  - Angle addition tables for small ω grid (if memory allows)
  - Otherwise: compute directly with range reduction

**Attention in continuous space:**
- SPCE reduces to rotary-style complex multiply with head-specific ω
- Attention kernel remains standard scaled dot-product on rotated Q, K
- For very long windows:
  - Combine block-sparse attention with low-rank kernel attention
  - Use FAVOR-style random features to approximate softmax on rotated features

**Custom CUDA kernels:**

Fused operations to write:
1. Rotary with per-token ω and per-head ω gates
2. Tick-gated rotary (reads `[TICK]` anchors and phase offsets)
3. Complex-safe LayerNorm (avoid phase-amplitude drift)
4. Diagonal SSM update per window shift (`float32` with `float64` accumulator)

**Implementation plan:**
- Start with Triton to prototype fused rotary + gate
- Move to CUDA with explicit vectorization (half-precision inputs, float accumulation)
- **Profile memory bandwidth first** (rotary multiply is memory-bound)
- Use shared memory tiles, align to cache lines

### 7. Guardrails and Keyframes

**Keyframes:**
- Emit scheduled `[KEYFRAME]` tokens every N ticks
- Store: camera pose, speaker ID, lighting, slow phase anchor `θ̂`
- Loss term penalizes deviation between carry-predicted `θ̂` and `ωt` at keyframes
- During generation: allow small corrections at keyframes to prevent long-tail drift

**Spectral regularization:**
- L1 on high-frequency gate mass per head (bias toward parsimonious ω use)
- Total variation on gates across layers (keep frequencies consistent through depth)
- Soft cap on chirp magnitude

---

## Minimal Viable SPCE Recipe

1. Global tick and `[TICK]` tokens
2. Absolute phase compute: `θ = ωt`
3. Complex rotary apply with per-head ω gates
4. Shared ω atoms: `K = 12` per pool, two pools
5. Scheduled keyframes every few seconds with slow phase anchors in carry
6. Diagonal SSM carry for low frequencies only
7. Next-token prediction on interleaved multimodal streams

---

## Success Metrics

**Evals that matter:**

- **Phase residual** at keyframes over 4-hour runs
- **A/V sync error** (milliseconds) over hour-long clips
- **Identity persistence** across occlusions and scene cuts
- **Cross-modal perplexity** and long-horizon perplexity
- **Throughput** (tokens/sec) with fused kernels on A100 and consumer GPUs

---

### Essence

**Do not integrate noise.** Compute phase from absolute time. Learn a tiny spectral palette and let heads gate it. Share one clock across all modalities with fractional offsets. Re-anchor with keyframes. Use fused complex rotary kernels. Keep the carry slow and stable. The rest is next-token prediction.

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
