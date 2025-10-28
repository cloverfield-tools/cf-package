# Cloverfield

**An unbounded streaming transformer for captioned physics video.**

## Vision

Cloverfield combines streaming cross-modal attention with physics-grounded training data to create a model that understands synchronized multimodal content—audio, video, and text—through continuous spectral phase coordinates.

Following the "Textbooks Are All You Need" philosophy (Phi-1, Phi-1.5, Phi-2), we prioritize **quality over scale**: curated, instructive, physics-grounded educational content over massive web scrapes.

## Training Philosophy: Quality Over Scale

### Inspired by "Textbooks Are All You Need"

Microsoft Research's Phi models (1.3B–2.7B parameters) demonstrated that **small models trained on high-quality synthetic data can match models 25× larger** on reasoning tasks.

Key principles:
- **Textbook quality**: Clear, self-contained, instructive content
- **Synthetic curation**: GPT-generated exercises and explanations
- **Data > Scale**: Phi-1's 7B tokens of curated data outperformed larger models on billions of web tokens

### Our Adaptation: Physics-Grounded Multimodal Data

We extend this philosophy to multimodal learning:

**Captioned educational videos** combining:
- Lecture narration (clear, instructive audio)
- Unreal Engine physics simulations (ground-truth dynamics)
- Synchronized captions and equation overlays (aligned text)

**Why physics simulations?**
- Perfect ground truth (forces, velocities, trajectories)
- Controllable complexity (start simple, scale systematically)
- Naturally aligned (engine timesteps = video frames = audio samples)
- Unlimited synthetic data at textbook quality

This is "textbook quality" for multimodal learning: precise, reproducible, instructive, and physically grounded.

---

## Core Architecture

### Primary Objective

**Unbounded streaming transformer** with:
1. **SPCE** (Spectral Phase-Coherent Encoding) — continuous field coordinates
2. **Spectral SSM carry** — long-range harmonics and entity memory
3. **Keyframe anchoring** — periodic drift correction

### 1. SPCE — Spectral Phase-Coherent Encoding

**Replace all positional encodings with absolute-time spectral phase:**

```
θ = ω·t + φ₀
```

**Not a positional encoding—a coordinate system.** Every modality (audio, video, text) shares the same spectral phase field. Beats, motion, and language stay naturally synchronized through phase coherence.

**Simplified design:**
- **Shared spectral palette**: 12–24 log-spaced ω atoms (global pool)
- **Per-head gates**: Each attention head learns softmax-weighted mixture of atoms
- **No per-token ω**: Frequencies are head-level, not token-level (reduces parameters)
- **Absolute time evaluation**: `θ = ω·t` computed directly (no cumulative drift)

**Why this works:**
- Audio beat at t=1.5s → phase φ(1.5)
- Video frame at t=1.5s → same phase φ(1.5)
- Caption word at t=1.5s → same phase φ(1.5)
- Phase coherence = automatic cross-modal alignment

### 2. Spectral SSM Carry

**Diagonal, low-rank state space model** for long-term memory:

```
x_next = exp(-a·Δt) ⊙ (b·x + gain·u)
```

**Properties:**
- Eigenvalues constrained to unit circle (bounded memory)
- Stores slow harmonics (low frequencies) and entity slots (objects, speakers)
- Updated once per sliding-window shift
- Provides the "carry state" that persists across streaming windows

**Coupling with SPCE:**
- High frequencies (ω ≈ 10³) handled by attention (fast refresh)
- Low frequencies (ω ≈ 10⁻⁴) handled by SSM carry (stable persistence)

### 3. Keyframes

**Periodic anchors at fixed interval T seconds** (e.g., every 2-5 seconds):

```
[KEYFRAME] <cam_pose> <object_ids> <θ̂_anchor>
```

**Purpose:**
- Refresh slow phase anchors (`θ̂`)
- Store camera pose, object IDs, lighting state
- Enforce drift penalty (loss term: `|θ̂_predicted - ω·t|`)
- Enable safe rewind/resume mid-stream

**During training**: Keyframes provide supervision for SSM carry
**During inference**: Small corrections at keyframes prevent long-tail drift

---

## Training Pipeline

### Curriculum

**Stage 1: Captioned lecture videos** (static scenes)
- Math lessons with voiceover + whiteboard
- Synchronized captions and equation overlays
- Focus: audio-text alignment, long-form reasoning

**Stage 2: Physics simulation videos** (dynamic scenes)
- Unreal Engine: projectile motion, collisions, rigid body dynamics
- Force vectors, velocity arrows, trajectory overlays
- Focus: visual dynamics, spatial reasoning, physics grounding

**Stage 3: Mixed datasets**
- Combined lectures + simulations
- Transfer learning and generalization
- Real-world educational content (Khan Academy, MIT OCW, etc.)

### Input Structure

Tokens are packed in **tick order** (1 tick ≈ 1/960 ms):

```
[TICK_0000] [KEYFRAME] <cam_pose> <speaker_id>
[TICK_0001] <video_patch_1> <audio_sample_1>
[TICK_0002] <video_patch_2> <audio_sample_2> <caption_word_1>
[TICK_0003] <video_patch_3> <audio_sample_3>
...
[TICK_1920] [KEYFRAME] <cam_pose> <object_pose_ball>
[TICK_1921] <force_vector> <velocity_arrow>
...
```

**Modality-specific encodings:**
- **Video**: Patch tokens @ 30fps → ticks with frame offset
- **Audio**: Waveform samples @ 16kHz → ticks with zero offset
- **Text**: Caption words with narrator timestamps → tick-aligned
- **Physics overlays**: Force vectors, equations → tick-aligned with video

**Key tokens:**
- `[TICK]`: Hard anchor for phase (emitted every N ticks)
- `[KEYFRAME]`: Store state, refresh anchors
- `<cam_pose>`, `<object_pose>`: 3D spatial grounding
- `<force_vector>`, `<equation>`: Physics annotations

---

## Simplified Implementation

### SPCE Core

**1. Shared ω atoms** (12–24 per pool, 2 pools: low + high freq)
```python
ω_low = log_uniform(1e-4, 1)     # Slow harmonics
ω_high = log_uniform(1, 1e3)     # Fast dynamics
```

**2. Per-head gates** (softmax-constrained mixture)
```python
ω_head = Σₖ softmax(α_head)[k] · ω_k
```

**3. Absolute phase**
```python
θ = ω_head · t  # Direct computation, no drift
rotary_apply(Q, K, θ)  # Standard RoPE-style attention
```

### SSM Carry

**Diagonal state space** (closed-form update)
```python
x_carry = exp(-a·Δt) * (b·x_prev + gain·input)
```

**Constraints:**
- `a` constrained to unit circle eigenvalues
- Low-rank projection for entity slots
- Updated once per window shift (not per token)

### Keyframes

**Scheduled emission** every `T` seconds:
```python
if t % keyframe_interval == 0:
    emit [KEYFRAME]
    store cam_pose, object_ids, θ̂_anchor
    loss += |θ̂_predicted - ω·t|  # Drift penalty
```

---

## Success Metrics

**What matters for captioned physics video:**

### Phase Stability
- **Phase residual** at keyframes over 4-hour streaming runs
- **Drift accumulation** (should be near-zero with absolute `θ = ω·t`)

### Cross-Modal Alignment
- **A/V sync error** (milliseconds) on hour-long lecture videos
- **Caption timing accuracy** (word-level alignment)

### Physics Understanding
- **Force vector prediction** from video (next-token prediction on physics overlays)
- **Trajectory extrapolation** (predict ball position from dynamics)
- **Equation grounding** (match visual motion to symbolic equations)

### Long-Horizon Reasoning
- **Cross-modal perplexity** over multi-hour videos
- **Identity persistence** across scene cuts, occlusions
- **Concept transfer** from static lectures to dynamic simulations

### Computational Efficiency
- **Throughput** (tokens/sec) with fused kernels
- **Memory footprint** during unbounded streaming
- **Keyframe overhead** (should be <5% of total compute)

---

## Implementation Roadmap

### Phase 1: SPCE Validation (1–2 months)
- Implement shared ω atoms + per-head gates in MLX/PyTorch
- Compare SPCE vs RoPE on audio-only task (music beat prediction)
- **Success criterion**: SPCE matches or beats RoPE on phase-sensitive tasks

### Phase 2: Single-Modal Streaming (2–3 months)
- Build SSM carry for audio streaming
- Add keyframe anchoring
- Test on long-form podcast transcription (audio + text)
- **Success criterion**: Unbounded streaming with <1ms drift/hour

### Phase 3: Physics-Grounded Video (3–6 months)
- Generate Unreal Engine physics datasets (100–1000 hours)
- Train on captioned physics simulations
- Evaluate force vector prediction and trajectory extrapolation
- **Success criterion**: Predict physics overlays from video with >80% accuracy

### Phase 4: Full Multimodal (6–12 months)
- Scale to 6B parameters with QLoRA on M4 Max
- Train on mixed curriculum (lectures + simulations + real videos)
- Publish results and open-source model
- **Success criterion**: Match or exceed general-purpose VLMs on physics reasoning benchmarks

---

## Why This Can Work

### ✅ Grounded in Proven Principles

**"Textbooks Are All You Need"** showed:
- 1.3B model + 7B tokens curated data > 10B+ model on web data
- Quality beats scale for reasoning tasks

**SPCE** extends this to multimodal:
- Physics simulations = "textbook quality" for video
- Continuous phase field = natural alignment
- Streaming SSM = unbounded context

### ✅ Technically Feasible

**Hardware**: 6B model trainable on M4 Max with QLoRA (~10GB VRAM)

**Data**: Unreal Engine enables unlimited synthetic physics data at textbook quality

**Architecture**: SPCE simplifies to RoPE-style kernels (proven, fast)

### ✅ Unique Advantages

**vs. Gemini/GPT-4o:**
- Physics-grounded understanding (force, motion, causality)
- Unbounded streaming (no context window limit)
- Continuous phase alignment (automatic A/V sync)

**vs. Academic models:**
- Real implementation focus (not just theoretical)
- Curated data strategy (not web-scale scraping)
- Open-source from day one

---

## Essence

**Do not integrate noise.** Compute phase from absolute time. Learn a tiny spectral palette and let heads gate it. Share one clock across all modalities. Re-anchor with keyframes. Use physics simulations as textbook-quality training data. The rest is next-token prediction.

---

## Status

🚧 **Active Development**

Current focus: SPCE prototype implementation in MLX

---

## References

- **Textbooks Are All You Need** (Gunasekar et al., 2023) — Quality over scale for code generation
- **Textbooks Are All You Need II** (Li et al., 2023) — Phi-1.5 for natural language
- **Phi-2** (Microsoft Research, 2023) — 2.7B model matching 25× larger models
- **Mamba** (Gu & Dao, 2023) — Linear-time SSMs for unbounded sequences
- **RoFormer** (Su et al., 2021) — Rotary position embeddings
