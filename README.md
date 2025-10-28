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

**Unbounded streaming transformer** with perfect token retrieval and efficient text encoding:

1. **SPCE** (Spectral Phase-Coherent Encoding) — continuous phase field coordinates
2. **Visual text encoding** — 10× compression following DeepSeek OCR lessons
3. **Integrated retrieval** — perfect recall of exact input tokens, unbounded context
4. **Windowed attention** — efficient O(w²) local processing
5. **SSM carry** — cross-window state persistence
6. **Keyframe anchoring** — periodic drift correction

### 1. SPCE — Spectral Phase-Coherent Encoding

**Replace all positional encodings with absolute-time spectral phase:**

```
θ = ω·t + φ₀
```

**Not a positional encoding—a coordinate system.** Every modality (audio, video, text) shares the same spectral phase field. Beats, motion, and language stay naturally synchronized through phase coherence.

**Design:**
- **Shared spectral palette**: 12–24 log-spaced ω atoms (global pool)
- **Per-head gates**: Each attention head learns softmax-weighted mixture of atoms
- **No per-token ω**: Frequencies are head-level, not token-level (reduces parameters)
- **Absolute time evaluation**: `θ = ω·t` computed directly (no cumulative drift)

**Why this works:**
- Audio beat at t=1.5s → phase φ(1.5)
- Video frame at t=1.5s → same phase φ(1.5)
- Caption word at t=1.5s → same phase φ(1.5)
- Phase coherence = automatic cross-modal alignment

**Key property:** SPCE rotation preserves semantic embeddings (orthogonal to content, like RoPE). Full NLU capability maintained.

### 2. Visual Text Encoding (DeepSeek OCR Approach)

**Lesson from DeepSeek:** Text as compressed visual tokens achieves 10× compression with 97% fidelity.

**Architecture:**
```
Text → Rendered Image → Windowed SAM (80M) → CLIP Global (300M) → 16× Compression → Visual Tokens
```

**Unified representation:**
- Video patches, audio spectrograms, and text ALL encoded as visual tokens
- Single modality-agnostic transformer processes everything
- Text compression: 5000 text tokens → 500 visual tokens (10×)

**Advantages:**
- Maintains semantic structure (layout, equations, formatting)
- Efficient: Fewer tokens without information loss
- Unified: No separate text vs vision pathways

### 3. Integrated Retrieval Architecture

**Three-tier memory for unbounded context:**

#### **Tier 1: Windowed Attention (Local Context)**
```
Window size: 2048 tokens
Complexity: O(w²) = O(4M) operations
Memory: O(w) = O(2K) tokens
```

Efficient local attention with SPCE phase rotation. Handles immediate context within sliding window.

#### **Tier 2: Chunked Knowledge Retrieval (RETRO-Style)**
```
Chunk size: 64 tokens
Retrieved neighbors: k=5 per chunk
Database: Physics knowledge base (millions of examples)
```

For every 64-token chunk, retrieve k=5 similar physics scenarios from pre-built database:
- Similar experimental setups
- Relevant equations and derivations
- Analogous phenomena from different domains

Cross-attention from window tokens to retrieved knowledge chunks.

#### **Tier 3: Exact Token Retrieval (kNN Memory)**
```
Search: k=8 nearest neighbors per query
Index: FAISS approximate nearest neighbors
Keys: Phase-rotated token embeddings (SPCE-encoded)
```

**Perfect recall:** Retrieve exact input tokens from arbitrarily long history via kNN search.

**Phase-aware similarity:** Tokens with similar phase patterns (temporal structure) retrieved together.

**Unbounded:** No context limit—memory grows with conversation/video length.

**Example:**
```
t=0:      User asks "What is projectile motion?"
t=5000:   User asks "Apply that formula here"
          → kNN retrieves exact tokens from t=0 ("projectile motion", formula)
          → SSM carry maintains topic state ("discussing mechanics")
          → Perfect long-range reference despite 2048-token window
```

### 4. Spectral SSM Carry

**Diagonal, low-rank state space model** for cross-window state:

```
x_next = exp(-a·Δt) ⊙ (b·x + gain·u)
```

**Properties:**
- Eigenvalues constrained to unit circle (bounded memory)
- Stores slow harmonics (low frequencies) and entity slots (objects, speakers, topics)
- Updated once per window shift
- Maintains conversational/narrative state across windows

**Coupling with SPCE:**
- High frequencies (ω ≈ 10³): Handled by attention (fast refresh)
- Low frequencies (ω ≈ 10⁻⁴): Handled by SSM carry (stable persistence)

### 5. Keyframes

**Periodic anchors at fixed interval T seconds** (e.g., every 2-5 seconds):

```
[KEYFRAME] <cam_pose> <object_ids> <θ̂_anchor>
```

**Purpose:**
- Refresh slow phase anchors (`θ̂`)
- Store camera pose, object IDs, lighting state
- Enforce drift penalty (loss term: `|θ̂_predicted - ω·t|`)
- Enable safe rewind/resume mid-stream
- Anchor points for retrieval index

**During training**: Keyframes provide supervision for SSM carry
**During inference**: Small corrections at keyframes prevent long-tail drift

### 6. Stateful Inference with Adaptive Capacity

The inference engine maintains state across conversations and supports **low-rank Hebbian adaptation**—dynamically expanding effective attention capacity through query-focused adaptation on current context. This enables efficient fine-tuning to specific conversations or domains during inference without full retraining.

*Implementation details deferred to inference optimization phase.*

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

### Retrieval Performance
- **kNN recall accuracy**: Exact token retrieval from 100K+ token history
- **Retrieval latency**: <10ms for k=8 neighbors via FAISS
- **Phase-aware similarity**: Temporal pattern matching accuracy
- **Knowledge retrieval relevance**: Chunked physics DB retrieval precision

### Computational Efficiency
- **Throughput** (tokens/sec) with fused kernels
- **Memory footprint** during unbounded streaming (should be O(w) constant)
- **Keyframe overhead** (should be <5% of total compute)
- **Retrieval overhead** (RETRO + kNN, should be <10% of total compute)

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

## Computational Efficiency Analysis

### TL;DR: Competitive with Mamba, Much Better Than Standard Transformers

**For streaming 4-hour video (≈400K tokens):**

| Architecture | Time Complexity | Memory | Throughput | Notes |
|--------------|----------------|---------|------------|-------|
| Standard Transformer | O(n²) = O(160B) | O(n²) | 1× baseline | Quadratic wall |
| FlashAttention | O(n²) = O(160B) | O(n) | 2-3× | Memory-efficient, still quadratic |
| Mamba | O(n) = O(400K) | O(1) | 5× | Linear time, constant memory |
| **Cloverfield** | **O(w²) ≈ O(4M)** | **O(1)** | **4-6×** | **Fixed window + SSM carry** |

*n = total sequence length, w = window size (e.g., 2048)*

---

### Detailed Breakdown

#### 1. Visual Text Compression (DeepSeek OCR)

**Text tokenization overhead:**
- Traditional BPE: 5000 text tokens for typical page
- DeepSeek visual encoding: 500 visual tokens (10× compression)
- Encoding cost: Windowed SAM (80M) + CLIP (300M) = ~400M ops per page
- **Amortized**: One-time encoding, 97% fidelity maintained

**Net benefit:**
- 90% reduction in sequence length for text-heavy content
- Unified visual representation simplifies architecture
- Preserves semantic structure (equations, layout, formatting)

#### 2. SPCE Overhead vs RoPE

**RoPE baseline:**
- Rotary embedding: ~1-3% overhead (negligible in practice)
- Applied per-layer with fused CUDA kernels
- Dominated by matrix multiplies, not position encoding

**SPCE overhead:**
- Absolute phase computation: `θ = ω_head · t` (same cost as RoPE)
- Per-head gate: Softmax over K=12-24 atoms (once per forward pass)
  - Cost: O(heads × K) ≈ 32 heads × 16 atoms = 512 ops
  - Negligible compared to attention (millions of ops)
- Complex rotary apply: Same as RoPE (fused kernel)

**Verdict:** SPCE ≈ RoPE overhead (1-3%), potentially slightly better since ω is per-head, not per-token.

#### 3. Windowed Attention + Integrated Retrieval vs Full Attention

**Standard attention over full context:**
```
Attention(Q, K, V) where K, V ∈ ℝⁿˣᵈ
Cost: O(n² · d) per layer
Memory: O(n² + n · d) for attention matrix + KV cache
```

**Cloverfield windowed attention + retrieval + SSM:**
```
Window attention: Q, K, V ∈ ℝʷˣᵈ where w << n
Cost: O(w² · d) per layer (constant w)

RETRO retrieval: k=5 neighbors per 64-token chunk
Cost: O((w/64) × k × d_encode) ≈ O(160 × d_encode)
Memory: O(k × chunk_size × d) = O(5 × 64 × d) (constant!)

kNN memory: k=8 nearest neighbors per query
Cost: O(w × log(n)) for FAISS search (amortized)
Memory: O(n × d) for full history index (grows with session)

SSM carry: x_next = exp(-a·Δt) ⊙ (b·x + gain·u)
Cost: O(d_ssm) per window shift (d_ssm ≈ 256-512)
Memory: O(d_ssm) (constant!)

Total per window: O(w² + w·log(n) + d_ssm)
```

**Example (4-hour video @ 100 tokens/sec):**
- n = 400,000 tokens (full context)
- w = 2,048 tokens (fixed window)

**Standard attention:**
- 400K² = 160 billion ops per layer
- 52 GB memory (KV cache)

**Cloverfield:**
- Window attention: 2K² = 4 million ops
- RETRO retrieval: 32 chunks × 5 neighbors × encode = ~160K ops
- kNN search: 2K × log(400K) ≈ 37K ops (FAISS)
- SSM update: 512 ops
- **Total: ~4.2M ops per layer (38,000× reduction!)**
- **Memory: 0.3 GB window + index (173× reduction in active memory)**

**Key insight:** Retrieval cost is negligible compared to attention savings.

#### 4. Keyframe Overhead

**Emission frequency:**
- Keyframe every T seconds (e.g., T=3s)
- At 100 tokens/sec: 1 keyframe per 300 tokens
- Overhead: ~0.33% extra tokens

**Keyframe operations:**
- Store state: O(d_ssm) = O(256-512)
- Drift penalty loss: O(heads) = O(32)
- Total: <1ms per keyframe (negligible)

**Verdict:** Keyframe overhead <1% of total compute.

#### 4. Streaming Comparison

**Scenario: Process 4-hour educational video (400K tokens)**

**Standard Transformer (e.g., GPT-4, Claude):**
- Must split into chunks (context window limit: 128K-1M tokens)
- Each chunk processed independently: O(chunk_size²)
- Total: O(n_chunks × chunk_size²)
- KV cache grows with context: O(n · d)
- **Problem:** Can't maintain state across hours without reprocessing

**FlashAttention Transformer:**
- IO-aware, memory-efficient, but still O(n²)
- Reduces HBM accesses by 10-20×, speeds up 2-3×
- Still quadratic: 400K² = 160B ops per layer
- **Problem:** Throughput degrades as sequence grows

**Mamba:**
- Linear time: O(n) = 400K ops per layer
- Constant memory: O(1) state
- 5× throughput vs standard attention
- **Limitation:** Weak at in-context learning (copying tasks)

**Cloverfield:**
- Fixed window: O(w²) = 2K² = 4M ops per layer
- SSM carry: O(d_ssm) per window shift = 256 ops
- Total per window: ~4M ops (constant!)
- Number of windows: n/w = 400K/2K = 200 windows
- Total: 200 × 4M = 800M ops (vs 160B for standard attention)
- **200× reduction in total ops**
- Memory: Constant O(w² + d_ssm)

#### 5. Memory Footprint

**Standard attention KV cache:**
```
Memory = n_layers × n_tokens × n_heads × head_dim × 2 (K+V) × bytes_per_param
For 6B model (32 layers, 400K tokens, 32 heads, 128 dim, fp16):
= 32 × 400K × 32 × 128 × 2 × 2 bytes
= 52 GB just for KV cache!
```

**Cloverfield:**
```
Memory = window_cache + SSM_state
Window cache = n_layers × w × n_heads × head_dim × 2 × 2
= 32 × 2048 × 32 × 128 × 2 × 2
= 268 MB (fixed!)

SSM state = n_layers × d_ssm × 2
= 32 × 512 × 2
= 33 KB (negligible!)

Total: ~270 MB (constant, independent of video length)
```

**Savings: 52 GB → 0.27 GB = 193× reduction**

#### 6. Throughput Estimates

**Benchmarks (from literature):**
- Standard attention: 1× baseline
- FlashAttention: 2-3× faster than standard
- Mamba: 5× faster than standard
- Hybrid (Mamba-2): 8× faster at inference

**Cloverfield estimates:**
- Attention on fixed window: Similar to FlashAttention (2-3×)
- SSM carry overhead: Negligible (<5%)
- Keyframe overhead: Negligible (<1%)
- Expected: **4-6× throughput vs standard attention**
- Memory-bound streaming: **No degradation with sequence length**

#### 7. Training Efficiency

**Standard multimodal transformer:**
- Batch size limited by memory (full context KV cache)
- Gradient checkpointing required for long sequences
- Slow convergence on long-range dependencies

**Cloverfield:**
- Fixed memory per window → larger batch sizes possible
- SSM carry trained with closed-form gradients (stable)
- Keyframe supervision provides dense signal for long-range learning
- Expected: **2-3× faster training convergence** on long videos

---

### When Cloverfield Wins

**✅ Excels at:**
1. **Long-form streaming** (hours of video)
   - Constant memory, no context limit
   - Standard transformers hit memory wall
2. **Phase-sensitive tasks** (A/V sync, music, speech)
   - SPCE naturally captures temporal coherence
3. **Real-time inference** (live video processing)
   - Fixed latency per window
   - Predictable compute budget
4. **Multi-hour reasoning** (educational lectures)
   - SSM carry maintains entity state
   - Keyframes provide periodic supervision

**⚠️ Potentially weaker at:**
1. **Strong copying/in-context learning**
   - Mamba's limitation applies here too
   - Mitigation: Hybrid attention-SSM (like Mamba-2 Hybrid)
2. **Very short sequences** (<2K tokens)
   - Full attention might be fine
   - SPCE/SSM overhead not justified
3. **Random access** (jumping around document)
   - Sequential processing assumption
   - Mitigation: Process multiple streams in parallel

---

### Architecture Comparison Summary

| Aspect | Standard Transformer | Mamba | Cloverfield |
|--------|---------------------|-------|-------------|
| **Time complexity** | O(n²) | O(n) | O(w² + w·log n) ≈ constant |
| **Memory (active)** | O(n²) → O(n) cache | O(1) | O(w) (constant) |
| **Memory (total)** | O(n) | O(1) | O(n) for kNN index |
| **Throughput** | 1× | 5× | 4-6× |
| **Context limit** | 128K-1M tokens | Unbounded | Unbounded |
| **Perfect recall** | Full attention | No | Yes (kNN retrieval) |
| **Knowledge retrieval** | No | No | Yes (RETRO-style) |
| **Phase alignment** | Learned | Learned | Built-in (SPCE) |
| **NLU capability** | Excellent | Good | Excellent (visual text) |
| **Streaming latency** | High (recompute) | Low (constant) | Low (constant) |
| **Memory @ 400K tokens** | 52 GB active | ~1 GB | 0.3 GB active + index |
| **Multi-hour stability** | Poor (no carry) | Good (SSM) | Excellent (SSM+keyframes+retrieval) |

---

### Implementation Efficiency Notes

**CUDA kernel priorities:**
1. **Fused SPCE rotary** (biggest impact)
   - Compute ω_head from gates + atoms
   - Apply complex rotation to Q, K
   - Target: Match RoPE performance (1-3% overhead)

2. **Diagonal SSM update** (second priority)
   - Closed-form `exp(-a·Δt) ⊙ ...`
   - Float64 accumulation for stability
   - Target: <5% overhead per window shift

3. **Keyframe injection** (low priority)
   - Simple token insertion
   - Can be done in PyTorch/MLX (not critical path)

**Expected development time:**
- Triton prototypes: 1-2 weeks
- Optimized CUDA: 2-4 weeks
- Performance validation: 1 week

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

### Training Philosophy: Quality Over Scale

- **Textbooks Are All You Need** (Gunasekar et al., 2023)
  Quality over scale for code generation. Phi-1 (1.3B params) with curated synthetic data outperforms larger models.
  *arXiv:2306.11644*

- **Textbooks Are All You Need II** (Li et al., 2023)
  Phi-1.5 for natural language reasoning tasks.
  *arXiv:2309.05463*

- **Phi-2** (Microsoft Research, 2023)
  2.7B model matching 25× larger models through data quality.
  *Microsoft Research Blog*

### Unified Multimodal Architectures

- **Gemini 1.5: Unlocking multimodal understanding** (Gemini Team, Google, 2024)
  Joint vision-language transformers with unified multimodal backbone. Direct cross-modal tokenization where image patches, speech spectrograms, and text sequences coexist in the same latent representation.
  *arXiv:2403.05530* | [PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)

- **Meta-Transformer: A Unified Framework for Multimodal Learning** (Zhang et al., 2023)
  Unified tokenizer across 12 modalities with shared token space. Performs multimodal perception without paired training data.
  *arXiv:2307.10802* | [Project](https://kxgong.github.io/meta_transformer/)

- **ImageBind: One Embedding Space To Bind Them All** (Girdhar et al., 2023)
  Joint embedding space across 6 modalities (vision, text, audio, depth, thermal, IMU).
  *CVPR 2023* | *arXiv:2305.05665*

- **Chameleon: Mixed-Modal Early-Fusion Foundation Models** (Meta AI, 2024)
  Early-fusion token-based architecture treating images and text as unified vocabulary.
  *arXiv:2405.09818*

- **UniForm: Unified Diffusion Transformer for Audio-Video** (Zhao et al., 2025)
  Unified latent space for audio and video with single diffusion process.
  *arXiv:2502.03897* | [Project](https://uniform-t2av.github.io/)

### Visual Text Encoding & Compression

- **DeepSeek-OCR: Context Optical Compression** (DeepSeek AI, 2024)
  10× text compression via visual token encoding with 97% fidelity. Windowed SAM (80M) for local detail + CLIP (300M) for global layout + 16× convolutional compression. Unified vision-language architecture processing text as images.
  *arXiv:2510.18234* | [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR) | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

### Retrieval-Augmented Architectures

- **RETRO: Retrieval-Enhanced Transformer** (Borgeaud et al., DeepMind, 2021)
  Chunked cross-attention to retrieved neighbors from 2T token database. 7.5B params achieve GPT-3 175B performance (25× smaller).
  *arXiv:2112.04426* | [Blog](https://deepmind.google/discover/blog/improving-language-models-by-retrieving-from-trillions-of-tokens/)

- **Memorizing Transformers** (Wu et al., Google, ICLR 2022)
  kNN-augmented attention with exact token retrieval from external memory. FAISS approximate nearest neighbors for 262K token memory with negligible overhead.
  *arXiv:2203.08913* | [GitHub](https://github.com/lucidrains/memorizing-transformers-pytorch)

- **REALM: Retrieval-Augmented Language Model Pre-Training** (Guu et al., Google, ICML 2020)
  End-to-end learned retrieval with backprop through millions of documents. 300M params outperform T5 11B (37× smaller).
  *arXiv:2002.08909*

- **Fusion-in-Decoder (FiD)** (Izacard & Grave, Meta, 2020)
  Independent encoding of retrieved passages with joint fusion in decoder cross-attention. State-of-the-art open-domain QA.
  *GitHub*: [facebookresearch/FiD](https://github.com/facebookresearch/FiD)

- **ATLAS: Few-shot Learning with Retrieval Augmented Language Models** (Izacard et al., Meta, JMLR 2023)
  Joint pre-training of Contriever retriever + FiD model with on-the-fly index updates. 11B params, 50× smaller than comparable models, 42% accuracy on NaturalQuestions with only 64 examples.
  *arXiv:2208.03299* | [GitHub](https://github.com/facebookresearch/atlas)

- **Perceiver IO** (Jaegle et al., DeepMind, 2021)
  Cross-attention from learned latent queries to arbitrary inputs. Latent bottleneck acts as learned retrieval with no quadratic dependence on input size.
  *arXiv:2107.14795*

### State Space Models for Streaming

- **Mamba: Linear-Time Sequence Modeling** (Gu & Dao, 2023)
  Selective state spaces with linear time complexity, 5× faster than transformers. Constant memory for unbounded sequences.
  *arXiv:2312.00752*

- **Mamba-2: State Space Duality** (Dao & Gu, 2024)
  Structured state space duality connecting SSMs and attention. 8× faster inference for hybrid models.
  *Technical Report*

### Spectral & Frequency Domain Methods

- **Fourier Neural Operator (FNO)** (Li et al., 2020)
  Learns mappings between function spaces in Fourier domain. Resolution-invariant with global convolutions via FFT.
  *ICLR 2021* | *arXiv:2010.08895*

- **Adaptive Fourier Neural Operator (AFNO)** (Guibas et al., 2021)
  Efficient token mixer learning in Fourier domain with quasi-linear complexity. Block-diagonal structure with adaptive weight sharing.
  *NeurIPS 2021* | *arXiv:2111.13587* | [GitHub](https://github.com/NVlabs/AFNO-transformer)

- **Global Filter Network (GFNet)** (Rao et al., 2021)
  Replaces self-attention with learnable Fourier filters for efficient long-range dependency modeling.
  *NeurIPS 2021*

- **SpectFormer** (Pinto et al., 2023)
  Combines spectral layers (FNet, GFNet, AFNO) with multi-headed attention for hybrid architecture.
  *arXiv:2304.06446*

### Complex-Valued & Phase-Aware Networks

- **Complex-valued Neural Networks for Non-Stationary Physical Data** (Toms et al., 2020)
  Preserves phase information in seismic and signal processing. Smaller complex networks outperform larger real-valued networks.
  *Computers & Geosciences* | *arXiv:1905.12321*

- **A Survey of Complex-Valued Neural Networks** (Hirose & Yoshida, 2021)
  Comprehensive review of phase-preserving architectures for audio, MRI, and communications.
  *arXiv:2101.12249*

- **Phase-Aware Deep Learning with Complex CNNs for Audio** (Komatsu et al., 2024)
  Complex-valued CNNs for superior audio signal processing with explicit phase modeling.
  *arXiv:2510.09926*

### Continuous & Coordinate-Based Representations

- **Neural Ordinary Differential Equations** (Chen et al., 2018)
  Continuous-depth models treating network depth as continuous variable. Constant memory cost with adaptive evaluation.
  *NeurIPS 2018* | *arXiv:1806.07366*

- **Implicit Neural Representations with Periodic Activation (SIREN)** (Sitzmann et al., 2020)
  Sinusoidal activation functions for coordinate-based networks. Accurately represents signals and their derivatives.
  *NeurIPS 2020* | *arXiv:2006.09661* | [Project](https://www.vincentsitzmann.com/siren/)

- **NeRF: Neural Radiance Fields** (Mildenhall et al., 2020)
  Continuous 3D scene representation using positional encodings for high-frequency details.
  *ECCV 2020*

### Positional Encoding Methods

- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
  Rotary position embeddings unifying absolute and relative approaches. Negligible 1-3% overhead.
  *arXiv:2104.09864*

- **ALiBi: Attention with Linear Biases** (Press et al., 2022)
  Integrates positional information directly in attention computation. Better extrapolation beyond training length.
  *ICLR 2022*

- **Time-Aware Positional Encoding for Video** (Disney Research, 2024)
  Encodes relative time distance between video frames, not just order. Critical for A/V synchronization.
  *Technical Report*

### Related Work on Temporal Coherence

- **TimeSformer** (Bertasius et al., 2021)
  Divided space-time attention for video understanding with fixed/learnable spatial and temporal encodings.
  *ICML 2021*

---

### SPCE's Unique Contribution

While these approaches use:
- **Learned discrete embeddings** (Gemini, Meta-Transformer)
- **Frequency domain operations** (FNO, AFNO, GFNet)
- **Complex phase in forward pass only** (Complex CNNs)
- **Continuous depth but discrete positions** (Neural ODEs)

**SPCE provides:**
- **Continuous spectral phase field** as the primary coordinate system
- **Absolute time evaluation** θ = ω·t (no cumulative drift)
- **Cross-modal phase coherence** built into representation (not learned)
- **Physics-grounded coordinates** from Unreal Engine timesteps
- **Unbounded streaming** with SSM carry and keyframe anchoring

Closest analogy: **Gemini's unified backbone + FNO's spectral learning + SIREN's continuous coordinates**, with **SPCE as the rotary encoding** (replacing RoPE with continuous time), unified for multimodal streaming.
