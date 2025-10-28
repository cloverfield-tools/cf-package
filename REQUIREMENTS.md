# Cloverfield Requirements & Goals

## Executive Summary

**Cloverfield** is a general-purpose multimodal foundation model designed to achieve state-of-the-art performance in its weight class (6B parameters, trainable on Apple M4 Max) across text, image, audio, video, and code modalities.

**Core Competitive Advantage**: Physics-grounded world model that ensures generated content follows real-world causality, dynamics, and constraints—not just visual or linguistic patterns.

**"SOTA in Weight Class"** means: Best performance among models with 4-10B parameters that can be trained from scratch on Apple M4 Max (≤128GB unified memory).

### Target Users

**Creative Professionals:**
- Image/video editors seeking natural language editing ("Give Mom a Christmas sweater")
- Game developers needing asset generation and scene creation
- Content creators wanting AI avatars and interactive storytelling

**Software Builders:**
- Developers using code generation with visual context
- Unity/Unreal developers seeking automated scene/effect creation

**Consumer Applications:**
- Interactive AI conversations with visual avatars
- Exploratory experiences in AI-generated virtual worlds (holodeck vision)

**Long-term vision**: Real-time generation and exploration of interactive virtual environments—holodeck on a laptop. Users can converse with AI characters, generate and modify scenes on-the-fly, and explore persistent worlds that maintain physical consistency.

---

## Model Class Definition

### Size Constraints
- **Parameter count**: 6B parameters (target)
- **Training hardware**: Apple M4 Max 128GB unified memory (50-60GB footprint during training)
- **Training approach**: From scratch with novel SPCE architecture (not compatible with pretrained weights)
- **Inference**: Efficient enough for edge deployment (M-series, consumer GPUs, <10GB VRAM with quantization)

### Competitive Weight Class
Models in the 4-10B parameter range:
- Phi-3 Medium (14B - upper bound reference)
- Gemma 7B
- Mistral 7B
- LLaMA 3 8B
- DeepSeek-Coder 6.7B
- Qwen 7B
- StableLM 7B

**Goal**: Match or exceed these models on capability-specific benchmarks while offering multimodal coherence and physics grounding they lack.

---

## Core Capabilities: SOTA Definitions

### 1. Natural Language Understanding & Generation

**"SOTA in weight class" means:**
- **MMLU (Massive Multitask Language Understanding)**: ≥70% (match Mistral 7B)
- **HellaSwag (Common Sense)**: ≥80% (match LLaMA 3 8B)
- **TruthfulQA**: ≥55% (exceed typical 7B models)
- **HumanEval (Code reasoning)**: ≥45% (competitive with DeepSeek-Coder)
- **Conversational coherence**: Multi-turn dialogue maintaining context over 32K tokens

**Use Cases:**
- Multi-turn technical support conversations
- Document analysis and summarization
- Creative writing assistance
- Real-time tutoring with physics grounding (explaining how things work)
- Code review and explanation

**Technical Requirements:**
- Window size: 8K-16K tokens (retrieval + SSM carry handle long-range dependencies)
- Visual text encoding (DeepSeek OCR approach) for 16× compression with 97% fidelity
- Three-tier retrieval for unbounded effective context
- SSM carry for cross-window conversational state (low-frequency phase coherence)

---

### 2. Video Understanding (Semantic Search, Q&A, Summarization)

**"SOTA in weight class" means:**
- **ActivityNet-QA**: Match GPT-4V performance on video question answering
- **Semantic search**: Sub-second retrieval of relevant moments in 4-hour videos
- **Action recognition**: ≥85% on Kinetics-400
- **Physics reasoning**: Novel capability—identify physical errors in synthetic videos (e.g., wrong gravitational acceleration, momentum violations)

**Use Cases:**
- Search across lecture recordings: "Show me where the instructor explains Newton's second law"
- Sports analysis: "Find all the dunks in this 2-hour game"
- Safety monitoring: "Identify unsafe behaviors in warehouse footage"
- Physics tutoring: "What's wrong with how this ball bounces in the simulation?"
- Content moderation: Semantic understanding of video context

**Technical Requirements:**
- 16× video compression for understanding pathway (DeepSeek approach)
- Window size: 4K-8K tokens (retrieval finds relevant moments in long-form video)
- SPCE phase alignment for automatic audio-visual synchronization
- Physics-grounded representations for causality reasoning

**Competitive Positioning:**
- Builds on user's Tout/Stanford 2012 expertise in video semantic search
- Unique advantage: Physics grounding enables error detection (GPT-4V, Gemini can't identify F≠ma violations)

---

### 3. Image Generation

**"SOTA in weight class" means:**
- **CLIP Score**: ≥0.32 (match Stable Diffusion 2.1)
- **FID (Fréchet Inception Distance)**: ≤12 on COCO (competitive with SDXL at similar param count)
- **Physical consistency**: Novel metric—generated images follow real-world physics (shadows match light sources, support structures adequate for weight, etc.)
- **Prompt adherence**: ≥85% on compositional prompts (multi-object scenes)

**Use Cases:**
- Educational content: Generate diagrams showing physics concepts (force vectors, trajectories)
- Product visualization: Realistic renders that obey physics (materials, lighting, weight distribution)
- Architectural previews: Structurally plausible building designs
- Game asset generation: Props and environments with realistic physics properties
- Scientific illustration: Accurate visual representations of physical phenomena

**Technical Requirements:**
- 4× video compression pathway repurposed for high-fidelity image generation
- Physics-grounded latent space (learned F=ma, optics, material properties)
- Diffusion or autoregressive generation conditioned on SPCE-encoded prompts

**Competitive Advantage:**
- Generated images are **structurally plausible**, not just visually convincing
- Shadows, reflections, support structures follow real physics
- Reduces AI-generated artifacts that violate common sense

---

### 4. Video Generation

**"SOTA in weight class" means:**
- **UCF-101 FVD (Fréchet Video Distance)**: ≤150 (match CogVideo/ModelScope)
- **Temporal coherence**: No flickering or object disappearance across frames
- **Physical realism**: **Novel capability**—generated motion follows F=ma, momentum conservation, realistic collision dynamics
- **4-second generation**: High-fidelity 30fps output at 512×512 resolution

**Use Cases:**
- **Educational animations**: Visualize physics problems (projectile motion, pendulums, collisions)
- **Game cutscenes**: Physically accurate character animations and object interactions
- **Product demos**: Show how products move and interact realistically
- **Synthetic training data**: Generate physics-correct videos for robotics/autonomous vehicle training
- **Special effects**: Background elements that obey real-world physics

**Technical Requirements:**
- 4× video compression for generation pathway (not 16×—too lossy)
- Window size: ~4K tokens for 4-second clips @ 30fps @ 4× compression
- Physics-grounded world model from Unreal Engine training
- Autoregressive or diffusion generation with momentum/causality constraints

**Competitive Advantage (THE MOAT):**
- **Sora/Runway/Pika generate visually impressive but physically implausible videos** (objects float, momentum ignored, causality violated)
- **Cloverfield learns F=ma, conservation laws, realistic dynamics** from Unreal Engine ground truth
- Generated videos are **reality-consistent**, not just pattern-matched
- Critical for applications requiring physical accuracy (education, simulation, robotics training)

---

### 5. Audio Generation

**"SOTA in weight class" means:**
- **FAD (Fréchet Audio Distance)**: ≤2.5 on MusicCaps (competitive with MusicLM)
- **Speech naturalness**: MOS (Mean Opinion Score) ≥4.0 (match VALL-E)
- **Audio-visual sync**: Perfect lip-sync and action-sound alignment via SPCE
- **Physics-based sound**: Novel capability—generated sounds match physical interactions (footsteps on gravel vs. wood, collision sounds based on material/velocity)

**Use Cases:**
- **Physics-correct sound effects**: Generate impact sounds based on object materials and velocities
- **Narration for generated video**: Synchronized voiceover with lip-sync
- **Music composition**: Background scores that match video pacing
- **Voice cloning**: Few-shot speaker adaptation
- **Sound design**: Realistic environmental audio for games/simulations

**Technical Requirements:**
- Audio tokenization at 50Hz (~3K tokens per minute)
- Window size: 4K-8K tokens (retrieval handles long-form audio context)
- SPCE phase alignment for audio-visual synchronization
- Physics-grounded audio synthesis (learned from Unreal Engine audio simulations)

**Competitive Advantage:**
- **Automatic audio-visual sync** through shared SPCE phase field (no separate alignment model needed)
- **Physics-grounded sound generation** (collision sounds match material properties and impact velocity)

---

### 6. Code Generation (via Distillation)

**"SOTA in weight class" means:**
- **HumanEval (Pass@1)**: ≥60% (match DeepSeek-Coder 6.7B)
- **MBPP (Mostly Basic Python Problems)**: ≥55%
- **Multi-language support**: Python, JavaScript, TypeScript, Rust, Go
- **Physics simulation code**: Novel capability—generate accurate physics simulation code (Unity/Unreal scripts, numerical solvers)

**Use Cases:**
- **Physics simulation scripting**: Generate Unreal Engine blueprints or Python physics simulations
- **Code explanation**: Multi-modal explanation combining code, diagrams, and physics reasoning
- **Bug detection**: Identify physics errors in simulation code
- **Robotics programming**: Generate control code with realistic dynamics constraints
- **Educational coding**: Teach programming through physics-based examples

**Technical Requirements:**
- Distilled from specialized code LLM (DeepSeek-Coder, StarCoder, etc.)
- Window size: ~8K tokens (retrieval pulls in relevant functions/classes as needed)
- Integrated with physics knowledge for simulation-specific code

**Competitive Advantage:**
- **Physics-aware code generation**: Understands F=ma in simulation contexts
- **Multi-modal code explanation**: Can show diagrams and animations alongside code

---

## Physics Grounding: The Competitive Moat

### What is Physics Grounding?

**Not a narrow domain focus.** Physics grounding means the model learns a **world model**—how reality works—from high-quality physics simulation data (Unreal Engine).

**Learned principles:**
- **F = ma** (force, mass, acceleration relationships)
- **Momentum conservation** (objects don't arbitrarily speed up/slow down)
- **Causality** (causes precede effects)
- **Material properties** (wood vs. steel, friction coefficients)
- **Optics** (light, shadows, reflections, refraction)
- **Collision dynamics** (elastic vs. inelastic impacts)

### Why This Matters

**Current SOTA models (GPT-4o, Gemini 1.5, Sora) learn from web data:**
- Visual patterns, not physics
- Linguistic patterns, not causality
- Generated content is **statistically plausible but physically implausible**

**Examples of failures:**
- Sora: Objects floating, momentum violations, causality errors
- GPT-4V: Cannot identify wrong gravitational acceleration in video
- DALL-E 3: Shadows don't match light sources, impossible support structures

**Cloverfield's advantage:**
- Trained on **ground-truth physics simulations** (Unreal Engine)
- Learns **how the world actually works**, not just how it looks
- Generated content is **reality-consistent**

### Applications Enabled by Physics Grounding

1. **Education**: Generate teaching materials that show correct physics (textbook quality)
2. **Simulation**: Create realistic training data for robotics, autonomous vehicles, industrial safety
3. **Content creation**: Video/image/audio that doesn't violate common sense
4. **Error detection**: Identify physical impossibilities in media (deepfake detection, simulation debugging)
5. **Scientific visualization**: Accurate representations of physical phenomena

### Training Philosophy: "Textbooks Are All You Need"

Following Microsoft's Phi-1/Phi-2 approach:
- **Quality over scale**: 100K hours of high-quality physics simulations > 1M hours of YouTube
- **Ground truth**: Unreal Engine provides perfect labels (forces, velocities, trajectories)
- **Synthetic diversity**: Generate unlimited variations of fundamental physics scenarios
- **Curriculum learning**: Start with simple mechanics, progress to complex interactions

---

## Technical Architecture Requirements

### Core Components

1. **SPCE (Spectral Phase-Coherent Encoding)**
   - Shared spectral phase field: θ = ω·t + φ₀
   - All modalities aligned through phase coherence
   - Preserves semantic embeddings (orthogonal to content, like RoPE)

2. **Windowed Attention + Integrated Retrieval**
   - Flexible window sizing: 4K-16K tokens depending on task (O(w²) scales accordingly)
   - Task-specific windows: Video gen (~4K), Code gen (~8K), NLU (~8K-16K)
   - Three-tier memory architecture compensates for finite windows:
     - Tier 1: Local windowed attention (immediate context)
     - Tier 2: RETRO-style chunked retrieval (physics knowledge base)
     - Tier 3: kNN exact token retrieval (phase-aware, unbounded context)

3. **Dual-Resolution Video Processing**
   - Understanding pathway: 16× compression (long context, semantic search)
   - Generation pathway: 4× compression (high fidelity)

4. **SSM Carry**
   - Diagonal state-space model for cross-window persistence
   - Low-frequency phase coherence captures context outside window boundaries
   - Maintains conversational/narrative state without requiring large windows

5. **Visual Text Encoding**
   - DeepSeek OCR approach: Text → Image → SAM → CLIP
   - 16× compression with 97% fidelity for understanding
   - Unified visual representation across modalities

### Training Data Requirements

1. **Physics Simulations (Unreal Engine)**
   - 100K hours of physics-grounded video (target)
   - Projectile motion, collisions, rigid body dynamics, fluid dynamics
   - Synchronized captions, equation overlays, force vector annotations

2. **Text Corpus**
   - 50B tokens of high-quality text (code, textbooks, scientific papers)
   - Emphasis on physics, mathematics, causal reasoning

3. **Code**
   - Distillation from specialized code LLM (DeepSeek-Coder, etc.)
   - Physics simulation code (Unity, Unreal, Python numerical solvers)

4. **Audio-Visual Data**
   - Synchronized audio-visual from Unreal Engine (physics-correct sound)
   - Music, speech, environmental audio

### Inference Requirements

- **Throughput**: ≥20 tokens/second on M4 Max (generation)
- **Latency**: ≤100ms first token (conversational)
- **Memory**: ≤32GB RAM for inference (fits M4 Max base model)
- **Streaming**: Constant-memory operation for unbounded input

---

## Success Criteria & Benchmarks

### Quantitative Metrics

| Capability | Benchmark | Target (SOTA in class) | Current SOTA 7B |
|------------|-----------|------------------------|-----------------|
| NLU | MMLU | ≥70% | 70% (Mistral 7B) |
| NLU | HellaSwag | ≥80% | 81% (LLaMA 3 8B) |
| Code | HumanEval Pass@1 | ≥60% | 65% (DeepSeek-Coder 6.7B) |
| Video | ActivityNet-QA | Match GPT-4V | GPT-4V (proprietary) |
| Image | FID on COCO | ≤12 | 11 (SDXL base) |
| Video | UCF-101 FVD | ≤150 | 140 (CogVideo) |
| Audio | FAD on MusicCaps | ≤2.5 | 2.3 (MusicLM) |

### Qualitative Metrics (Novel Capabilities)

1. **Physics Reasoning** (video understanding)
   - Human evaluation: Identify physical errors in synthetic videos
   - Target: ≥90% accuracy on basic mechanics errors (F=ma, momentum)

2. **Physical Consistency** (generation)
   - Human evaluation: Generated content obeys real-world physics
   - Target: ≥80% "physically plausible" rating from physics teachers

3. **Audio-Visual Sync** (generation)
   - Automatic via SPCE (no fine-tuning needed)
   - Target: Perfect sync (0 frame offset) with no separate alignment model

4. **Conversational Coherence** (NLU)
   - Multi-turn physics tutoring dialogue (SSM carry maintains state across windows)
   - Target: ≥4.0/5.0 coherence rating in unbounded-length conversations

### Competitive Positioning Summary

| Model | Params | Multimodal | Physics-Grounded | Trainable on M4 Max |
|-------|--------|------------|------------------|---------------------|
| **Cloverfield** | **6B** | **✅ (5 modalities)** | **✅ (world model)** | **✅** |
| GPT-4o | 1.8T (est.) | ✅ | ❌ | ❌ |
| Gemini 1.5 Pro | Unknown | ✅ | ❌ | ❌ |
| LLaMA 3 8B | 8B | ❌ (text only) | ❌ | ✅ |
| Mistral 7B | 7B | ❌ (text only) | ❌ | ✅ |
| Phi-3 Medium | 14B | ❌ (text only) | ❌ | ⚠️ (limited) |

**Unique value proposition**: Only model in weight class with full multimodal capabilities + physics-grounded world model + consumer hardware trainability.

---

## Open Questions & Future Work

1. **Training data licensing**: Unreal Engine synthetic data vs. real-world augmentation
2. **Distillation strategy**: Which code LLM to distill from? (DeepSeek-Coder, StarCoder 2, etc.)
3. **Stateful inference**: Low-rank Hebbian adaptation design (deferred)
4. **Retrieval database**: Size and composition of physics knowledge base (RETRO tier 2)
5. **Evaluation datasets**: Create custom physics reasoning benchmarks (current benchmarks don't measure physical consistency)

---

## References to Other Documents

- **Architecture details**: See [`README.md`](./README.md)
- **Implementation guide**: See README section "Implementation Details"
- **Computational analysis**: See README section "Why This Architecture?"

---

**Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: Requirements definition phase
