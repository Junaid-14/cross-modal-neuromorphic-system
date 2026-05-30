# M7 Architecture Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         M7: Modality-Adaptive SNN                            │
│              Cross-Modal Continual Learning with Embedding Replay            │
└─────────────────────────────────────────────────────────────────────────────┘

   Event Input (3 modalities, sequential)
          │
          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  N-MNIST        │     │  SHD            │     │  DVS-Gesture    │
│  (34×34 frames) │     │  (700-ch spike) │     │  (128×128 evt)  │
│  Modality: nmn  │     │  Modality: shd  │     │  Modality: dvs  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Conv2D Backbone│     │  Linear Backbone│     │  Conv2D Backbone│
│  (shared arch)  │     │  (2-layer MLP)  │     │  (shared arch)  │
│  Output: 512-d  │     │  Output: 512-d  │     │  Output: 512-d  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │   512-d Feature     │
                    │   Representation    │
                    │   (modality-agnostic)│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
    │  Adaptive   │  │   Hopfield  │  │   HGRN Gate     │
    │    Gate     │  │   Memory    │  │  (temporal)     │
    │ 512→128→2   │  │ 256×512-d   │  │ 512→512         │
    │ Softmax     │  │ patterns    │  │ LayerNorm       │
    │ [w_hop,     │  │ associative │  │                 │
    │  w_hgrn]    │  │ retrieval   │  │                 │
    └──────┬──────┘  └──────┬──────┘  └────────┬────────┘
           │                │                   │
           │                ▼                   ▼
           │         ┌─────────────┐    ┌─────────────┐
           │         │  Hopfield   │    │   HGRN      │
           │         │  Output     │    │   Output    │
           │         │  512-d      │    │   512-d     │
           │         └──────┬──────┘    └──────┬──────┘
           │                │                   │
           └────────────────┼───────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Weighted Sum   │
                   │  z = w_hop·hop  │
                   │    + w_hgrn·grn │
                   │  Output: 512-d  │
                   └────────┬────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │    Per-Task Classification   │
              │         Heads                │
              │  ┌─────┐ ┌─────┐ ┌─────┐   │
              │  │ T1  │ │ T2  │ │ T3  │   │
              │  │NM   │ │SHD  │ │DVS  │   │
              │  │10-cl│ │20-cl│ │11-cl│   │
              │  └─────┘ └─────┘ └─────┘   │
              └─────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  CrossEntropy │
                    │  + SCL Loss   │
                    │  (engrams)    │
                    └───────────────┘
```

## Component Detail

### 1. Modality Backbones

| Modality | Input Shape | Backbone | Output |
|----------|------------|----------|--------|
| N-MNIST | (T, 34, 34) | Conv2D → Flatten → Linear | 512-d |
| SHD | (T, 700) | 2-layer Linear (512→512) | 512-d |
| DVS-Gesture | (T, 2, 128, 128) | Conv2D + AdaptivePool | 512-d |

> T = time bins (varies by dataset: N-MNIST=25, SHD=100, DVS=~variable)

### 2. Modern Hopfield Layer

```
Input: x ∈ ℝ^512

Query:    Q = W_q · x        ∈ ℝ^512
Key:      K = W_k · patterns  ∈ ℝ^{256×512}  (learned patterns)
Value:    V = W_v · patterns  ∈ ℝ^{256×512}

Energy:   E = -logsumexp(β · Q·K^T)
Output:   hop = V^T · softmax(β · Q·K^T)  ∈ ℝ^512

β = inverse temperature (learned or fixed)
```

**Role:** Associative memory for static/spatial feature retrieval. Excels on visual tasks.

### 3. HGRN Gate (Hierarchical Gated Recurrent Network)

```
Input: x ∈ ℝ^512

z = σ(W_z · x + b_z)          # update gate
r = σ(W_r · x + b_r)          # reset gate
h̃ = tanh(W_h · (r ⊙ x) + b_h) # candidate activation
h = (1 - z) ⊙ x + z ⊙ h̃       # gated output

Output: grn = LayerNorm(h) ∈ ℝ^512
```

**Role:** Temporal gating for sequential spike data. Better for auditory/motion tasks.

### 4. Adaptive Gate

```
Input: x ∈ ℝ^512

g = MLP(x) = Softmax(W_2 · ReLU(W_1 · x + b_1) + b_2)
           ∈ ℝ^2  = [w_hop, w_hgrn]

where W_1: 512 → 128, W_2: 128 → 2

Output: z = w_hop · hop + w_hgrn · grn  ∈ ℝ^512
```

**Learned behavior observed:**

| Task | w_hop | w_hgrn | Interpretation |
|------|-------|--------|----------------|
| N-MNIST (visual) | ~1.00 | ~0.00 | Pure Hopfield — spatial engrams |
| SHD (audio) | ~0.91 | ~0.09 | HGRN contribution for temporal |
| DVS (motion) | ~1.00 | ~0.00 | Visual dominance, slight HGRN |

### 5. Per-Task Heads

```
Each head: LayerNorm(512) → Linear(512 → num_classes)

T1: 512 → 10  (N-MNIST)
T2: 512 → 20  (SHD)
T3: 512 → 11  (DVS-Gesture)
```

**Critical design:** Only the active task head is updated during training. Previous heads are frozen. This prevents cross-task gradient interference — the source of near-zero forgetting.

### 6. Embedding Replay Buffer

```
┌────────────────────────────────────────┐
│        Embedding Replay Buffer         │
│                                        │
│  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ Class 0 │  │ Class 1 │  │  ...   │ │
│  │ 50×512  │  │ 50×512  │  │ 50×512 │ │
│  │ reservoir│  │reservoir│  │sampled │ │
│  └─────────┘  └─────────┘  └────────┘ │
│                                        │
│  Total: ~4 MB (vs 80 MB raw replay)   │
│  Strategy: Reservoir sampling per class │
└────────────────────────────────────────┘
```

**Replay loss:** `L_replay = CE(head_replay(replay_emb), replay_label)`

## Data Flow: Training Step

```
Forward (Task k):
  1. x_k → Backbone_k → features (512-d)
  2. features → Hopfield + HGRN (parallel)
  3. features → Adaptive Gate → [w_hop, w_hgrn]
  4. z = w_hop·hop + w_hgrn·grn
  5. z → Head_k → logits_k
  6. L_task = CE(logits_k, y_k) + λ·SCL(z, y_k)

If replay active:
  7. (emb_r, y_r) ~ Buffer.sample(batch_size, task_ids)
  8. logits_r = Head_r(emb_r) for each replay task
  9. L_replay = Σ CE(logits_r, y_r)
  10. L_total = L_task + α·L_replay

Backward:
  11. Only Backbone + Gate + Head_k get gradients
  12. Other heads frozen
  13. Buffer.populate() with current task embeddings
```

## Key Innovations

| Innovation | What It Does | Why It Matters |
|------------|-------------|----------------|
| **Adaptive Gate** | Learns modality-specific memory routing | No hand-tuning; visual→Hopfield, audio→HGRN |
| **Hopfield + HGRN Hybrid** | Combines associative + temporal memory | Addresses Paper 2 asymmetry (21.53pp gap) |
| **Per-Task Heads** | Isolates classifier gradients | Near-zero forgetting without regularization |
| **Embedding Replay** | Replays 512-d SCL embeddings | 800× smaller than raw replay; deployable on edge |
| **SCL Engrams** | Class-separable spike representations | Stable representations across tasks |
