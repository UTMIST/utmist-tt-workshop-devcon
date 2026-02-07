# Tenstorrent: Building the Future of AI Acceleration in Open Source

> A comprehensive guide to understanding LLM inference challenges and how Tenstorrent's RISC-V based architecture addresses them. Written for anyone from ML beginners to experienced systems engineers.

---

## Table of Contents

- [1. What is Tenstorrent?](#1-what-is-tenstorrent)
- [2. The Challenge of LLM Inference](#2-the-challenge-of-llm-inference)
  - [2.1 What is a Language Model Doing?](#21-what-is-a-language-model-doing)
  - [2.2 The Math Inside a Language Model](#22-the-math-inside-a-language-model)
  - [2.3 Anatomy of a Forward Pass](#23-anatomy-of-a-forward-pass)
  - [2.4 Prefill vs Decode: Two Very Differ
  ent Phases](#24-prefill-vs-decode-two-very-different-phases)
  - [2.5 The KV Cache Explosion](#25-the-kv-cache-explosion)
  - [2.6 Memory Bandwidth: The Real Bottleneck](#26-memory-bandwidth-the-real-bottleneck)
  - [2.7 Latency vs Throughput Tradeoff](#27-latency-vs-throughput-tradeoff)
  - [2.8 Why We Need Paged Attention](#28-why-we-need-paged-attention)
  - [2.9 Scaling to Multiple Devices](#29-scaling-to-multiple-devices)
- [3. How Tenstorrent Hardware Works](#3-how-tenstorrent-hardware-works)
  - [3.1 The Software Stack](#31-the-software-stack)
  - [3.2 GPUs vs Tenstorrent: A Comparison](#32-gpus-vs-tenstorrent-a-comparison)
  - [3.3 Memory: DRAM and L1 (SRAM)](#33-memory-dram-and-l1-sram)
  - [3.4 Interleaved vs Sharded Memory](#34-interleaved-vs-sharded-memory)
  - [3.5 The Network-on-Chip (NoC)](#35-the-network-on-chip-noc)
  - [3.6 Tiles: The Native Data Unit](#36-tiles-the-native-data-unit)
  - [3.7 Inside a Tensix Core](#37-inside-a-tensix-core)
  - [3.8 Circular Buffers: How Kernels Communicate](#38-circular-buffers-how-kernels-communicate)
  - [3.9 Writing Kernels: CUDA vs TT-Metal](#39-writing-kernels-cuda-vs-tt-metal)
- [4. Collective Communications Library (CCL)](#4-collective-communications-library-ccl)
  - [4.1 Why CCLs?](#41-why-ccls)
  - [4.2 All-Gather](#42-all-gather)
  - [4.3 All-Reduce](#43-all-reduce)
  - [4.4 Reduce-Scatter](#44-reduce-scatter)
- [5. High-Performance Computing Techniques](#5-high-performance-computing-techniques)
  - [5.1 Pipelining: Overlapping Compute and Data Movement](#51-pipelining-overlapping-compute-and-data-movement)
  - [5.2 Double and Triple Buffering](#52-double-and-triple-buffering)
  - [5.3 SFPU Operation Chaining](#53-sfpu-operation-chaining)
  - [5.4 DRAM Bandwidth Saturation](#54-dram-bandwidth-saturation)
  - [5.5 Metal Trace and Multiple Command Queues](#55-metal-trace-and-multiple-command-queues)
- [6. Building with TTNN: Models, Use Cases, and Next Steps](#6-building-with-ttnn-models-use-cases-and-next-steps)

---

## 1. What is Tenstorrent?

Tenstorrent is an AI accelerator company building hardware and software from the ground up to run AI workloads -- particularly Large Language Models (LLMs) -- with extreme efficiency. What sets Tenstorrent apart:

- **Open Source**: The entire software stack, from the low-level runtime ([TT-Metal](https://github.com/tenstorrent/tt-metal)) to the high-level neural network library (TT-NN), is open source.
- **RISC-V Based**: Every compute core is built on the open RISC-V ISA. This means the hardware is programmable at every level -- no black-box fixed-function units.
- **Scalable Architecture**: Devices connect directly via Ethernet, without needing expensive NVSwitch or InfiniBand NICs. Two chips connect to make an N300, eight make a T3000, and thirty-two make a Galaxy.
- **Tile-Native Computing**: The hardware operates natively on 32x32 matrix tiles, perfectly matching the data patterns of deep learning.

Tenstorrent currently ships two chip architectures:
- **Wormhole**: 8x10 grid of Tensix cores, 1.5MB L1 per core (120MB total on-chip SRAM), 12 DRAM channels at 288 GB/s Theoretical BW, 16 Ethernet links for scale-out.
- **Blackhole**: 11x10 grid of Tensix cores, 1.5MB L1 per core (120MB total on-chip SRAM), 8 DRAM channels at 512 GB/s Theoretical BW, 16 Ethernet links for scale-out.

---

## 2. The Challenge of LLM Inference

### 2.1 What is a Language Model Doing?

When you type a prompt into ChatGPT, here is what happens under the hood:

```
You type: "Explain quantum computing in simple terms"

Step 1 - Tokenization:
  "Explain quantum computing in simple terms"
  → [15339, 18426, 25213, 304, 4382, 3878]  (6 tokens)

Step 2 - Prefill (process entire prompt at once):
  Feed all 6 tokens through the model simultaneously.
  The model maps your tokens to high dimension vectors with some dimension size of D (Shape transformation: [6,1] -> [6,D]) 
  The model performs a series of mathemtical transformations to build up internal representations (KV cache) for each token.
  Output: probability distribution over the next token.

Step 3 - Decode (generate one token at a time):
  The same model is run but we feed the model with only 1 token of sequence length.
  Sample from the distribution → "Quantum"
  Feed "Quantum" back into the model → "computing"
  Feed "computing" back into the model → "is"
  Feed "is" back into the model → "like"
  ... and so on, one token at a time, until done.
```

This **autoregressive** nature is fundamental: each new token depends on *all* previous tokens. You cannot parallelize the generation of token N+1 before token N is produced. This serial dependency is what makes LLM inference fundamentally challenging.

### 2.2 The Math Inside a Language Model

A transformer-based language model (the most common architecture seen in Language Models) is a stack of **decoder layers** (typically 32-80 layers for modern LLMs). Each layer contains the same sequence of operations. Let's go through each one:

#### RMS Norm (Root Mean Square Layer Normalization)

**Purpose**: Stabilize activations before each sub-layer. Without normalization, values would explode or vanish as they pass through dozens of layers.

**Formula**:

```
RMSNorm(x) = x * (1 / sqrt(mean(x²) + ε)) * γ

Where:
  x     = input vector of dimension d (e.g., 8192 for Llama-70B)
  ε     = small constant (1e-5) for numerical stability
  γ     = learnable scale parameter (per-dimension)
  mean  = average over the d dimensions
```

**Intuition**: Think of it as normalizing the "energy" of each token's representation to a consistent scale. If one token's hidden state has values in the range [-100, 100] and another is in [-0.01, 0.01], the math downstream would behave very differently for each. RMSNorm brings them to the same scale.

```python
# PyTorch implementation
def rms_norm(x, weight, eps=1e-5):
    # x shape: [batch, seq_len, hidden_dim]
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight
```

<details>
<summary><strong>Why RMSNorm instead of LayerNorm?</strong></summary>

LayerNorm computes both mean and variance: `(x - mean(x)) / sqrt(var(x) + ε) * γ + β`. RMSNorm drops the mean subtraction and the β bias, relying only on the root-mean-square for normalization. This removes one reduction operation (computing the mean) and one addition (the bias), making it faster while producing nearly identical results in practice. Llama, Qwen, and most modern LLMs use RMSNorm.

</details>

#### Scaled Dot-Product Attention (SDPA)

**Purpose**: Allow each token to "look at" every other token and decide which ones are most relevant for predicting the next token.

**Formula**:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Where:
  Q = queries  [batch, n_heads, seq_len, head_dim]
  K = keys     [batch, n_kv_heads, seq_len, head_dim]
  V = values   [batch, n_kv_heads, seq_len, head_dim]
  d_k = head_dim (e.g., 128)
```

**Intuition**: Imagine you are reading a sentence. For each word, you ask "which other words in this sentence should I pay attention to?" The Query is your question, the Keys are labels on all the other words, and the Values are the actual content of those words. The dot product Q@K^T computes how well each query matches each key (higher = more relevant). Softmax turns these scores into probabilities, and then we take a weighted sum of Values.

**What do Q, K, and V heads represent?**

Think of **heads** as independent "perspectives" or "experts" that look at the data in different ways:

- **Query heads (Q)**: Each Q head represents a different *question* being asked. Head 1 might learn to ask "what's the subject of this sentence?", Head 2 might ask "what adjectives modify me?", Head 3 might focus on "what's the verb associated with me?". Having multiple Q heads means each token can ask multiple different questions simultaneously.

- **Key heads (K)**: Keys are like *labels* or *tags* that tokens advertise about themselves. Each K head provides a different type of label. If Q head 1 is asking about subjects, K head 1 provides "I am/am not a subject" signals.

- **Value heads (V)**: Values are the actual *content* that gets retrieved. When a query matches a key, the corresponding value is what gets passed along. Different V heads carry different aspects of the token's meaning.

**Why might we want different numbers of Q and KV heads?**

In standard **Multi-Head Attention (MHA)**, we have equal numbers: `n_q_heads = n_kv_heads` (e.g., 32 Q heads and 32 KV heads). Each Q head has its own dedicated K and V heads. This is maximally expressive but expensive.

The key insight is: **queries need to be diverse, but keys/values can often be shared**.

- Queries represent the *questions* each token asks -- these need to be varied because different aspects of a token might need different information.
- Keys/Values represent *what information a token offers* -- this is more stable. A noun is a noun regardless of who's asking.

This asymmetry motivates reducing KV heads while keeping Q heads high.

**The Three Attention Variants:**

| Variant | Q Heads | KV Heads | Ratio | KV Cache Size |
|---------|---------|----------|-------|---------------|
| **MHA** (Multi-Head Attention) | 32 | 32 | 1:1 | 100% (baseline) |
| **GQA** (Grouped-Query Attention) | 32 | 8 | 4:1 | 25% |
| **MQA** (Multi-Query Attention) | 32 | 1 | 32:1 | 3.125% |

**Multi-Head Attention (MHA)** - The Original
- Every Q head has its own K and V heads
- Maximum expressiveness, maximum memory cost
- Used in: Original Transformer, GPT-2, BERT
- *Analogy*: 32 students each have their own personal tutor (K) and textbook (V)

**Multi-Query Attention (MQA)** - The Extreme
- All Q heads share a *single* K and V head
- Minimal memory, but can hurt model quality
- Used in: PaLM, Falcon
- *Analogy*: 32 students share one tutor and one textbook -- efficient but crowded

**Grouped-Query Attention (GQA)** - The Sweet Spot
- Q heads are divided into groups; each group shares one K and V head
- Balances memory savings with model quality
- Used in: Llama 2/3, Mistral, Mixtral
- *Analogy*: 32 students split into 8 study groups of 4, each group shares a tutor and textbook

**When to use each:**

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Training from scratch, memory is abundant | MHA | Maximum model quality |
| Inference on long sequences, memory-bound | MQA | Smallest KV cache |
| Production LLMs, balance quality + efficiency | GQA | Best tradeoff -- Llama 3 uses this |
| Converting existing MHA model | GQA | Can "uptrain" MHA → GQA with minimal quality loss |

**Why this matters for inference:**

During autoregressive decoding, we must store K and V for all previous tokens (the "KV cache"). For a 70B model with 128K context:
- MHA (64 KV heads): ~40GB KV cache
- GQA (8 KV heads): ~5GB KV cache ← 8x reduction!
- MQA (1 KV head): ~0.6GB KV cache

This is why GQA is dominant in modern LLMs -- it makes long-context inference practical.

**Pseudocode**:
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: [B, H, S, D], K: [B, H_kv, S, D], V: [B, H_kv, S, D]
    # If GQA: H > H_kv, and we repeat K/V to match Q's head count
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, S, S]
    scores = scores / math.sqrt(d_k)                # Scale down

    # Step 2: Apply causal mask (prevent attending to future tokens)
    if mask is not None:
        scores = scores + mask  # mask has -inf for future positions

    # Step 3: Normalize to probabilities
    attn_weights = torch.softmax(scores, dim=-1)    # [B, H, S, S]

    # Step 4: Weighted sum of values
    output = torch.matmul(attn_weights, V)           # [B, H, S, D]
    return output
```

<details>
<summary><strong>GQA Implementation Detail: How K/V heads are shared</strong></summary>

When `n_q_heads > n_kv_heads`, we need to "expand" the K and V tensors so each Q head knows which K/V head to use. For Llama-70B with 64 Q heads and 8 KV heads (ratio 8:1), each KV head is shared by 8 Q heads:

```python
# Q heads 0-7 use KV head 0
# Q heads 8-15 use KV head 1
# ... and so on

def repeat_kv(x, n_rep):
    """Repeat KV heads to match Q head count."""
    # x: [B, n_kv_heads, S, D]
    # returns: [B, n_kv_heads * n_rep, S, D]
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D).reshape(B, H_kv * n_rep, S, D)
```

This "repeat" operation is cheap (just a view/broadcast) and happens during attention computation, not in the KV cache storage.

</details>

#### MLP (Multi-Layer Perceptron / Feed-Forward Network)

**Purpose**: Apply non-linear transformations to each token independently. While attention mixes information *between* tokens, the MLP processes each token's representation *independently* through a wider intermediate space.

**Formula (SwiGLU variant used in Llama)**:

```
MLP(x) = W2 @ (SiLU(W1 @ x) * (W3 @ x))

Where:
  x   = input                     [batch, seq_len, hidden_dim]
  W1  = gate projection weight    [hidden_dim, intermediate_dim]
  W3  = up projection weight      [hidden_dim, intermediate_dim]
  W2  = down projection weight    [intermediate_dim, hidden_dim]
  SiLU(z) = z * sigmoid(z)       (smooth activation function)
```

**Intuition**: Think of the MLP as a "thinking" step. Attention gathered information from other tokens; now the MLP processes that gathered information through a non-linear transformation. The intermediate dimension is typically 4x the hidden dimension (e.g., 8192 → 28672 for Llama-70B), so the data first expands into a wider space where complex patterns can be captured, then compresses back down.

```python
def mlp_forward(x, w1, w2, w3):
    # x: [batch, seq_len, hidden_dim]
    w1_out = F.silu(x @ w1.T)      # Gate: [batch, seq_len, intermediate_dim]
    w3_out = x @ w3.T               # Up:   [batch, seq_len, intermediate_dim]
    return (w1_out * w3_out) @ w2.T  # Down: [batch, seq_len, hidden_dim]
```

### 2.3 Anatomy of a Forward Pass

Here is a simplified PyTorch implementation of a complete Llama-style transformer forward pass:

```python
import torch
import torch.nn.functional as F
import math

class TransformerBlock(torch.nn.Module):
    """One decoder layer of a Llama-style transformer."""

    def __init__(self, hidden_dim, n_heads, n_kv_heads, intermediate_dim):
        super().__init__()
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        # Weights
        self.attention_norm = RMSNorm(hidden_dim)
        self.wq = torch.nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(n_heads * self.head_dim, hidden_dim, bias=False)

        self.ffn_norm = RMSNorm(hidden_dim)
        self.w1 = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)

        # KV Cache
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, start_pos, freqs_cis, mask=None):
        """
        x: [batch, seq_len, hidden_dim]

        The compute graph for one decoder layer:

          x ──► RMSNorm ──► QKV Projections ──► RoPE ──► KV Cache Update
          │                                                    │
          │                                              ┌─────▼─────┐
          │                                              │   SDPA    │
          │                                              │ Q@K^T/√d  │
          │                                              │ softmax   │
          │                                              │  @V       │
          │                                              └─────┬─────┘
          │                                                    │
          │◄─────────── + (residual) ◄──── Output Projection ◄─┘
          │
          ├──► RMSNorm ──► MLP (W1/W3 projections → SiLU*gate → W2)
          │                    │
          ▼◄─────────── + (residual) ◄──────────────────────────┘

        """
        # ═══ Attention Block ═══
        residual = x
        x = self.attention_norm(x)

        # QKV projections (large matmuls)
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply Rotary Position Embeddings (RoPE)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Update KV Cache
        self.cache_k[:bsz, start_pos:start_pos+seqlen] = k
        self.cache_v[:bsz, start_pos:start_pos+seqlen] = v
        keys = self.cache_k[:bsz, :start_pos+seqlen]
        values = self.cache_v[:bsz, :start_pos+seqlen]

        # Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)

        # Output projection
        attn_output = attn_output.reshape(bsz, seqlen, -1)
        x = residual + self.wo(attn_output)

        # ═══ MLP Block ═══
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.w2(F.silu(self.w1(x)) * self.w3(x))

        return x
```

### 2.4 Prefill vs Decode: Two Very Different Phases

#### Prefill Phase (Compute-Bound)

During prefill, the model processes the entire input prompt at once. For a 1024-token prompt with hidden dimension 8192:

```
QKV Projection:  [1, 1024, 8192] @ [8192, 8192] = massive matmul
Attention:       [1, 64, 1024, 128] @ [1, 8, 1024, 128]^T
                 = [1, 64, 1024, 1024] attention matrix
MLP FF1:         [1, 1024, 8192] @ [8192, 28672] = huge matmul
```

The matrices are large enough that the FLOPs required far exceed the time to load data. The **arithmetic intensity** (FLOPs per byte of data movement) is high. The compute units are the bottleneck, not memory.

**Analogy**: Prefill is like a factory with a huge batch order. The machines are running at full capacity, and the warehouse (memory) can keep up with delivering raw materials.

#### Decode Phase (Memory-Bound)

During decode, the model generates one token at a time. For batch=32 users with hidden dimension 8192:

```
QKV Projection:  [1, 32, 8192] @ [8192, 8192]
                 ─── activation is tiny (32 tokens)
                 ─── weight is huge (8192 × 8192 = 67M params = 134MB in fp16)
                 ─── Must load 134MB of weights to do only 32 × 8192 × 8192 = 4.3B FLOPs
```

The arithmetic intensity is catastrophically low. For every byte of weight loaded from memory, you only do a handful of operations. **Memory bandwidth becomes the bottleneck**.

<details>
<summary><strong>Deep Dive into performance: Arithmetic Intensity and the Roofline Model</strong></summary>

**How to Calculate Performance?**

The roofline model gives us a simple formula for the maximum achievable performance:

```
Perf = min(Peak FLOPs/s, AI × Sustained BW)
```

Where:
- **Peak FLOPs/s**: The theoretical maximum compute throughput of your hardware
- **AI**: Arithmetic intensity of your workload (FLOPs/byte)
- **Sustained BW**: The actual measured memory bandwidth (bytes/s)

**What is Sustained Bandwidth?**

Sustained bandwidth is *not* the peak spec on the datasheet — it's what you actually measure:

```
Sustained BW = Bytes actually transferred / Time it took to transfer them
```

For example, if you transfer 128 MiB and it takes 0.5 ms:
```
Sustained BW = 128 × 2²⁰ bytes / 0.0005 s = 268 GB/s
```

This is typically lower than peak bandwidth due to:
- Memory access patterns (random vs sequential)
- Bank conflicts and row buffer misses
- NoC congestion and routing overhead
- Software/scheduling inefficiencies

**Why does this matter?** When estimating real performance, using peak BW gives you an optimistic upper bound. Using measured sustained BW gives you a realistic estimate. For Wormhole, peak DRAM BW is ~288 GB/s, but sustained BW for typical workloads might be 200-250 GB/s depending on access patterns.

**What is Arithmetic Intensity?**

Arithmetic Intensity (AI) measures how much computation you do per byte of data moved:

```
AI = FLOPs / Bytes moved    (units: FLOPs/byte)
```

"Bytes moved" typically means bytes transferred to/from DRAM, unless you explicitly specify another level (L2 cache, L1/SRAM, etc.).

**What is the Ridge Point (Machine Balance)?**

Every processor has a characteristic "ridge point" — the arithmetic intensity where compute and memory bandwidth are perfectly balanced:

```
I* = Peak FLOPs/s ÷ Peak Memory BW (bytes/s)    (units: FLOPs/byte)
```

This is the key number in the **roofline model**:
- If **AI < I\***: Your workload is **memory-bound** (waiting for data)
- If **AI > I\***: Your workload is **compute-bound** (ALUs are the bottleneck)

**Step-by-Step: Calculating AI for Decode GEMM**

For a single matrix multiplication during decode:
- Input activation: `[32, 8192]` in fp16
- Weight matrix: `[8192, 8192]` in fp16
- Output: `[32, 8192]` in fp16

*Step 1: Count FLOPs*
```
Output elements = 32 × 8192 = 262,144
Each output element requires 8192 multiply-accumulate operations (MACs)
Total MACs = 262,144 × 8192 = 2,147,483,648

Counting 1 MAC = 2 FLOPs (one multiply + one add):
FLOPs = 2 × 2,147,483,648 = 4,294,967,296 ≈ 4.29 × 10⁹
```

*Step 2: Count Bytes Moved*
```
Weights (dominant term):
  8192 × 8192 × 2 bytes = 134,217,728 bytes ≈ 128 MiB

Activations (negligible in comparison):
  Input:  32 × 8192 × 2 ≈ 0.5 MiB
  Output: 32 × 8192 × 2 ≈ 0.5 MiB

Total ≈ 134 × 10⁶ bytes (weights dominate)
```

*Step 3: Compute Arithmetic Intensity*
```
AI = 4.29 × 10⁹ FLOPs / 1.34 × 10⁸ bytes ≈ 32 FLOPs/byte
```

**Step-by-Step: Is Decode Memory-Bound or Compute-Bound?**

For Wormhole:
- Peak compute: ~320 TFLOPs/s (4 TFLOPs × 80 Tensix cores)
- Peak DRAM BW: 288 GB/s

*Calculate the ridge point:*
```
I* = 320 × 10¹² FLOPs/s ÷ 288 × 10⁹ bytes/s
   = 1,111 FLOPs/byte
```

*Compare workload AI to ridge point:*
```
Decode AI:     32 FLOPs/byte
Ridge point: 1111 FLOPs/byte

32 << 1111  →  Decode is deeply memory-bound
```

The decode phase operates at only ~3% of the ridge point, meaning we're using a tiny fraction of the available compute. The processor spends most of its time waiting for weights to arrive from DRAM.

**Visual: The Roofline Model**

```
Performance │                        ┌─────────── Compute ceiling (320 TFLOPs/s)
(TFLOPs/s)  │                       ╱
            │                      ╱
    320 ────│─────────────────────●─────────────────────
            │                    ╱│
            │                   ╱ │
            │                  ╱  │
            │                 ╱   │
            │                ╱    │
            │               ╱     │
            │              ╱      │
            │             ╱       │
            │            ╱        │
            │           ╱ Memory  │
            │          ╱  ceiling │
            │         ╱   (slope  │
            │        ╱    = BW)   │
            │       ╱             │
            │      ╱              │
            │     ╱               │
            │    ╱                │
            │   ╱                 │
            │  ╱                  │
            │ ╱ ▲                 │
            │╱  │Decode           │
            └───┴─────────────────┴────────────────────→
                32              1111            Arithmetic Intensity
              (here)        (ridge point)        (FLOPs/byte)
```

**Important Caveat**

This analysis assumes weights are streamed from DRAM for each decode step. If weights are reused from cache (L2/L1/SRAM) across multiple decode steps or via clever prefetching, the effective bytes moved decreases, and the effective AI increases. But for the typical "load weights from DRAM per layer per step" pattern, ~32 FLOPs/byte is the right order of magnitude.

</details>

```
Arithmetic Intensity (decode, batch=32):
  FLOPs = 2 × 32 × 8192 × 8192 = 4.29 × 10⁹
  Bytes loaded = 8192 × 8192 × 2 = 1.34 × 10⁸
  AI = FLOPs / Bytes = 32 FLOPs/byte

For Wormhole (288 GB/s DRAM BW, ~4 TFLOPS × 80 cores):
  Roofline crossover ≈ 320 TFLOPS / 288 GB/s ≈ 1111 FLOPs/byte
  32 << 1111, so decode is deeply memory-bound.
```

**Analogy**: Decode is like a factory that has to process a tiny custom order every minute. Each time, the warehouse has to deliver the entire catalog of raw materials (all the weights), but the machines only run for a fraction of a second. Most time is spent waiting for the warehouse. But the key idea is that decode requires you to re-read weights from slower access memory (DRAM) into fast access memory (L1) every time every decode iteration. In an ideal world we would fit all weights in SRAM. However Tenstorrent hardware has a limited budget of L1.

### 2.5 The KV Cache Explosion

Every token generated must attend to *all* previous tokens. Without caching, you'd recompute the Key and Value projections for every past token at every step. The KV cache stores these computed K and V tensors.

**KV Cache Size Formula**:

```
KV Cache (bytes) = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch_size × bytes_per_element

Example: Llama-70B, batch=32, seq_len=8192, bfloat16:
  = 2 × 80 × 8 × 128 × 8192 × 32 × 2
  = 2 × 80 × 8 × 128 × 8192 × 32 × 2
  = 68,719,476,736 bytes
  ≈ 64 GB

At 128K context length:
  = 64 GB × (128K / 8K)
  = 1,024 GB = 1 TB!
```

**Transfer Time**:
```
At 288 GB/s (Wormhole DRAM bandwidth):
  8K context:  64 GB / 288 GB/s  ≈ 222 ms per decode step (just for KV cache!)
  128K context: 1 TB / 288 GB/s  ≈ 3.6 seconds per token!
```

This is why long-context inference is so challenging. The KV cache doesn't just grow linearly -- it must be *read* in its entirety at every single decode step.

### 2.6 Memory Bandwidth: The Real Bottleneck

For decode, the time to generate one token is dominated by:

```
Time per token ≈ max(
    (model_weights_bytes / memory_bandwidth),   ← loading all weights
    (kv_cache_bytes / memory_bandwidth),         ← loading KV cache for attention
    (total_FLOPs / compute_throughput)           ← actually computing
)
```

For Llama-70B on a single Wormhole chip (which doesn't have enough memory, but illustrating the math):
```
Weight load time:  70B params × 2 bytes / 288 GB/s = 486 ms
KV cache time:     depends on context length
Compute time:      ~140 TFLOPs needed / 320 TFLOPS = 0.44 ms  ← negligible!
```

The compute is ~1000x faster than the memory can deliver data. **This is the fundamental challenge of LLM inference.**

### 2.7 Latency vs Throughput Tradeoff

There is a fundamental tension between:
- **Latency** (time to generate one token for one user): Minimize by processing one user as fast as possible.
- **Throughput** (tokens generated per second across all users): Maximize by batching many users together.

Batching helps because when you increase batch size from 1 to 32, you load the same weights once but use them for 32 users. The arithmetic intensity increases by 32x, moving you closer to the compute-bound regime. But each user's response takes longer because you're sharing the hardware.

```
Batch=1:   Time ≈ weight_load_time       (memory-bound, fast for this user)
Batch=32:  Time ≈ weight_load_time × 1    (same weight loading!)
           But now you generated 32 tokens instead of 1.
           Throughput increased 32x, latency barely changed!

Batch=512: Now activations don't fit in L1 → spill to DRAM → performance cliff
```

The sweet spot depends on hardware memory capacity and bandwidth.

### 2.8 Why We Need Paged Attention

Traditional KV cache allocation is **static**: you pre-allocate the maximum sequence length for every request in the batch. This is incredibly wasteful:

```
Without paged attention:
  User A: prompt="Hi" (2 tokens), max_len=8192
  → Allocates 8192 × KV_entry_size bytes, using only 2 entries (0.02% utilization!)

  User B: prompt="Write me a novel..." (50 tokens), max_len=8192
  → Also allocates full 8192 entries

  Result: Most of allocated memory is wasted padding.
```

**Paged attention** (from vLLM) treats the KV cache like virtual memory in an operating system:
- The KV cache is divided into fixed-size **pages** (blocks of tokens).
- Pages are allocated on demand as sequences grow.
- A **page table** maps logical sequence positions to physical memory pages.
- When a sequence finishes, its pages are freed for reuse.

In the TT-Metal implementation:
```python
# Paged KV cache update (from models/tt_transformers/tt/attention.py)
ttnn.experimental.paged_update_cache(
    keys, k_heads,
    update_idxs_tensor=current_pos,
    page_table=page_table
)

# Paged attention decode
attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    Q, K, V,
    page_table_tensor=page_table,
    cur_pos_tensor=current_pos,
    scale=self.scale,
)
```

### 2.9 Scaling to Multiple Devices

Modern LLMs are too large for a single chip. Llama-70B has 70 billion parameters (140 GB in fp16), but a single Wormhole has 12 GB of DRAM. You need multiple devices.

**Tensor Parallelism** is the primary strategy: split each weight matrix across devices. For N devices:
- Each device stores 1/N of each weight matrix.
- Each device computes a partial result.
- Collective communication (All-Reduce, Reduce-Scatter, All-Gather) combines partial results.

The challenge: communication between devices adds latency. On 8 devices, you need at least 7 communication rounds per layer for the data to circulate. Minimizing this overhead while keeping compute units busy is a major engineering challenge.

---

## 3. How Tenstorrent Hardware Works

### 3.1 The Software Stack

```
┌──────────────────────────────────────────────────────┐
│  User Code (PyTorch, TTNN Python API)                │
│  model = ttnn.from_torch(torch_model, device=device) │
├──────────────────────────────────────────────────────┤
│  TTNN  (ttnn/)                                       │
│  High-level NN operations: matmul, softmax,          │
│  attention, norm, etc. Op dispatch & optimization.    │
├──────────────────────────────────────────────────────┤
│  TT-Metal  (tt_metal/)                               │
│  Runtime: program creation, kernel dispatch, memory   │
│  allocation, circular buffers, multi-device coord.    │
├──────────────────────────────────────────────────────┤
│  LLK  (Low-Level Kernels, tt_metal/third_party/tt_llk)│
│  Tensix-native primitives: FPU matmul, SFPU ops,     │
│  tile packing/unpacking, data format conversion.      │
├──────────────────────────────────────────────────────┤
│  TT Instructions  (Custom Tensix ISA)                │
│  Instruction streams for FPU/SFPU/unpacker/packer.   │
├──────────────────────────────────────────────────────┤
│  UMD / KMD  (User/Kernel Mode Driver)                │
│  Hardware abstraction: NOC config, core management,  │
│  PCIe communication, device reset/init.              │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │              HARDWARE (Silicon)                │  │
│  │  Tensix Cores ─ NoC ─ DRAM ─ Ethernet ─ PCIe │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 3.2 GPUs vs Tenstorrent: A Comparison

| Aspect | GPU (e.g., NVIDIA H100) | Tenstorrent (Wormhole) |
|--------|------------------------|----------------------|
| **Compute Unit** | Streaming Multiprocessor (SM) with thousands of CUDA cores | Tensix core with 5 RISC-V processors + FPU + SFPU |
| **Grid Structure** | Grid of blocks → warps of 32 threads | 8x10 grid of independent Tensix cores |
| **Memory Hierarchy** | HBM → L2 cache → Shared Memory/L1 → Registers | DRAM → L1 SRAM (per core, 1.5MB) → Registers |
| **Memory** | 80GB HBM3 @ 3.35 TB/s | 12GB GDDR6 @ 288 GB/s |
| **On-chip SRAM** | ~50MB shared L2 + ~256KB shared memory/SM | **120MB** total (1.5MB × 80 cores) |
| **Data Movement** | Implicit (hardware cache hierarchy) | **Explicit** (programmer-controlled via NoC) |
| **Kernel Model** | Single kernel = read+compute+write | 3 separate kernels: reader + compute + writer |
| **Inter-op Data** | Must go through global memory | Can stay in L1 (sharded) |
| **Scale-out** | NVLink + NVSwitch (proprietary) | Direct Ethernet (no switch needed) |

**Key Insight**: On a GPU, data between operations must round-trip to global memory (HBM). On Tenstorrent, data can stay in L1 SRAM between operations via **L1 sharding**. This means a chain of operations (like RMSNorm → QKV projection → RoPE → Attention) can keep intermediates in fast L1 without ever touching slow DRAM. This is equivalent to "operator fusion" but without writing a single fused kernel.

### 3.3 Memory: DRAM and L1 (SRAM)

#### Why is DRAM Slow and SRAM Fast? (A Digital Electronics View)

**DRAM (Dynamic Random Access Memory)**:
- Each bit is stored as charge in a tiny capacitor + 1 transistor.
- Capacitors leak charge and must be **refreshed** periodically (hence "Dynamic").
- Reading is destructive: you must sense, amplify, and rewrite the charge.
- Organized in rows and columns; you must first "open" a row, then access a column.
- **Latency**: ~10-20ns per random access.
- **Advantage**: Very dense (1 transistor per bit), so you get lots of capacity cheaply.

**SRAM (Static Random Access Memory) / L1**:
- Each bit is stored in a latch made of 6 transistors.
- No refresh needed (hence "Static").
- Reading is non-destructive: data remains after read.
- Direct access without row/column precharge.
- **Latency**: ~1-2ns per access.
- **Disadvantage**: 6 transistors per bit = 6x less dense than DRAM.


**Analogy**: DRAM is like a huge warehouse where you have to open a crate (row activate), find the item (column select), take a photo (sense amplify), and put the item back (refresh). SRAM is like items sitting on your desk -- just reach over and grab them instantly.

#### DRAM Banks on Wormhole

Wormhole has **12 DRAM banks**, each connected to the NoC (Network on Chip). The banks are physically distributed across the chip:

```
   Wormhole DRAM Bank Layout (simplified):
   ┌─────────────────────────────────────┐
   │ D0   [tensix cores]   [tensix]  D6 │
   │ D1   [tensix cores]   [tensix]  D7 │
   │ D2   [tensix cores]   [tensix]  D8 │
   │ D3   [tensix cores]   [tensix]  D9 │
   │ D4   [tensix cores]   [tensix] D10 │
   │ D5   [tensix cores]   [tensix] D11 │
   └─────────────────────────────────────┘

   Total: 12 banks × 1GB = 12GB, each bank at 24 GB/s
   Aggregate: 12 × 24 = 288 GB/s
```

#### L1 Memory

Each Tensix core has **1.5MB of L1 SRAM**. This is where:
- Circular buffers live (for kernel communication)
- Input/output data for the compute engine
- Intermediate results during computation
- Firmware code for the 5 RISC-V processors

Total on-chip SRAM across 80 cores = **120MB**. This is massive compared to GPU shared memory.

### 3.4 Interleaved vs Sharded Memory

#### DRAM Interleaved

When a tensor is stored as "interleaved in DRAM," its pages are distributed round-robin across all DRAM banks:

```
Tensor: [4 pages: P0, P1, P2, P3] across 3 DRAM banks

   Bank 0:  [P0] [P3]
   Bank 1:  [P1]
   Bank 2:  [P2]

   └── Pages assigned round-robin: P0→Bank0, P1→Bank1, P2→Bank2, P3→Bank0
```

```python
# Create an interleaved DRAM tensor in TTNN
import ttnn, torch

torch_tensor = torch.randn(64, 64, dtype=torch.bfloat16)
tt_tensor = ttnn.from_torch(
    torch_tensor,
    layout=ttnn.TILE_LAYOUT,     # 32x32 tile format
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG  # Interleaved across DRAM banks
)
# Result: 4 tiles (64/32=2 rows × 64/32=2 cols), distributed across 12 banks
```

#### DRAM Sharded

For performance-critical DRAM-bound operations (like decode matmul), we **shard** tensors so each DRAM reader only reads from its own assigned bank. This avoids NoC congestion:

```
DRAM Sharded (12 banks):
   Bank 0:  [Shard 0]    ← Reader core 0 only reads from Bank 0
   Bank 1:  [Shard 1]    ← Reader core 1 only reads from Bank 1
   ...
   Bank 11: [Shard 11]   ← Reader core 11 only reads from Bank 11

   No cross-bank traffic = maximum bandwidth utilization (92%+ achieved!)
```

#### L1 Interleaved

Pages distributed across L1 banks (one per core):

```python
tt_tensor = ttnn.from_torch(
    torch_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG  # Interleaved across L1 banks
)
```

#### L1 Sharded

Each core gets a specific, contiguous portion of the tensor. This is the most performance-optimal layout because the data is already local to the compute core that needs it.

```python
# Width-sharded across 8 cores: each core gets a slice of columns
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))
        }),
        shard_shape=[32, 128],  # Each core: 32 rows × 128 columns
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# Each core has its shard ready in L1 -- zero data movement needed!
tt_tensor = ttnn.from_torch(
    torch.randn(32, 1024),
    dtype=ttnn.bfloat16,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=memory_config
)
```

**Three sharding strategies**:

| Strategy | How it splits | Best for |
|----------|--------------|----------|
| **Height Sharding** | Consecutive rows per core | Operations processing data row-wise |
| **Width Sharding** | Consecutive columns per core | Decode matmul (weight columns split) |
| **Block Sharding** | 2D blocks per core | Prefill matmul (2D parallelism) |

### 3.5 The Network-on-Chip (NoC)

The NoC is a 2D mesh/torus network connecting all cores, DRAM banks, and Ethernet ports. Every Tensix core is a NoC endpoint.

```
   Wormhole NoC Topology (simplified):

   ┌────┐   ┌────┐   ┌────┐   ┌────┐
   │Core│───│Core│───│Core│───│Core│──►(wraps around)
   │0,0 │   │1,0 │   │2,0 │   │3,0 │
   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
     │         │         │         │
   ┌─▼──┐   ┌─▼──┐   ┌─▼──┐   ┌─▼──┐
   │Core│───│Core│───│Core│───│Core│
   │0,1 │   │1,1 │   │2,1 │   │3,1 │
   └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
     │         │         │         │
     ▼         ▼         ▼         ▼
   (wraps    around     in       both
    dimensions for      2D       torus)
```

**Why a NoC instead of a bus or crossbar?**

- **Scalability**: A bus becomes a bottleneck as you add more cores. A crossbar has O(N²) wire complexity. A mesh/torus scales linearly.
- **Locality**: Data can move between neighboring cores in one hop without traversing the whole chip.
- **Bandwidth**: Multiple independent data transfers can happen simultaneously on different parts of the mesh.
- **Multicast**: A core can send data to multiple destinations efficiently.

**Two NoC networks**: Wormhole has two independent NoC networks (NoC 0 and NoC 1) with different routing priorities (one routes right-then-down, the other left-then-up). This avoids deadlocks and doubles available bandwidth.

**Key NoC operations**:
```cpp
// In a TT-Metal kernel:

// Read data from DRAM to local L1
noc_async_read(dram_noc_addr, local_l1_addr, num_bytes);
noc_async_read_barrier();  // Wait for completion

// Write data from local L1 to another core's L1
noc_async_write(local_l1_addr, remote_core_noc_addr, num_bytes);
noc_async_write_barrier();

// Multicast: send data to a rectangular region of cores
noc_async_write_multicast(local_addr, mcast_noc_addr, num_bytes, num_dests);
```

### 3.6 Tiles: The Native Data Unit

Tenstorrent hardware operates natively on **32x32 tiles**. This design choice has deep implications:

#### Why 32x32?

The matrix engine (FPU) processes 8x16 × 16x16 matrix multiplications per cycle. A 32x32 tile is decomposed into 16x16 **faces** for computation:

```
   32×32 Tile Structure:
   ┌───────────────┬───────────────┐
   │               │               │
   │   Face 0      │   Face 1      │
   │   (16×16)     │   (16×16)     │
   │               │               │
   ├───────────────┼───────────────┤
   │               │               │
   │   Face 2      │   Face 3      │
   │   (16×16)     │   (16×16)     │
   │               │               │
   └───────────────┴───────────────┘

   Memory layout (contiguous):
   [Face 0 data (256 elements)] [Face 1 data] [Face 2 data] [Face 3 data]

   Each face: 16 rows × 16 columns = 256 elements
   Total tile: 4 × 256 = 1024 elements
```

**Why faces?** The matrix engine natively multiplies 16x16 matrices. A 32x32 tile multiplication decomposes into multiple 16x16 face multiplications:

```
C[32×32] = A[32×32] × B[32×32]

Decomposed into face operations:
  C_face00 = A_face00 × B_face00 + A_face01 × B_face20
  C_face01 = A_face00 × B_face01 + A_face01 × B_face21
  C_face10 = A_face10 × B_face00 + A_face11 × B_face20
  C_face11 = A_face10 × B_face01 + A_face11 × B_face21
```

**Tilized registers**: The compute engine's source and destination registers hold data in this tiled, face-based format. The unpacker converts from memory layout to the register format, and the packer converts back. This means:
- NoC transfers are large contiguous bursts (whole tiles) → high bandwidth utilization
- Compute operates on tile-sized chunks → high throughput
- All operations in the pipeline are tile-aligned

```python
# Creating a tiled tensor in TTNN:
tt_tensor = ttnn.from_torch(
    torch.randn(64, 64),
    layout=ttnn.TILE_LAYOUT,  # Converts to 32x32 tiles (4 tiles for 64x64)
    dtype=ttnn.bfloat16,
    device=device
)
```

### 3.7 Inside a Tensix Core

Each Tensix core is a self-contained processor with 5 RISC-V CPUs and specialized hardware:

```
                    ┌─────────────────────────────────────────────┐
                    │              TENSIX CORE                     │
                    │                                             │
  ┌──────────┐     │  ┌─────────┐  ┌─────────┐                  │
  │          │     │  │ BRISC   │  │ NCRISC  │  ← Data Movement │
  │   NoC    │◄───►│  │ (RISC0) │  │ (RISC1) │    (reader/writer)│
  │ Network  │     │  │ Reader  │  │ Writer  │                  │
  │          │     │  └────┬────┘  └────┬────┘                  │
  └──────────┘     │       │            │                        │
                    │       ▼            ▼                        │
                    │  ┌──────────────────────┐                  │
                    │  │   L1 SRAM (1.5 MB)   │                  │
                    │  │                      │                  │
                    │  │  ┌─CB0─┐  ┌─CB1─┐   │                  │
                    │  │  │input│  │input│   │  ← Circular      │
                    │  │  └──┬──┘  └──┬──┘   │    Buffers       │
                    │  │     │        │      │                  │
                    │  └─────┼────────┼──────┘                  │
                    │        │        │                          │
                    │  ┌─────▼────────▼──────────────────┐      │
                    │  │       COMPUTE ENGINE             │      │
                    │  │                                  │      │
                    │  │  ┌──────────┐  ┌──────────┐     │      │
                    │  │  │Unpacker 0│  │Unpacker 1│     │      │
                    │  │  │(formats) │  │(formats) │     │      │
                    │  │  └────┬─────┘  └────┬─────┘     │      │
                    │  │       │              │           │      │
                    │  │  ┌────▼────┐   ┌────▼────┐      │      │
                    │  │  │ SRC_A   │   │ SRC_B   │      │      │
                    │  │  │Register │   │Register │      │      │
                    │  │  └────┬────┘   └────┬────┘      │      │
                    │  │       │              │           │      │
                    │  │  ┌────▼──────────────▼────┐     │      │
                    │  │  │         FPU            │     │      │
                    │  │  │  8×16 × 16×16 matmul   │     │      │
                    │  │  │  per cycle (4 TFLOPS)   │     │      │
                    │  │  └───────────┬────────────┘     │      │
                    │  │              │                   │      │
                    │  │  ┌───────────▼────────────┐     │      │
                    │  │  │       DST Register      │     │      │
                    │  │  │    (Destination/Accum)   │     │      │
                    │  │  └───────────┬────────────┘     │      │
                    │  │              │                   │      │
                    │  │  ┌───────────▼────────────┐     │      │
                    │  │  │        SFPU            │     │      │
                    │  │  │  exp, log, sqrt, silu, │     │      │
                    │  │  │  sigmoid, tanh, etc.   │     │      │
                    │  │  └───────────┬────────────┘     │      │
                    │  │              │                   │      │
                    │  │  ┌───────────▼────────────┐     │      │
                    │  │  │       Packer           │     │      │
                    │  │  │  (format conversion)    │     │      │
                    │  │  └───────────┬────────────┘     │      │
                    │  │              │                   │      │
                    │  │  TRISC0  TRISC1  TRISC2         │      │
                    │  │  (RISC2) (RISC3) (RISC4)        │      │
                    │  │  unpack   math    pack           │      │
                    │  └──────────────────────────────────┘      │
                    │              │                              │
                    │         ┌────▼────┐                        │
                    │         │ CB_out  │  ← Output CB           │
                    │         └─────────┘                        │
                    └─────────────────────────────────────────────┘
```

**The 5 RISC-V processors**:

| RISC-V | Name | Role | What it does |
|--------|------|------|-------------|
| RISC 0 | BRISC | Reader | Issues NoC reads to pull data from DRAM/other L1 into local L1 circular buffers |
| RISC 1 | NCRISC | Writer | Issues NoC writes to push results from L1 to DRAM/other cores |
| RISC 2 | TRISC0 | Unpack | Controls the unpacker hardware: converts data from memory format to register format |
| RISC 3 | TRISC1 | Math | Controls FPU and SFPU: issues the actual compute instructions (matmul, exp, etc.) |
| RISC 4 | TRISC2 | Pack | Controls the packer: converts results from register format back to memory format |

**Key registers**:
- **SRC_A** and **SRC_B**: Source operand registers, fed by the two unpackers. These hold the input tiles for computation.
- **DST (Destination)**: The accumulator register. Results from FPU/SFPU operations land here. Can hold 16 tiles in bfloat16 mode or 8 tiles in float32 mode.
- All registers hold data in **tilized format** matching the face structure.

### 3.8 Circular Buffers: How Kernels Communicate

Circular buffers are the synchronization mechanism between the reader, compute, and writer kernels running on the same Tensix core. Think of them as **thread-safe producer-consumer queues** implemented in L1 memory.

#### How a Circular Buffer Works

```
Circular Buffer (4 slots, each holds 1 tile):

Initial state (empty):
   ┌──────┬──────┬──────┬──────┐
   │      │      │      │      │
   │ Slot │ Slot │ Slot │ Slot │
   │  0   │  1   │  2   │  3   │
   │      │      │      │      │
   └──────┴──────┴──────┴──────┘
     ▲ WR                 ▲ RD
     └── Write Pointer     └── Read Pointer

   tiles_received = 0
   tiles_acked = 0
   Available for writing: 4 slots
   Available for reading: 0 slots


After reader pushes 2 tiles:
   ┌──────┬──────┬──────┬──────┐
   │██████│██████│      │      │
   │ Tile │ Tile │      │      │
   │  A   │  B   │      │      │
   │██████│██████│      │      │
   └──────┴──────┴──────┴──────┘
                  ▲ WR    ▲ RD

   tiles_received = 2
   tiles_acked = 0
   Available for reading: 2 tiles (received - acked = 2)


After compute consumes 1 tile:
   ┌──────┬──────┬──────┬──────┐
   │      │██████│      │      │
   │(free)│ Tile │      │      │
   │      │  B   │      │      │
   │      │██████│      │      │
   └──────┴──────┴──────┴──────┘
                  ▲ WR           ▲ RD (wrapped!)

   tiles_received = 2
   tiles_acked = 1
   Available for writing: 3 slots
   Available for reading: 1 tile


After reader pushes 2 more, compute consumes 1:
   ┌──────┬──────┬──────┬──────┐
   │      │      │██████│██████│
   │      │(free)│ Tile │ Tile │
   │      │      │  C   │  D   │
   │      │      │██████│██████│
   └──────┴──────┴──────┴──────┘
   ▲ WR                         ▲ RD

   The write pointer wraps around!
```

#### CB API in Kernel Code

```cpp
// ═══ Reader Kernel (BRISC) ═══
// Reserve space in the CB, blocking if full
cb_reserve_back(cb_id, num_tiles);
// Get the L1 address to write to
uint32_t l1_addr = get_write_ptr(cb_id);
// Read data from DRAM into the reserved space
noc_async_read(dram_addr, l1_addr, tile_size_bytes);
noc_async_read_barrier();
// Signal to compute: "I produced num_tiles tiles"
cb_push_back(cb_id, num_tiles);

// ═══ Compute Kernel (TRISC) ═══
// Wait until reader has produced at least 1 tile
cb_wait_front(cb_in, 1);
// Acquire DST registers for computation
tile_regs_acquire();
// Copy tile from CB to source register
copy_tile(cb_in, 0, 0);  // CB index, tile index, DST index
// Compute (e.g., exponential)
exp_tile(0);  // Operates on DST register 0
// Commit and wait for result
tile_regs_commit();
tile_regs_wait();
// Write result to output CB
cb_reserve_back(cb_out, 1);
pack_tile(0, cb_out);      // Pack DST register 0 to output CB
cb_push_back(cb_out, 1);
// Free the consumed input tile
cb_pop_front(cb_in, 1);
tile_regs_release();

// ═══ Writer Kernel (NCRISC) ═══
// Wait for compute to produce a tile
cb_wait_front(cb_out, 1);
// Get L1 address of the produced tile
uint32_t l1_addr = get_read_ptr(cb_out);
// Write to DRAM
noc_async_write(l1_addr, dram_addr, tile_size_bytes);
noc_async_write_barrier();
// Free the consumed tile
cb_pop_front(cb_out, 1);
```

<details>
<summary><strong>How RISCs Read Pointers to Determine Data Availability</strong></summary>

The circular buffer interface has two key counters stored in L1:
- `tiles_received`: Incremented by the producer (reader or compute) after `cb_push_back`
- `tiles_acked`: Incremented by the consumer (compute or writer) after `cb_pop_front`

**Producing** (`cb_push_back`): Atomically increments `tiles_received`. The consumer RISC polls this value.

**Consuming** (`cb_wait_front`): The consumer RISC spins on `(tiles_received - tiles_acked) >= requested_tiles`. Once enough tiles are available, it proceeds.

**Freeing** (`cb_pop_front`): Atomically increments `tiles_acked`. The producer RISC uses this to calculate free space: `free_space = total_slots - (tiles_received - tiles_acked)`.

This lock-free design means the 5 RISCs never need to take locks. The reader and compute kernels can run fully asynchronously, with the CB counters providing implicit synchronization -- the fundamental enabler of pipelining.

</details>

### 3.9 Writing Kernels: CUDA vs TT-Metal

Let's compare a simple vector exponential (element-wise `exp`) in both programming models:

#### CUDA Kernel

```cuda
// CUDA: Single kernel handles everything
__global__ void exp_kernel(float* input, float* output, int n) {
    // Each thread computes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}

// Launch: grid of blocks, each with 256 threads
exp_kernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);
```

**GPU model**: One kernel, each thread reads one element from global memory, computes exp, writes one element back. Data lives in HBM. The hardware cache hierarchy (L2, L1) helps, but is implicit.

#### TT-Metal Kernels (3 separate kernels)

**Reader kernel** (runs on BRISC):
```cpp
// reader.cpp - Moves data from DRAM to L1 circular buffer
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_addr = get_write_ptr(cb_in);
        noc_async_read_tile(i, src_accessor, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
```

**Compute kernel** (runs on TRISC0/1/2):
```cpp
// compute.cpp - Performs exp on tiles in registers
void kernel_main() {
    uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in, cb_out);
    exp_tile_init();

    for (uint32_t i = 0; i < n_tiles; i++) {
        tile_regs_acquire();

        cb_wait_front(cb_in, 1);       // Wait for reader
        copy_tile(cb_in, 0, 0);         // CB → DST register 0
        exp_tile(0);                    // exp on DST register 0

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);           // DST register 0 → output CB
        cb_push_back(cb_out, 1);        // Signal writer

        cb_pop_front(cb_in, 1);         // Free input slot
        tile_regs_release();
    }
}
```

**Writer kernel** (runs on NCRISC):
```cpp
// writer.cpp - Moves results from L1 circular buffer to DRAM
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);       // Wait for compute
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, dst_accessor, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);        // Free output slot
    }
}
```

**Key differences**:
1. **Separation of concerns**: Reader, compute, and writer are independent programs running on independent RISC-V cores. They synchronize through circular buffers, not memory barriers.
2. **Explicit data movement**: You control exactly when and where data moves. No implicit cache hierarchy.
3. **Tile-native**: Operations like `exp_tile(0)` operate on whole 32x32 tiles at once, not individual elements.
4. **Natural pipelining**: While compute processes tile N, the reader can pre-fetch tile N+1 and the writer can write tile N-1. This happens automatically because the 3 kernels run concurrently.

---

## 4. Collective Communications Library (CCL)

### 4.1 Why CCLs?

When a model is distributed across multiple devices using tensor parallelism, each device has only a fraction of each weight matrix and computes a partial result. To produce correct outputs, devices must exchange data. CCLs provide optimized implementations of these communication patterns.

For example, in Llama-70B on 8 devices:
- The QKV projection weight is split across 8 devices (each has 1/8).
- Each device computes a partial QKV output.
- An **All-Gather** combines the partial results so each device has the full QKV.
- After the output projection (also split), a **Reduce-Scatter** sums partial results and distributes the reduced output.

### 4.2 All-Gather

**What it does**: Each device has a piece of the data. After All-Gather, every device has the complete data (concatenation of all pieces).

```
BEFORE All-Gather (4 devices, each has 1 chunk):

  Device 0: [A]
  Device 1: [B]
  Device 2: [C]
  Device 3: [D]

AFTER All-Gather (every device has the full tensor):

  Device 0: [A][B][C][D]
  Device 1: [A][B][C][D]
  Device 2: [A][B][C][D]
  Device 3: [A][B][C][D]
```

**Ring All-Gather Algorithm** (used in TT-Metal):

```
Step 0: Each device sends its local chunk to the next device in the ring.
  D0→D1: [A]     D1→D2: [B]     D2→D3: [C]     D3→D0: [D]

Step 1: Each device forwards the chunk it received.
  D0→D1: [D]     D1→D2: [A]     D2→D3: [B]     D3→D0: [C]

Step 2: Forward again.
  D0→D1: [C]     D1→D2: [D]     D2→D3: [A]     D3→D0: [B]

After 3 steps (N-1 for N devices), every device has all chunks!

Timeline (bidirectional ring doubles throughput):
  Step 0: D0 sends [A] right, receives [D] from left
  Step 1: D0 sends [D] right, receives [C] from left
  Step 2: D0 sends [C] right, receives [B] from left
  D0 now has: [A][B][C][D] ✓
```

```python
# TTNN usage
output = ttnn.all_gather(
    input_tensor,
    dim=3,                              # Gather along width dimension
    num_links=1,                        # Ethernet links to use
    cluster_axis=0,                     # Which mesh dimension to gather along
    mesh_device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,        # Ring topology for efficiency
)
```

### 4.3 All-Reduce

**What it does**: Each device has partial results. After All-Reduce, every device has the sum (or other reduction) of all partial results.

```
BEFORE All-Reduce (4 devices, partial sums):

  Device 0: [A0]    (partial result from device 0's weight shard)
  Device 1: [A1]    (partial result from device 1's weight shard)
  Device 2: [A2]
  Device 3: [A3]

AFTER All-Reduce (every device has the complete sum):

  Device 0: [A0+A1+A2+A3]
  Device 1: [A0+A1+A2+A3]
  Device 2: [A0+A1+A2+A3]
  Device 3: [A0+A1+A2+A3]
```

**Implementation in TT-Metal**: Decomposed as Reduce-Scatter + All-Gather:

```
Phase 1 - Reduce-Scatter:
  Each device gets the fully-reduced version of 1/N of the data.
  D0: [sum(A0..A3) chunk0]
  D1: [sum(A0..A3) chunk1]
  D2: [sum(A0..A3) chunk2]
  D3: [sum(A0..A3) chunk3]

Phase 2 - All-Gather:
  Gather all reduced chunks so every device has the full result.
  D0: [chunk0][chunk1][chunk2][chunk3] = full reduced tensor
  D1: [chunk0][chunk1][chunk2][chunk3]
  ...
```

This decomposition is bandwidth-optimal and allows overlapping computation with communication.

### 4.4 Reduce-Scatter

**What it does**: Reduces data across devices AND scatters the result -- each device gets 1/N of the reduced output.

```
BEFORE Reduce-Scatter (4 devices):

  Device 0: [A0 | B0 | C0 | D0]    (split into 4 chunks)
  Device 1: [A1 | B1 | C1 | D1]
  Device 2: [A2 | B2 | C2 | D2]
  Device 3: [A3 | B3 | C3 | D3]

AFTER Reduce-Scatter:

  Device 0: [A0+A1+A2+A3]          (sum of all devices' "A" chunk)
  Device 1: [B0+B1+B2+B3]          (sum of all devices' "B" chunk)
  Device 2: [C0+C1+C2+C3]
  Device 3: [D0+D1+D2+D3]
```

**Ring Reduce-Scatter Algorithm**:

```
Step 0: Each device sends one chunk to the next, and adds what it receives.
  D0 sends D0 to D1, receives D3 from D3 → accumulates into local buffer
  D1 sends A1 to D2, receives A0 from D0 → accumulates
  ...

Step 1: Continue ring rotation with accumulated partial sums.

Step 2: Final ring step completes the reduction.

After N-1 steps, each device has the fully-reduced version of one chunk.
```

```python
# Used after row-parallel matmul (FF2 in MLP)
w2_out_reduced = ttnn.reduce_scatter(
    w2_out,
    scatter_dim=3,                       # Scatter along width
    math_op=ttnn.ReduceType.Sum,         # Sum reduction
    num_links=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## 5. High-Performance Computing Techniques

### 5.1 Pipelining: Overlapping Compute and Data Movement

The fundamental advantage of the TT-Metal 3-kernel model is that the reader, compute, and writer run **concurrently** on separate RISC-V processors:

```
Traditional (sequential):
  |--Read A--|--Compute A--|--Write A--|--Read B--|--Compute B--|--Write B--|

Pipelined (TT-Metal):
  Reader:   |--  Read A --|--  Read B --|--  Read C --|--Read D--|
  Compute:                |--Compute A--|--Compute B--|--Compute C--|--Compute D--|
  Writer:                               |-- Write A --|--Write B--|--Write C--|--Write D--|

  Total time: dramatically reduced!
  The longest stage determines throughput, and latency of other stages is hidden.
```

From the FlashAttention implementation on Wormhole/Blackhole:

> "Like all kernels written in the TT-Metal stack, our FlashAttention kernel takes advantage of concurrent reader, writer, and compute kernels to overlap data movement with compute. The RISCs within a Tensix synchronize using circular buffers, which can be thought of as thread-safe producer/consumer queues."
> -- *FlashAttention tech report*

### 5.2 Double and Triple Buffering

By sizing circular buffers to hold 2 or 3 tiles instead of 1, we enable the reader to pre-fetch the next tile while compute is processing the current one:

```
Single buffer (stalls on every tile):
  Reader:   |Read T0|      stall      |Read T1|      stall      |Read T2|
  Compute:          |  Compute T0  |          |  Compute T1  |

Double buffer (reader can stay ahead):
  Reader:   |Read T0|Read T1|Read T2|Read T3|Read T4|
  Compute:          |Comp T0|Comp T1|Comp T2|Comp T3|
                     ↑ T1 already in buffer, no stall!
```

Real example from the prefetcher kernel using **triple buffering** with transaction IDs:

```cpp
// From ttnn/cpp/ttnn/operations/prefetcher/device/kernels/reader_dram.cpp

constexpr uint32_t total_num_blocks_in_buffer = 3;  // Triple buffered!

for (uint32_t block = 0; block < num_blocks; block++) {
    // Set transaction ID for this block
    noc_async_read_set_trid(curr_block_trid);

    // Issue all read commands for this block
    for (uint32_t h = 0; h < block_num_pages; ++h) {
        noc_async_read_one_packet_with_state_with_trid(
            src_addr, read_addr, l1_write_addr, curr_block_trid);
    }

    // Don't wait for current block -- wait for the OLDEST block instead
    // This keeps 2 blocks in flight at all times
    if (num_free_blocks < 3) {
        noc_async_read_barrier_with_trid(oldest_block_trid);
        cb_push_back(cb_id, max_block_num_tiles);
    }

    // Advance transaction ID (wraps: 1 → 2 → 3 → 1)
    curr_block_trid = (curr_block_trid == 3) ? 1 : (curr_block_trid + 1);
}
```

### 5.3 SFPU Operation Chaining

When computing complex functions like `softplus(x) = log(1 + exp(x))`, you could call three separate operations, each requiring a round-trip through circular buffers. Instead, SFPU chaining keeps intermediate results in the DST register:

```cpp
// From programming_examples/sfpu_eltwise_chain/kernels/compute/compute.cpp

// Load data into DST register 0
copy_tile(src_cb_index, 0, 0);      // Input data → DST[0]
copy_tile(ones_cb_index, 0, 1);     // Ones tile → DST[1]

// Chain SFPU operations -- results stay in DST, no memory round-trips!
exp_tile_init();
exp_tile(0);                        // DST[0] = exp(DST[0])

add_binary_tile_init();
add_binary_tile(0, 1, 0);          // DST[0] = DST[0] + DST[1] = exp(x) + 1

log_tile_init();
log_tile(0);                       // DST[0] = log(DST[0]) = log(exp(x) + 1)

// Only NOW do we write back to memory
pack_tile(0, result_cb_index);
```

**Benefits**:
- Eliminated 2 intermediate memory round-trips
- Data stays in registers (fastest storage)
- Reduces memory bandwidth pressure
- Lower latency for the whole computation

### 5.4 DRAM Bandwidth Saturation

For decode-phase matmuls (memory-bound), Tenstorrent achieves **92%+ of theoretical DRAM bandwidth**. Key techniques:

1. **Place DRAM readers next to their banks**: Each of the 12 DRAM banks gets a dedicated reader core placed physically adjacent to it. This minimizes NoC congestion.

2. **Use different NoC virtual channels**: Two readers in the same row use different VCs so the NoC can fairly serve both.

3. **Transaction ID pipelining**: Issue read requests with IDs and barrier on older IDs, keeping DRAM continuously busy.

```
Achieved bandwidth (from tech report):
| Test              | DRAM BW @12GBps | DRAM BW @14GBps |
|-------------------|-----------------|-----------------|
| DRAM spec         | 288 GB/s        | 336 GB/s        |
| DRAM benchmark    | 267 GB/s (92%)  | 310 GB/s (92%)  |
| Llama3-70 decode  | 239-260 GB/s    | 247-294 GB/s    |
| Mixtral decode    | 243-261 GB/s    | 267-300 GB/s    |
```

### 5.5 Metal Trace and Multiple Command Queues

#### Metal Trace

For models that repeat the same sequence of operations (like decode iterations), Metal Trace records the dispatch commands into a DRAM buffer and replays them, eliminating host overhead:

```
Without trace (host-bound model):
  Host: |dispatch op1|dispatch op2|dispatch op3|...
  Device: |     wait    |op1| wait |op2| wait |op3|

With trace (commands pre-recorded):
  Host: |replay trace|        (done instantly)
  Device: |op1|op2|op3|op4|op5|  (no gaps!)
```

```python
# Capture trace
tid = ttnn.begin_trace_capture(device, cq_id=0)
output = run_model(input_tensor)  # Record all operations
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute trace (near-zero host overhead)
ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

#### Multiple Command Queues

Two independent command queues allow overlapping I/O with computation:

```
Single CQ:
  |Write Input|Run Model|Read Output|Write Input|Run Model|Read Output|

Two CQs (CQ0=ops, CQ1=I/O):
  CQ0 (ops):    |  Run Model  |  Run Model  |  Run Model  |
  CQ1 (I/O):  |Wr|         |Wr|Rd|      |Wr|Rd|
                 ↑ overlapped! ↑

  Gap between model runs eliminated!
```

---

## 6. Building with TTNN: Models, Use Cases, and Next Steps

### What You Can Build

TTNN supports a wide range of model architectures:

| Category | Examples | Repository Path |
|----------|---------|-----------------|
| **Large Language Models** | Llama 3/3.1 (8B, 70B, 405B), Qwen 2.5, DeepSeek V3, Falcon 7B, Mixtral 8x7B | `models/demos/`, `models/tt_transformers/` |
| **Vision-Language Models** | Qwen 2.5 VL, Qwen 3 VL | `models/demos/qwen25_vl/`, `models/demos/qwen3_vl/` |
| **Encoder Models** | BERT (multiple variants), Sentence-BERT, BGE-Large | `models/demos/bert/`, `models/demos/sentence_bert/` |
| **Vision Models** | ViT, YOLOv4, OWL-ViT | `models/demos/vision/`, `tech_reports/ViT-TTNN/` |
| **CNNs** | ResNet, custom CNNs | `models/tt_cnn/`, `tech_reports/CNNs/` |

### Getting Started with Your Own Model

```python
import torch
import ttnn

# 1. Open a device
device = ttnn.open_device(device_id=0)

# 2. Convert PyTorch tensors to TTNN
torch_input = torch.randn(1, 1, 32, 8192)
tt_input = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# 3. Use TTNN operations (same feel as PyTorch)
# RMSNorm
output = ttnn.rms_norm(tt_input, epsilon=1e-5, weight=tt_gamma)

# Linear projection
output = ttnn.linear(tt_input, tt_weight, bias=tt_bias)

# Attention
attn_out = ttnn.transformer.scaled_dot_product_attention(
    Q, K, V, is_causal=True
)

# Softmax
probs = ttnn.softmax(scores, dim=-1)

# 4. Convert back to PyTorch
result = ttnn.to_torch(output)

# 5. Close device
ttnn.close_device(device)
```

### Techniques for Writing High-Performance Kernels

Based on analysis of production kernels in the TT-Metal codebase, here are the key techniques developers use:

1. **L1 Sharding over DRAM Interleaving**: Keep data in L1 between operations. This alone can eliminate DRAM round-trips and provide massive speedups (the FlashAttention report showed 20x speedup).

2. **DRAM-Sharded Matmul for Decode**: Place reader cores next to their DRAM banks, use transaction ID pipelining, and shard weights across banks. Achieves 92%+ bandwidth utilization.

3. **Double/Triple Buffering**: Size circular buffers for 2-3 tiles to enable full pipelining between reader, compute, and writer.

4. **SFPU Operation Chaining**: Keep intermediate results in DST registers to avoid memory round-trips for multi-step computations.

5. **Fused Operations**: Use TTNN's built-in fusion support (e.g., `input_tensor_a_activation=ttnn.UnaryOpType.SILU` on multiply) to combine operations.

6. **Math Fidelity Tuning**: Use LoFi or HiFi2 for speed-critical operations where slight precision loss is acceptable. Reserve HiFi4 for accuracy-sensitive layers.

7. **Metal Trace**: Record and replay operation sequences to eliminate host dispatch overhead.

8. **Multiple Command Queues**: Overlap I/O transfers with model execution using 2 CQs.

9. **Multicast for Shared Data**: When multiple cores need the same data (e.g., KV heads in multi-query attention), use NoC multicast instead of point-to-point.

10. **Weight Prefetching**: Pre-load the next layer's weights while computing the current layer (used in `models/tt_transformers/tt/prefetcher.py`).

### Key Resources in the Repository

| Resource | Path | Description |
|----------|------|-------------|
| Tech Reports | `tech_reports/` | In-depth guides on LLMs, FlashAttention, performance optimization, tensor layouts, etc. |
| Programming Examples | `tt_metal/programming_examples/` | Working examples: matmul, eltwise, NoC transfers, multi-core, distributed |
| Jupyter Tutorials | `ttnn/tutorials/` | Interactive notebooks: intro, basic ops, matmul, attention, MLP, CNN |
| LLM Guide | `tech_reports/LLMs/llms.md` | Complete guide to bringing up LLMs on TT hardware |
| Model Bringup | `tech_reports/ttnn/TTNN-model-bringup.md` | Step-by-step model porting guide |
| ViT Guide | `tech_reports/ViT-TTNN/vit.md` | Beginner-friendly ViT walkthrough |
| Performance Guide | `tech_reports/AdvancedPerformanceOptimizationsForModels/` | Metal Trace, multi-CQ, and advanced optimization techniques |
| Transformer Code | `models/tt_transformers/tt/` | Production attention, MLP, decoder, and model implementations |

### vLLM Integration

Tenstorrent hardware integrates with [vLLM](https://github.com/vllm-project/vllm) for production LLM serving with paged attention, continuous batching, and efficient scheduling. See `tech_reports/LLMs/vLLM_integration.md` for details.

---

> **This document is open source, just like the TT-Metal stack it describes. Contributions, corrections, and improvements are welcome at [github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal).**
