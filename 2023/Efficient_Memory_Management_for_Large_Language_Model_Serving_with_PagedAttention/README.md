# Efficient Memory Management for Large Language Model Serving with PagedAttention

- **Authors:** Woosuk Kwon*, Zhuohan Li*, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Affiliations:** UC Berkeley, Stanford University, UC San Diego, Independent Researcher
- **Published:** SOSP 2023 (arXiv:2309.06180), October 2023
- **Keywords:** LLM serving, KV cache, memory management, virtual memory, paging, continuous batching, throughput
- **Webpage:** https://blog.vllm.ai/2023/06/20/vllm.html
- **GitHub:** https://github.com/vllm-project/vllm

---

## Pass 1 — Bird's-Eye View

| C | Assessment |
|---|-----------|
| **Category** | Systems paper (SOSP): memory management for high-throughput LLM inference serving. Introduces the PagedAttention algorithm and the vLLM serving engine built on it. |
| **Context** | Builds on iteration-level (continuous) batching from Orca (OSDI 2022), optimized transformer kernels from FasterTransformer, Megatron-LM tensor parallelism, and — conceptually — classic OS virtual memory with paging (Kilburn's one-level storage, 1962) and copy-on-write. |
| **Correctness** | Sound and well quantified. The core premise — serving throughput is memory-bound because KV cache[^1] is stored contiguously and pre-allocated at maximum length — is backed by profiling (only 20.4–38.2% of KV memory holds actual token states in prior systems). One caveat: Orca was not publicly available, so all Orca baselines are the authors' own reimplementations. |
| **Contributions** | (1) Quantifying KV-cache memory waste (reserved slots, internal and external fragmentation) in existing serving systems; (2) PagedAttention — attention over KV cache stored in fixed-size, non-contiguous blocks; (3) vLLM — an end-to-end distributed serving engine with block tables, copy-on-write sharing, and preemptive scheduling; (4) 2–4× throughput over state of the art at equal latency, without changing model outputs. |
| **Clarity** | Excellently written. The OS analogy (blocks = pages, tokens = bytes, requests = processes) carries the whole paper; worked examples (Figs. 6–9) make each mechanism concrete. A model systems paper. |

**30-second summary.** LLM serving throughput is limited by how many requests can be batched, which is limited by GPU memory for the KV cache. Prior systems (FasterTransformer, Orca) store each request's KV cache as one contiguous tensor pre-allocated at maximum sequence length, wasting 60–80% of KV memory to reservation and fragmentation. PagedAttention borrows OS virtual-memory paging: the KV cache is split into fixed-size blocks (default 16 tokens) that live anywhere in GPU memory, addressed through per-request block tables, with new blocks allocated on demand. This bounds internal fragmentation to one block per sequence, eliminates external fragmentation entirely, and — because multiple logical blocks can map to one physical block with reference counting and block-level copy-on-write — enables KV sharing across parallel samples, beam candidates, and shared prompt prefixes. The vLLM engine adds all-or-nothing preemption (swap to CPU or recompute) and Megatron-style tensor parallelism. Result: near-zero KV waste (96.3% utilization), 2–4× higher throughput than Orca at the same latency (up to 22× vs FasterTransformer), with bigger gains for long sequences, large models, and beam search.

---

## Pass 2 — Careful Read

### Core Idea in One Sentence

Store the KV cache in fixed-size non-contiguous blocks addressed through a per-request block table — exactly like OS virtual-memory pages — so KV memory can be allocated on demand and shared across sequences, letting far more requests fit in a batch.

### Method / Approach

- **PagedAttention kernel:** partition each sequence's keys and values into KV blocks of $B$ tokens; the attention of query $q_i$ is computed block-by-block against key block $K_j$ and value block $V_j$ , fetched via the block table, so blocks need not be contiguous in physical GPU memory.
- **KV cache manager (paging):** each request holds *logical* blocks filled left to right; a block table maps them to *physical* blocks allocated only when needed. Waste is bounded by one partially-filled block per sequence; all blocks are the same size, so external fragmentation cannot occur.
- **Block-level sharing with copy-on-write:** physical blocks carry reference counts; parallel samples share all prompt blocks, beam-search candidates share dynamically evolving block trees, and provider-defined shared prefixes (system prompts) are cached once. Writes to a shared block trigger a copy of just that block.
- **Scheduling and preemption:** first-come-first-serve at sequence-group granularity; under memory pressure, vLLM evicts *all* blocks of the latest-arrived sequences (all-or-nothing, since all blocks of a sequence are accessed together) and recovers them either by swapping to CPU RAM or by recomputing the KV cache in a single prompt-phase pass over the already-generated tokens.

### Key Results

Workloads: ShareGPT and Alpaca traces with Poisson arrivals; metric is *normalized latency* (mean per-request end-to-end latency ÷ output length) vs sustainable request rate.

| Comparison | Setting | Result |
|------------|---------|--------|
| vLLM vs Orca (Oracle) | OPT-13B/66B/175B, ShareGPT | 1.7–2.7× higher sustainable request rate |
| vLLM vs Orca (Max) | same | 2.7–8× higher request rate |
| vLLM vs FasterTransformer | same | up to 22× higher request rate |
| Batched requests | OPT-13B, ShareGPT | 30.4 vs 13.6 (Orca Oracle) / 7.0 (Orca Max) requests per batch |
| KV utilization | vs Orca variants | 96.3% of KV memory holds real token states vs 20.4–38.2% |
| Beam search (width 6) | OPT-13B, Alpaca | 2.3× over Orca (Oracle); 37.6–55.2% KV memory saved by sharing |
| Shared prefix (5-shot) | LLaMA-13B, WMT16 translation | 3.58× over Orca (Oracle) |
| Chatbot | OPT-13B, ShareGPT history | 2× over all Orca variants |

- **Kernel overhead is real but contained:** paged attention kernels are 20–26% slower than FasterTransformer's contiguous kernels, but attention is only one operator — end-to-end, vLLM still wins decisively.
- **Block size 16 is the sweet spot:** small blocks underutilize GPU parallelism, large blocks increase internal fragmentation and reduce sharing; 16 works across workloads and became the default.
- **Recomputation vs swapping:** recomputation is more efficient with small blocks (swapping many tiny blocks throttles PCIe), swapping wins for large blocks; comparable in the 16–64 range, and recomputation never exceeds 20% higher overhead than swapping.

### Strengths

- **Right abstraction, perfectly borrowed:** the virtual-memory analogy is not decorative — pages/bytes/processes map one-to-one onto blocks/tokens/requests, and copy-on-write and swapping carry over with LLM-specific twists (all-or-nothing eviction, recomputation as a recovery path that has no OS analogue).
- **Exact, not approximate:** unlike quantization or pruning, paging changes memory layout only — model outputs are bit-identical, so the 2–4× throughput is free of accuracy trade-offs.
- **Honest accounting of overheads:** the paper measures its own kernel slowdown (20–26%) and the indirection costs rather than hiding them behind end-to-end numbers.
- **Complete system, released:** scheduler, distributed execution, OpenAI-compatible frontend, and CUDA kernels (8.5K lines Python + 2K C++/CUDA), open-sourced — which is why it became infrastructure rather than just a paper.
- **Waste taxonomy:** the reserved / internal-fragmentation / external-fragmentation breakdown (Fig. 2) gave the field shared vocabulary for reasoning about KV memory.

### Weaknesses / Open Questions

1. **Reimplemented baselines:** Orca is not public, so the headline 2–4× is measured against the authors' own three Orca variants (Max / Pow2 / Oracle); the Oracle variant is a best-case bound, but reimplementation bias is hard to rule out.
2. **Attention-kernel tax:** every attention kernel must be rewritten to walk block tables; the 20–26% kernel overhead grows in relative importance as attention kernels get faster (later exposed by vAttention, which showed GPU virtual memory can provide paging without software indirection).
3. **Compute-bound regimes:** when memory is plentiful relative to sequence length (e.g., OPT-175B on 8×A100-80GB with short Alpaca sequences), Orca-style systems nearly close the gap — paging only helps when KV memory is the binding constraint.
4. **Prefill/decode interference unaddressed:** scheduling is FCFS over whole sequence groups; the paper predates concerns about prompt-phase compute blocking decode latency (later addressed by chunked prefill and disaggregation).
5. **CPU-side and cross-request cache reuse is manual:** shared prefixes must be registered by the provider in advance; automatic detection of arbitrary shared prefixes across requests came only with later systems (e.g., RadixAttention).

### References to Follow Up

1. **Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al., OSDI 2022: introduced iteration-level (continuous) batching; vLLM's complementary counterpart and main baseline.
2. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao et al., NeurIPS 2022: the other landmark attention-efficiency work of the era, optimizing on-chip IO where PagedAttention optimizes off-chip memory management.
3. **High-throughput Generative Inference of Large Language Models with a Single GPU (FlexGen)** — Sheng et al., ICML 2023: offloading-based batch inference; contrasts the offline-throughput setting with vLLM's online serving.
4. **Efficiently Scaling Transformer Inference** — Pope et al., MLSys 2023: Google's analysis of serving trade-offs (multi-query attention, partitioning) that frames the memory-bound decode problem PagedAttention attacks.
5. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** — Shoeybi et al., 2019: the tensor-parallel execution scheme vLLM adopts for multi-GPU serving.

---

## Pass 3 — Virtual Re-implementation

### Detailed Technical Summary

**Why KV memory is the bottleneck.** Serving a 13B model on an A100-40GB: ~65% of memory is static weights, ~30% is KV cache, and activations are ephemeral. Each token's KV cache for OPT-13B costs $2 \times 5120 \times 40 \times 2$ bytes $= 800$ KB (key+value × hidden size × layers × FP16), so one 2048-token request needs up to 1.6 GB. Decode is autoregressive — one token per step, matrix-vector rather than matrix-matrix — so GPUs are underutilized and throughput comes from batching; batch size is capped by KV memory. Prior systems allocate each request's KV cache as one contiguous tensor sized to the *maximum possible* length (e.g., 2048), producing three kinds of waste: *reserved* slots held for future tokens for the whole request lifetime, *internal fragmentation* from over-provisioning (request finishes earlier than the maximum), and *external fragmentation* from the buddy allocator (chunks have request-specific sizes). Measured on real traces, only 20.4–38.2% of KV memory in Orca-style systems holds live token states.

**PagedAttention.** Split each sequence's KV cache into *KV blocks* of block size $B$ tokens: key block $K_j = (k_{(j-1)B+1}, \dots, k_{jB})$ , value block $V_j$ likewise. Standard attention

```math
a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i} \exp(q_i^\top k_t / \sqrt{d})}, \qquad o_i = \sum_{j=1}^{i} a_{ij} v_j
```

is regrouped into a block-wise computation

```math
A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t 1 / \sqrt{d})}, \qquad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top ,
```

where $A_{ij}$ is the row vector of attention scores over the $j$ -th block. The kernel fetches each $K_j, V_j$ through the block table, so physical placement is arbitrary. Keys/values of all layers and heads for the same token positions are kept in one block (per-layer/head splitting would work too; this choice simplifies the implementation).

**KV cache manager.** The OS analogy: blocks ↔ pages, tokens ↔ bytes, requests ↔ processes. A request's sequence is a list of *logical* blocks filled left to right; the *block engine* on each GPU worker pre-carves GPU DRAM (and CPU RAM, for swap space) into physical blocks. A per-sequence *block table* stores, for each logical block, the physical block number and the count of filled slots. Physical blocks are allocated only when the previous block is full — no reservation for the maximum length. Waste per sequence is at most $B-1$ slots in the final block ( $< 4$ % at $B{=}16$ for typical lengths), and uniform block size eliminates external fragmentation. Freed blocks return to a free list when a request completes.

**Decoding walk-through.** For a 7-token prompt with $B{=}4$ : prefill fills logical blocks 0–1 (4 + 3 tokens) mapped to arbitrary physical blocks, computing the prompt KV with a conventional (matrix-matrix) attention kernel and generating the first token into the free slot of block 1. Decode steps append into the last block until full, then a new physical block is allocated and appended to the block table. The scheduler batches all sequences' token IDs and block tables into one control message per step; workers run the model and read/write KV strictly through the tables.

**Sharing and copy-on-write.** Multiple logical blocks (from different sequences) may map to the same physical block, which carries a *reference count*.

- *Parallel sampling:* all samples of a request share the prompt's physical blocks (ref count = number of samples). When a sample writes into a shared last block, vLLM allocates a new physical block, copies the shared block's contents, decrements the ref count — block-granularity copy-on-write, as in OS `fork`. At most one block is ever copied per divergence.
- *Beam search:* candidates share blocks along the beam tree (prompt block + prefixes of surviving candidates), with sharing patterns changing every step; freeing happens when a candidate's ref counts hit zero. This replaces the frequent bulk KV copies prior systems needed on beam divergence, saving up to 55% of KV memory (Alpaca) / 66% (ShareGPT) at beam width 6.
- *Shared prefix / system prompt:* the provider pre-computes and caches physical blocks for common prefixes; requests map their first logical blocks onto them (last block copy-on-write), skipping prefix prefill entirely.

**Scheduling, preemption, recovery.** Policy is first-come-first-serve over *sequence groups* (all sequences of one request, e.g., all beam candidates — they are scheduled and preempted together, since they may share blocks). When free physical blocks run out, vLLM preempts the most-recently-arrived groups (evicting *all* their blocks — an all-or-nothing policy justified by the fact that all blocks of a sequence are accessed together every step) and stops admitting new requests until preempted ones finish. Two recovery mechanisms: (1) *swapping* — evicted blocks are copied to a CPU-RAM block pool (bounded by the GPU KV space, so swap space never exceeds GPU KV memory); (2) *recomputation* — drop the blocks and, when rescheduled, re-run a single prefill over prompt + already-generated tokens (much cheaper than the original decode since it is one matrix-matrix pass). Microbenchmarks: swapping is bad at small block sizes (many tiny PCIe transfers), recomputation is constant across block sizes; they are comparable for $B \in [16, 64]$ .

**Distributed execution.** Megatron-LM SPMD tensor parallelism: attention is split by head, so every worker handles the same token positions but a subset of heads — the block tables are *shared* across workers, and each worker stores the same physical block IDs holding only its heads' slice ( $1/N$ of each block). The centralized scheduler broadcasts token IDs + block tables each step; workers need no memory-management coordination beyond that, synchronizing activations with all-reduce as usual.

**Implementation and kernels.** 8.5K lines of Python (scheduler, block manager, FastAPI OpenAI-compatible frontend) + 2K lines of C++/CUDA. Three custom kernels: (1) *fused reshape + block write* — new KV vectors are split, laid out for efficient block reads, and scattered to block-table positions in one kernel; (2) *fused block read + attention* — adapted from FasterTransformer's attention kernel to walk block tables, one GPU warp per block, supporting variable sequence lengths in a batch; (3) *fused block copy* — batches all copy-on-write copies of a step into one kernel launch instead of many small `cudaMemcpyAsync` calls. The engine exposes exactly three sequence-level methods — `fork`, `append`, `free` — from which parallel sampling, beam search, and prefix sharing are all composed.

**Evaluation setup.** Models: OPT-13B (1×A100), OPT-66B (4×A100), OPT-175B (8×A100-80GB), LLaMA-13B. Workloads synthesized from ShareGPT (mean input 161 tokens, output 338) and Alpaca (input 19, output 58) with Poisson arrival rates; 1-hour traces (15 min for 175B). Baselines: FasterTransformer with a dynamic-batching scheduler, and three in-house Orca variants differing in output-length knowledge: Oracle (true lengths — infeasible upper bound), Pow2 (reserves ≤2× true length), Max (reserves model maximum, 2048). Metric: normalized latency (s/token) vs request rate; throughput is the highest rate sustaining low normalized latency.

### Hidden Assumptions

1. **Decode dominates and is memory-bound:** the design targets the autoregressive phase; if workloads were prefill-heavy (very long prompts, short outputs), the KV-residency benefit shrinks and prefill compute becomes the bottleneck — a regime the paper does not evaluate.
2. **Uniform, known block granularity works for all layers:** one global block size (16) is assumed adequate across models, layers, and head configurations; no per-layer adaptation is considered.
3. **The GPU memory pool for KV is statically pre-carved:** the block engine assumes weights and activation workspace have fixed, known sizes so the remaining DRAM can be pre-allocated as KV blocks — true for static transformer serving, false for workloads with dynamic weight residency (LoRA swapping, MoE offloading).
4. **Output lengths are unpredictable:** the whole waste argument assumes serving systems cannot know output lengths in advance (hence Orca-Oracle being labeled infeasible); later length-prediction schedulers relax this.
5. **Sequences in a group live and die together:** all-or-nothing eviction and gang scheduling assume intra-request sequences share enough blocks that partial eviction never pays off.
6. **PCIe (not NVLink) connects CPU and GPU for swap:** the swapping-vs-recomputation trade-off is measured on PCIe bandwidth; on hosts with faster CPU–GPU links (Grace Hopper class) the balance shifts toward swapping.
7. **Homogeneous FP16 dense transformers:** per-token KV cost is assumed identical across requests and layers; multi-query/grouped-query attention, sliding-window layers, and MoE routing (which change KV geometry per layer) are out of scope.

### Reproducibility Notes

- **Code:** fully open source at https://github.com/vllm-project/vllm (the paper's artifact became the production project); kernels, scheduler, and frontend all included.
- **Data:** ShareGPT and Alpaca datasets (public); request traces are synthesized — tokenized input/output lengths + Poisson arrivals — so exact traces depend on seed and are not shipped, but the recipe is fully described.
- **Compute:** Google Cloud A2 instances — 1×A100-40GB (13B), 4×A100-40GB (66B), 8×A100-80GB (175B); NCCL for tensor-parallel communication.
- **Baselines:** FasterTransformer is public, but its scheduler is custom-built by the authors; all three Orca variants are reimplementations (Orca itself was never released) — the largest reproducibility gap, though the Oracle variant at least bounds Orca from above.
- **Hyperparameters:** block size 16 (default; ablated 1–256), FCFS scheduling, max 2048-token sequences for OPT; recovery via recomputation or swap both implemented.
- **Underspecified:** exact scheduler batching limits (max tokens per batch), the memory fraction reserved for activations, and the ShareGPT filtering/tokenization details are only loosely described — all visible in the released code, which effectively serves as the appendix.

### Ideas for Future Work

1. **Hardware-assisted paging:** use GPU virtual-memory APIs (CUDA VMM) to get on-demand physical allocation while keeping virtually contiguous KV tensors — eliminating the block-table indirection and the need to rewrite every attention kernel (this became vAttention, 2024).
2. **Automatic cross-request prefix sharing:** replace provider-registered shared prefixes with a global prefix tree over physical blocks so arbitrary common prefixes are deduplicated on the fly (realized by RadixAttention/SGLang).
3. **Smarter preemption:** length-prediction-aware scheduling and partial (per-block) eviction with cost models over swap vs recompute, instead of FCFS + all-or-nothing.
4. **Paging beyond one GPU tier:** extend the block abstraction to multi-tier memory (HBM ↔ CPU ↔ NVMe ↔ remote memory pools) for very long contexts and disaggregated prefill/decode clusters.
5. **Co-designing with attention variants:** block layouts specialized for grouped-query attention, sliding-window layers, and quantized (FP8/INT4) KV storage — each changes the per-token block geometry and sharing calculus.

---

## Pass 4 — Modern Perspective Review (as of July 2026)

### What Has Changed Since Publication

- **vLLM became default infrastructure:** the artifact grew into one of the most widely deployed open-source LLM inference engines (OpenAI-compatible serving, now a community project with a V1 engine rewrite), and "paged KV cache" became a standard feature in essentially every serving stack — TensorRT-LLM, HuggingFace TGI, DeepSpeed-FastGen, LightLLM, llama.cpp all adopted block-based KV management.
- **Prefix caching generalized:** SGLang's RadixAttention (2024) extended vLLM's manual shared-prefix blocks into an automatic radix tree over the KV cache, deduplicating arbitrary common prefixes across requests; vLLM followed with automatic prefix caching. Multi-turn and agentic workloads made this the single biggest lever after paging itself.
- **The scheduling frontier moved past FCFS:** chunked prefill (Sarathi-Serve, OSDI 2024) interleaves prompt and decode compute to fix prefill-blocking-decode latency; prefill/decode *disaggregation* (DistServe, Splitwise, Mooncake) splits the two phases across machine pools with KV cache transfer — questions the paper's single-engine framing entirely.
- **Paging itself was challenged:** vAttention (2024) showed CUDA virtual-memory APIs can provide on-demand physical allocation while keeping KV tensors virtually contiguous, avoiding PagedAttention's software indirection and per-kernel rewrites; it sparked a debate about whether user-space paging was the right layer, though block tables remain dominant in practice.
- **KV geometry changed under the abstraction:** grouped-query attention became universal (shrinking KV per token 4–8×), FP8/INT4 KV quantization became routine, and sliding-window/hybrid-attention models (Mistral, Gemma-class) plus KV compression methods (H2O, InfiniGen-style eviction) reduced the pressure that motivated paging — while million-token contexts and inference-time reasoning (long chain-of-thought) increased it again.
- **Evaluation standards shifted:** normalized latency gave way to TTFT (time-to-first-token) / TPOT (time-per-output-token) SLOs and *goodput* under SLO attainment; weight quantization (e.g., [Extreme Compression of Large Language Models via Additive Quantization](../../2024/Extreme_Compression_of_Large_Language_Models_via_Additive_Quantization/) and successors) now routinely combines with paged serving, trading weight memory for more KV block space.

### Has the Community Accepted the Claims?

Overwhelmingly. PagedAttention is one of the most influential systems ideas of the LLM era: the waste taxonomy, the block-table abstraction, and copy-on-write KV sharing were adopted essentially verbatim across the industry, and the paper's 2–4× throughput claim was reproduced in practice wherever KV memory was the binding constraint. The refinements that followed sharpened rather than overturned it: vAttention showed the same benefits can be had through hardware virtual memory with less kernel complexity (a legitimate architectural critique — the 20–26% kernel overhead the paper itself measured is the cost being contested); Sarathi-Serve and the disaggregation line showed that once memory waste is solved, *scheduling* (prefill/decode interference) becomes the next bottleneck; and RadixAttention showed the sharing mechanism generalizes far beyond what the paper shipped. Notably, the paper's own honest framing — paging helps when and only when KV memory binds — has held up exactly.

---

### Comparison Papers

#### Predecessors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu et al. | 2022 | Iteration-level (continuous) batching; the complementary technique and the paper's main baseline (reimplemented as Max/Pow2/Oracle variants) |
| FasterTransformer | NVIDIA | 2021–2023 | Highly optimized latency-oriented inference kernels; baseline, and the attention kernel PagedAttention's kernel is adapted from |
| One-Level Storage System | Kilburn et al. | 1962 | The original virtual-memory-with-paging design the whole approach is modeled on |
| Megatron-LM | Shoeybi et al. | 2019 | Tensor model parallelism scheme vLLM adopts for distributed execution |
| FlashAttention | Dao et al. | 2022 | IO-aware exact attention kernels; the complementary on-chip half of attention efficiency |

#### Contemporaries / Competitors

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| FlexGen: High-Throughput Generative Inference with a Single GPU | Sheng et al. | 2023 | Offloading-based *offline* batch inference; same memory-wall problem, opposite latency regime |
| Text Generation Inference (TGI) | HuggingFace | 2023 | Production serving engine developed concurrently; later adopted paged KV cache |
| DeepSpeed Inference / ZeRO-Inference | Aminabadi et al. | 2022–2023 | Kernel + offloading optimizations for transformer inference at scale |
| Efficiently Scaling Transformer Inference | Pope et al. | 2023 | Analytical treatment of serving trade-offs (partitioning, multi-query attention) at Google scale |
| AlpaServe | Li et al. | 2023 | Model-parallel statistical multiplexing for serving; general model serving rather than KV-centric |

#### Successors / Extensions

| Paper | Authors | Year | Relation |
|-------|---------|------|----------|
| SGLang / RadixAttention | Zheng et al. | 2024 | Radix tree over KV blocks for automatic cross-request prefix sharing; generalizes vLLM's manual prefix caching |
| vAttention: Dynamic Memory Management without PagedAttention | Prabhu et al. | 2024 | Challenges the software-paging design — CUDA virtual memory gives on-demand allocation with contiguous virtual KV tensors and unmodified kernels |
| Sarathi-Serve | Agrawal et al. | 2024 | Chunked prefill + stall-free batching; fixes prefill/decode interference that vLLM's FCFS scheduling leaves open |
| DistServe | Zhong et al. | 2024 | Disaggregates prefill and decode onto separate GPU pools with KV transfer; goodput-oriented serving |
| Mooncake | Qin et al. | 2024 | KV-cache-centric disaggregated architecture (production system at Moonshot); extends paging to a multi-tier, cluster-wide KV pool |

---

### Bottom Line

A foundational classic — arguably *the* systems paper of the LLM serving era. It identified the right bottleneck (KV cache memory waste, not compute), imported the right abstraction from a 60-year-old literature, quantified everything honestly, and shipped code that became industry-standard infrastructure. Reading it remains the fastest way to understand how modern inference engines actually manage memory, and its vocabulary (KV blocks, block tables, copy-on-write sharing, swap-vs-recompute) is now the field's lingua franca. The frontier has moved — to hardware-assisted paging, prefix trees, chunked prefill, and disaggregated clusters — but every one of those works defines itself relative to this paper. Read it first; read vAttention and Sarathi-Serve for the counterpoints.

[^1]: **KV cache** — Key-Value cache. See the [glossary](../../common/terms/).
