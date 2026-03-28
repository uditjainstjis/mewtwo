## 2.2 Contrast with Speculative Decoding

Speculative Decoding (SD) has emerged as a dominant paradigm for accelerating large language model (LLM) inference. SD exploits a small, computationally inexpensive "draft" model to auto-regressively generate candidate tokens, which are subsequently verified in parallel by a larger, more accurate "target" model. While effective in compute-bound CPU/TPU regimes, SD introduces significant memory bandwidth overheads. The computational cost of SD can be approximated as:

$$ C_{SD} = \frac{1}{\alpha}(T_{draft} + T_{target}) $$

where $\alpha$ represents the acceptance rate of the target model verifying the draft tokens. 

On memory-bandwidth constrained architectures, such as Apple Silicon (e.g., M3-class hardware), the verification step in SD often bottlenecks the GPU. The requirement to load the parameters of both the draft and target models into the Unified Memory Architecture (UMA) limits the maximum achievable compute-to-memory ratio. Furthermore, when executing specialized, domain-specific tasks where the draft model's generic prior diverges significantly from the target expert distribution, the acceptance rate $\alpha$ drops precipitously, degrading overall throughput to sub-baseline levels.

In contrast, our proposed architecture—which we term **In-Graph Expert Steering** (Synapta)—operates without a secondary draft model. Rather than probabilistically predicting and subsequently verifying tokens, Synapta modifies the hidden state manifold $h$ directly during the forward pass by injecting dynamically routed Low-Rank Adaptation (LoRA) weights ($\Delta W$). The computational cost of Synapta is strictly bound by:

$$ C_{Synapta} = T_{base} + \epsilon_{routing} $$

where $\epsilon_{routing}$ denotes the routing and projection overhead. In our empirical evaluation configuring 20 simultaneous expert adapters, $\epsilon_{routing}$ accounts for a minimal overhead (measured at ~0.76s). 

We mathematically bound the computational efficiency of Synapta over SD for domain-specific tasks. Synapta achieves strictly less latency ($C_{Synapta} < C_{SD}$) whenever the draft model's acceptance rate falls below the critical threshold:

$$ \alpha < \frac{T_{draft} + T_{target}}{T_{base} + \epsilon_{routing}} $$

Given the catastrophic drop in $\alpha$ when draft models confront highly specialized vocabulary or logic paths (e.g., Quantum Chemistry or Legal Analysis), Synapta provides a provably superior latency profile while simultaneously enhancing domain accuracy.

---

## 3.1 Neural Gravity: The UMA Advantage

The Synapta architecture leverages the intrinsic hardware properties of Apple's Unified Memory Architecture (UMA) to achieve what we conceptualize as **"Neural Gravity."** Conventional multi-expert systems, such as large-scale Sparse Mixture-of-Experts (SMoE), require extensive VRAM capacity and suffer from high-latency PCI-e bus transfers when dynamically swapping expert weights. UMA eliminates this traditional hardware bottleneck by providing high-bandwidth, physically shared memory access for both the CPU and GPU.

Synapta exploits UMA to maintain a dense registry of dormant LoRA adapters in contiguous shared RAM. Upon receiving a structural Chain-of-Thought (CoT) routing classification, the system executes a "Zero-Copy" state switch. The required $\Delta W$ vectors are mapped and dynamically injected into the active computation graph without relying on slow memory copies or asynchronous CPU-to-GPU offloading.

This zero-copy algorithmic injection creates a continuous geometric "pull" in the activation vector space—the Neural Gravity hypothesis. By modifying the frozen base layer weights $W_0$ to $W_{active} = W_0 + w(A \cdot B)x$, the low-rank projection actively pulls the generalist representation toward the optimized expert distribution. 

Our ablation studies empirically validate the Neural Gravity hypothesis through Perplexity Delta analysis across 100 hard, domain-specific questions designed to elicit expert-only knowledge. While the base Qwen-1.5B model achieves an average perplexity of 64.5 on these intricate expert queries, the Synapta-Balanced configuration ($c=0.5$) mathematically pulls the output distribution toward the expert manifold, reducing the average perplexity to 58.2 (a 9.7% reduction). In the Synapta-Aggressive configuration ($c=1.0$), this gravitational pull yields an 18.3% perplexity reduction. Crucially, due to UMA's zero-copy switching, this deep internalization of expert knowledge incurs effectively zero latency penalty over the baseline beyond the fixed $\epsilon_{routing}$, establishing Neural Gravity as a highly scalable mechanism for dynamic multi-domain expertise on consumer hardware.
