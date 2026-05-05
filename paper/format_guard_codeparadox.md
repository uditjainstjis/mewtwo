\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\title{The Code Paradox and Format Guard:\\Token-Level LoRA Adapter Routing as a Substitute for Failed Static Composition on a 30B Base}
\author{Anonymous}
\date{May 2026}

\begin{document}
\maketitle

\begin{abstract}
Multi-adapter composition is conventionally studied at the weight level (e.g., DARE, TIES, AdapterFusion, LoRAHub) or via prompt-level mixture-of-experts routing. We report two empirical findings on a Nemotron-Nano-30B-A3B base model in 4-bit NF4 quantisation trained with four task-specialised LoRA adapters (math, code, science, BFSI-extract) at rank 16. First, the \emph{Code Paradox}: training on Python code degrades HumanEval pass@1 under our original v1 scorer by 23 percentage points and continues to underperform the unfine-tuned base under a corrected v2 scorer, while simultaneously improving ARC-Challenge by 11 percentage points and MATH-500 by 14.5 percentage points. Training on math, conversely, lifts HumanEval. The cross-domain transfer is asymmetric and counter-intuitive: the worst adapter at code generation is the best adapter at non-code reasoning. Second, static weight-merging of these four adapters via DARE, TIES, and uniform linear interpolation never exceeds the best single expert on a four-benchmark grid and yields a +0.0 percentage point gain over best-single on a 45-question mixed-domain probe.

We propose \emph{Format Guard}, a HuggingFace \texttt{LogitsProcessor} that holds all four adapters resident in VRAM and swaps the active adapter every ten generated tokens via a regex-driven, paradox-aware router. Warm swaps are an $\mathcal{O}(1)$ pointer flip, and a four-adapter process fits in approximately 18 GB of VRAM on a single 32 GB consumer GPU. On the full HumanEval test set (n=164) under our corrected v2 scorer, Format Guard improves pass@1 from 56.1\% to 73.2\% (+17.1 percentage points; McNemar $\chi^2 = 15.68$, $p < 10^{-3}$). The empirical zero-overhead property---Format Guard matching a dedicated adapter when the right adapter is in the pool---is replicated on a 664-question held-out BFSI evaluation and a 60-question hand-curated benchmark. We disclose a HumanEval extraction bug in our v1 pipeline that inflated the original Format Guard headline by approximately 31 percentage points, and we report a $-3$ percentage point Format Guard regression on MBPP that exposes a real failure mode of mid-sequence routing on format-rigid tasks.
\end{abstract}

\section{Introduction}

The dominant paradigm for adapting a single foundation model to multiple downstream domains is to train one parameter-efficient adapter per domain and to compose them in deployment. Low-rank adaptation (LoRA) injects rank-$r$ updates into the attention and MLP weight matrices of a frozen backbone, producing a small (typically $\sim1$--2\% of base parameters) trainable artifact per domain~\cite{hu2021lora}. Compared to full fine-tuning, LoRA reduces trainable parameters by orders of magnitude while maintaining competitive downstream accuracy.

The literature offers two broad families of composition. \emph{Static weight-merging} methods combine adapter or model weights ahead of time into a single artifact. Examples include DARE and TIES-MERGE, which sparsify and combine parameter deltas~\cite{edalati2023absorbing,quarantiello2024adaptive}; model soups, which average full model weights~\cite{wortsman2022model}; and AdapterFusion, which learns a fusion layer over multiple adapters~\cite{pfeiffer2020adapterfusion}. \emph{Routing} methods instead select among adapters at inference time, sometimes per prompt and sometimes per token; these include AdapterFusion's fusion networks~\cite{pfeiffer2020adapterfusion} and sparse mixture-of-experts architectures that route tokens via a learned gate~\cite{shazeer2017moe,fedus2021switch}. Static merging is attractive because it produces a single deployable artifact with a single forward-pass cost; routing is attractive because it preserves expert specialisation at the cost of additional bookkeeping.

Both paradigms are typically evaluated under benign assumptions: that the per-adapter contribution to a downstream task is approximately monotonic in adapter quality, and that combining experts can only help. We report empirical evidence from a 30B-class base model that contradicts both assumptions in specific, reproducible ways, and we propose a working mechanism that recovers most of the lost composition gain on one benchmark family.

Our contributions are three-fold.
\begin{itemize}
  \item First, we document the \emph{Code Paradox} on a 4-bit Nemotron-Nano-30B-A3B base with four LoRA adapters. Training on Python code data degrades the model's in-domain HumanEval pass@1 while simultaneously lifting its out-of-domain ARC-Challenge and MATH-500 scores; training on math data, conversely, lifts HumanEval. The transfer is sufficiently asymmetric that the \emph{worst} adapter at code generation is the \emph{best} adapter at non-code reasoning.
  \item Second, we show that uniform-weight DARE and TIES merging of these four adapters does not exceed the best single expert on any of four target benchmarks and yields a flat +0.0 percentage point gain on a 45-question mixed-domain probe. Under this configuration and objective, the static-composition geometry is unfavourable.
  \item Third, we propose \emph{Format Guard}, a LogitsProcessor that swaps the active LoRA adapter every ten generated tokens via a regex-driven heuristic over the recent decoded suffix. Format Guard holds all adapters resident in VRAM so that swaps are pointer flips, and exploits the Code Paradox by routing code-generation contexts to the math adapter and math contexts to the code adapter. Format Guard reaches +17.1 percentage points over base on full HumanEval under a corrected scorer (paired McNemar $p<10^{-3}$ at $n=164$) and matches a dedicated adapter on two BFSI-domain evaluations at empirically zero cost.
\end{itemize}

We deliberately scope these claims tightly. Format Guard has been evaluated only on a single 30B base; the asymmetric positive arm of the Code Paradox is established only on that base (the in-domain code-on-code regression replicates at $n=200$ on Qwen-3.5-0.8B but the asymmetric positive transfer is not yet replicated cross-family); and our DARE/TIES experiments use only uniform weights, leaving learned-weight composition (e.g., LoRAHub-style~\cite{quarantiello2024adaptive}) untested at 30B. Section~\ref{sec:limitations} discloses these and other limitations explicitly. We also disclose an extraction bug in our v1 HumanEval scoring pipeline that inflated the original Format Guard headline by approximately 31 percentage points; the corrected delta of +17.1 percentage points is what we stand behind.

\section{Background and Related Work}

\subsection{LoRA and parameter-efficient fine-tuning}

Low-rank adapter (LoRA) tuning injects rank-$r$ updates into the attention and MLP weight matrices of a frozen backbone: given a weight matrix $W$, LoRA replaces $W x$ with $(W + BA)x$, where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ are low-rank trainable matrices and $d$ is the hidden dimension~\cite{hu2021lora}. This yields a small trainable artifact per domain, typically 1--2\% of the base parameters, enabling multi-task fine-tuning without duplicating the full model. We adopt LoRA with rank 16 and standard hyperparameters for all adapters.

\subsection{Static composition of fine-tuned models}

Several works study weight-level composition of fine-tuned models or adapters. Model soups average the weights of multiple fine-tuned models and can improve accuracy without additional inference cost when composing models fine-tuned on similar tasks~\cite{wortsman2022model}. DARE and TIES-MERGE absorb abilities from "homologous" models by sparsifying and merging delta weights; DARE randomly drops a fraction of deltas and rescales, while TIES-MERGE resolves sign conflicts and retains top-magnitude entries~\cite{edalati2023absorbing}. AdapterFusion learns a non-destructive composition of task adapters via a learned fusion layer over adapter outputs~\cite{pfeiffer2020adapterfusion}. LoRAHub-style approaches learn mixing weights for multiple adapters rather than fixing them uniformly~\cite{quarantiello2024adaptive}.

Our static baselines consider uniform-weight linear averaging, DARE and TIES over four LoRA adapters at 30B. We do not explore learned-weight composition at this scale; our conclusions are restricted to uniform-weight merges on this base and adapter set.

\subsection{Routing and mixture-of-experts}

Routing-based architectures select among experts at inference time rather than merging them statically. Sparse mixture-of-experts layers route tokens to a small subset of experts via a learned gate, enabling very large models with manageable compute cost~\cite{shazeer2017moe,fedus2021switch}. AdapterFusion uses a learned fusion module over adapter outputs, effectively implementing a layer-level routing mechanism~\cite{pfeiffer2020adapterfusion}. Prompt-level routers select a single adapter per query using a classifier or retrieval rule.

Our approach sits in the routing branch but differs in two ways. First, we operate at the token level rather than at the layer or prompt level. Second, we use a hand-designed, paradox-aware heuristic router implemented as a HuggingFace \texttt{LogitsProcessor}, rather than a learned gate, to separate the effects of the routing \emph{mechanism} from the effects of router training.

\subsection{Benchmarks}

We evaluate on four public reasoning benchmarks and two BFSI-domain benchmarks. HumanEval evaluates functional correctness on 164 hand-written Python problems via unit tests~\cite{chen2021evaluating}. MATH is a competition-math dataset with over 12,000 problems; we use a curated MATH-500 subset following prior work~\cite{hendrycks2021math}. ARC-Challenge evaluates science-question answering beyond simple retrieval~\cite{clark2018arc}. MBPP is a 500-problem Python synthesis benchmark, of which we use the standard 100-problem subset for pass@1~\cite{austin2021mbpp}. Our BFSI datasets are constructed from Indian regulatory text (Reserve Bank of India master directions and Securities and Exchange Board of India master circulars) using a deterministic QA pipeline described in companion work; they serve here primarily to test Format Guard's claimed zero-overhead property.

\section{Methods and System Design}

\subsection{Base model, adapters, and training recipe}

We use Nemotron-Nano-30B-A3B in 4-bit NF4 quantisation as the base for all primary experiments. Four LoRA adapters are trained, one per domain: \texttt{math} (numeric reasoning and worked solutions), \texttt{code} (Python code generation and infilling), \texttt{science} (factual scientific QA), and \texttt{bfsi\_extract} (extractive question answering over Indian regulatory text). All adapters use LoRA rank 16 with scaling parameter $\alpha = 32$, targeting all attention and MLP modules.

The bfsi\_extract training run uses paged AdamW 8-bit at learning rate $2 \cdot 10^{-4}$ with cosine schedule, one epoch, maximum sequence length 1024, and effective batch size 4. Training completes in 3h28m on a single RTX 5090 with 17.81 GB peak VRAM and produces a 1.74 GB bf16 adapter artifact; trainable parameters are 434.6M out of 32B (1.36\%). The math, code, and science adapters use comparable budgets. We treat this configuration as representative for our 30B experiments.

\subsection{Static composition baselines}

We construct three static-merge baselines from the four adapters:
\begin{itemize}
  \item \textbf{Uniform linear}: per-parameter mean of the four adapter deltas.
  \item \textbf{DARE}: random delta dropping with rescale, uniform weights~\cite{edalati2023absorbing,quarantiello2024adaptive}.
  \item \textbf{TIES}: sign-conflict resolution and top-magnitude retention, uniform weights~\cite{edalati2023absorbing}.
\end{itemize}

All three baselines are evaluated against the per-benchmark best single adapter and against the unmerged base. Section~5.2 reports the results.

\subsection{Format Guard: a token-level LoRA routing LogitsProcessor}

The core mechanism we propose is \emph{Format Guard}, a HuggingFace \texttt{LogitsProcessor} instantiated at generation time. Format Guard relies on three implementation properties.

\paragraph{VRAM residency.} All four adapters are loaded simultaneously via PEFT's multi-adapter API. Switching the active adapter during generation is therefore implemented as an $\mathcal{O}(1)$ pointer flip on the model's adapter table; no weights are touched and no additional parameters are allocated per swap. We measured the cold-load path (NVMe SSD \textrightarrow{} GPU) at 315.9 ms per swap over 44 swaps under realistic generation traffic, which is prohibitive for token-level routing. The warm-VRAM path is in the microsecond regime.

\paragraph{Pointer-swap routing primitive.} Each candidate target adapter is selected via a call analogous to \texttt{model.set\_adapter(target)}. This is the same mechanism used by PEFT for multi-adapter inference and incurs no kernel re-compilation or weight movement.

\paragraph{Regex- and paradox-aware router.} Every $K = 10$ generated tokens, Format Guard inspects the most recent decoded suffix and chooses the next active adapter. The router is intentionally simple and paradox-aware: it consults the empirical Phase 1 cross-domain table (Section~5.1) rather than a topical one-to-one mapping. Concretely:
\begin{itemize}
  \item Code-like text in the suffix (Python keywords, \texttt{def}-headers, indented blocks) \textrightarrow{} switch to the math adapter, because math-on-code improves HumanEval while code-on-code regresses.
  \item Math notation in the suffix (LaTeX-style equation patterns) \textrightarrow{} switch to the code adapter, because code-on-math yields the strongest MATH-500 gains.
  \item BFSI keywords (regulator names, \texttt{Section X(Y)}, master direction phrasing) \textrightarrow{} switch to bfsi\_extract.
  \item Default fallback \textrightarrow{} code adapter, used as a generic reasoner.
\end{itemize}

The router is a deterministic regex cascade. We deliberately do not learn the routing rule so that any composition gain can be attributed to the \emph{mechanism} of token-level swapping rather than to a learned classifier. A learned router is a natural follow-up (Section~7).

\subsection{Memory budget and hardware}

A four-adapter Format Guard process peaks at approximately 18 GB of VRAM (base $\approx$ 17 GB plus a small delta from the additional adapters, since the underlying frozen base weights are shared). A five-adapter variant that also includes a BFSI-recall adapter peaks at approximately 19.7 GB. Both configurations fit on a single 32 GB consumer GPU.

\subsection{Evaluation protocol}

We evaluate four public reasoning benchmarks plus two BFSI replication benchmarks:
\begin{itemize}
  \item \textbf{HumanEval} (n=164), pass@1 with greedy decoding~\cite{chen2021evaluating}.
  \item \textbf{MATH-500} (n=200), exact match on a curated subset of the MATH benchmark~\cite{hendrycks2021math}.
  \item \textbf{ARC-Challenge} (n=100), accuracy on the challenge subset of the AI2 Reasoning Challenge~\cite{clark2018arc}.
  \item \textbf{MBPP} (n=100), pass@1 on the widely used subset of the MBPP benchmark~\cite{austin2021mbpp}.
  \item \textbf{BFSI-extract held-out} (n=664), case-insensitive substring match plus token F1 on a document-disjoint split (26 held-out PDFs) of Indian regulatory QA pairs.
  \item \textbf{Synapta Indian BFSI Benchmark v1} (n=60), a hand-curated benchmark with mixed scoring (30 substring items, 30 token-F1 $\geq 0.5$ items).
\end{itemize}

All comparisons are paired: we reuse the same prompts and random seeds across modes. Statistical tests are paired McNemar exact-binomial on binary outcomes and Wilcoxon signed-rank on continuous F1 where applicable.

\subsection{The HumanEval scoring bug}

In an earlier iteration of our pipeline (\emph{v1 buggy scoring}), we used a HumanEval scorer that exhibited two extraction bugs. First, \emph{import-stripping}: the \texttt{extract\_code} function returned only the contents of the model's fenced code block and dropped the prompt's pre-definition header (e.g., \texttt{from typing import List}), producing \texttt{NameError} at runtime when tests relied on those imports. Second, \emph{indent-stripping}: when the model returned a body-only fenced block (e.g., an indented \texttt{return} statement), a call to \texttt{body.strip()} removed the leading 4-space indent so the body parsed as a top-level statement when prepended to \texttt{def f():}, producing \texttt{IndentationError}. We patched both bugs and re-ran the same saved completions through the corrected extractor to produce a v2 score for both base and Format Guard. The corrected harness is what we use throughout this paper. Section~5.3 reports the magnitude of the correction and its implications.

\section{Experiments}

The experimental layout is as follows. Section~5.1 reports Phase 1 single-adapter benchmarks on Nemotron-30B, which establish the Code Paradox. Section~5.2 reports static-composition failure across DARE, TIES, and uniform linear merging. Section~5.3 reports the headline Format Guard HumanEval result with the v2 scorer and details the v1 $\rightarrow$ v2 correction. Section~5.4 reports the full benchmark grid for Format Guard, including the MBPP regression. Section~5.5 reports cold-swap and warm-swap latency. Section~5.6 reports the Code Paradox replication on Qwen-3.5-0.8B and a rank-scaling spot-check. Section~5.7 reports the BFSI replication of the Format Guard zero-overhead property at n=664 and n=60.

\section{Results}

% (For brevity, we include only a subset of the detailed tables here. The full paper draft you provided already contains the necessary tables and narrative; you can paste Sections 5.1--5.7 from your draft into this LaTeX skeleton.)

% You should insert your detailed Tables 1--6 and the text for Sections 5.1--5.7 here, unchanged except for minor typography.

We refer the reader to our detailed per-benchmark tables and per-category breakdowns (HumanEval categories, BFSI per-tier and per-regulator breakdowns) as provided in the accompanying experiment descriptions. The high-level findings are:
\begin{itemize}
  \item The Code Paradox is clearly visible in Phase 1 single-adapter baselines: the code adapter underperforms the base on HumanEval while outperforming it on ARC and MATH-500, and the math adapter improves HumanEval.
  \item Uniform-weight DARE/TIES and linear merges never exceed the best single expert and can degrade performance substantially on ARC, HumanEval, and MBPP on this 30B base.
  \item Format Guard improves HumanEval pass@1 by +17.1 percentage points over base under a corrected scorer (paired McNemar $p<10^{-3}$), matches the best single adapter on ARC and MATH-500, and regresses MBPP by 3 percentage points.
  \item On two BFSI benchmarks (n=664 and n=60, paired), Format Guard with bfsi\_extract in the pool matches the dedicated adapter's performance up to small statistical noise, validating an empirical zero-overhead property in this domain.
\end{itemize}

\section{Limitations and Negative Findings}
\label{sec:limitations}

We summarise key limitations and negative findings; these scope our claims and are intended to prevent over-generalisation.

\begin{itemize}
  \item \textbf{Single base for Format Guard headline.} Format Guard's HumanEval result is established only on Nemotron-Nano-30B-A3B in 4-bit NF4 quantisation. We have not yet trained math/code/science adapters on a second 30B-class base (e.g., Qwen3-32B, Llama-3-70B, Mixtral-8x7B) and re-run the n=164 evaluation.
  \item \textbf{Asymmetric positive arm of Code Paradox is single-base.} The in-domain code-on-code regression replicates at $n=200$ on Qwen-3.5-0.8B. The asymmetric positive transfer (code helping math and vice versa) is established only on Nemotron-30B and is not yet replicated across base families.
  \item \textbf{MBPP regression and format-rigid tasks.} Format Guard regresses MBPP by 3 percentage points because mid-sequence adapter swaps perturb Python indentation conventions. This is a real failure mode of the current heuristic router.
  \item \textbf{Heuristic regex router.} Our router is deliberately simple and hand-designed. A learned router conditioned on the last-$K$ decoded tokens may perform better but introduces an additional trainable component whose generalisation is itself non-trivial. We do not claim our regex router is optimal.
  \item \textbf{Uniform-weight static merging only.} Our static baselines use uniform DARE, TIES, and linear merges. We do not test learned-weight composition (e.g., LoRAHub) at 30B.
  \item \textbf{HumanEval scoring correction.} All v1 HumanEval numbers in earlier internal documents are stale. We use only v2-corrected scores in this paper and explicitly report the shrinkage of the headline delta.
  \item \textbf{Small-n benchmarks and strict cutoffs.} Our Synapta Indian BFSI Benchmark v1 has $n=60$; McNemar $p=0.0313$ for adapter vs base is marginal at $\alpha=0.05$. Token-F1 $\geq 0.5$ cutoffs are arguably too strict for verbose paragraph extraction.
\end{itemize}

\section{Discussion and Future Work}

Our results suggest that, on the 30B base and adapter set we study, static composition via uniform-weight DARE/TIES does not yield emergent gains over the best single expert, whereas token-level routing with Format Guard recovers a substantial portion of the lost performance on HumanEval and preserves accuracy on BFSI when a dedicated adapter exists. We emphasise that these are \emph{conditional} statements: they hold for this base, this adapter set, and this evaluation protocol, and should not be over-generalised.

Two directions stand out for future work. First, replacing the heuristic router with a small learned classifier over recent decoded tokens may close the MBPP regression and improve robustness on format-rigid tasks, at the cost of an additional trainable component. Second, training comparable adapter sets on multiple 30B-class bases and re-running the HumanEval evaluation is essential to test whether the Code Paradox and Format Guard gains generalise beyond Nemotron-30B.

\section{Ethics, Risks, and Reproducibility}

This work focuses on model combination and evaluation methodology; we do not deploy any system in user-facing or safety-critical settings. Our experiments use publicly released base models (Nemotron-Nano-30B-A3B, Qwen-3.5-0.8B) and standard benchmarks (HumanEval, MATH-500, ARC-Challenge, MBPP) without involving personal data or private user content.

We take two steps to reduce the risk of misleading claims. First, all headline numbers in this paper trace to a single internal artifact table with explicit provenance and confidence intervals; any number not in that table is treated as non-citable. Second, we explicitly disclose evaluation bugs, rolled-back claims, and missing experiments. In particular, we retract earlier v1 HumanEval results and adopt only v2-corrected scores, and we avoid making cross-family claims about the Code Paradox or Format Guard until $n \geq 200$ replications exist on additional base models.

To aid reproducibility, we intend to release the LoRA adapter weights, evaluation scripts, and corrected HumanEval scoring harness needed to reproduce all Category 1 experiments, subject to venue policy and third-party licenses. Where artifacts depend on vendor base-model licenses, we rely on the original distribution terms and do not redistribute proprietary weights.

\begin{thebibliography}{99}

\bibitem{austin2021mbpp}
J.~Austin, A.~Odena, M.~Nye, M.~Bosma, H.~Michalewski, D.~Dohan, et~al.
\newblock Program synthesis with large language models.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2021.

\bibitem{chen2021evaluating}
M.~Chen, J.~Tworek, H.~Jun, Q.~Yuan, H.~P. de~Oliveira~Pinto, J.~Kaplan, et~al.
\newblock Evaluating large language models trained on code.
\newblock \emph{arXiv preprint arXiv:2107.03374}, 2021.

\bibitem{clark2018arc}
P.~Clark, I.~Cowhey, O.~Etzioni, T.~Khot, A.~Sabharwal, O.~Tafjord, et~al.
\newblock Think you have solved question answering? {T}ry {ARC}, the {AI2} reasoning challenge.
\newblock In \emph{Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)}, 2018.

\bibitem{edalati2023absorbing}
A.~Edalati, C.~Zhang, S.~I. Mirzadeh, M.~Farajtabar, and A.~Ghasemi.
\newblock Absorbing abilities from homologous models as a free lunch.
\newblock \emph{arXiv preprint arXiv:2311.03099}, 2023.

\bibitem{fedus2021switch}
W.~Fedus, B.~Zoph, and N.~Shazeer.
\newblock Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.
\newblock \emph{arXiv preprint arXiv:2101.03961}, 2021.

\bibitem{hendrycks2021math}
D.~Hendrycks, S.~Basart, S.~Kadavath, M.~Mazeika, A.~Arora, E.~Guo, et~al.
\newblock Measuring mathematical problem solving with the {MATH} dataset.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2021.

\bibitem{hu2021lora}
E.~J. Hu, Y.~Shen, P.~Wallis, Z.~Allen-Zhu, Y.~Li, L.~Wang, and W.~Chen.
\newblock {LoRA}: Low-rank adaptation of large language models.
\newblock \emph{arXiv preprint arXiv:2106.09685}, 2021.

\bibitem{pfeiffer2020adapterfusion}
J.~Pfeiffer, A.~Kamath, A.~R{"u}ckl{"e}, K.~Cho, and I.~Gurevych.
\newblock Adapterfusion: Non-destructive task composition for transfer learning.
\newblock \emph{arXiv preprint arXiv:2005.00247}, 2020.

\bibitem{quarantiello2024adaptive}
L.~Quarantiello, et~al.
\newblock Adaptive {LoRA} merging for efficient domain incremental learning.
\newblock In \emph{Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)}, 2024.

\bibitem{shazeer2017moe}
N.~Shazeer, A.~Mirhoseini, K.~Maziarz, A.~Davis, Q.~Le, G.~Hinton, and J.~Dean.
\newblock Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
\newblock \emph{arXiv preprint arXiv:1701.06538}, 2017.

\bibitem{wortsman2022model}
M.~Wortsman, G.~Ilharco, S.~Y. Gadre, R.~Roelofs, R.~Gontijo-Lopes, A.~S. Morcos, et~al.
\newblock Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
\newblock In \emph{Proceedings of the 39th International Conference on Machine Learning (ICML)}, 2022.

\end{thebibliography}

\end{document}