\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[numbers,sort&compress]{natbib}

\title{Sovereign Regulatory NLP for Indian BFSI:\\A Deterministic Pipeline, Per-Customer LoRA Adapters, and Honest Out-of-Distribution Disclosure}
\author{Anonymous}
\date{May 2026}

\begin{document}
\maketitle

\begin{abstract}
Sovereign deployment of regulatory AI in Indian banking, financial services, and insurance (BFSI) faces two structural constraints: data localisation laws restrict cloud APIs, and per-customer compliance corpora differ in style and structure from any single general LLM's training distribution. We report a complete, reproducible pipeline that scrapes 130 public Reserve Bank of India (RBI) Master Directions and Securities and Exchange Board of India (SEBI) Master Circulars, deterministically constructs 4{,}477 question--answer pairs through a 3-tier regex template grammar, gates every candidate through a 10-check validator (98.45\% pass), and trains a single 4-bit LoRA adapter on a consumer GPU in 3h~28min. To eliminate paraphrase contamination and chunk-neighbour memorisation, our train/eval split is \emph{document-disjoint}: 26 entire PDFs (20\%) are held out before any pair is constructed. On 664 paired held-out questions, the adapter lifts substring match from 58.7\% to 89.6\% (+30.9~pp; McNemar $b_{10}{=}14$, $b_{01}{=}219$, $p{=}1.66\times10^{-48}$). The same recipe transfers to a structurally different no-context recall task ($n{=}214$, $+38.4\%$ relative F1, Wilcoxon $p{=}1.50\times10^{-16}$). On the contemporary public benchmark IndiaFinBench, the same adapter scores only 32.1\% (Wilson 95\% CI [27.3, 37.4]) versus 89.7\% reported for Gemini~2.5 Flash, a 57.6~pp gap~\citep{pall2026indiafinbench}. We frame this gap as the central evidence \emph{for} per-customer training rather than against the methodology. We release a 60-question hand-curated paired benchmark under CC-BY-SA-4.0 on which the adapter scores 50\% versus 40\% (McNemar $p{=}0.0313$). Code and data manifests are open-sourced.
\end{abstract}

\section{Introduction}

The Indian BFSI sector is governed by a thicket of regulators---RBI, SEBI, IRDAI, PFRDA, and the FEMA framework---that issue thousands of Master Directions, Master Circulars, and amendments each year. Compliance teams at banks, NBFCs, asset managers, and insurance brokers must locate, interpret, and cite specific paragraphs from these documents under audit pressure. Two structural constraints shape any deployable AI assistant in this market.

First, India's Digital Personal Data Protection Act and sectoral RBI circulars on cloud usage place strong pressure toward on-premise or sovereign-cloud deployment. Many regulated entities are unwilling to send customer or transaction data to general-purpose frontier APIs, and many are operationally restricted to on-device or regional inference. This rules out the typical recipe of ``few-shot a frontier model behind a thin retrieval layer'' for a sizeable fraction of the addressable market.

Second, every customer's compliance corpus differs in style. RBI Master Directions are heading-numbered, paragraph-extractive, and verbose; SEBI Master Circulars are tabular and clause-numbered; an individual NBFC's internal compliance handbook adds yet another stylistic layer on top. A single fine-tuned model that wins on one customer's questions can fail on another's---not because the model has lost knowledge, but because the answer surface form has shifted.

We argue these constraints together motivate \emph{per-customer adapters}: small (1--2\% trainable parameters) LoRA adapters trained on each customer's regulatory corpus, served from a single base model on commodity hardware, with rigorous paired evaluation against a base baseline before deployment. We make four concrete contributions.

\begin{enumerate}
  \item \textbf{A deterministic, contamination-resistant pipeline} that turns 130 public regulator PDFs into 4{,}477 raw QA pairs without LLM-generated questions (Section~\ref{sec:methods}).
  \item \textbf{Document-disjoint paired evaluation at $n{=}664$} demonstrating $+30.9$~pp substring match with extreme statistical significance ($p{=}1.66\times10^{-48}$), and a generalisation result on a structurally different recall task at $n{=}214$ (Section~\ref{sec:results}).
  \item \textbf{Honest out-of-distribution disclosure}: on IndiaFinBench, an externally authored, contemporary public benchmark, the same adapter scores 32.1\% vs Gemini Flash's reported 89.7\%~\citep{pall2026indiafinbench}. We argue this gap supports rather than undermines the per-customer thesis (Sections~\ref{sec:results}, \ref{sec:discussion}).
  \item \textbf{An open 60-question hand-curated benchmark} (CC-BY-SA-4.0) with paired baselines, alternative-answer lists, and a gated reference scorer.
\end{enumerate}

We position the work as a use-inspired evaluation paper: methodology and dataset/pipeline construction are central; the OOD result is treated as evidence about distribution shift, not as an apples-to-apples comparison.

\section{Background and Related Work}

\paragraph{Domain adaptation via parameter-efficient fine-tuning.} Low-rank adaptation (LoRA) injects rank-$r$ updates into attention and MLP weight matrices of a frozen backbone, yielding a small trainable module per domain that is typically 1--2\% of the base parameter count~\citep{hu2021lora}. QLoRA extends this idea to 4-bit quantised backbones, enabling fine-tuning of 30B--65B models on a single GPU by combining NF4 quantisation, double quantisation, and paged optimisers~\citep{dettmers2023qlora}. We use these as off-the-shelf primitives; our contribution is not a new fine-tuning method but a faithful and contamination-resistant evaluation of one well-tuned LoRA adapter on a regulator-specific QA task.

\paragraph{Data contamination in QA evaluation.} Several works document contamination of downstream benchmarks in pretraining corpora and the resulting memorisation and exploitation effects~\citep{magar2022data}. Beyond pretraining-level contamination, \citet{magar2022data} and others highlight that even within a fixed corpus, random-row train/test splits can enable ``chunk-neighbour'' memorisation: an answer span memorised in one paragraph supports correct generation in a paraphrased neighbour. Standard random-row splits on regulatory QA inherit this pathology. Our document-disjoint split (Section~\ref{sec:split}) is a direct response.

\paragraph{Out-of-distribution evaluation of fine-tuned models.} Holistic evaluation work such as HELM~\citep{liang2022helm} repeatedly finds that models tuned on one benchmark style transfer poorly to differently-styled questions on the same domain. Our IndiaFinBench result is consistent with this literature; we make the OOD gap the centre of the discussion rather than an appendix.

\paragraph{Regulatory NLP and Indian BFSI benchmarks.} IndiaFinBench is, to our knowledge, the first publicly available benchmark for evaluating LLMs on Indian financial regulatory text~\citep{pall2026indiafinbench}. It comprises 406 expert-annotated QA pairs from 192 SEBI and RBI documents spanning 1992--2026, with four task types: regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning. The IndiaFinBench authors evaluate twelve models under zero-shot conditions and find that frontier models such as Gemini 2.5 Flash reach 89.7\% overall accuracy, while smaller models lag behind. We use the $n{=}324$ IndiaFinBench test split as our OOD probe. To our knowledge no other open Indian-BFSI benchmark with paired baselines existed at the time of our adapter training.

\paragraph{Adapter-routing logits processors.} Format Guard is a token-level adapter-routing logits processor we developed in companion work: it swaps the active LoRA adapter every $K$ tokens via regex over the decoded suffix and an $\mathcal{O}(1)$ pointer flip on the base model's adapter table, enabling token-level routing among multiple LoRA experts on a single 30B base~\citep{ours2026fg}. We reuse Format Guard here only as a routing primitive; the BFSI cluster's substantive contribution is the corpus-and-adapter pipeline, not Format Guard's mechanism, which is described in detail elsewhere.

\section{Methods and System Design}
\label{sec:methods}

\subsection{Corpus}

We scrape 80 RBI Master Directions and 50 SEBI Master Circulars (130 PDFs, 115~MB), all in the public domain. Text is extracted with \texttt{pdfplumber} and \texttt{PyMuPDF} under multiprocessing, yielding 8.06~M characters. We detect 7{,}329 numbered section boundaries and emit 4{,}185 chunks with median 384 tokens via a smart chunker that preserves clause numbering. Provenance for every chunk is logged: source PDF, page span, section heading.

\subsection{Three-tier deterministic QA construction}

We construct QA pairs deterministically from regulator text using three regex template families. We deliberately avoid LLM-generated questions to prevent paraphrase contamination of a downstream LLM evaluator and to preserve full provenance for auditability.

\begin{itemize}
  \item \textbf{Tier 1 (native FAQ, $n{=}1{,}142$).} Native ``Q./A.'' style FAQ blocks already present in some Master Directions are pulled directly with a strict template.
  \item \textbf{Tier 2 (numeric extractive, $n{=}2{,}219$).} Sentences containing numeric facts (rupee amounts, days, percentages, ratios, ISIN/PAN-style identifiers) are converted into ``\emph{What is the X for Y?}''-style questions where the numeric token is the gold answer. Patterns are anchored to surrounding clause structure.
  \item \textbf{Tier 3 (heading-extractive, $n{=}1{,}116$).} Section/sub-section headings are converted into ``\emph{What does \textless heading\textgreater{} say?}'' questions whose gold answer is the corresponding paragraph.
\end{itemize}

This yields 4{,}477 raw QA pairs.

\subsection{Ten-check validator}

Each candidate pair passes a 10-check validator before it is allowed into the dataset. Checks include: question/answer minimum and maximum length, gold answer present in source chunk, chunk--PDF--section provenance fields populated, no markup leakage in answer span, language-detector pass for Hindi/Devanagari spillover, and basic sanity heuristics (no all-caps blocks, no orphan footnote markers). Pass rate after a v2 cleaner pass is 98.45\%. Failed candidates are discarded, not paraphrased; this preserves the determinism of the pipeline.

\subsection{Document-disjoint train/eval split}
\label{sec:split}

We split at the PDF level, not at the QA-pair level: 104 PDFs (80\%) are assigned to train and 26 PDFs (20\%) are quarantined as held-out. Every QA pair is then routed to train/eval based solely on which PDF its context came from. This eliminates the chunk-neighbour memorisation pathway: no eval-time paragraph appears (paraphrased or otherwise) in the train set. The final paired dataset is 2{,}931 train + 664 eval pairs, with split manifest stored at \texttt{data/rbi\_corpus/qa/split\_manifest\_v2.json}.

\subsection{BFSI extract adapter}

We train a single LoRA adapter (\texttt{bfsi\_extract}) on top of Nemotron-Nano-30B-A3B (4-bit NF4):

\begin{itemize}
  \item \textbf{Architecture.} LoRA $r{=}16$, $\alpha{=}32$, target modules over both attention and MLP projections. Trainable parameters: 434.6~M of 32~B = 1.36\%.
  \item \textbf{Training.} One epoch, paged AdamW 8-bit, learning rate $2\times10^{-4}$ with cosine schedule, sequence length 1024, on a single RTX~5090.
  \item \textbf{Cost.} Wall-clock 3h~28min, peak VRAM 17.81~GB, adapter file size 1.74~GB.
\end{itemize}

\subsection{BFSI recall adapter}

To probe whether the recipe generalises beyond extractive-with-context QA, we build a second dataset of no-context recall questions ($n{=}214$ paired, document-disjoint) where the gold answer is a numeric or short factual span and the question must be answered without an accompanying paragraph. We train an attention-only LoRA (same $r{=}16$, $\alpha{=}32$, sequence length 384, all other settings identical) on the recall train split.

\subsection{IndiaFinBench OOD probe}

For the external OOD probe, we evaluate the same \texttt{bfsi\_extract} adapter---no further fine-tuning, no prompt engineering, identical decoding settings---on the IndiaFinBench $n{=}324$ test split~\citep{pall2026indiafinbench}. Sample contexts in IndiaFinBench range 200--8000 tokens, longer than our internal training contexts. Evaluation runs in 73~min on a single RTX~5090.

\subsection{Indian BFSI Benchmark v1}

We hand-curate a small open benchmark of 60 questions (RBI 30, SEBI 30; Tier 2 numeric 30, Tier 3 heading-extractive 30) drawn from 22 distinct held-out PDFs. Each question carries a hand-curated alternative-answer list (e.g., ``Rs. 1,00,000'', ``one lakh rupees'', ``INR 100000'') so that scoring is robust to surface variation. The benchmark uses a mixed metric: 30 questions scored by substring match, 30 scored by token-F1 with threshold 0.5. We release the data, scorer, and metadata under CC-BY-SA-4.0.

\subsection{Frontier comparison methodology}

For a directional sanity check against frontier APIs, we run a stratified $n{=}15$ comparison (Tier 2 / Tier 3 $\times$ RBI / SEBI) of \texttt{bfsi\_extract} against Anthropic Claude Opus and Claude Sonnet, where the frontier models are accessed via a subagent harness because we have no API budget. All systems receive identical context paragraphs and identical prompt formats. Both substring match and token F1 are reported. We treat $n{=}15$ as too small for statistical claims and report only directional differences.

\subsection{Evaluation metrics}

Across all evaluations we report case-insensitive substring match as the primary citation-faithful metric, token-F1 as a complementary semantic metric, and Wilson 95\% confidence intervals on the binary substring rate. For paired comparisons we report the McNemar $2\times2$ contingency $(b_{11}, b_{10}, b_{01}, b_{00})$ together with the exact-binomial $p$-value. Where token F1 is the primary outcome (Section~\ref{sec:results-recall}), we use the Wilcoxon signed-rank test for paired F1 differences.

\subsection{Format Guard as a routing primitive}

We additionally evaluate Format Guard, a token-level logits processor we developed in companion work~\citep{ours2026fg}, in the BFSI setting. Format Guard simultaneously loads four adapters (math, code, science, \texttt{bfsi\_extract}) into a single 30B-class base, swaps the active adapter every $K{=}10$ generated tokens via a regex over the decoded suffix, and uses an $\mathcal{O}(1)$ pointer-level swap on the model's adapter table. We report Format Guard runs on every BFSI evaluation here only to demonstrate that it incurs zero overhead relative to the dedicated single adapter; we do not re-derive its core HumanEval lift, which is reported elsewhere~\citep{ours2026fg}.

\section{Experiments}

We run four paired evaluations and one directional comparison:

\begin{itemize}
  \item \textbf{E1 -- BFSI extract held-out ($n{=}664$).} Three modes: base, \texttt{bfsi\_extract} direct, Format Guard with four adapters. Primary metric substring match, secondary token F1.
  \item \textbf{E2 -- BFSI recall held-out ($n{=}214$).} Three modes: base, \texttt{bfsi\_recall} direct, Format Guard. Primary metric token F1.
  \item \textbf{E3 -- IndiaFinBench OOD ($n{=}324$).} \texttt{bfsi\_extract} only, evaluated against published Gemini Flash zero-shot scores from IndiaFinBench~\citep{pall2026indiafinbench}.
  \item \textbf{E4 -- Indian BFSI Benchmark v1 ($n{=}60$).} Three modes (base, \texttt{bfsi\_extract}, Format Guard) on the released benchmark.
  \item \textbf{E5 -- Frontier directional ($n{=}15$).} \texttt{bfsi\_extract} vs Claude Opus and Claude Sonnet via the subagent harness.
\end{itemize}

All paired evaluations share base decoding settings (temperature 0, identical max-new-tokens, identical prompt template) and identical seeds. McNemar contingencies are reported in full. We do not re-tune for any single benchmark.

\section{Results}
\label{sec:results}

\subsection{BFSI extract on held-out paired ($n{=}664$)}

Table~\ref{tab:extract-main} reports substring and token-F1 scores.

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Mode & Substring & Wilson 95\% CI & Token F1 \\
\midrule
Base Nemotron-30B & 58.7\% & [54.95, 62.42] & 0.132 \\
+ \texttt{bfsi\_extract} (direct) & \textbf{89.6\%} & [87.06, 91.71] & 0.172 \\
Format Guard (4 adapters) & 88.7\% & [86.07, 90.89] & 0.171 \\
\bottomrule
\end{tabular}
\caption{Held-out paired evaluation, $n{=}664$ document-disjoint.}
\label{tab:extract-main}
\end{table}

The adapter--vs--base McNemar contingency is $(b_{11}, b_{10}, b_{01}, b_{00}) = (376,14,219,55)$, giving $p{=}1.66\times10^{-48}$. The improvement-to-regression ratio is 219:14; for every question the adapter regresses, it newly answers roughly 15.6 questions correctly that the base did not.

Table~\ref{tab:extract-pairs} shows the three pairwise comparisons.

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Comparison & $b_{10}$ & $b_{01}$ & $\Delta$ & $p$ \\
\midrule
adapter vs base & 14 & 219 & $+30.9$~pp & $1.66\times10^{-48}$ \\
Format Guard vs base & 19 & 218 & $+30.0$~pp & $5.12\times10^{-44}$ \\
Format Guard vs adapter & 6 & 0 & $-0.9$~pp & $0.031$ \\
\bottomrule
\end{tabular}
\caption{Pairwise McNemar comparisons on $n{=}664$ held-out extract questions.}
\label{tab:extract-pairs}
\end{table}

Format Guard and the dedicated adapter disagree on only 6 of 664 questions, all of which are Format-Guard losses. Format Guard never improves over the dedicated adapter on BFSI; the marginal $p{=}0.031$ is driven by the 6:0 split. The empirical zero-overhead claim for Format Guard on BFSI follows.

Per-tier, Tier~2 numeric questions ($n{=}386$) gain $+24.6$~pp; Tier~3 heading-extractive questions ($n{=}278$) gain $+39.5$~pp. Per-regulator, RBI questions gain $+31.5$~pp; SEBI questions gain $+30.1$~pp.

\subsection{BFSI recall paired ($n{=}214$)}
\label{sec:results-recall}

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Mode & F1 mean & Lift & Wilcoxon vs base \\
\midrule
Base & 0.158 & --- & --- \\
+ \texttt{bfsi\_recall} & 0.219 & $+0.061$ ($+38.4\%$ rel.) & $p{=}1.50\times10^{-16}$ \\
Format Guard (4 adapters) & 0.219 & $+0.061$ & vs adapter: $p{=}0.55$ \\
\bottomrule
\end{tabular}
\caption{No-context recall paired, $n{=}214$. Substring is uninformative on this task (near-zero for all modes).}
\label{tab:recall}
\end{table}

In 74.3\% of paired questions, adapter F1 strictly exceeds base F1. The Wilcoxon $p{=}1.50\times10^{-16}$ confirms that the recipe generalises to a structurally different task type (no-context recall, attention-only LoRA, shorter sequence length). Format Guard vs dedicated adapter is statistically indistinguishable ($p{=}0.55$): Format Guard's zero-overhead behaviour replicates.

\subsection{IndiaFinBench OOD ($n{=}324$)}

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Metric & Score & Wilson 95\% CI & vs Gemini Flash \\
\midrule
Substring & \textbf{32.1\%} & [27.3, 37.4] & 89.7\% \\
Normalised match & 32.7\% & [27.8, 38.0] & --- \\
Token F1 (mean) & 0.288 & --- & --- \\
\bottomrule
\end{tabular}
\caption{IndiaFinBench overall performance for \texttt{bfsi\_extract}; Gemini 2.5 Flash zero-shot value from \citet{pall2026indiafinbench}.}
\label{tab:indiafin-overall}
\end{table}

Per task type, we observe large gaps on regulatory interpretation, numerical reasoning, and temporal reasoning; contradiction detection substring scores are inflated by yes/no lexical overlap and contradicted by very low F1. By regulator, the adapter scores SEBI 33.0\% (\(n{=}273\)) and RBI 27.5\% (\(n{=}51\)).

The 57.6~pp gap between our 32.1\% on IndiaFinBench and Gemini Flash's reported 89.7\% is the predicted out-of-distribution failure mode of a fine-tuned adapter~\citep{liang2022helm}. Our 89.6\% on the in-distribution held-out set and Gemini's 89.7\% on IndiaFinBench are not directly comparable: different benchmarks, different scoring, different question styles. The apples-to-apples number for our adapter on IndiaFinBench is 32.1\%, and we report it without softening.

\subsection{Indian BFSI Benchmark v1 ($n{=}60$)}

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Mode & Score & Wilson 95\% CI & Substring & F1 \\
\midrule
Base & 40.0\% (24/60) & [28.6, 52.6] & 76.7\% & 0.122 \\
+ \texttt{bfsi\_extract} & \textbf{50.0\%} (30/60) & [37.7, 62.3] & 98.3\% & 0.157 \\
Format Guard & 50.0\% (30/60) & [37.7, 62.3] & 98.3\% & 0.157 \\
\bottomrule
\end{tabular}
\caption{Released benchmark, mixed scoring. Paired McNemar adapter vs base $p{=}0.0313$; Format Guard vs adapter $p{=}1.0$ (identical 30/60).}
\label{tab:benchmark}
\end{table}

On the 30 substring-scored questions, the adapter goes from 80\% to a perfect 100\%, a clean +20~pp lift. On the 30 questions scored under the strict F1$\geq0.5$ cutoff, both base and adapter score 0\%, despite mean F1 rising from \(\sim0.12\) to \(\sim0.16\). This is a metric artefact: a 1--2 sentence gold span almost never reaches F1$\geq0.5$ against a full-paragraph quotation.

\subsection{Frontier directional comparison ($n{=}15$)}

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
Model & Substring & Token F1 \\
\midrule
Nemotron-30B + \texttt{bfsi\_extract} & \textbf{87\%} & 0.38 \\
Anthropic Claude Opus & 7\% & \textbf{0.65} \\
Anthropic Claude Sonnet & 27\% & 0.49 \\
\bottomrule
\end{tabular}
\caption{Stratified $n{=}15$ directional comparison via subagent harness. NOT a statistical claim.}
\label{tab:frontier}
\end{table}

The two metrics measure different deliverables. Our adapter dominates the citation-faithful substring metric by 60--80~pp (it is configured to quote the regulator verbatim and does so reliably). Claude Opus wins token F1 by \(\sim0.27\) (it is configured to produce semantically polished answers in its own voice). With $n{=}15$ this is directional only; we deliberately do not run a hypothesis test.

\section{Limitations and Negative Findings}

We summarise key limitations; these scope our claims and are intended to prevent over-generalisation.

\begin{itemize}
  \item \textbf{Regulator coverage.} Our corpus covers RBI and SEBI only. India's BFSI regulatory perimeter additionally includes IRDAI (insurance), PFRDA (pensions), and the FEMA framework (foreign exchange). We do not claim multi-regulator coverage.
  \item \textbf{Small-$n$ on the released benchmark.} Benchmark v1 has $n{=}60$. The McNemar test on adapter vs base is $p{=}0.0313$---positive but marginal at $\alpha{=}0.05$.
  \item \textbf{Frontier comparison is directional only.} The $n{=}15$ frontier comparison is too small for any statistical claim. We do not claim our system ``beats'' Claude on Indian BFSI.
  \item \textbf{Benchmark-incomparability caveat.} We deliberately do not report ``our adapter 89.6\% $\approx$ Gemini Flash 89.7\%''. These are different benchmarks; the apples-to-apples numbers are 89.6\% vs base 58.7\% on our held-out, and 32.1\% vs Gemini's 89.7\% on IndiaFinBench.
  \item \textbf{Format Guard zero-overhead replicates, not improves.} On all three BFSI paired evaluations, Format Guard either matches the dedicated adapter or differs only on a handful of questions, all of which are Format-Guard losses. It does not improve over the best single adapter on BFSI.
  \item \textbf{F1 cutoff metric artefact.} The F1$\geq0.5$ cutoff in Benchmark v1 is too strict for verbose paragraph-extraction style; a revised scoring would likely yield non-zero scores.
  \item \textbf{Single base model.} All BFSI results are on Nemotron-Nano-30B-A3B 4-bit. Multi-base replication is not reported here.
  \item \textbf{Long-context behaviour at the OOD boundary.} IndiaFinBench sample contexts (up to 8k tokens) exceed our training context (1024). We did not retrain at longer context; long-context degradation may inflate the OOD gap.
\end{itemize}

\section{Discussion and Future Work}
\label{sec:discussion}

\subsection{Reframing the OOD gap}

The natural reading of a $-57.6$~pp gap to Gemini Flash on IndiaFinBench is ``your model is much worse.'' We argue the more accurate reading is: a model fine-tuned on one customer's regulatory corpus does not transfer to a different question style on the same domain---the failure mode predicted by OOD evaluation work such as HELM~\citep{liang2022helm}. This is also the structural argument for per-customer adapters: if a single fine-tuned model could handle every Indian-BFSI customer's questions, then a single frontier API call could too, and the sovereignty problem would be vacuous.

\subsection{Use-inspired evaluation as a contribution type}

We position this paper as a use-inspired evaluation paper rather than a methods paper. The pipeline uses well-known primitives (LoRA, QLoRA, regex extraction, McNemar, Wilcoxon). The contribution is in their disciplined assembly: deterministic QA construction without LLM paraphrase, document-disjoint splitting before any pair is constructed, paired evaluation against the same base in three independent settings, and honest disclosure of the OOD result rather than an inflated cross-benchmark headline.

\subsection{Future work}

Concrete near-term extensions include:

\begin{enumerate}
  \item \textbf{Multi-regulator extension.} Extending the pipeline to IRDAI, PFRDA, and FEMA is likely to yield a larger train corpus and shrink the IndiaFinBench OOD gap on regulatory-interpretation tasks.
  \item \textbf{Frontier comparison at $n{=}60$.} Running Anthropic Claude, OpenAI GPT-4o, and Google Gemini against Benchmark v1 would replace the directional $n{=}15$ comparison with a properly paired $n{=}60$ result.
  \item \textbf{Per-customer pilot.} Applying the pipeline to a real customer's internal compliance corpus and reporting the held-out paired McNemar from that customer's adapter.
  \item \textbf{Long-context retraining.} Extending \texttt{bfsi\_extract}'s context window and re-running IndiaFinBench would disentangle style-shift from length-shift in the OOD gap.
  \item \textbf{Multi-base replication.} Repeating \texttt{bfsi\_extract} on additional 30B-class bases (e.g., Qwen3-32B, Llama-3-70B) to test base-invariance.
\end{enumerate}

\section*{Ethics, Risks, and Reproducibility}

This work focuses on regulatory-text QA methodology for BFSI; we do not deploy any system in user-facing or safety-critical settings. Our experiments use publicly available regulatory PDFs (RBI and SEBI portals), publicly released base models, and the IndiaFinBench benchmark. We do not process personal data or customer-identifying information.

We take two steps to reduce the risk of misleading claims. First, all headline numbers trace to internal tables with explicit provenance and confidence intervals; any number not in those tables is treated as non-citable. Second, we explicitly disclose limitations, evaluation artefacts, and missing experiments. In particular, we avoid cross-benchmark comparisons (e.g., our held-out vs IndiaFinBench) and clearly separate in-distribution and out-of-distribution results.

To aid reproducibility, we intend to release the corpus-construction scripts, the 10-check validator, the document-disjoint split manifest, the Benchmark v1 data and scorer, and the evaluation harnesses for all BFSI experiments, subject to venue policy and upstream licenses. Where artifacts depend on vendor base-model licenses, we rely on the original distribution terms and do not redistribute proprietary weights.

\begin{thebibliography}{99}

\bibitem{austin2021mbpp}
J.~Austin, A.~Odena, M.~Nye, M.~Bosma, H.~Michalewski, D.~Dohan, et~al.
\newblock Program synthesis with large language models.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2021.

\bibitem{chen2021evaluating}
M.~Chen, J.~Tworek, H.~Jun, Q.~Yuan, H.~P. de~Oliveira~Pinto, J.~Kaplan, et~al.
\newblock Evaluating large language models trained on code.
\newblock \emph{arXiv preprint arXiv:2107.03374}, 2021.

\bibitem{dettmers2023qlora}
T.~Dettmers, A.~Pagnoni, A.~Holtzman, and L.~Zettlemoyer.
\newblock {QLoRA}: Efficient finetuning of quantized {LLMs}.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2023.

\bibitem{hu2021lora}
E.~J. Hu, Y.~Shen, P.~Wallis, Z.~Allen-Zhu, Y.~Li, L.~Wang, and W.~Chen.
\newblock {LoRA}: Low-rank adaptation of large language models.
\newblock \emph{arXiv preprint arXiv:2106.09685}, 2021.

\bibitem{liang2022helm}
P.~Liang, R.~Bommasani, T.~Zhang, R.~R. Guo, J.~Yu, F.~X. He, et~al.
\newblock Holistic evaluation of language models.
\newblock \emph{arXiv preprint arXiv:2211.09110}, 2022.

\bibitem{magar2022data}
I.~Magar and R.~Schwartz.
\newblock Data contamination: From memorization to exploitation.
\newblock In \emph{Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)}, 2022.

\bibitem{pall2026indiafinbench}
S.~Pall, et~al.
\newblock IndiaFinBench: An evaluation benchmark for large language model performance on Indian financial regulatory text.
\newblock \emph{arXiv preprint arXiv:2604.19298}, 2026.

\bibitem{ours2026fg}
Anonymous.
\newblock The Code Paradox and Format Guard: Token-level {LoRA} adapter routing as a substitute for failed static composition on a 30B base.
\newblock \emph{Manuscript in preparation}, 2026.

\end{thebibliography}

\end{document}