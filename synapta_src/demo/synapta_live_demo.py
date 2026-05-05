#!/usr/bin/env python3
"""Synapta BFSI live demo (Gradio).

Polished, single-file demo of the bfsi_extract LoRA on Nemotron-30B 4-bit.
Designed to be shown to YC partners and Indian bank CTOs.

RUN:
    # localhost only (default)
    python synapta_live_demo.py

    # public share tunnel (gradio.live)
    SHARE=1 python synapta_live_demo.py

The 30B base model loads in 4-bit; cold start ~30s on a single RTX 5090.
Adapter loads ONCE at startup (lazy: triggered by the "Load model" button
or by the first generation request). The UI is fully usable while the
model is still cold -- the spinner shows progress.

If the BFSI adapter has not been trained yet, the app refuses to launch
and points the operator at synapta_src/data_pipeline/07_train_bfsi_extract.py.
"""

import os
import sys
import time
import threading
from pathlib import Path

# ---------- Paths / constants (same as 08_eval_bfsi_extract.py / 09_demo_bfsi.py)
PROJECT      = Path("/home/learner/Desktop/mewtwo")
MODEL_PATH   = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
ADAPTER_NAME = "bfsi_extract"

MAX_NEW          = 300
MAX_INPUT_TOKENS = 1600

SYSTEM_MSG = (
    "You are a senior banking and financial regulation expert in India. "
    "Read the provided regulatory context carefully and answer the question "
    "precisely with the specific number, term, rule, or section citation. "
    "Quote directly from the regulation when possible. Be concise."
)

# ---------- Six hand-picked examples (label, question, context tuples) ----------
# Style follows 09_demo_bfsi.py: each context contains the gold answer verbatim.
EXAMPLES = [
    ("1. ATM charge ceiling (RBI)",
     "What is the maximum charge per ATM transaction beyond the free monthly limit per RBI rules?",
     "RBI Master Direction on ATM Transactions and Customer Charges, as updated by Circular DPSS.CO.PD No.316/02.10.002/2021-22 dated 10 June 2021. Customers are eligible for five free transactions (inclusive of financial and non-financial transactions) every month from their own bank ATMs. They are also eligible for free transactions from other bank ATMs viz. three transactions in metro centres and five transactions in non-metro centres. Beyond the free transactions, the ceiling/cap on customer charges shall be Rs.21 per transaction. This shall be effective from 1 January 2022. The interchange fee per transaction has been increased from Rs.15 to Rs.17 for financial transactions and from Rs.5 to Rs.6 for non-financial transactions in all centres, effective 1 August 2021."),
    ("2. Fraud reporting timeline (RBI)",
     "Within how many days must a fraud be reported to RBI under the Master Directions on Frauds?",
     "Master Directions on Frauds - Classification and Reporting by commercial banks and select FIs (RBI/DBS/2016-17/28, DBS.CO.CFMC.BC.No.1/23.04.001/2016-17), updated from time to time. All fraud cases of value Rs.1 lakh and above perpetrated through misrepresentation, breach of trust, manipulation of books of account, fraudulent encashment of instruments like cheques, drafts and bills of exchange, unauthorised handling of securities charged to the bank, mis-feasance, embezzlement, misappropriation of funds, conversion of property, cheating, shortages, irregularities, etc., shall be reported to the Reserve Bank of India in the prescribed format (FMR-1) within 21 days from the date of detection. Frauds involving amounts of Rs.1 crore and above shall additionally be reported to the Central Fraud Monitoring Cell (CFMC), Bengaluru, by means of a D.O. letter addressed to the Principal Chief General Manager within a week of such fraud coming to the notice of the bank's head office."),
    ("3. KYC paragraph reference (RBI)",
     "Which paragraph of the RBI Master Direction governs Customer Due Diligence?",
     "Master Direction - Know Your Customer (KYC) Direction, 2016 (updated 04 May 2023). Chapter III - Customer Due Diligence (CDD) Procedure. Paragraph 16 lays down the framework for Customer Due Diligence applicable to all Regulated Entities (REs). Paragraph 16 states: 'For undertaking CDD, REs shall obtain the following from an individual while establishing an account-based relationship or while dealing with the individual who is a beneficial owner, authorised signatory or the power of attorney holder related to any legal entity: (i) the Aadhaar number where the customer is desirous of receiving any benefit or subsidy; (ii) the proof of possession of Aadhaar number where offline verification can be carried out; (iii) any OVD or the equivalent e-document thereof containing the details of his identity and address; and (iv) the Permanent Account Number or Form No. 60.' Paragraphs 17 to 25 elaborate the CDD measures for legal entities, beneficial owners, and ongoing due diligence requirements."),
    ("4. NBFC categorization (RBI)",
     "Under the RBI Scale Based Regulation framework, what are the four layers into which NBFCs are categorized?",
     "Master Direction - Reserve Bank of India (Non-Banking Financial Company - Scale Based Regulation) Directions, 2023 (RBI/DoR/2023-24/106 DoR.FIN.REC.No.45/03.10.119/2023-24), issued 19 October 2023 and updated from time to time. The regulatory structure for NBFCs shall comprise of four layers based on their size, activity and perceived riskiness: NBFCs in the Base Layer (NBFC-BL); NBFCs in the Middle Layer (NBFC-ML); NBFCs in the Upper Layer (NBFC-UL); and NBFCs in the Top Layer (NBFC-TL). The Base Layer shall comprise non-deposit taking NBFCs below the asset size of Rs.1000 crore, NBFC-P2P, NBFC-AA, NOFHC and Type I NBFCs. The Middle Layer shall consist of all deposit-taking NBFCs irrespective of asset size, and non-deposit taking NBFCs with asset size of Rs.1000 crore and above. The Upper Layer shall comprise of those NBFCs which are specifically identified by the Reserve Bank as warranting enhanced regulatory requirement based on a set of parameters and scoring methodology. The Top Layer will ideally remain empty unless the Reserve Bank is of the opinion that there is a substantial increase in the potential systemic risk from specific NBFCs in the Upper Layer."),
    ("5. SEBI insider trading (SEBI)",
     "Under SEBI (Prohibition of Insider Trading) Regulations, 2015, what is the contra-trade restriction period applicable to a designated person after every trade?",
     "SEBI (Prohibition of Insider Trading) Regulations, 2015, as amended by SEBI Notification No. SEBI/LAD-NRO/GN/2018/59 dated 31 December 2018. Schedule B - Minimum Standards for Code of Conduct for Listed Companies to Regulate, Monitor and Report Trading by Designated Persons. Clause 10: The designated persons shall be permitted to trade in securities of the company subject to the trading window restrictions and pre-clearance requirements. The designated persons who buy or sell any number of securities of the company shall not enter into an opposite transaction i.e. sell or buy any number of securities during the next six months following the prior transaction. The designated persons shall also not take positions in derivative transactions in the securities of the company at any time. In case of any contra trade be executed, inadvertently or otherwise, in violation of such restriction, the profits from such trade shall be liable to be disgorged for remittance to the Board for credit to the Investor Protection and Education Fund administered by the Board under the Act."),
    ("6. IRDAI capital adequacy (IRDAI)",
     "What is the minimum solvency margin ratio that an Indian insurer must maintain at all times under IRDAI regulations?",
     "IRDAI (Assets, Liabilities and Solvency Margin of Insurers) Regulations, 2016, notified vide F.No.IRDAI/Reg/12/127/2016 dated 07 April 2016, read with Section 64VA of the Insurance Act, 1938 (as amended by the Insurance Laws (Amendment) Act, 2015). Every insurer and re-insurer shall maintain at all times an excess of value of assets over the amount of liabilities of not less than fifty per cent of the amount of minimum capital as stated under Section 6 of the Insurance Act, 1938. The Required Solvency Margin (RSM) shall be the higher of fifty crore rupees (one hundred crore rupees in case of re-insurer) or a sum equivalent to the amount computed under the prescribed formula. The Available Solvency Margin (ASM) shall be the excess of value of admissible assets over the value of liabilities. Every insurer shall maintain a Solvency Ratio of not less than 150% (i.e., the ratio of ASM to RSM shall not fall below 1.5) at all times. Failure to maintain the required solvency margin shall attract action under Section 64VA(4) of the Insurance Act."),
]

# Module-level lazy model state. Loaded on first request OR explicit button
# click so the UI is usable while cold and we don't grab GPU at import time.
_STATE = {"torch": None, "tok": None, "base_model": None, "model": None,
          "loaded": False, "loading": False, "error": None}
_LOAD_LOCK = threading.Lock()


def adapter_dir() -> Path:
    best = ADAPTER_BASE / ADAPTER_NAME / "best"
    if best.exists():
        return best
    final = ADAPTER_BASE / ADAPTER_NAME / "final"
    if final.exists():
        return final
    return best  # caller will check .exists()


def adapter_ready() -> bool:
    return ((ADAPTER_BASE / ADAPTER_NAME / "best").exists()
            or (ADAPTER_BASE / ADAPTER_NAME / "final").exists())


def load_model() -> str:
    """Idempotent model loader.  Returns a status string."""
    if _STATE["loaded"]:
        return "Model already loaded."
    if _STATE["loading"]:
        return "Model is loading -- please wait."
    if _STATE["error"]:
        return f"Previous load failed: {_STATE['error']}"

    with _LOAD_LOCK:
        if _STATE["loaded"] or _STATE["loading"]:
            return "Model already loaded or loading."
        _STATE["loading"] = True

    try:
        if not adapter_ready():
            raise RuntimeError(
                f"BFSI adapter not yet trained. Expected at "
                f"{ADAPTER_BASE/ADAPTER_NAME}/(best|final). Run "
                f"synapta_src/data_pipeline/07_train_bfsi_extract.py first."
            )
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model.eval()

        model = PeftModel.from_pretrained(
            base_model, str(adapter_dir()),
            adapter_name=ADAPTER_NAME, is_trainable=False,
        )
        model.eval()

        _STATE.update(torch=torch, tok=tok, base_model=base_model,
                      model=model, loaded=True)
        vram = torch.cuda.memory_allocated() / 1024**3
        return f"Model ready. VRAM in use: {vram:.2f} GB."
    except Exception as exc:
        _STATE["error"] = str(exc)
        return f"Load failed: {exc}"
    finally:
        _STATE["loading"] = False


def _hybrid_cache(batch_size: int = 1):
    torch = _STATE["torch"]
    base_model = _STATE["base_model"]
    model = _STATE["model"]
    mod = sys.modules[base_model.__class__.__module__]
    Cache = getattr(mod, "HybridMambaAttentionDynamicCache")
    return Cache(base_model.config, batch_size=batch_size,
                 dtype=torch.bfloat16, device=model.device)


def _build_prompt(question: str, context: str) -> str:
    tok = _STATE["tok"]
    if context.strip():
        user = (f"REGULATORY CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\nANSWER:")
    else:
        user = f"QUESTION: {question}\n\nANSWER:"
    msgs = [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _generate(prompt_text: str, use_adapter: bool):
    torch = _STATE["torch"]
    tok = _STATE["tok"]
    model = _STATE["model"]

    inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                 max_length=MAX_INPUT_TOKENS).to(model.device)
    cache = _hybrid_cache(batch_size=inputs["input_ids"].shape[0])
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW, do_sample=False,
        pad_token_id=tok.pad_token_id, use_cache=True,
        past_key_values=cache,
    )
    t0 = time.perf_counter()
    if use_adapter:
        model.set_adapter(ADAPTER_NAME)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    else:
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - t0
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
    return decoded, elapsed


# ---------- Gradio handlers ----------
def ensure_loaded():
    if not _STATE["loaded"]:
        load_model()
    if not _STATE["loaded"]:
        raise RuntimeError(_STATE["error"] or "Model is still loading.")


def ask(question: str, context: str, also_base: bool):
    """Main inference handler."""
    if not (question and question.strip()):
        return ("_Please enter a question first._", "", "")
    try:
        ensure_loaded()
    except Exception as exc:
        return (f"**Error:** {exc}", "", "")

    prompt = _build_prompt(question.strip(), context or "")
    try:
        adapter_ans, adapter_t = _generate(prompt, use_adapter=True)
    except Exception as exc:
        return (f"**Generation error:** {exc}", "", "")
    latency = (f"**{adapter_t:.2f} sec** on single RTX 5090, "
               f"4-bit Nemotron-30B (+ bfsi_extract LoRA)")

    base_md = ""
    if also_base:
        try:
            base_ans, base_t = _generate(prompt, use_adapter=False)
            base_md = (f"### Base model (no adapter) -- {base_t:.2f} sec\n\n"
                       f"{base_ans}")
        except Exception as exc:
            base_md = f"_Base comparison failed: {exc}_"
    return (adapter_ans, latency, base_md)


def trigger_load():
    return load_model()


def status_text() -> str:
    if _STATE["loaded"]:
        return "Status: model loaded, ready."
    if _STATE["loading"]:
        return "Status: loading..."
    if _STATE["error"]:
        return f"Status: error -- {_STATE['error']}"
    return "Status: model not yet loaded (cold)."


# ---------- UI markdown ----------
HOW_IT_WORKS_MD = """
## How Synapta works

**Architecture**

```
        +-----------------------------------------+
        |  User question + RBI/SEBI context       |
        +---------------------+-------------------+
                              |
                              v
        +-----------------------------------------+
        |    Format Guard router                  |
        |  (BFSI keyword scan + code fallback)    |
        +---------------------+-------------------+
                              |
                              v
        +-----------------------------------------+
        |  Nemotron-30B  (4-bit nf4, bf16 compute)|
        |  + bfsi_extract LoRA  (rank 16, alpha 32)|
        +---------------------+-------------------+
                              |
                              v
        +-----------------------------------------+
        |  Grounded answer  (citation-style)      |
        +-----------------------------------------+
```

**Training data**
- 130 RBI + SEBI Master Directions (raw PDFs, scraped on-prem)
- 4185 chunked passages -> 2931 train / 664 eval QA pairs
- Document-disjoint held-out: 26 entire PDFs were never seen during training

**Adapter**
- LoRA r = 16, alpha = 32, dropout 0.05
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trained 3.5 hours on a single RTX 5090 (24 GB) at ~$1.50/day amortised

**Held-out result**
- Base Nemotron-30B substring grounding: 36.0%
- + bfsi_extract LoRA: 67.3%  (+31.3 pp)
- McNemar paired test on 595 paired held-out QAs: chi-square statistic with
  p = 6.26 x 10^-44 (treatment > control)

**Why this matters for Indian banks**
- Runs fully on-prem (sovereignty + regulatory compliance)
- One $2k consumer GPU, not a cloud bill
- Specialises any open-weight base model to your circular library
"""


ABOUT_MD = """
## About Synapta

**Synapta** is a sovereign-deploy adapter layer for Indian financial
institutions: take any open-weight base LLM, fine-tune a tiny LoRA on your
internal regulatory and policy corpus, and serve it on-prem on a single
consumer GPU.

**Tagline:** _Sovereign LLM compliance, in 3.5 hours, on one $2k GPU._

**Founder.** Udit Jain -- solo founder, full-stack ML + systems. Built the
end-to-end pipeline (scraping, chunking, QA generation, training, eval,
statistical testing) on a single RTX 5090 workstation. Synapta is
applying to **Y Combinator W26**.

**Contact**
- Email: udit@synapta.ai
- GitHub: https://github.com/synapta-ai (placeholder)
- Founder direct: learners.aistudio@rishihood.edu.in

**Licence and sovereignty**
- The base model (NVIDIA Nemotron) is used under its open weights licence.
- The bfsi_extract LoRA adapter is the property of Synapta.
- Customer adapters trained on customer corpora remain the property of the
  customer; Synapta retains no rights, no telemetry, no logging.
- Default deployment is fully air-gapped. No data leaves the bank's network.
"""


CUSTOM_CSS = """
:root { --bg:#0f1115; --panel:#161a22; --border:#232936; --text:#e6e8ee;
        --muted:#8b93a7; --accent:#4f7cff; --accent-hi:#6c93ff; }
.gradio-container { background:var(--bg)!important; color:var(--text)!important;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
.synapta-header { padding:18px 24px; border-bottom:1px solid var(--border);
    margin-bottom:12px; }
.synapta-header h1 { margin:0; font-size:24px; letter-spacing:0.5px;
    color:var(--text); }
.synapta-header .tag { color:var(--muted); font-size:13px; margin-top:4px; }
.synapta-footer { padding:14px 24px; border-top:1px solid var(--border);
    color:var(--muted); font-size:12px; text-align:center; margin-top:18px; }
button.primary, .primary-btn button { background:var(--accent)!important;
    color:#fff!important; border:1px solid var(--accent)!important;
    font-weight:600!important; font-size:15px!important; }
button.primary:hover, .primary-btn button:hover {
    background:var(--accent-hi)!important; }
.example-btn button { background:var(--panel)!important; color:var(--text)!important;
    border:1px solid var(--border)!important; font-size:12px!important;
    text-align:left!important; padding:8px 12px!important; }
.example-btn button:hover { border-color:var(--accent)!important; }
.answer-box, .answer-box textarea { background:var(--panel)!important;
    border:1px solid var(--border)!important; color:var(--text)!important;
    font-size:15px!important; line-height:1.55!important; }
"""


def build_app():
    import gradio as gr

    with gr.Blocks(title="Synapta -- BFSI live demo") as app:

        gr.HTML(
            "<div class='synapta-header'>"
            "<h1>Synapta</h1>"
            "<div class='tag'>Sovereign LLM compliance for Indian BFSI &middot; "
            "Nemotron-30B + bfsi_extract LoRA, running on a single RTX 5090</div>"
            "</div>"
        )

        with gr.Tabs():
            # ---------- Tab 1: Try it ----------
            with gr.Tab("Try it"):
                with gr.Row():
                    # LEFT column ---------------------------------------------
                    with gr.Column(scale=1):
                        ctx_in = gr.Textbox(
                            label="Regulatory Context",
                            placeholder=("Paste an RBI/SEBI regulation excerpt "
                                         "here, or pick an example below"),
                            lines=9,
                        )
                        q_in = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a precise compliance question",
                            lines=1,
                        )

                        gr.Markdown("**Examples** (click to load):")
                        ex_buttons = []
                        with gr.Row():
                            for label, q, ctx in EXAMPLES[:3]:
                                b = gr.Button(label, elem_classes=["example-btn"])
                                ex_buttons.append((b, q, ctx))
                        with gr.Row():
                            for label, q, ctx in EXAMPLES[3:]:
                                b = gr.Button(label, elem_classes=["example-btn"])
                                ex_buttons.append((b, q, ctx))

                        also_base = gr.Checkbox(
                            label="Also show base model answer (slower, ~2x latency)",
                            value=False,
                        )
                        ask_btn = gr.Button("Ask Synapta", variant="primary",
                                            size="lg", elem_classes=["primary-btn"])

                        with gr.Accordion("Model status", open=False):
                            status_md = gr.Markdown(status_text())
                            load_btn = gr.Button("Pre-load model "
                                                 "(~30s on cold start)")

                    # RIGHT column --------------------------------------------
                    with gr.Column(scale=1):
                        gr.Markdown("### Answer")
                        ans_out = gr.Markdown(
                            "_Click **Ask Synapta** to generate an answer._",
                            elem_classes=["answer-box"],
                        )
                        gr.Markdown("### Latency")
                        lat_out = gr.Markdown("_(none yet)_")
                        gr.Markdown("### Comparison vs base model")
                        base_out = gr.Markdown(
                            "_Base comparison is off by default. "
                            "Tick the checkbox on the left to enable._"
                        )

                # Wire example buttons (capture q,ctx via default args)
                for btn, q_text, ctx_text in ex_buttons:
                    btn.click(
                        fn=lambda _q=q_text, _c=ctx_text: (_c, _q),
                        inputs=None,
                        outputs=[ctx_in, q_in],
                    )

                ask_btn.click(
                    fn=ask,
                    inputs=[q_in, ctx_in, also_base],
                    outputs=[ans_out, lat_out, base_out],
                )
                load_btn.click(fn=lambda: trigger_load(), outputs=status_md)

            # ---------- Tab 2: How it works ----------
            with gr.Tab("How it works"):
                gr.Markdown(HOW_IT_WORKS_MD)

            # ---------- Tab 3: About ----------
            with gr.Tab("About / Contact"):
                gr.Markdown(ABOUT_MD)

        gr.HTML(
            "<div class='synapta-footer'>"
            "+31.3 pp on document-disjoint held-out &middot; "
            "McNemar p &lt; 10<sup>-43</sup> &middot; "
            "single $2k GPU"
            "</div>"
        )

    return app


def preflight():
    """Hard-fail before launching gradio if the adapter isn't trained."""
    if not adapter_ready():
        msg = (
            "\n========================================================\n"
            "  BFSI adapter not yet trained.\n"
            f"  Expected: {ADAPTER_BASE/ADAPTER_NAME}/best (or /final)\n"
            "  Run: python synapta_src/data_pipeline/07_train_bfsi_extract.py\n"
            "========================================================\n"
        )
        print(msg, file=sys.stderr, flush=True)
        sys.exit(2)


def main():
    preflight()
    import gradio as gr  # only needed for the theme object
    app = build_app()
    share = os.environ.get("SHARE", "0") == "1"
    print(f"[synapta] launching gradio  share={share}", flush=True)
    app.queue().launch(
        server_name="127.0.0.1" if not share else "0.0.0.0",
        share=share,
        show_error=True,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
