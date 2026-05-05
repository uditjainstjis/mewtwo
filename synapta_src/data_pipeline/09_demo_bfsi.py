#!/usr/bin/env python3
"""Interactive BFSI extract adapter demo (Nemotron-30B 4-bit + bfsi_extract LoRA).

Polished side-by-side comparison: BASE model vs +bfsi_extract adapter on real
RBI / SEBI regulatory questions. Designed for live demos to a CTO / YC reviewer.

USAGE:
  # Default: scripted run through 5 hand-picked killer questions
  python 09_demo_bfsi.py

  # REPL mode (paste a question, optional context PDF):
  python 09_demo_bfsi.py --mode interactive

  # One-shot ad-hoc question:
  python 09_demo_bfsi.py --mode interactive --question "What is the KYC threshold?"

  # Provide a context PDF (text extracted via PyPDF2 if installed; else .txt allowed):
  python 09_demo_bfsi.py --mode interactive --question "..." --context-pdf /path/file.pdf

  # Skip base comparison (faster, adapter-only):
  python 09_demo_bfsi.py --no-base

NOTE: Adapter must already be trained at adapters/nemotron_30b/bfsi_extract/best.
If missing, the script exits loudly without loading anything heavy.
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
ADAPTER_NAME = "bfsi_extract"

MAX_NEW = 220
MAX_INPUT_TOKENS = 1600

# ---------- Pretty printing (colorama optional) ----------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    C_GREEN = Fore.GREEN
    C_RED = Fore.RED
    C_CYAN = Fore.CYAN
    C_YELLOW = Fore.YELLOW
    C_MAGENTA = Fore.MAGENTA
    C_DIM = Style.DIM
    C_BRIGHT = Style.BRIGHT
    C_RESET = Style.RESET_ALL
except Exception:
    C_GREEN = C_RED = C_CYAN = C_YELLOW = C_MAGENTA = ""
    C_DIM = C_BRIGHT = C_RESET = ""

CHECK = "PASS"
CROSS = "FAIL"
try:
    "✓".encode(sys.stdout.encoding or "utf-8")
    CHECK = "✓"
    CROSS = "✗"
except Exception:
    pass

BAR = "=" * 72

SYSTEM_MSG = (
    "You are a senior banking and financial regulation expert in India. "
    "Read the provided regulatory context carefully and answer the question "
    "precisely with the specific number, term, rule, or section citation. "
    "Quote directly from the regulation when possible. Be concise."
)

# ---------- Hand-picked killer demo questions ----------
# Each context excerpt is hand-written in the style of the relevant RBI Master
# Direction / SEBI circular and contains the gold answer verbatim.
DEMO_QUESTIONS = [
    {
        "id": "atm_charge",
        "question": "What is the maximum charge per ATM transaction beyond the "
                    "free monthly limit per RBI rules?",
        "context": (
            "RBI Master Direction on ATM Transactions and Customer Charges, "
            "as updated by Circular DPSS.CO.PD No.316/02.10.002/2021-22 dated "
            "10 June 2021. Customers are eligible for five free transactions "
            "(inclusive of financial and non-financial transactions) every "
            "month from their own bank ATMs. They are also eligible for free "
            "transactions (inclusive of financial and non-financial "
            "transactions) from other bank ATMs viz. three transactions in "
            "metro centres and five transactions in non-metro centres. Beyond "
            "the free transactions, the ceiling/cap on customer charges shall "
            "be Rs.21 per transaction. This shall be effective from 1 January "
            "2022. The interchange fee per transaction has been increased "
            "from Rs.15 to Rs.17 for financial transactions and from Rs.5 to "
            "Rs.6 for non-financial transactions in all centres, effective 1 "
            "August 2021. Banks are advised to display the applicable "
            "transaction charges in a clear and conspicuous manner at the ATM "
            "premises and on their websites."
        ),
        "gold": "Rs.21",
        "alts": ["Rs. 21", "21 per transaction", "rupees 21", "INR 21", "21/-"],
    },
    {
        "id": "fraud_reporting",
        "question": "Within how many days must a fraud be reported to RBI under "
                    "the Master Directions on Frauds?",
        "context": (
            "Master Directions on Frauds - Classification and Reporting by "
            "commercial banks and select FIs (RBI/DBS/2016-17/28, "
            "DBS.CO.CFMC.BC.No.1/23.04.001/2016-17), updated from time to "
            "time. All fraud cases of value Rs.1 lakh and above perpetrated "
            "through misrepresentation, breach of trust, manipulation of "
            "books of account, fraudulent encashment of instruments like "
            "cheques, drafts and bills of exchange, unauthorised handling of "
            "securities charged to the bank, mis-feasance, embezzlement, "
            "misappropriation of funds, conversion of property, cheating, "
            "shortages, irregularities, etc., shall be reported to the "
            "Reserve Bank of India in the prescribed format (FMR-1) within "
            "21 days from the date of detection. Frauds involving amounts of "
            "Rs.1 crore and above shall additionally be reported to the "
            "Central Fraud Monitoring Cell (CFMC), Bengaluru, by means of a "
            "D.O. letter addressed to the Principal Chief General Manager "
            "within a week of such fraud coming to the notice of the bank's "
            "head office. Delay in reporting of frauds will be viewed "
            "seriously by the Reserve Bank and may attract penal action."
        ),
        "gold": "21 days",
        "alts": ["within 21 days", "twenty-one days", "21-day"],
    },
    {
        "id": "pmla_threshold",
        "question": "What is the threshold above which NBFCs must report cash "
                    "transactions under PMLA?",
        "context": (
            "Master Direction - Know Your Customer (KYC) Direction, 2016 "
            "(RBI/DBR/2015-16/18, Master Direction DBR.AML.BC.No.81/14.01."
            "001/2015-16), as amended. Pursuant to Rule 3 of the Prevention "
            "of Money-laundering (Maintenance of Records) Rules, 2005, every "
            "NBFC, banking company, financial institution and intermediary "
            "shall maintain the record of, and furnish information to the "
            "Director, Financial Intelligence Unit-India (FIU-IND), of: (a) "
            "all cash transactions of the value of more than rupees ten lakh "
            "or its equivalent in foreign currency; (b) all series of cash "
            "transactions integrally connected to each other which have been "
            "individually valued below rupees ten lakh or its equivalent in "
            "foreign currency where such series of transactions have taken "
            "place within a month and the monthly aggregate exceeds rupees "
            "ten lakh or its equivalent in foreign currency; (c) all "
            "suspicious transactions whether or not made in cash. The Cash "
            "Transaction Report (CTR) for each month shall be submitted to "
            "FIU-IND by 15th of the succeeding month."
        ),
        "gold": "ten lakh",
        "alts": ["Rs.10 lakh", "10 lakh", "rupees ten lakh", "Rs. 10 lakh"],
    },
    {
        "id": "cdd_paragraph",
        "question": "Which paragraph of the RBI Master Direction governs Customer "
                    "Due Diligence?",
        "context": (
            "Master Direction - Know Your Customer (KYC) Direction, 2016 "
            "(updated 04 May 2023). Chapter III - Customer Due Diligence "
            "(CDD) Procedure. Paragraph 16 lays down the framework for "
            "Customer Due Diligence applicable to all Regulated Entities "
            "(REs). Paragraph 16 states: 'For undertaking CDD, REs shall "
            "obtain the following from an individual while establishing an "
            "account-based relationship or while dealing with the individual "
            "who is a beneficial owner, authorised signatory or the power of "
            "attorney holder related to any legal entity: (i) the Aadhaar "
            "number where the customer (a) is desirous of receiving any "
            "benefit or subsidy under any scheme notified under section 7 of "
            "the Aadhaar Act, 2016; or (b) decides to submit his Aadhaar "
            "number voluntarily; or (ii) the proof of possession of Aadhaar "
            "number where offline verification can be carried out; or (iii) "
            "the proof of possession of Aadhaar number where offline "
            "verification cannot be carried out; or (iv) any OVD or the "
            "equivalent e-document thereof containing the details of his "
            "identity and address; and (v) the Permanent Account Number or "
            "the equivalent e-document thereof or Form No. 60.' Paragraphs "
            "17 to 25 elaborate the CDD measures for legal entities, "
            "beneficial owners, and ongoing due diligence requirements."
        ),
        "gold": "Paragraph 16",
        "alts": ["Para 16", "para. 16", "paragraph 16"],
    },
    {
        "id": "car_minimum",
        "question": "What is the minimum capital adequacy ratio (CAR) for "
                    "scheduled commercial banks?",
        "context": (
            "Master Circular - Basel III Capital Regulations (RBI/2015-16/58, "
            "DBR.No.BP.BC.1/21.06.201/2015-16), updated annually. The Basel "
            "III capital regulations are applicable to all Scheduled "
            "Commercial Banks (SCBs) (excluding Local Area Banks and Regional "
            "Rural Banks) on an ongoing basis. Banks are required to maintain "
            "a minimum Pillar 1 Capital to Risk-weighted Assets Ratio (CRAR) "
            "of 9% on an ongoing basis (other than capital conservation "
            "buffer and countercyclical capital buffer etc.). The Reserve "
            "Bank will take into account the relevant risk factors and the "
            "internal capital adequacy assessments of each bank to ensure "
            "that the capital held by a bank is commensurate with the bank's "
            "overall risk profile. In addition to the minimum total CRAR of "
            "9%, banks are required to maintain a Capital Conservation "
            "Buffer (CCB) of 2.5% of RWAs in the form of Common Equity Tier "
            "1 capital, taking the total minimum capital requirement to "
            "11.5% of RWAs. The minimum Common Equity Tier 1 (CET1) ratio "
            "shall be 5.5% of RWAs and the minimum Tier 1 capital ratio "
            "shall be 7% of RWAs."
        ),
        "gold": "9%",
        "alts": ["9 per cent", "nine percent", "9 percent", "9.0%"],
    },
]


# ---------- Logging helpers ----------
def info(msg):
    print(f"{C_DIM}[demo] {msg}{C_RESET}", flush=True)


def fatal(msg):
    print(f"{C_RED}{C_BRIGHT}[FATAL] {msg}{C_RESET}", file=sys.stderr, flush=True)
    sys.exit(2)


# ---------- Lazy heavy imports ----------
def load_model_stack():
    info("Importing torch / transformers / peft ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_dir = ADAPTER_BASE / ADAPTER_NAME / "best"
    if not adapter_dir.exists():
        adapter_dir = ADAPTER_BASE / ADAPTER_NAME / "final"
    if not adapter_dir.exists():
        fatal(
            f"Adapter not yet trained -- expected at "
            f"{ADAPTER_BASE / ADAPTER_NAME}/(best|final). "
            "Re-run after training completes."
        )

    info(f"Loading tokenizer from {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    info("Loading Nemotron-30B in 4-bit (nf4 + double quant, bf16 compute) ...")
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

    info(f"Attaching adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model, str(adapter_dir),
        adapter_name=ADAPTER_NAME, is_trainable=False,
    )
    model.eval()
    info(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return torch, tok, base_model, model


def get_hybrid_cache(torch, base_model, model, batch_size=1):
    """Build the HybridMambaAttentionDynamicCache that Nemotron requires."""
    model_module = sys.modules[base_model.__class__.__module__]
    HybridCache = getattr(model_module, "HybridMambaAttentionDynamicCache")
    return HybridCache(
        base_model.config,
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=model.device,
    )


# ---------- Generation ----------
def build_prompt(tok, question, context):
    if context:
        user = (
            f"REGULATORY CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
    else:
        user = f"QUESTION: {question}\n\nANSWER:"
    msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_once(torch, tok, base_model, model, prompt_text, use_adapter):
    inputs = tok(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=MAX_INPUT_TOKENS,
    ).to(model.device)
    cache = get_hybrid_cache(torch, base_model, model,
                             batch_size=inputs["input_ids"].shape[0])
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
        past_key_values=cache,
    )
    t0 = time.time()
    if use_adapter:
        model.set_adapter(ADAPTER_NAME)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    else:
        with model.disable_adapter():
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0
    decoded = tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return decoded, elapsed


# ---------- Scoring ----------
def is_grounded(answer, gold, alts):
    if not answer:
        return False
    a = answer.lower()
    if gold.lower() in a:
        return True
    return any(alt.lower() in a for alt in alts)


def mark(passed):
    return f"{C_GREEN}{CHECK}{C_RESET}" if passed else f"{C_RED}{CROSS}{C_RESET}"


# ---------- Display ----------
def render(question, context, base_ans, base_t, adapter_ans, adapter_t,
           gold=None, alts=None, show_base=True):
    print()
    print(f"{C_CYAN}{BAR}{C_RESET}")
    print(f"  {C_BRIGHT}Q:{C_RESET} {question}")
    if context:
        snippet = context[:100].replace("\n", " ")
        print(f"  {C_DIM}Context:{C_RESET} {snippet}...")
    print(f"{C_CYAN}{BAR}{C_RESET}\n")

    if show_base:
        print(f"  {C_YELLOW}{C_BRIGHT}[BASE]{C_RESET}   "
              f"{C_DIM}elapsed {base_t:.2f}s{C_RESET}")
        print(f"  {base_ans}\n")

    print(f"  {C_MAGENTA}{C_BRIGHT}[+BFSI]{C_RESET}  "
          f"{C_DIM}elapsed {adapter_t:.2f}s{C_RESET}")
    print(f"  {adapter_ans}\n")

    if gold is not None:
        print(f"  {C_DIM}Gold:{C_RESET} {gold}")
        if show_base:
            base_pass = is_grounded(base_ans, gold, alts or [])
            adapter_pass = is_grounded(adapter_ans, gold, alts or [])
            print(f"  Match:  BASE={mark(base_pass)}   "
                  f"ADAPTER={mark(adapter_pass)}")
        else:
            adapter_pass = is_grounded(adapter_ans, gold, alts or [])
            print(f"  Match:  ADAPTER={mark(adapter_pass)}")
    print()


def banner():
    print()
    print(f"{C_CYAN}{C_BRIGHT}" + "#" * 72 + C_RESET)
    print(f"{C_CYAN}{C_BRIGHT}#  Synapta BFSI Adapter Live Demo "
          f"(Nemotron-30B + bfsi_extract LoRA){C_RESET}")
    print(f"{C_CYAN}{C_BRIGHT}#  " +
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + C_RESET)
    print(f"{C_CYAN}{C_BRIGHT}" + "#" * 72 + C_RESET)


# ---------- Context loaders ----------
def read_context_file(path):
    p = Path(path)
    if not p.exists():
        fatal(f"Context file not found: {path}")
    if p.suffix.lower() == ".pdf":
        try:
            import PyPDF2
        except Exception:
            fatal("PDF context requires PyPDF2 -- pip install PyPDF2")
        reader = PyPDF2.PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


# ---------- Modes ----------
def run_scripted(torch, tok, base_model, model, show_base):
    banner()
    info(f"Scripted mode: {len(DEMO_QUESTIONS)} hand-picked questions.")
    base_passes = adapter_passes = 0
    for i, item in enumerate(DEMO_QUESTIONS, 1):
        info(f"-- {i}/{len(DEMO_QUESTIONS)}  id={item['id']}")
        prompt = build_prompt(tok, item["question"], item["context"])
        if show_base:
            base_ans, base_t = generate_once(
                torch, tok, base_model, model, prompt, use_adapter=False)
        else:
            base_ans, base_t = "(skipped)", 0.0
        adapter_ans, adapter_t = generate_once(
            torch, tok, base_model, model, prompt, use_adapter=True)
        render(
            item["question"], item["context"],
            base_ans, base_t, adapter_ans, adapter_t,
            gold=item["gold"], alts=item["alts"], show_base=show_base,
        )
        if show_base and is_grounded(base_ans, item["gold"], item["alts"]):
            base_passes += 1
        if is_grounded(adapter_ans, item["gold"], item["alts"]):
            adapter_passes += 1
        torch.cuda.empty_cache()
    n = len(DEMO_QUESTIONS)
    print(f"{C_CYAN}{BAR}{C_RESET}")
    if show_base:
        print(f"  {C_BRIGHT}Final tally:{C_RESET}  "
              f"BASE {base_passes}/{n}    ADAPTER {adapter_passes}/{n}")
    else:
        print(f"  {C_BRIGHT}Final tally:{C_RESET}  ADAPTER {adapter_passes}/{n}")
    print(f"{C_CYAN}{BAR}{C_RESET}\n")


def run_interactive(torch, tok, base_model, model, show_base,
                    one_shot_q=None, context_path=None):
    banner()
    if one_shot_q is not None:
        ctx = read_context_file(context_path) if context_path else ""
        prompt = build_prompt(tok, one_shot_q, ctx)
        if show_base:
            base_ans, base_t = generate_once(
                torch, tok, base_model, model, prompt, use_adapter=False)
        else:
            base_ans, base_t = "(skipped)", 0.0
        adapter_ans, adapter_t = generate_once(
            torch, tok, base_model, model, prompt, use_adapter=True)
        render(one_shot_q, ctx, base_ans, base_t, adapter_ans, adapter_t,
               show_base=show_base)
        return

    info("Interactive REPL.  Type 'quit' or Ctrl-D to exit.")
    info("Optional: prefix a line with 'context:' to set/replace the context.")
    info("Optional: 'load <path>' to load context from a .txt or .pdf file.")
    ctx = ""
    while True:
        try:
            line = input(f"{C_CYAN}question> {C_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", ":q"}:
            break
        if line.lower().startswith("context:"):
            ctx = line[len("context:"):].strip()
            info(f"Context set ({len(ctx)} chars).")
            continue
        if line.lower().startswith("load "):
            ctx = read_context_file(line[5:].strip())
            info(f"Context loaded ({len(ctx)} chars).")
            continue
        prompt = build_prompt(tok, line, ctx)
        if show_base:
            base_ans, base_t = generate_once(
                torch, tok, base_model, model, prompt, use_adapter=False)
        else:
            base_ans, base_t = "(skipped)", 0.0
        adapter_ans, adapter_t = generate_once(
            torch, tok, base_model, model, prompt, use_adapter=True)
        render(line, ctx, base_ans, base_t, adapter_ans, adapter_t,
               show_base=show_base)
        torch.cuda.empty_cache()


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="BFSI extract adapter side-by-side demo.")
    p.add_argument("--mode", choices=["interactive", "scripted"],
                   default="scripted",
                   help="scripted (default) or interactive REPL")
    p.add_argument("--question", default=None,
                   help="One-shot question for interactive mode (skips REPL)")
    p.add_argument("--context-pdf", default=None,
                   help="Optional context file (.pdf or .txt) for the question")
    p.add_argument("--no-base", action="store_true",
                   help="Skip the base-model comparison generation")
    return p.parse_args()


def main():
    args = parse_args()
    show_base = not args.no_base

    # Pre-flight check before loading 30B weights.
    adapter_dir = ADAPTER_BASE / ADAPTER_NAME
    if not (adapter_dir / "best").exists() and not (adapter_dir / "final").exists():
        fatal(
            f"Adapter not yet trained -- expected at "
            f"{adapter_dir}/(best|final). Re-run after training completes."
        )

    torch, tok, base_model, model = load_model_stack()

    if args.mode == "scripted":
        run_scripted(torch, tok, base_model, model, show_base)
    else:
        run_interactive(
            torch, tok, base_model, model, show_base,
            one_shot_q=args.question, context_path=args.context_pdf,
        )


if __name__ == "__main__":
    main()
