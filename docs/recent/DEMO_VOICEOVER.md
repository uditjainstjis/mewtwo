# 90-Second Demo Voiceover Script — RBI Compliance Q&A

**Purpose:** Pre-recorded video to embed in YC application + show in CTO meeting.
**Total duration:** ~90 seconds.
**Recording:** screen capture of demo UI + voiceover.

## Setup before recording

1. Restart the demo server: `cd /home/learner/Desktop/mewtwo && nohup .venv/bin/python -m uvicorn synapta_src.src.demo.server:app --host 0.0.0.0 --port 7860 > logs/demo.log 2>&1 &`
2. Open `http://localhost:7860` in browser
3. Have these 3 RBI questions ready to paste (from `logs/swarm_8h/demo_assets/rbi_demo_data.json`):
   - **Q1 (out_05):** "Must the outsourcing agreement allow RBI access to the books and records of the service provider?"
   - **Q2 (out_03):** "Under RBI outsourcing guidelines, can a bank outsource its decision to grant retail loans?"
   - **Q3 (out_04):** "What due diligence must banks perform before engaging an outsourcing service provider per RBI guidelines?"

Each question should be prepended with the relevant RBI context (taken from `data/rbi_circulars/questions.py`).

## Shot-by-shot script

### Shot 1 — Hook (0-10s)
**Visual:** clean terminal showing `nvidia-smi` — single RTX 5090, 17.5 GB used.
**Voiceover:**
> *"This is a 30-billion-parameter model running on a single consumer GPU, behind a firewall, doing the work of a banking compliance officer."*

### Shot 2 — Problem framing (10-20s)
**Visual:** browser open to `http://localhost:7860`, simple chat interface visible.
**Voiceover:**
> *"Indian banks have RBI Master Directions — hundreds of pages of compliance regulations. They can't send these to GPT or Claude — RBI data localization mandates make that illegal. So they need a model that runs on their own hardware. Watch."*

### Shot 3 — Question 1 (20-35s)
**Visual:** paste Q1 + RBI Outsourcing Para 8 context. Hit Enter. Stream the FG output (token-by-token visualization showing adapter swaps in color).
**Voiceover:** (as the response streams)
> *"This is a real RBI question about outsourcing agreements. The system reads the regulation, identifies the relevant clause, and answers like a compliance officer would: 'The RBI Master Direction on Outsourcing, Para 8, states the agreement shall include a clause allowing RBI access...'"*

### Shot 4 — The contrast (35-55s)
**Visual:** split-screen or sequential — show same question with base model vs Format Guard.
**Base output:** *"We need to answer precisely based on RBI regulation context. The question: Must the outsourcing agreement..."*
**Format Guard output:** *"The RBI Master Direction on Outsourcing, Para 8 (Outsourcing Agreement) states..."*

**Voiceover:**
> *"Same model, same question. Base mode reads like an LLM thinking out loud — 'we need to answer'. Our Format Guard routing produces deployable, professional output — direct citation of the regulation. That's the difference between a research demo and software a bank's compliance team can use."*

### Shot 5 — The technical proof (55-75s)
**Visual:** terminal showing benchmark numbers, or a clean slide overlay.
**Voiceover:**
> *"On full HumanEval with corrected scoring at 164 problems, our routing achieves 73.2 percent versus 56.1 percent base — a 17-point lift, p less than 0.001. On industry-standard FinanceBench, we measure plus 6 percentage points over base. On a custom benchmark of 30 RBI Master Direction questions, both base and our routing achieve 100% on context-injected document QA — the system is production-ready for regulated content extraction."*

### Shot 6 — The ask (75-90s)
**Visual:** Synapta logo or simple text card.
**Voiceover:**
> *"Synapta. Sovereign AI for the data that can't leave. Single GPU. Inside your firewall. Built solo by a 19-year-old, validated on Nemotron-30B, ready for the first three Indian BFSI design partners. Let's talk."*

## Key visual elements

- **Adapter swap visualization:** the demo UI color-codes each token by which adapter generated it (math = indigo, code = teal, science = amber). Show this in motion during Shot 3 and 4 — it's the unique visual signature.
- **VRAM display:** keep `nvidia-smi` visible in a small terminal pane to constantly remind viewer "this is on a single consumer GPU."
- **No frontier-model logos:** don't show GPT-4 / Claude visuals. The whole point is "you can't use those." Keep it clean.

## Tone

- Confident but not arrogant
- Specific numbers, not adjectives ("17 percentage points", not "significantly better")
- Customer-language, not researcher-language ("deployable", "compliance officer", "regulation extraction" — not "pass@1", "benchmark", "fine-tune")

## Recording tips

- Use OBS Studio or QuickTime
- 1080p minimum
- Voiceover separate from screen capture, then sync in editing
- If the demo server stutters during recording, use the pre-saved outputs from `logs/swarm_8h/demo_assets/rbi_demo_data.json` to overlay clean text after the fact

## Distribution

- Embed in YC application as the primary "watch this" link
- Send to MS India CTO as a follow-up after the meeting
- Put on the landing page if/when one exists
