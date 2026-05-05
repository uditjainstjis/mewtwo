# YC Submission Checklist — 4 things only Udit can do

**Status: every artifact is built. Submission is the bottleneck. Do these 4 in order.**

---

## ⏱ Time required: ~2 hours total

## ✅ Step 1 — Record the Loom demo (~30 min)

The wine-glass-on-tank moment. The single highest-leverage thing in this whole project.

### Setup (5 min)
```bash
cd /home/learner/Desktop/mewtwo
source .venv/bin/activate

# Verify GPU is free (no training running)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
# If > 5000 MB, wait or kill background processes

# Launch the live demo (loads Nemotron-30B + adapter, ~30 sec cold start)
python synapta_src/demo/synapta_live_demo.py
# UI opens at http://localhost:7860
```

### Recording (15 min)
Open Loom (loom.com or desktop app). Use the script at `docs/recent/DEMO_VOICEOVER_BFSI.md` for the voiceover. Three takes max — ship the third.

**Cover these 5 questions in 90 seconds**:
1. "Click 'ATM charge ceiling' example, ask Synapta, show answer Rs. 21 with citation"
2. "Click 'Fraud reporting timeline', show 21 days answer"
3. "Click 'KYC paragraph reference', show paragraph 16"
4. "Type your own question — paste any RBI/SEBI clause, ask anything"
5. "Toggle base-model-comparison checkbox, show Synapta + cited answer vs base hedging"

**Voiceover hook** (open with this exact line):
> "This is Synapta. One $2k GPU. Trained yesterday on 130 RBI and SEBI Master Directions. Ask it any compliance question — the answer comes back with a paragraph citation in under 5 seconds. Watch."

### Upload (5 min)
- Loom → "Get share link"
- Set to "Public" (no login required)
- Copy the URL — you'll paste it in Step 3

---

## ✅ Step 2 — Push to public GitHub (~15 min)

Public reproducibility = YC technical-credibility signal. Repository must be public before Step 3.

### Commands
```bash
cd /home/learner/Desktop/mewtwo

# Verify clean state
git status
# Should show "nothing to commit, working tree clean" — if not, decide what to commit/discard

# Create the GitHub repo via web UI first:
# Go to https://github.com/new
# Name: synapta
# Description: "Sovereign AI for Indian BFSI compliance — adapter + methodology"
# Public
# DO NOT initialize with README (we already have one)
# Click "Create repository"

# GitHub gives you commands. Use these (replace USERNAME):
git remote add origin git@github.com:YOUR_USERNAME/synapta.git
git branch -M main
git push -u origin main
```

### What's in the repo (verify before pushing if you care)
- `README.md` — polished, headline-numbers-first
- `synapta_src/` — all pipeline scripts
- `data/rbi_corpus/` — manifests + extracted text + chunks + train/eval QA (115 MB raw PDFs may be too big — `.gitignore` should handle this)
- `adapters/nemotron_30b/bfsi_extract/best/` — the 1.74 GB adapter (LFS or omit)
- `docs/recent/` — all the briefs

### .gitignore caveat
Large files (115 MB PDFs, 1.74 GB adapter weights) need either git-lfs or `.gitignore` exclusion. Quick option:
```bash
cat >> .gitignore << 'EOF'
data/*/pdfs/
adapters/nemotron_30b/*/best/adapter_model.safetensors
adapters/nemotron_30b/*/training/
*.log
.venv/
__pycache__/
EOF
git add .gitignore
git commit -m "ignore large artifacts"
git push
```

---

## ✅ Step 3 — Submit YC application (~45 min)

### Source of truth
`docs/recent/YC_APPLICATION_DRAFT.md` — V3, integrated with competitive landscape + drift answer + methodology hook.

### Read it once before pasting
The application leads with: *"the only Indian BFSI AI stack with auditor-grade methodology — McNemar p < 10⁻⁴⁸ on 664 paired held-out RBI/SEBI questions across all 3 inference modes (base, +adapter, Format Guard routing) where the entire vendor surface ships case-study marketing numbers."*

### Field-by-field paste guide
1. **Company name**: Synapta
2. **Company URL**: github.com/YOUR_USERNAME/synapta (from Step 2)
3. **What does your company do?**: paste from YC_APPLICATION_DRAFT.md "What you're doing" section
4. **Founders**: Udit Jain (solo, 19, Bangalore)
5. **Where do you live?**: Bangalore, India
6. **Demo video URL**: paste Loom URL from Step 1
7. **Why this idea?**: paste from YC_APPLICATION_DRAFT.md "Why us" section
8. **What's new about your approach?**: paste the methodology-empty-space paragraph
9. **Competition**: paste the OnFinance + Signzy + IDfy + Big-4 paragraph
10. **Risks**: paste the 5-risk section (regulatory drift is Risk #1, with auto-retrain as the answer)
11. **Anything else?**: include the demo URL again + "Live demo runs on a single $2k GPU. The benchmark numbers are in the appendix; the only test that matters is the one you run yourself."

### Final check before submit
- ✅ Demo URL works (open it in incognito)
- ✅ GitHub repo loads in incognito
- ✅ Application copy < character limits (YC fields have specific limits — trim if needed)

### Click submit. The deadline is today.

---

## ✅ Step 4 — Tweet the result (~10 min, do AFTER submission, ideally 9am IST tomorrow May 5)

### Why tweet?
Public proof signal. Some YC partners check applicant Twitter activity before interviews.

### Source
`docs/recent/TWEET_THREAD.md` — 8 tweets, all ≤280 chars, posting instructions included.

### Posting timing
- Best: 9am IST May 5 (catches Indian and US morning Twitter overlap)
- Acceptable: any time after YC submission

### Tag
- @ycombinator
- @paulg (if you can stomach it — high upside, low downside)
- Include the Loom URL in tweet 7 or 8

### After thread
- Pin tweet 1 to your profile
- Reply to your own thread with the GitHub repo URL + Loom link
- Retweet your own tweet 1 four hours later (for US west-coast timezone)

---

## What happens AFTER submission

You wait. YC W26 decisions usually come within 2-4 weeks of deadline. While you wait:
- Cold-email 5 mid-tier Indian BFSI institutions (Federal Bank, Bandhan, AU Small Finance Bank, RBL, Equitas) using `BFSI_PIPELINE_NARRATIVE.md` as email body. Even one warm reply changes your YC interview chances meaningfully.
- Follow up with the YC alum from your earlier meeting (skip the deck/report I drafted — they were poor crafts. Just a 2-line "submitted, here's the demo URL" note).
- Don't tweak the application after submission. Don't keep training adapters. Sleep.

---

## What I (Claude) am doing while you do these 4 things

- ⏳ bfsi_recall training (~50 min remaining as of writing)
- 🎯 Building the Indian BFSI Benchmark v1 (60 questions, gated)
- 📝 Adding F1-gap honesty paragraph to YC app
- 🔄 If GPU frees, attempting FG eval mode auto-chain

Everything I produce will be on disk by the time you finish Step 4.

**Now go record the Loom.** That's the single highest-leverage minute of work in this entire project.
