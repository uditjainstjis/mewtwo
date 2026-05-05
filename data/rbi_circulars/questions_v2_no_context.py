"""BFSI v2 questions — NO CONTEXT (recall test).

Tests whether the model has RBI/SEBI/IRDAI domain knowledge from pretraining.
Unlike v1 (where the regulation was provided as context), v2 asks the model
to recall the rule. This is the MORE REALISTIC test — a compliance officer
typically asks "what does RBI say about X?" without pasting the regulation.

If Format Guard helps here over base, that's a real differentiating signal
for the YC pitch: "Our routing helps the model recall and apply BFSI domain
knowledge better than vanilla."

Each question:
  - id: stable identifier
  - question: just the question text
  - gold_answer: the canonical answer
  - alternatives: substring/term matches that count
  - scoring: 'contains' / 'multi_term'
  - source: which regulation it's testing (for our reference)
"""

QUESTIONS_V2 = [
    # ============== RBI questions (no context) ==============
    {
        "id": "v2_rbi_01",
        "question": "Under RBI KYC norms, how often must a high-risk customer's KYC be updated?",
        "gold_answer": "two years",
        "alternatives": ["two years", "2 years", "every two", "every 2 years"],
        "scoring": "contains",
        "source": "RBI KYC Master Direction",
    },
    {
        "id": "v2_rbi_02",
        "question": "What is the maximum balance allowed in an RBI Small Account at any point in time?",
        "gold_answer": "fifty thousand rupees",
        "alternatives": ["50,000", "50000", "fifty thousand", "Rs. 50,000", "₹50,000"],
        "scoring": "contains",
        "source": "RBI KYC Master Direction Section 23",
    },
    {
        "id": "v2_rbi_03",
        "question": "Within how many hours must an Indian bank report a cyber security incident to the Reserve Bank of India?",
        "gold_answer": "2 to 6 hours",
        "alternatives": ["2 to 6", "2-6 hours", "within 6 hours", "six hours", "two to six"],
        "scoring": "contains",
        "source": "RBI Cybersecurity Framework",
    },
    {
        "id": "v2_rbi_04",
        "question": "Under RBI Digital Lending guidelines, can a Lending Service Provider's app store biometric data on its servers?",
        "gold_answer": "no",
        "alternatives": ["no", "shall not", "cannot", "prohibited", "not allowed", "must not"],
        "scoring": "contains",
        "source": "RBI Digital Lending Guidelines Para 8.4",
    },
    {
        "id": "v2_rbi_05",
        "question": "What core management functions are Indian banks prohibited from outsourcing per RBI guidelines? Name at least 2.",
        "gold_answer": "internal audit, compliance, KYC decisions, loan sanction",
        "alternatives": ["internal audit", "compliance", "KYC", "loan sanction", "loans", "investment", "decision-making"],
        "scoring": "multi_term",
        "must_match_count": 2,
        "source": "RBI Outsourcing Master Direction",
    },
    {
        "id": "v2_rbi_06",
        "question": "What is the RBI cooling-off (look-up) period during which a borrower can exit a digital loan without penalty?",
        "gold_answer": "three days minimum",
        "alternatives": ["three days", "3 days", "minimum of three", "look-up period"],
        "scoring": "contains",
        "source": "RBI Digital Lending Guidelines Para 6.2",
    },
    {
        "id": "v2_rbi_07",
        "question": "How many free ATM transactions per month must Indian banks provide to their savings account customers at their own bank's ATMs?",
        "gold_answer": "five free transactions",
        "alternatives": ["five", "5 free", "5 transactions", "free of charge"],
        "scoring": "contains",
        "source": "RBI ATM charges circular",
    },
    {
        "id": "v2_rbi_08",
        "question": "What is the maximum charge per ATM transaction beyond the free monthly limit, per RBI rules?",
        "gold_answer": "Rs. 21",
        "alternatives": ["21", "Rs. 21", "Rs.21", "twenty-one", "₹21"],
        "scoring": "contains",
        "source": "RBI ATM charges circular",
    },
    {
        "id": "v2_rbi_09",
        "question": "What rank/seniority level must the Chief Information Security Officer of an Indian bank be, per RBI cybersecurity guidelines?",
        "gold_answer": "GM/DGM/CGM",
        "alternatives": ["GM", "DGM", "CGM", "general manager", "senior level"],
        "scoring": "contains",
        "source": "RBI Cybersecurity Framework",
    },
    {
        "id": "v2_rbi_10",
        "question": "Within how many working days must a failed ATM transaction be reversed to the customer's account, per RBI rules?",
        "gold_answer": "five working days",
        "alternatives": ["five", "5 working days", "5 days", "T+5"],
        "scoring": "contains",
        "source": "RBI Customer Protection circular",
    },

    # ============== SEBI questions (mutual funds, insider trading) ==============
    {
        "id": "v2_sebi_01",
        "question": "Under SEBI Insider Trading Regulations, who is considered an 'insider' in a listed company?",
        "gold_answer": "person with access to unpublished price-sensitive information (UPSI), connected persons, designated persons",
        "alternatives": ["unpublished price-sensitive", "UPSI", "connected person", "designated person", "directors", "officers"],
        "scoring": "multi_term",
        "must_match_count": 2,
        "source": "SEBI (Prohibition of Insider Trading) Regulations 2015",
    },
    {
        "id": "v2_sebi_02",
        "question": "Under SEBI mutual fund regulations, what is the maximum total expense ratio (TER) chargeable on an equity-oriented mutual fund scheme?",
        "gold_answer": "depends on AUM tier; first 500 crore at 2.25%",
        "alternatives": ["2.25", "2.5", "expense ratio", "AUM slab", "tiered"],
        "scoring": "contains",
        "source": "SEBI Mutual Fund Regulations",
    },
    {
        "id": "v2_sebi_03",
        "question": "What is the minimum subscription period for a New Fund Offer (NFO) of a closed-ended mutual fund scheme per SEBI rules?",
        "gold_answer": "minimum 3 days, maximum 15 days",
        "alternatives": ["3 days", "three days", "15 days", "fifteen days", "minimum 3"],
        "scoring": "contains",
        "source": "SEBI Mutual Fund Regulations",
    },
    {
        "id": "v2_sebi_04",
        "question": "Under SEBI LODR Regulations, within how many days of the close of a quarter must a listed company file its quarterly financial results?",
        "gold_answer": "45 days",
        "alternatives": ["45 days", "forty-five days", "within 45"],
        "scoring": "contains",
        "source": "SEBI LODR Regulation 33",
    },
    {
        "id": "v2_sebi_05",
        "question": "Per SEBI guidelines, what is the maximum permissible exposure of a mutual fund scheme to a single issuer's debt instruments?",
        "gold_answer": "10% of NAV (12% with trustee approval)",
        "alternatives": ["10%", "12%", "ten percent", "single issuer", "exposure limit"],
        "scoring": "contains",
        "source": "SEBI Mutual Fund Regulations Schedule VII",
    },

    # ============== IRDAI / Insurance questions ==============
    {
        "id": "v2_irda_01",
        "question": "Under IRDAI rules, what is the minimum solvency ratio required for life insurance companies in India?",
        "gold_answer": "150%",
        "alternatives": ["150", "150%", "1.5", "one hundred fifty"],
        "scoring": "contains",
        "source": "IRDAI Solvency Margin Regulations",
    },
    {
        "id": "v2_irda_02",
        "question": "What is the free-look period in days during which a customer can return a life insurance policy in India per IRDAI rules?",
        "gold_answer": "15 days (30 days for distance marketing)",
        "alternatives": ["15 days", "fifteen days", "30 days", "thirty days", "free-look"],
        "scoring": "contains",
        "source": "IRDAI Protection of Policyholders' Interests Regulations",
    },
    {
        "id": "v2_irda_03",
        "question": "Within how many days must an insurance claim under a health insurance policy be settled or rejected per IRDAI guidelines?",
        "gold_answer": "30 days from receipt of all documents",
        "alternatives": ["30 days", "thirty days", "from receipt", "all documents"],
        "scoring": "contains",
        "source": "IRDAI Health Insurance Regulations",
    },

    # ============== NBFC questions ==============
    {
        "id": "v2_nbfc_01",
        "question": "What is the minimum Net Owned Fund (NOF) required for an NBFC to commence operations in India per RBI rules?",
        "gold_answer": "10 crore (raised from 2 crore in October 2022)",
        "alternatives": ["10 crore", "ten crore", "Rs. 10 crore", "₹10 crore", "10,00,00,000"],
        "scoring": "contains",
        "source": "RBI Master Direction NBFC",
    },
    {
        "id": "v2_nbfc_02",
        "question": "What is the minimum Capital to Risk-weighted Assets Ratio (CRAR) for NBFCs in India per RBI?",
        "gold_answer": "15%",
        "alternatives": ["15%", "15 percent", "fifteen percent", "CRAR of 15"],
        "scoring": "contains",
        "source": "RBI NBFC Master Direction",
    },

    # ============== Cross-domain reasoning (harder) ==============
    {
        "id": "v2_cross_01",
        "question": "If an Indian bank outsources its IT operations and a cyber incident occurs at the service provider, who is responsible for reporting it to RBI within 2-6 hours?",
        "gold_answer": "the bank (regulated entity)",
        "alternatives": ["the bank", "bank is responsible", "regulated entity", "RE", "outsourcing does not"],
        "scoring": "contains",
        "source": "RBI Outsourcing + Cybersecurity Framework combined",
    },
    {
        "id": "v2_cross_02",
        "question": "An NBFC wants to launch a digital lending app. Per RBI rules, must the loan disbursal go through the LSP's account or directly to the borrower's account?",
        "gold_answer": "directly to the borrower",
        "alternatives": ["directly to the borrower", "borrower's account", "borrower's bank account", "not through LSP", "no pass-through"],
        "scoring": "contains",
        "source": "RBI Digital Lending Guidelines",
    },
    {
        "id": "v2_cross_03",
        "question": "A bank's small account customer wants to open a fixed deposit. What's the maximum FD they can open given the small account balance limit?",
        "gold_answer": "Rs. 50,000 (the balance ceiling for small accounts)",
        "alternatives": ["50,000", "fifty thousand", "Rs. 50,000", "balance limit"],
        "scoring": "contains",
        "source": "RBI KYC Section 23 (Small Accounts)",
    },

    # ============== Numeric extraction (high precision required) ==============
    {
        "id": "v2_num_01",
        "question": "What is the threshold above which Indian banks must report a Suspicious Transaction Report (STR) under PMLA rules?",
        "gold_answer": "no monetary threshold; STR based on suspicion regardless of amount",
        "alternatives": ["no threshold", "no monetary", "regardless of amount", "based on suspicion", "any amount"],
        "scoring": "contains",
        "source": "PMLA Rules + RBI KYC Master Direction",
    },
    {
        "id": "v2_num_02",
        "question": "What is the cash transaction reporting (CTR) threshold for Indian banks under PMLA?",
        "gold_answer": "Rs. 10 lakh",
        "alternatives": ["10 lakh", "ten lakh", "Rs. 10 lakh", "₹10 lakh", "10,00,000"],
        "scoring": "contains",
        "source": "PMLA Rules",
    },
]
