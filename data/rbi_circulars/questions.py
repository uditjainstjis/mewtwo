"""Hand-curated BFSI compliance questions across 5 publicly-available RBI Master Directions.

Each question has:
  - id: stable identifier
  - circular: which RBI document it tests
  - question: what we ask the model
  - context: the relevant excerpt from the actual RBI circular (200-800 words)
  - gold_answer: what a compliance officer would consider correct
  - scoring: how to grade ('contains' / 'numeric_match' / 'multi_term')

Sources (verifiable):
  RBI Master Direction on KYC (DBR.AML.BC.No.81/14.01.001/2015-16, updated)
  RBI Master Direction on Digital Lending (RBI/2022-23/82, Sept 2022)
  RBI Master Direction on Outsourcing (DBR.No.BP.BC.76/21.04.158/2014-15)
  RBI Master Direction on Cybersecurity (DBS.CO.CSITE.BC.No.11/33.01.001/2015-16)
  RBI Master Direction on Banking Operations (RBI/DBOD/2015-16/13)
"""

QUESTIONS = [
    # ============== KYC Master Direction ==============
    {
        "id": "kyc_01",
        "circular": "kyc",
        "question": "Under the RBI KYC Master Direction, what is the maximum aggregate balance a Small Account holder is permitted to maintain at any point in time?",
        "context": """RBI Master Direction on KYC, Section 23 (Small Accounts):
A 'Small Account' means a savings bank account in which the aggregate of all credits in a financial year does not exceed Rupees one lakh, the aggregate of all withdrawals and transfers in a month does not exceed Rupees ten thousand, and the balance at any point of time does not exceed Rupees fifty thousand. Small Accounts shall be opened only at Core Banking Solution (CBS) linked branches. The threshold of Rupees fifty thousand for balance, Rupees one lakh for credits, and Rupees ten thousand per month for withdrawals shall be subject to such enhancements as RBI may prescribe.""",
        "gold_answer": "fifty thousand",
        "scoring": "contains",
        "alternatives": ["50,000", "50000", "Rs. 50,000", "₹50,000", "Rupees fifty thousand"],
    },
    {
        "id": "kyc_02",
        "circular": "kyc",
        "question": "When a customer is classified as 'high risk' under RBI KYC norms, how frequently must their KYC be updated?",
        "context": """RBI Master Direction on KYC, Section 38 (Periodic Updation of KYC):
Periodic updation of KYC shall be carried out at least once in every two years for high risk customers, once in every eight years for medium risk customers and once in every ten years for low risk customers as per the time intervals specified by RBI. The policy shall be approved by the Board. In case of accounts of customers categorized as high risk, banks shall conduct enhanced due diligence and continuous monitoring on an ongoing basis.""",
        "gold_answer": "two years",
        "scoring": "contains",
        "alternatives": ["2 years", "every 2 years", "every two years", "once in two years"],
    },
    {
        "id": "kyc_03",
        "circular": "kyc",
        "question": "What documents are acceptable as Officially Valid Documents (OVDs) for individual KYC under RBI norms? List them.",
        "context": """RBI Master Direction on KYC, Section 3 (Definitions):
"Officially Valid Document" (OVD) means the passport, the driving licence, proof of possession of Aadhaar number, the Voter's Identity Card issued by the Election Commission of India, job card issued by NREGA duly signed by an officer of the State Government, and the letter issued by the National Population Register containing details of name and address. Provided that where the OVD furnished by the customer does not have the updated address, certain other documents shall be deemed to be OVDs for the limited purpose of proof of address.""",
        "gold_answer": "passport, driving licence, Aadhaar, Voter ID, NREGA job card, NPR letter",
        "scoring": "multi_term",
        "alternatives": ["passport", "driving licence", "driving license", "Aadhaar", "voter", "NREGA", "NPR"],
        "must_match_count": 4,
    },
    {
        "id": "kyc_04",
        "circular": "kyc",
        "question": "Under RBI KYC norms, what is the maximum monthly withdrawal limit for a Small Account?",
        "context": """RBI Master Direction on KYC, Section 23: A 'Small Account' means a savings bank account in which the aggregate of all credits in a financial year does not exceed Rupees one lakh, the aggregate of all withdrawals and transfers in a month does not exceed Rupees ten thousand, and the balance at any point of time does not exceed Rupees fifty thousand.""",
        "gold_answer": "ten thousand",
        "scoring": "contains",
        "alternatives": ["10,000", "10000", "Rs. 10,000", "₹10,000", "Rupees ten thousand"],
    },
    {
        "id": "kyc_05",
        "circular": "kyc",
        "question": "What enhanced due diligence is required for Politically Exposed Persons (PEPs) under RBI KYC?",
        "context": """RBI Master Direction on KYC, Section 41 (Politically Exposed Persons):
In respect of accounts of Politically Exposed Persons (PEPs) — defined as individuals who are or have been entrusted with prominent public functions in a foreign country, e.g., Heads of State/Governments, senior politicians, senior government/judicial/military officers, senior executives of state-owned corporations and important political party officials — banks shall: (a) Have appropriate risk management systems to determine whether a customer is a PEP, (b) Obtain senior management approval for establishing/continuing business relationship with a PEP, (c) Take reasonable measures to establish source of funds/wealth, (d) Conduct enhanced ongoing monitoring of the relationship.""",
        "gold_answer": "senior management approval, source of funds verification, enhanced ongoing monitoring",
        "scoring": "multi_term",
        "alternatives": ["senior management", "approval", "source of funds", "source of wealth", "enhanced", "ongoing monitoring"],
        "must_match_count": 3,
    },

    # ============== Digital Lending Master Direction ==============
    {
        "id": "digi_01",
        "circular": "digital_lending",
        "question": "Under RBI Digital Lending guidelines (Sept 2022), can loan disbursement be done directly to a third-party account at the lender's discretion?",
        "context": """RBI Digital Lending Guidelines, Para 5.1:
All loan disbursals and repayments shall be executed only between the bank accounts of the borrower and the Regulated Entity (RE) without any pass-through/pool account of the Lending Service Provider (LSP) or any third party. The disbursements shall always be made into the bank account of the borrower, except for disbursals covered exclusively under statutory or regulatory mandate, flow of money between Regulated Entities for co-lending transactions and disbursals where loans are mandated for specific end use such as direct payment to merchants.""",
        "gold_answer": "no, must be to borrower account",
        "scoring": "contains",
        "alternatives": ["no", "borrower's bank account", "directly to borrower", "not allowed", "prohibited", "shall not"],
    },
    {
        "id": "digi_02",
        "circular": "digital_lending",
        "question": "What is the cooling-off period under RBI Digital Lending guidelines during which a borrower can exit a loan without penalty?",
        "context": """RBI Digital Lending Guidelines, Para 6.2 (Cooling-off period):
The borrower shall be given an explicit option to exit digital loan by paying the principal and the proportionate APR without any penalty during the look-up period. The look-up period shall be determined by the Board of the Regulated Entity, subject to a minimum period of three days for loans having tenor of seven days or more, and one day for loans with tenor of less than seven days.""",
        "gold_answer": "three days minimum (one day for short loans)",
        "scoring": "contains",
        "alternatives": ["three days", "3 days", "minimum of three", "look-up period"],
    },
    {
        "id": "digi_03",
        "circular": "digital_lending",
        "question": "Are Lending Service Providers permitted to access the borrower's mobile phone resources such as files, contact list, call logs, and telephony functions under RBI Digital Lending guidelines?",
        "context": """RBI Digital Lending Guidelines, Para 8.4 (Data Privacy):
The Digital Lending Apps (DLAs) shall desist from accessing mobile phone resources such as file and media, contact list, call logs, telephony functions, etc. A one-time access can be taken for camera, microphone, location or any other facility necessary for the purpose of on-boarding/KYC requirements only, with explicit consent of the borrower. The Regulated Entity shall ensure that DLAs do not store biometric data on their servers.""",
        "gold_answer": "no, prohibited",
        "scoring": "contains",
        "alternatives": ["no", "shall desist", "not allowed", "prohibited", "should not", "must not"],
    },
    {
        "id": "digi_04",
        "circular": "digital_lending",
        "question": "Under RBI Digital Lending guidelines, who is responsible for grievance redressal when a borrower has a complaint about a Lending Service Provider?",
        "context": """RBI Digital Lending Guidelines, Para 9.1 (Grievance Redressal):
The Regulated Entity (RE) shall ensure that they and the Lending Service Provider (LSP) engaged by them shall have a suitable nodal grievance redressal officer to deal with FinTech / digital lending related complaints. Such grievance redressal officer shall also deal with complaints against their respective DLAs. The contact details of grievance redressal officer shall be prominently displayed on the website of the RE/LSP, on its DLA, and also in the Key Fact Statement provided to the borrower. If any complaint lodged by the borrower against the RE or the LSP engaged by the RE is not resolved by the RE within the stipulated period of 30 days, the borrower can lodge a complaint over the Complaint Management System portal under the Reserve Bank Integrated Ombudsman Scheme.""",
        "gold_answer": "regulated entity (the bank/NBFC)",
        "scoring": "contains",
        "alternatives": ["regulated entity", "RE", "bank", "NBFC", "responsibility of the regulated entity"],
    },
    {
        "id": "digi_05",
        "circular": "digital_lending",
        "question": "What is the time limit within which an RBI Digital Lending complaint must be resolved before the borrower can escalate to the RBI Ombudsman?",
        "context": """Same as digi_04 above: 'If any complaint lodged by the borrower against the RE or the LSP engaged by the RE is not resolved by the RE within the stipulated period of 30 days, the borrower can lodge a complaint over the Complaint Management System portal under the Reserve Bank Integrated Ombudsman Scheme.'""",
        "gold_answer": "30 days",
        "scoring": "contains",
        "alternatives": ["30 days", "thirty days", "stipulated period of 30"],
    },

    # ============== Outsourcing Master Direction ==============
    {
        "id": "out_01",
        "circular": "outsourcing",
        "question": "Under RBI's Outsourcing Master Direction, what core management functions cannot be outsourced by a bank?",
        "context": """RBI Master Direction on Outsourcing of Financial Services, Para 5 (Core Management Functions):
Banks shall not outsource core management functions including Internal Audit, Compliance function and decision-making functions such as determining compliance with KYC norms for opening deposit accounts, according sanction for loans (including retail loans) and management of investment portfolio. However, for special cases like outsourcing of internal audit by foreign banks operating in India and the outsourcing of risk management functions by entities forming part of a financial conglomerate, RBI may consider granting case-by-case permissions.""",
        "gold_answer": "internal audit, compliance, KYC decision-making, loan sanction, investment management",
        "scoring": "multi_term",
        "alternatives": ["internal audit", "compliance", "KYC", "loan sanction", "investment", "decision-making"],
        "must_match_count": 3,
    },
    {
        "id": "out_02",
        "circular": "outsourcing",
        "question": "What is the bank's responsibility regarding the actions of its outsourcing service provider?",
        "context": """RBI Master Direction on Outsourcing, Para 4 (Responsibilities of the Bank):
The outsourcing of any activity by the bank does not diminish its obligations, and those of its Board and senior management, who have the ultimate responsibility for the outsourced activity. Banks would, therefore, be responsible for the actions of their service provider including Direct Sales Agents/ Direct Marketing Agents/ Recovery Agents and the confidentiality of information pertaining to the customers that is available with the service provider. Banks shall retain ultimate control of the outsourced activity.""",
        "gold_answer": "bank retains ultimate responsibility for service provider's actions",
        "scoring": "multi_term",
        "alternatives": ["ultimate responsibility", "bank is responsible", "ultimate control", "actions of service provider", "obligations not diminished"],
        "must_match_count": 2,
    },
    {
        "id": "out_03",
        "circular": "outsourcing",
        "question": "Under RBI outsourcing guidelines, can a bank outsource its decision to grant retail loans?",
        "context": """Same as out_01 above: 'Banks shall not outsource core management functions including ... according sanction for loans (including retail loans)...'""",
        "gold_answer": "no",
        "scoring": "contains",
        "alternatives": ["no", "cannot", "shall not", "not allowed", "prohibited"],
    },
    {
        "id": "out_04",
        "circular": "outsourcing",
        "question": "What due diligence must banks perform before engaging an outsourcing service provider per RBI guidelines?",
        "context": """RBI Master Direction on Outsourcing, Para 7 (Due Diligence):
The bank shall undertake appropriate due diligence in selecting the third party (service provider). The structure of the activity, technical and financial competence of the service provider, business reputation and culture of the service provider, conformance with the provisions of the laws of India and review of any past performance ratings or independent reviews of the service provider, must form a key component of the due diligence process. Due diligence shall be conducted prior to selection and at periodic intervals during the engagement.""",
        "gold_answer": "financial competence, technical competence, business reputation, legal compliance, past performance",
        "scoring": "multi_term",
        "alternatives": ["financial", "technical", "competence", "reputation", "legal", "compliance", "past performance"],
        "must_match_count": 3,
    },
    {
        "id": "out_05",
        "circular": "outsourcing",
        "question": "Must the outsourcing agreement allow RBI access to the books and records of the service provider?",
        "context": """RBI Master Direction on Outsourcing, Para 8 (Outsourcing Agreement):
The agreement shall provide for continuous monitoring and assessment by the bank of the service provider so that any necessary corrective measure can be taken immediately. The agreement shall also include a clause allowing the RBI or persons authorised by it to access the bank's documents, records of transactions, and other necessary information given to, stored or processed by the service provider within a reasonable time. The agreement should also include termination clauses, including the bank's right to terminate the contract.""",
        "gold_answer": "yes",
        "scoring": "contains",
        "alternatives": ["yes", "must allow", "shall allow", "RBI access", "agreement shall include"],
    },

    # ============== Cybersecurity Master Direction ==============
    {
        "id": "cyber_01",
        "circular": "cybersecurity",
        "question": "Under RBI's Cybersecurity Framework, within what time must banks report a cyber incident to the RBI?",
        "context": """RBI Cybersecurity Framework, Annex 3 (Reporting):
Banks shall report all unusual cyber-security incidents (whether successful or attempts) to the Reserve Bank of India (Department of Banking Supervision) within 2 to 6 hours of detection. The bank shall also report on the steps taken/plans for containment and the impact assessment. Detailed report on the incident along with the corrective actions taken shall be furnished within 24 hours.""",
        "gold_answer": "2 to 6 hours",
        "scoring": "contains",
        "alternatives": ["2 to 6 hours", "2-6 hours", "within 6 hours", "within 2 hours", "two to six hours"],
    },
    {
        "id": "cyber_02",
        "circular": "cybersecurity",
        "question": "Are banks required under the RBI Cybersecurity Framework to have a Board-approved Cyber Crisis Management Plan?",
        "context": """RBI Cybersecurity Framework, Para 1 (Cyber Security Policy):
Banks should immediately put in place a cyber-security policy elucidating the strategy containing an appropriate approach to combat cyber threats given the level of complexity of business and acceptable levels of risk, duly approved by their Board. A Cyber Crisis Management Plan (CCMP) should be immediately evolved and should be a part of the overall Board approved strategy. CERT-IN guidelines should be referred to in this regard.""",
        "gold_answer": "yes, board-approved",
        "scoring": "contains",
        "alternatives": ["yes", "board approved", "Board approved", "approved by their Board", "CCMP"],
    },
    {
        "id": "cyber_03",
        "circular": "cybersecurity",
        "question": "What is the role of the Chief Information Security Officer (CISO) under RBI cybersecurity guidelines?",
        "context": """RBI Cybersecurity Framework, Para 5 (Organisational Arrangements):
Banks should review the organisational arrangements so that the security concerns are appreciated, receive adequate attention and get escalated to appropriate levels in the hierarchy to enable quick action. A senior level officer of the rank of GM/DGM/CGM, etc. may be designated as Chief Information Security Officer (CISO) responsible for articulating and enforcing the policies that bank uses to protect their information assets apart from coordinating the security related issues / implementation within the organization as well as relevant external agencies.""",
        "gold_answer": "articulating and enforcing information security policies, coordinating security implementation",
        "scoring": "multi_term",
        "alternatives": ["articulating", "enforcing", "policies", "protect", "information assets", "coordinating", "security"],
        "must_match_count": 3,
    },
    {
        "id": "cyber_04",
        "circular": "cybersecurity",
        "question": "What seniority level must the CISO hold per RBI cybersecurity guidelines?",
        "context": """Same as cyber_03 above: 'A senior level officer of the rank of GM/DGM/CGM, etc. may be designated as Chief Information Security Officer (CISO)...'""",
        "gold_answer": "GM/DGM/CGM",
        "scoring": "contains",
        "alternatives": ["GM", "DGM", "CGM", "General Manager", "senior level"],
    },
    {
        "id": "cyber_05",
        "circular": "cybersecurity",
        "question": "Within what time must a detailed cyber incident report (with corrective actions) be submitted to RBI after an incident?",
        "context": """Same as cyber_01: 'Detailed report on the incident along with the corrective actions taken shall be furnished within 24 hours.'""",
        "gold_answer": "24 hours",
        "scoring": "contains",
        "alternatives": ["24 hours", "twenty-four hours", "within 24"],
    },

    # ============== Banking Operations Master Direction ==============
    {
        "id": "bank_01",
        "circular": "banking_ops",
        "question": "Under RBI Banking Operations guidelines, what is the maximum cash withdrawal limit per day from an ATM by a debit cardholder for an account-holding customer of the same bank (RBI default ceiling)?",
        "context": """RBI Master Direction on Banking Operations:
The cash withdrawal limit per day from ATMs by debit card holders is set by individual banks, subject to RBI's overall guidelines. Banks may permit cash withdrawal of up to Rs.10,000 per transaction at ATMs of the same bank for own customers. For non-customers (interoperable transactions at other banks' ATMs), the limit per transaction is Rs.10,000. The daily cumulative limit shall be set by the issuing bank within the framework of risk management. Banks shall also notify the customer regarding the daily withdrawal limit applicable on the card.""",
        "gold_answer": "10,000 per transaction",
        "scoring": "contains",
        "alternatives": ["10,000", "10000", "Rs. 10,000", "ten thousand"],
    },
    {
        "id": "bank_02",
        "circular": "banking_ops",
        "question": "Per RBI guidelines, how many free ATM transactions per month must banks provide to their savings account customers at their own bank's ATMs?",
        "context": """RBI Master Direction on ATM Charges:
Customers are eligible for five free transactions (financial and non-financial) at the ATMs of the bank where they hold their accounts every month. They are also eligible for free transactions (inclusive of financial and non-financial transactions) at any other bank's ATMs — three transactions in metro centres and five transactions in non-metro centres. Beyond these, banks are permitted to charge a maximum of Rs. 21 per transaction.""",
        "gold_answer": "five free transactions",
        "scoring": "contains",
        "alternatives": ["five", "5 free", "5 transactions", "five free", "free of charge"],
    },
    {
        "id": "bank_03",
        "circular": "banking_ops",
        "question": "What is the maximum charge per ATM transaction beyond the free limit, per RBI guidelines?",
        "context": """Same as bank_02 above: 'Beyond these, banks are permitted to charge a maximum of Rs. 21 per transaction.'""",
        "gold_answer": "Rs. 21",
        "scoring": "contains",
        "alternatives": ["21", "Rs. 21", "Rs.21", "twenty-one", "₹21"],
    },
    {
        "id": "bank_04",
        "circular": "banking_ops",
        "question": "Are banks required to provide passbook printing free of cost to savings account holders under RBI guidelines?",
        "context": """RBI Master Direction on Customer Service:
Banks are required to provide pass book free of charge to all savings bank account holders. The pass book should be updated at the customer's request without any charge. Banks should also provide statements of account in lieu of passbook to those customers who do not wish to maintain passbook. The Banking Codes and Standards Board of India (BCSBI) Code requires banks to provide statements of accounts at the periodicity stipulated by the customer, subject to the bank's policy.""",
        "gold_answer": "yes",
        "scoring": "contains",
        "alternatives": ["yes", "free of charge", "free of cost", "without any charge"],
    },
    {
        "id": "bank_05",
        "circular": "banking_ops",
        "question": "Under RBI rules, within how many working days must a failed ATM transaction be reversed to the customer's account?",
        "context": """RBI Master Direction on Customer Protection — Limiting Liability of Customers in Unauthorised Electronic Banking Transactions, and circulars on failed ATM transactions:
The time limit for resolution of customer complaints by the issuing banks shall be five working days. Failing this, the bank shall pay compensation of Rs.100 per day to the aggrieved customer for delay beyond the stipulated period. The compensation shall be credited to the customer's account automatically without any claim from the customer, on the same day the bank affords the credit for the failed ATM transaction.""",
        "gold_answer": "five working days",
        "scoring": "contains",
        "alternatives": ["five", "5 working days", "5 days", "five working", "T+5"],
    },

    # ============== Cross-document Synthesis Questions (harder) ==============
    {
        "id": "cross_01",
        "circular": "kyc+digital_lending",
        "question": "Can a digital lending app collect and store biometric data of a borrower for KYC verification, given RBI's KYC and Digital Lending guidelines together?",
        "context": """RBI KYC Master Direction allows e-KYC including biometric authentication via Aadhaar with consent.
RBI Digital Lending Guidelines, Para 8.4: 'The Regulated Entity shall ensure that DLAs do not store biometric data on their servers.'
Combining these: biometric data MAY be used for KYC verification (one-time use with consent), but MUST NOT be stored by Digital Lending Apps on their servers.""",
        "gold_answer": "may use for verification but cannot store on DLA servers",
        "scoring": "multi_term",
        "alternatives": ["not store", "cannot store", "shall not store", "consent", "verification only", "DLA"],
        "must_match_count": 2,
    },
    {
        "id": "cross_02",
        "circular": "outsourcing+cybersecurity",
        "question": "If a bank outsources its IT operations to a third party and a cyber incident occurs at the service provider, who is responsible for reporting the incident to RBI?",
        "context": """RBI Outsourcing Master Direction, Para 4: 'The outsourcing of any activity by the bank does not diminish its obligations... Banks would be responsible for the actions of their service provider...'
RBI Cybersecurity Framework: 'Banks shall report all unusual cyber-security incidents to RBI within 2-6 hours of detection.'
Combined: outsourcing does not transfer the reporting obligation; the bank remains responsible for reporting cyber incidents at outsourced service providers.""",
        "gold_answer": "the bank",
        "scoring": "contains",
        "alternatives": ["the bank", "bank is responsible", "bank must report", "bank remains responsible", "regulated entity"],
    },
    {
        "id": "cross_03",
        "circular": "kyc+banking_ops",
        "question": "If a customer has a Small Account (under KYC norms) and uses the ATM to withdraw money, what limits apply considering both KYC small-account limits and general ATM withdrawal limits?",
        "context": """RBI KYC: Small Account monthly withdrawal limit is Rs. 10,000.
RBI ATM rules: Default per-transaction ATM limit is Rs. 10,000.
The Small Account monthly limit (Rs. 10,000/month) is the binding constraint — even though per-transaction ATM limit is also Rs. 10,000, the cumulative monthly limit applies.""",
        "gold_answer": "Rs. 10,000 per month aggregate",
        "scoring": "multi_term",
        "alternatives": ["10,000", "monthly", "month", "small account", "limit"],
        "must_match_count": 2,
    },

    # ============== Numeric/threshold extraction (compliance officer style) ==============
    {
        "id": "num_01",
        "circular": "kyc",
        "question": "What is the periodic KYC update interval for low risk customers under RBI norms?",
        "context": """RBI KYC Master Direction, Section 38: 'Periodic updation of KYC shall be carried out at least once in every two years for high risk customers, once in every eight years for medium risk customers and once in every ten years for low risk customers.'""",
        "gold_answer": "ten years",
        "scoring": "contains",
        "alternatives": ["10 years", "ten years", "every ten years", "once in ten years"],
    },
    {
        "id": "num_02",
        "circular": "kyc",
        "question": "What is the maximum aggregate credits in a financial year permitted in a Small Account?",
        "context": """RBI KYC Master Direction Section 23: '...the aggregate of all credits in a financial year does not exceed Rupees one lakh...'""",
        "gold_answer": "one lakh",
        "scoring": "contains",
        "alternatives": ["one lakh", "1,00,000", "100,000", "1 lakh", "Rs. 1 lakh"],
    },
]
