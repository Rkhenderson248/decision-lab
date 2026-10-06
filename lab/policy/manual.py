"""Kestrel Valley Credit Union (fictional): consumer lending policy manual.

Written for the policy-assistant demo. It is illustrative, simplified and not
legal or regulatory guidance; where real rules apply it says so generically.
"""

SECTIONS = [
    ("GEN-1.1", "Purpose and scope",
     "This manual governs consumer lending at Kestrel Valley Credit Union: credit cards, new and used auto loans, "
     "unsecured personal loans and home equity lines of credit (HELOCs). First-lien mortgages are governed by the "
     "separate Mortgage Lending Manual and are out of scope here. Where this manual and applicable law differ, the "
     "law prevails and the Chief Credit Officer must be told within five business days."),
    ("GEN-1.2", "Membership requirement",
     "Every borrower and co-borrower must be a member in good standing before a loan is funded. An applicant may "
     "apply before joining, but membership must be opened and the share account funded with at least $5 before "
     "closing. Members whose accounts caused a loss to the credit union in the past seven years are not eligible "
     "unless the loss has been repaid in full."),
    ("UW-2.1", "Minimum credit scores by product",
     "Minimum credit scores are: credit cards 640; unsecured personal loans 660; auto loans 600; HELOCs 680. "
     "Scores are taken from the primary applicant's middle score when three are available, or the lower score "
     "when two are available. Applications below the minimum may only be approved as a documented exception "
     "under section UW-2.8."),
    ("UW-2.2", "Debt-to-income limits",
     "Total debt-to-income (DTI), including the proposed payment, may not exceed 43% for unsecured products and "
     "45% for auto loans and HELOCs. For applicants with a credit score of 740 or higher and at least six months of "
     "verified reserves, the limit rises to 50%. Monthly debts include all installment and revolving minimum "
     "payments, housing payments, alimony and child support; utilities and insurance are excluded."),
    ("UW-2.3", "Income verification",
     "Salaried and hourly income is verified with the most recent 30 days of pay stubs and either the prior year's "
     "W-2 or a verbal verification of employment within ten days of closing. Overtime, bonus and commission income "
     "count only with a two-year history and must be averaged over that period. Income that cannot be documented "
     "may not be used to qualify."),
    ("UW-2.4", "Self-employed borrowers",
     "Self-employed applicants must provide two years of complete personal and business tax returns. Qualifying "
     "income is the two-year average of net income after adding back depreciation and depletion. If income "
     "declined by more than 20% from the first year to the second, the lower year is used. Applicants self-employed "
     "for less than two years may qualify only with a prior two-year history in the same line of work."),
    ("UW-2.5", "Co-applicants and guarantors",
     "A co-applicant shares full liability and ownership; their income and debts are both included and the lower "
     "of the two qualifying scores is used for pricing. Guarantors are accepted only for applicants aged 18 to 21 "
     "with no credit history, are limited to auto loans and credit cards, and must themselves meet all underwriting "
     "standards for the requested product."),
    ("UW-2.6", "Bankruptcy and foreclosure seasoning",
     "Applicants must be at least two years past a Chapter 7 bankruptcy discharge and at least one year into a "
     "Chapter 13 repayment plan with trustee approval. A foreclosure or deed in lieu requires four years of "
     "seasoning before a HELOC can be considered. Re-established credit since the event, with no late payments, "
     "is required in every case."),
    ("UW-2.7", "Recent delinquencies",
     "Any 60-day or worse delinquency in the past 12 months, or more than two 30-day delinquencies in the past 24 "
     "months, requires a written explanation and senior underwriter approval. Medical collections under $500 and "
     "accounts in documented disaster forbearance are disregarded."),
    ("UW-2.8", "Exceptions and approval authority",
     "Exceptions to score, DTI or seasoning standards require written compensating factors such as substantial "
     "reserves, long membership tenure or a lower loan-to-value. Loan officers may approve exceptions up to "
     "$15,000; senior underwriters up to $50,000; the Credit Committee above $50,000. Every exception is recorded "
     "with its reason code and reviewed monthly for patterns."),
    ("UW-2.9", "Thin files and no-score applicants",
     "Applicants without a credit score may be considered for auto loans and secured credit cards using "
     "alternative data: twelve months of on-time rent, utility or telecom payments, or a twelve-month deposit "
     "history with the credit union. Maximum amounts are $15,000 for auto loans and $2,000 for secured cards."),
    ("COL-3.1", "Auto loan collateral and loan-to-value",
     "Auto loans may finance up to 120% of the vehicle's retail value for new vehicles and 110% for used "
     "vehicles, including taxes, title and approved add-ons. Vehicles older than ten model years or with more than "
     "150,000 miles are not eligible as collateral. The credit union must be listed as first lienholder."),
    ("COL-3.2", "Auto loan terms",
     "Maximum terms are 84 months for new vehicles and amounts above $30,000, 72 months for used vehicles up to "
     "five model years old, and 60 months for older vehicles. Terms longer than 72 months require a credit score "
     "of 700 or higher."),
    ("COL-3.3", "HELOC limits",
     "Combined loan-to-value (CLTV), including all liens, may not exceed 85% of the appraised value for primary "
     "residences and 70% for second homes. Investment properties are not eligible. Lines above $150,000 require a "
     "full interior appraisal; smaller lines may use an automated valuation."),
    ("PR-4.1", "Risk-based pricing tiers",
     "Rates are set by credit score tier: Tier A (740 and above), Tier B (680 to 739), Tier C (620 to 679) and Tier D (below "
     "620). Each tier's rate is the product base rate plus a published tier margin. Loan officers may not adjust "
     "rates outside the published sheet except through an approved promotion."),
    ("PR-4.2", "Risk-based pricing notice",
     "When an applicant receives a rate higher than the credit union's best available rate because of their "
     "credit report, they must receive a risk-based pricing or credit score disclosure as required by applicable "
     "law, delivered before the loan is consummated."),
    ("PR-4.3", "Relationship discounts",
     "Members with direct deposit and at least two other products receive a 0.25 percentage point discount on auto "
     "and personal loans. Autopay from a credit union account earns an additional 0.25 points. Discounts may not "
     "reduce any rate below the product floor."),
    ("FL-5.1", "Fair lending",
     "Credit decisions must be based only on creditworthiness and must never consider race, color, religion, "
     "national origin, sex, marital status, age (provided the applicant can contract), receipt of public assistance "
     "or the good-faith exercise of consumer protection rights. Standards and exceptions must be applied "
     "consistently; exception patterns are monitored by group monthly."),
    ("FL-5.2", "Adverse action notices",
     "When an application is declined, approved on different terms or withdrawn as incomplete, the applicant must "
     "receive a written notice within 30 days stating the action taken and up to four principal reasons, using the "
     "standard reason statements generated by the scorecard. Vague reasons such as 'did not meet internal "
     "standards' are not permitted."),
    ("FL-5.3", "Servicemembers",
     "Active-duty servicemembers and their dependents receive the protections required by applicable law on rate "
     "caps, fees and arbitration. Loan officers must check covered-borrower status before closing and contact "
     "Compliance with any question before proceeding."),
    ("FR-6.1", "Identity verification",
     "Identity is verified with an unexpired government-issued photo ID and a second form of identification. "
     "Online applications also pass a knowledge-based or document-scan verification. Applications that fail "
     "verification are not decisioned until identity is confirmed in person or by video."),
    ("FR-6.2", "Fraud red flags",
     "Red flags include a phone number or address shared with other recent applications, stated income far above "
     "what the applicant's profile supports, an email address created within the past 30 days and several "
     "applications from one device within 24 hours. Any two red flags send the application to the fraud queue; a "
     "shared address alone never declines an application, since households share addresses."),
    ("SV-7.1", "Hardship assistance",
     "Members facing a documented hardship such as job loss, medical expense or natural disaster may receive up to "
     "three months of payment deferral, a term extension of up to 12 months or a temporary rate reduction. Hardship "
     "plans are approved by the collections supervisor and reported to the credit bureaus as agreed, not as late."),
    ("SV-7.2", "Collection contact standards",
     "Collectors may contact members between 8 a.m. and 9 p.m. in the member's local time, no more than seven call "
     "attempts in seven days per account, and must honor a request to stop calls at work. Text messages are used "
     "only with documented consent. Every contact is logged with its outcome."),
    ("SV-7.3", "Charge-off",
     "Unsecured loans and credit cards are charged off at 180 days past due; auto loans at 120 days past due or "
     "earlier once the vehicle is repossessed and sold. Charge-off does not end collection efforts and recoveries "
     "are tracked against the original account."),
]

# Questions with the section that answers them, for evaluation. None = not covered (should refuse).
GOLDEN = [
    ("What is the minimum credit score for a HELOC?", "UW-2.1"),
    ("How high can debt-to-income go for an auto loan?", "UW-2.2"),
    ("Can a borrower with a 760 score and reserves go above 45% DTI?", "UW-2.2"),
    ("How do we count bonus and overtime income?", "UW-2.3"),
    ("What do self-employed applicants need to provide?", "UW-2.4"),
    ("How long after a Chapter 7 discharge can someone get a loan?", "UW-2.6"),
    ("Who can approve an exception on a $40,000 loan?", "UW-2.8"),
    ("Can we lend to someone with no credit score?", "UW-2.9"),
    ("What is the maximum loan-to-value on a used car?", "COL-3.1"),
    ("How long can the term be on a new car loan?", "COL-3.2"),
    ("Can a HELOC be opened on an investment property?", "COL-3.3"),
    ("What credit score range is Tier B?", "PR-4.1"),
    ("What discount do members get for autopay?", "PR-4.3"),
    ("What must an adverse action notice include?", "FL-5.2"),
    ("When does an application go to the fraud queue?", "FR-6.2"),
    ("What hours can collectors call members?", "SV-7.2"),
    ("When is a credit card charged off?", "SV-7.3"),
    ("What hardship options can a member get after losing their job?", "SV-7.1"),
    ("What is the maximum loan amount for a jumbo first mortgage?", None),
    ("What is the credit union's dividend rate on savings?", None),
    ("How do I reset my online banking password?", None),
    ("Which vendor do we use for appraisals in Texas?", None),
]
