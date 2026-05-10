# shared/corpus.py

from langchain_core.documents import Document

CORPUS = [

# =========================
# HEALTH INSURANCE
# =========================

Document(
    page_content=(
        "Pre-existing diseases like diabetes and hypertension are excluded for 36 months "
        "from first policy inception. Conditions diagnosed within 48 months before policy "
        "start are considered pre-existing."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "exclusion",
        "section": "waiting_period",
        "doc_id": "H001"
    }
),

Document(
    page_content=(
        "Cashless hospitalisation requires pre-authorisation from the TPA at least 48 "
        "hours before planned admission. Emergency admissions must be intimated within "
        "24 hours. Cashless available at 7000+ network hospitals."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "cashless",
        "section": "claims_procedure",
        "doc_id": "H002"
    }
),

Document(
    page_content=(
        "Maternity benefit covers normal delivery up to Rs 50,000 and Caesarean up to "
        "Rs 75,000. Available after 2 continuous policy years. Newborn is covered from day 1."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "maternity",
        "section": "benefits",
        "doc_id": "H003"
    }
),

Document(
    page_content=(
        "No Claim Bonus in health insurance gives 5% increase in sum insured for each "
        "claim-free year up to a maximum of 50% of the original sum insured."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "bonus",
        "section": "premium",
        "doc_id": "H004"
    }
),

Document(
    page_content=(
        "Day care procedures requiring less than 24 hours hospitalisation are covered. "
        "Over 540 listed procedures are covered including cataract, chemotherapy, dialysis."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "daycare",
        "section": "benefits",
        "doc_id": "H005"
    }
),

Document(
    page_content=(
        "Reimbursement claims must be submitted within 30 days of hospital discharge. "
        "Required: discharge summary, original bills, diagnostic reports, prescription, "
        "and completed claim form signed by treating doctor."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "reimbursement",
        "section": "claims_procedure",
        "doc_id": "H006"
    }
),

Document(
    page_content=(
        "Cosmetic surgery and dental treatment are excluded unless required due to an accident. "
        "Infertility treatments, IVF, and weight loss surgeries are also excluded."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "exclusion",
        "section": "exclusions",
        "doc_id": "H007"
    }
),

Document(
    page_content=(
        "Restoration benefit reinstates the full sum insured once per year after complete "
        "exhaustion, for unrelated illnesses only. Ensures family coverage is not lost."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "restoration",
        "section": "add_ons",
        "doc_id": "H008"
    }
),

Document(
    page_content=(
        "Air ambulance cover for life-threatening emergencies is available up to Rs 2,50,000 "
        "per policy year for inter-city air transfer to a higher-equipped hospital."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "emergency",
        "section": "add_ons",
        "doc_id": "H009"
    }
),

Document(
    page_content=(
        "Annual preventive health check-up is covered up to Rs 2,000 per adult member. "
        "Tests: CBC, lipid profile, blood sugar, ECG at empanelled diagnostic centres."
    ),
    metadata={
        "policy_type": "health",
        "claim_type": "wellness",
        "section": "benefits",
        "doc_id": "H010"
    }
),

# =========================
# MOTOR INSURANCE
# =========================

Document(
    page_content=(
        "Third party liability is mandatory under the Motor Vehicles Act. It covers death, "
        "bodily injury or property damage to a third party. Property damage capped Rs 7.5L."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "third_party",
        "section": "coverage",
        "doc_id": "M001"
    }
),

Document(
    page_content=(
        "Own damage cover protects against accidental damage from collision, fire, theft, "
        "natural calamities, riots and malicious acts. Claim subject to IDV minus depreciation."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "own_damage",
        "section": "coverage",
        "doc_id": "M002"
    }
),

Document(
    page_content=(
        "No Claim Bonus for motor insurance: 20% after 1 claim-free year, 25% after 2 years, "
        "35% after 3 years, 45% after 4 years, and 50% after 5+ consecutive claim-free years."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "bonus",
        "section": "premium",
        "doc_id": "M003"
    }
),

Document(
    page_content=(
        "Cashless repair network covers 5000+ garages across India. Intimate insurer within "
        "24 hours of accident. Surveyor inspection is mandatory before vehicle is moved."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "cashless",
        "section": "claims_procedure",
        "doc_id": "M004"
    }
),

Document(
    page_content=(
        "Zero depreciation add-on eliminates depreciation deductions on replaced parts. "
        "Available for vehicles up to 5 years old. Increases premium by 15-20%."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "add_on",
        "section": "add_ons",
        "doc_id": "M005"
    }
),

Document(
    page_content=(
        "Engine protection add-on covers damage due to water ingression and hydrostatic lock. "
        "Essential for vehicles in flood-prone areas and during monsoon season in India."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "flood",
        "section": "add_ons",
        "doc_id": "M006"
    }
),

Document(
    page_content=(
        "Driving under influence of alcohol or drugs voids the motor insurance claim. "
        "Other exclusions: no valid licence, commercial use of private vehicle, racing."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "exclusion",
        "section": "exclusions",
        "doc_id": "M007"
    }
),

Document(
    page_content=(
        "Roadside assistance provides towing up to 50 km, battery jump-start, flat tyre "
        "change and emergency fuel delivery. Available 24x7 on highways and city roads."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "add_on",
        "section": "add_ons",
        "doc_id": "M008"
    }
),

Document(
    page_content=(
        "Insured Declared Value is the current market value of the vehicle minus depreciation. "
        "IDV decreases 5-20% per year by age. It determines the maximum claim amount."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "valuation",
        "section": "coverage",
        "doc_id": "M009"
    }
),

Document(
    page_content=(
        "Personal accident cover provides Rs 15 lakh to owner-driver for accidental death "
        "or permanent total disability. Mandatory under IRDAI regulations since 2019."
    ),
    metadata={
        "policy_type": "motor",
        "claim_type": "personal_accident",
        "section": "coverage",
        "doc_id": "M010"
    }
),

# =========================
# LIFE INSURANCE
# =========================

Document(
    page_content=(
        "Term life insurance provides pure death benefit with no maturity value. Sum assured "
        "is paid to nominee on insured death during policy term. Premiums are much lower."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "death_benefit",
        "section": "product_type",
        "doc_id": "L001"
    }
),

Document(
    page_content=(
        "Critical illness rider pays lump sum on diagnosis of cancer, heart attack, stroke, "
        "kidney failure or major organ transplant. Paid regardless of actual treatment cost."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "critical_illness",
        "section": "riders",
        "doc_id": "L002"
    }
),

Document(
    page_content=(
        "Suicide clause: death by suicide within the first year of policy is excluded. "
        "From year 2 onwards, 80% of total premiums paid are returned to the nominee."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "exclusion",
        "section": "exclusions",
        "doc_id": "L003"
    }
),

Document(
    page_content=(
        "Death claim process: nominee submits death certificate, original policy bond, "
        "NEFT details and ID proof within 90 days. Settled within 30 days if complete."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "death_claim",
        "section": "claims_procedure",
        "doc_id": "L004"
    }
),

Document(
    page_content=(
        "Free look period allows policy cancellation within 15 days of receiving the document. "
        "Full premium refunded after deducting proportionate risk premium and stamp duty."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "cancellation",
        "section": "policy_terms",
        "doc_id": "L005"
    }
),

Document(
    page_content=(
        "Accidental death benefit rider doubles the sum assured if death is accidental. "
        "Also covers permanent total disability with 100% of sum assured. Age 18-65."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "accidental_death",
        "section": "riders",
        "doc_id": "L006"
    }
),

Document(
    page_content=(
        "Premium waiver rider waives future premiums if policyholder becomes permanently "
        "disabled or critically ill. Policy continues in force with all benefits intact."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "disability",
        "section": "riders",
        "doc_id": "L007"
    }
),

Document(
    page_content=(
        "Grace period for life insurance is 30 days for monthly premium mode and 15 days "
        "for other frequencies. Policy lapses if not paid. Revival possible within 5 years."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "lapse",
        "section": "policy_terms",
        "doc_id": "L008"
    }
),

Document(
    page_content=(
        "Surrender value is available after 3 years of premiums paid. Guaranteed surrender "
        "value is 30% of premiums paid excluding first year. Special surrender value may be higher."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "surrender",
        "section": "policy_terms",
        "doc_id": "L009"
    }
),

Document(
    page_content=(
        "Nomination allows designating a nominee to receive the death benefit. "
        "Nominee can be changed at any time. Without nomination, legal heirs receive the amount."
    ),
    metadata={
        "policy_type": "life",
        "claim_type": "nomination",
        "section": "policy_terms",
        "doc_id": "L010"
    }
),

]

if __name__ == "__main__":
    print(f"Corpus: {len(CORPUS)} documents")

    for pt in ["health", "motor", "life"]:
        n = sum(1 for d in CORPUS if d.metadata["policy_type"] == pt)
        print(f"{pt}: {n} docs")