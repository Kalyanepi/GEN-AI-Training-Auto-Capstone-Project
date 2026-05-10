# truncation_experiment.py

import os
import time
import json
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Token encoder
enc = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def build_large_policy() -> str:
    """Build a realistic ~8,000 token insurance policy document."""

    sections = {
        "SECTION 1 - DEFINITIONS": """
Insured Person: Any person named in the Schedule as an Insured Person.

Sum Insured: The maximum amount payable by the Company during a Policy Year.

Pre-Existing Disease (PED): Any condition, ailment, injury or disease diagnosed
within 48 months prior to first policy issuance. PED includes diabetes mellitus
type 1 and 2, hypertension, cardiac conditions, thyroid disorders, asthma,
and any other chronic condition requiring ongoing treatment or monitoring.

Network Hospital: A Hospital empanelled with the Company for cashless treatment.

Day Care Treatment: Medical treatment or surgical procedure that requires less than
24 hours hospitalisation due to technological advancements. Over 540 day-care
procedures are covered under this policy.
""",

        "SECTION 2 - COVERAGE AND BENEFITS": """
2.1 In-Patient Hospitalisation:
Expenses for treatment requiring continuous hospitalisation for a minimum period
of 24 hours.

Covered expenses include:
- room rent up to single AC room
- ICU charges
- surgeon and anaesthetist fees
- diagnostic tests
- medicines
- blood
- oxygen
- OT charges
- implants

2.2 Pre-Hospitalisation:
Medical expenses incurred 60 days prior to admission related to the condition
for which hospitalisation occurred.

2.3 Post-Hospitalisation:
Medical expenses incurred 90 days after discharge related to the condition
for which hospitalisation occurred.

2.4 Organ Donor Expenses:
Medical expenses of the organ donor for harvesting of the donated organ
covered up to Sum Insured.

2.5 Maternity Benefit:
Available after 2 policy years.
- Normal delivery: Rs 50,000
- Caesarean delivery: Rs 75,000
- Newborn baby covered from day 1
""",

        "SECTION 3 - EXCLUSIONS": """
3.1 Pre-Existing Diseases:
Excluded for 36 months from policy inception.

3.2 Specific Disease Waiting Period:
24-month waiting period applies to:
- cataracts
- benign prostatic hypertrophy
- hernia
- fistula in anus
- piles
- sinusitis
- tonsillitis
- joint replacement surgery
- varicose veins

3.3 Cosmetic and Aesthetic Treatments:
Any surgery or treatment to change physical appearance unless medically necessary
due to an accident.

3.4 Dental Treatment:
Dental procedures unless requiring hospitalisation.

3.5 Infertility Treatments:
IVF, IUI, ICSI and all assisted reproduction excluded.

3.6 Mental Illness:
Psychiatric and psychosomatic disorders excluded under base plan.
Available as an optional add-on rider at additional premium.
""",

        "SECTION 4 - PREMIUM AND RENEWAL": """
4.1 Premium Calculation:
Premium is calculated based on:
- Sum Insured
- age of eldest member
- number of members
- zone of residence
- medical history

Zone A:
Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata

Zone B:
All other cities and towns across India.

4.2 No Claim Bonus (NCB):
5% increase in Sum Insured for each claim-free year,
up to a maximum of 50% of original Sum Insured.

4.3 Cumulative Bonus Protector:
NCB is protected even if one claim is filed,
subject to optional rider selection.

4.4 Grace Period:
30 days from policy expiry date for renewal.
""",

        "SECTION 5 - CLAIMS PROCEDURE": """
5.1 Cashless Claims:
Pre-authorisation required from TPA at least 48 hours
before planned admission.

For emergencies, inform within 24 hours of admission.

5.2 Reimbursement Claims:
Submit all original documents within 30 days of discharge.

Required documents:
- discharge summary
- all bills and receipts
- diagnostic reports
- prescription
- completed claim form

5.3 Claim Settlement Timeline:
Company shall settle or reject within 30 days
of receiving all documents.

If surveyor appointed, within 45 days.

5.4 Grievance Redressal:
Lodge complaint via toll-free 1800-XXX-XXXX or
grievance@insuresafe.com.
Escalate to IRDAI Bima Bharosa portal if unresolved.
""",

        "SECTION 6 - SPECIAL CONDITIONS AND ADD-ONS": """
6.1 Restoration Benefit:
Full Sum Insured automatically reinstated once per year
for unrelated illnesses after complete exhaustion.

6.2 Air Ambulance:
Covered up to Rs 2,50,000 per policy year
for life-threatening emergencies requiring inter-city transfer.

6.3 Second Opinion:
One free second medical opinion per policy year
for critical illnesses including:
- cancer
- cardiac conditions
- neurological conditions

6.4 Preventive Health Check:
Annual health check-up at empanelled centres
covered up to Rs 2,000 per adult member per year.
"""
    }

    doc_parts = [
        "InsureSafe HealthGuard Gold Policy — Policy Wording Document\n"
    ]

    for section, content in sections.items():
        sep = "=" * 70
        doc_parts.append(
            f"\n{sep}\n{section}\n{sep}\n{content}\n"
        )

    full_doc = "\n".join(doc_parts)

    while count_tokens(full_doc) < 7500:
        full_doc += (
            "\n[This page intentionally contains expanded policy schedules, "
            "rate tables, and annexures as part of the complete policy wording.]"
        )
        full_doc += " " * 200

    return full_doc


def truncate_to_tokens(text: str, max_tokens: int):
    """Truncate text to max_tokens."""

    tokens = enc.encode(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = enc.decode(truncated_tokens)

    return truncated_text, len(truncated_tokens)


def query_with_context(context: str, question: str, label: str):

    token_count = count_tokens(context)

    print(f"\n[{label}] Sending {token_count:,} tokens...")

    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question based ONLY on the provided document. "
                    "If the answer is not in the document, say "
                    "'NOT FOUND IN CONTEXT'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT:\n{context}\n\nQUESTION: {question}"
                )
            },
        ],
        temperature=0,
        max_tokens=300
    )

    elapsed = time.time() - start

    answer = response.choices[0].message.content

    return {
        "label": label,
        "tokens_sent": token_count,
        "answer": answer,
        "latency_s": round(elapsed, 2),
        "found": "NOT FOUND IN CONTEXT" not in answer
    }


def run_experiment():

    full_doc = build_large_policy()

    total_tokens = count_tokens(full_doc)

    print(f"Document size: {total_tokens:,} tokens")

    question = (
        # "What is the timeline for claim settlement after all documents "
        # "are submitted? Also, how many days after discharge must "
        # "reimbursement documents be submitted?"
        
    )

    levels = {
        "100% (Full)": total_tokens,
        "75%": int(total_tokens * 0.75),
        "50%": int(total_tokens * 0.50),
        "25%": int(total_tokens * 0.25),
        "05%": int(total_tokens * 0.05)
    }

    results = []

    for label, max_tok in levels.items():

        ctx, actual = truncate_to_tokens(full_doc, max_tok)

        result = query_with_context(ctx, question, label)

        results.append(result)

        time.sleep(1)

    print("\n" + "=" * 70)
    print("TRUNCATION EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"Question: {question[:80]}...")
    print("Level Tokens Found? Latency")
    print("-" * 50)

    for r in results:

        found_str = "YES" if r["found"] else "NO"

        print(
            f"{r['label']:<12} "
            f"{r['tokens_sent']:>8,} "
            f"{found_str:>8} "
            f"{r['latency_s']:>9.1f}s"
        )

    with open("truncation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to truncation_results.json")


if __name__ == "__main__":
    run_experiment()