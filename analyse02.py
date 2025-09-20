import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import textdistance

# =========================
# 1. Load data
# =========================
cust_a = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project02/data/customer_A_doctors.csv")
cust_b = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project02/data/customer_B_doctors.csv")
master = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project02/data/veeva_master_doctors.csv")

# Clean whitespace in column names
for df in [cust_a, cust_b, master]:
    df.columns = df.columns.str.strip()

# =========================
# 2. Preprocessing function
# =========================
def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    # Convert full-width characters to half-width
    return ''.join([chr(ord(c) - 65248) if 65281 <= ord(c) <= 65374 else c for c in x])

# Standardize fields
cust_a['name_norm'] = cust_a["doctor_name"].apply(normalize_text)
cust_a['hospital_norm'] = cust_a["work_unit"].apply(normalize_text)
cust_a['specialty_norm'] = cust_a["dept"].apply(normalize_text)

cust_b['name_norm'] = cust_b["physician_name"].apply(normalize_text)
cust_b['hospital_norm'] = cust_b["hospital_name"].apply(normalize_text)
cust_b['specialty_norm'] = cust_b["specialty"].apply(normalize_text)

master['name_norm'] = master["name"].apply(normalize_text)
master['hospital_norm'] = master["hospital"].apply(normalize_text)
master['specialty_norm'] = master["department"].apply(normalize_text)

# =========================
# 3. Matching function (with department constraint)
# =========================
def match_doctors(source_df, master_df, top_k=1, threshold=0.75):
    results = []
    for _, row in source_df.iterrows():
        best_score = -1
        best_match = None
        
        for _, m in master_df.iterrows():
            # Name similarity
            name_score_fuzzy = fuzz.token_sort_ratio(row["name_norm"], m["name_norm"]) / 100
            name_score_jaro = textdistance.jaro_winkler(row["name_norm"], m["name_norm"])
            name_score = (name_score_fuzzy + name_score_jaro) / 2

            # Hospital similarity
            if row["hospital_norm"] == m["hospital_norm"]:
                hosp_score = 1.0
            elif row["hospital_norm"] in m["hospital_norm"] or m["hospital_norm"] in row["hospital_norm"]:
                hosp_score = 0.8
            else:
                hosp_score = fuzz.partial_ratio(row["hospital_norm"], m["hospital_norm"]) / 100

            # Department similarity (using standardized specialty_norm)
            if row["specialty_norm"] == m["specialty_norm"]:
                spec_score = 1.0
            elif row["specialty_norm"] in m["specialty_norm"] or m["specialty_norm"] in row["specialty_norm"]:
                spec_score = 0.8
            else:
                spec_score = fuzz.partial_ratio(row["specialty_norm"], m["specialty_norm"]) / 100

            # Weighted score (Name 0.6, Hospital 0.25, Department 0.15)
            score = 0.6 * name_score + 0.25 * hosp_score + 0.15 * spec_score

            if score > best_score:
                best_score = score
                best_match = m

        # Match status
        match_status = "no_match"
        if best_score >= threshold:
            match_status = "match"
        elif best_score >= 0.6:
            match_status = "possible_match"

        results.append({
            "source_id": row.get("id", row.get("internal_id", None)),
            "source_name": row.get("doctor_name", row.get("physician_name", None)),
            "source_hospital": row.get("hospital_norm", None),
            "source_specialty": row.get("dept", row.get("specialty", None)),
            "master_doctor_id": best_match.get("doctor_id", None) if best_match is not None else None,
            "master_name": best_match.get("name", None) if best_match is not None else None,
            "master_hospital": best_match.get("hospital", None) if best_match is not None else None,
            # "master_specialty": best_match.get("specialty_norm", None) if best_match is not None else None,
            "master_department": best_match.get("department", None) if best_match is not None else None,  # original department column
            "score": round(best_score, 3),
            "match_status": match_status
        })

    return pd.DataFrame(results)

# =========================
# 4. Run matching
# =========================
matches_a = match_doctors(cust_a, master)
matches_b = match_doctors(cust_b, master)

# =========================
# 5. Matching statistics
# =========================
def report_stats(df, customer_name):
    total = len(df)
    matched = len(df[df['match_status'] == 'match'])
    possible = len(df[df['match_status'] == 'possible_match'])
    unmatched = len(df[df['match_status'] == 'no_match'])
    success_rate = matched / total * 100
    print(f"\n{customer_name} Matching Stats:")
    print(f"Total records: {total}")
    print(f"Matched: {matched}")
    print(f"Possible match: {possible}")
    print(f"Unmatched: {unmatched}")
    print(f"Match success rate: {success_rate:.2f}%")
    # Show some unmatched examples
    print("\nSample unmatched records:")
    print(df[df['match_status']=='no_match'].head(5))

report_stats(matches_a, "Customer A")
report_stats(matches_b, "Customer B")

# =========================
# 6. Save results
# =========================
matches_a.to_csv("F:/veeva_assessment/VeevaAssessment/project02/analyse/matched_customer_A.csv", index=False, encoding="utf-8-sig")
matches_b.to_csv("F:/veeva_assessment/VeevaAssessment/project02/analyse/matched_customer_B.csv", index=False, encoding="utf-8-sig")

print("\n 👌 Matching finished, results saved")
