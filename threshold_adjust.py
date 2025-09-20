import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Load matching results
# =========================
matches_a = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project02/analyse/matched_customer_A.csv")
matches_b = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project02/analyse/matched_customer_B.csv")

# =========================
# 2. Threshold sensitivity analysis function
# =========================
def evaluate_thresholds(matches_df, thresholds=np.arange(0.5, 0.91, 0.05)):
    results = []
    for t in thresholds:
        matched = len(matches_df[matches_df['score'] >= t])
        total = len(matches_df)
        match_rate = matched / total * 100
        results.append({
            "threshold": round(t, 2),
            "matched_records": matched,
            "total_records": total,
            "match_rate(%)": round(match_rate, 2)
        })
    return pd.DataFrame(results)

# =========================
# 3. Run analysis
# =========================
threshold_analysis_a = evaluate_thresholds(matches_a)
threshold_analysis_b = evaluate_thresholds(matches_b)

print("\nThreshold Sensitivity Analysis - Customer A")
print(threshold_analysis_a)
print("\nThreshold Sensitivity Analysis - Customer B")
print(threshold_analysis_b)

# =========================
# 4. Plotting (Threshold vs Match Rate)
# =========================
plt.figure(figsize=(6,4))
plt.plot(threshold_analysis_a["threshold"], threshold_analysis_a["match_rate(%)"], marker='o', label="Customer A")
plt.plot(threshold_analysis_b["threshold"], threshold_analysis_b["match_rate(%)"], marker='s', label="Customer B")
plt.xlabel("Threshold")
plt.ylabel("Match Rate (%)")
plt.title("Threshold Sensitivity Analysis")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("F:/veeva_assessment/VeevaAssessment/project02/analyse/threshold_sensitivity.png", dpi=300)
plt.show()
