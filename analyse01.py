from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ---------------------------
# 1️ Data Loading
# ---------------------------
brickmetrics = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project01/task01_data/brickmetrics.csv")
hcptobrick = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project01/task01_data/hcptobrick.csv")
hcpcompanymetrics = pd.read_csv("F:/veeva_assessment/VeevaAssessment/project01/task01_data/hcpcompanymetrics.csv")

# ---------------------------
# 2️ Data Merging
# ---------------------------
hcpbrickmetrics = pd.merge(hcptobrick, brickmetrics, on="brick_id", how="left")
hcp_full = pd.merge(hcpbrickmetrics, hcpcompanymetrics, on="veevaid", how="left")
hcpbrickmetrics.to_csv("hcpbrickmetrics.csv", index=False, encoding="utf-8-sig")

# ---------------------------
# 3️ Descriptive Statistics (Enhanced)
# ---------------------------
desc_stats = hcp_full[["company_hcp_calls","brick_average_calls","brick_average_access"]].describe().T
desc_stats["median"] = hcp_full[["company_hcp_calls","brick_average_calls","brick_average_access"]].median()
desc_stats["skew"] = hcp_full[["company_hcp_calls","brick_average_calls","brick_average_access"]].skew()
desc_stats["kurt"] = hcp_full[["company_hcp_calls","brick_average_calls","brick_average_access"]].kurt()
print("=== Basic Descriptive Statistics ===")
print(desc_stats)
desc_stats.to_csv("desc_stats.csv", index=True, encoding="utf-8-sig")

# Group statistics by Brick
brick_stats = hcp_full.groupby("brick_id")[["company_hcp_calls","brick_average_calls","brick_average_access"]].mean()
print("=== Brick-level Mean Statistics ===")
print(brick_stats.sort_values("company_hcp_calls", ascending=False).head(10))

# ---------------------------
# 4️ Calls Gap Calculation
# ---------------------------
hcp_full["calls_gap"] = hcp_full["brick_average_calls"] - hcp_full["company_hcp_calls"]

# ---------------------------
#  Figure 1: Overall Average Comparison
# ---------------------------
overall_means = {
    "Company A": hcp_full["company_hcp_calls"].mean(),
    "Industry Avg": hcp_full["brick_average_calls"].mean()
}

plt.figure(figsize=(6,4))
sns.barplot(x=list(overall_means.keys()), y=list(overall_means.values()), palette="Set2")
plt.title("Company A vs Industry Avg Calls (Overall)")
plt.ylabel("Average Calls")
plt.savefig("overall_compare.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------
# Figure 2: Brick-level Average Comparison (saved separately)
# ---------------------------
brick_compare = hcp_full.groupby("brick_id")[["company_hcp_calls","brick_average_calls"]].mean().reset_index()

plt.figure(figsize=(10,6))
brick_compare_melted = brick_compare.melt(
    id_vars="brick_id", 
    value_vars=["company_hcp_calls","brick_average_calls"],
    var_name="Source", value_name="Avg Calls"
)

sns.barplot(data=brick_compare_melted, x="brick_id", y="Avg Calls", hue="Source", palette="Set2")
plt.title("Company A vs Industry Avg Calls by Brick")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Brick ID")
plt.ylabel("Average Calls")
plt.savefig("brick_compare.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------
# 5️ Feature Normalization
# ---------------------------
scaler = MinMaxScaler()
hcp_full[["s_calls_gap","s_brick_calls","s_brick_access"]] = scaler.fit_transform(
    hcp_full[["calls_gap","brick_average_calls","brick_average_access"]].fillna(0)
)

# ---------------------------
# 6️ Rule-based Scoring for Potential Score
# ---------------------------
hcp_full["potential_score"] = (
    0.6*hcp_full["s_calls_gap"] + 
    0.3*hcp_full["s_brick_calls"] + 
    0.1*hcp_full["s_brick_access"]
)

# ---------------------------
# 7️ KMeans Clustering
# ---------------------------
X_cluster = hcp_full[["s_calls_gap","s_brick_calls","s_brick_access"]]
kmeans = KMeans(n_clusters=3, random_state=42)
hcp_full["potential_segment"] = kmeans.fit_predict(X_cluster)

# Calculate the average potential score per cluster and sort descending
cluster_scores = hcp_full.groupby("potential_segment")["potential_score"].mean().sort_values(ascending=False)

# Assign High/Medium/Low labels based on ranking
cluster_labels = ["High", "Medium", "Low"]
label_map = {cluster: label for cluster, label in zip(cluster_scores.index, cluster_labels)}
hcp_full["potential_segment_label"] = hcp_full["potential_segment"].map(label_map)

print(hcp_full[["potential_segment","potential_segment_label"]].head(10))
print(hcp_full["potential_segment_label"].value_counts())
hcp_full.to_csv("hcp_full.csv", index=False, encoding="utf-8-sig")

# ---------------------------
# 8️ Weight Sensitivity Analysis
# ---------------------------
weight_combinations = [
    (0.6, 0.3, 0.1),
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.5, 0.4, 0.1),
    (0.7, 0.2, 0.1)
]

top10_counter = {vid: 0 for vid in hcp_full["veevaid"]}

for w_calls_gap, w_brick_calls, w_brick_access in weight_combinations:
    hcp_full["potential_score_temp"] = (
        w_calls_gap * hcp_full["s_calls_gap"] +
        w_brick_calls * hcp_full["s_brick_calls"] +
        w_brick_access * hcp_full["s_brick_access"]
    )
    top10_temp = hcp_full.sort_values("potential_score_temp", ascending=False).head(10)
    for vid in top10_temp["veevaid"]:
        top10_counter[vid] += 1

top10_stability = pd.DataFrame.from_dict(top10_counter, orient="index", columns=["Top10_count"])
top10_stability = top10_stability.sort_values("Top10_count", ascending=False)
print("=== Weight Sensitivity Analysis: Top10 Stability ===")
print(top10_stability.head(20))

# ---------------------------
# 9️ Final Top10 Potential HCPs
# ---------------------------
final_top10 = hcp_full.sort_values("potential_score", ascending=False).head(10)
print("=== Final Top10 Potential HCPs ===")
print(final_top10[["veevaid","first_name","last_name","potential_score","potential_segment_label"]])

# Save as CSV
final_top10.to_csv("top10_potential_hcps.csv", index=False, encoding="utf-8-sig")

print("✅ Top10 Potential HCPs table saved as top10_potential_hcps.csv")

# ---------------------------
# 10 Unified Subplot Panel (4 Key Figures)
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fig1: Company HCP Calls Distribution
sns.histplot(hcp_full["company_hcp_calls"], bins=30, kde=True, ax=axes[0,0])
axes[0,0].set_title("Distribution of Company HCP Calls")
axes[0,0].set_xlabel("Company HCP Calls")
axes[0,0].set_ylabel("Frequency")

# Fig2: Calls Gap Distribution
sns.histplot(hcp_full["calls_gap"], bins=30, kde=True, ax=axes[0,1])
axes[0,1].set_title("Distribution of Calls Gap (Brick Avg - Company Calls)")
axes[0,1].set_xlabel("Calls Gap")
axes[0,1].set_ylabel("Frequency")

# Fig3: Overall Average Comparison
sns.barplot(x=list(overall_means.keys()), y=list(overall_means.values()), 
            palette="Set2", ax=axes[1,0])
axes[1,0].set_title("Company A vs Industry Avg Calls (Overall)")
axes[1,0].set_ylabel("Average Calls")

# Fig4: Potential Segment Distribution
sns.countplot(data=hcp_full, x="potential_segment_label", 
              order=["High","Medium","Low"], ax=axes[1,1], palette="Set3")
axes[1,1].set_title("Potential Segment Distribution")
axes[1,1].set_xlabel("Potential Segment")
axes[1,1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("summary_panel.png", dpi=300, bbox_inches="tight")
plt.show()

