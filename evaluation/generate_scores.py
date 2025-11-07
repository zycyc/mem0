import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate scores from evaluation metrics")
parser.add_argument("--input_file", type=str, default="results/rag_500_k1_evaluation_metrics.json", help="Path to the input dataset file")
args = parser.parse_args()

# Load the evaluation metrics data
with open(args.input_file, "r") as f:
    data = json.load(f)

# Flatten the data into a list of question items
all_items = []
for key in data:
    all_items.extend(data[key])

# Convert to DataFrame
df = pd.DataFrame(all_items)

# Convert category to numeric type
df["category"] = pd.to_numeric(df["category"])

# Map numeric categories to category names (1: multi_hop, 2: temporal, 3: open_domain, 4: single_hop)
category_map = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
df["category_name"] = df["category"].map(category_map)

# Calculate mean scores by category
result = df.groupby("category_name").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

# Add count of questions per category
result["count"] = df.groupby("category_name").size()

# Print the results
print("Mean Scores Per Category:")
print(result)

# Calculate overall means
overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)
