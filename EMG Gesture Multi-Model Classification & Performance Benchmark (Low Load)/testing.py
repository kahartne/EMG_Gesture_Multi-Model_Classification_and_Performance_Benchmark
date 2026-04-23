import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("features.csv")

# Counts
counts = df["label"].value_counts().sort_index()
percent = counts / len(df)

print("Label Counts:\n", counts)
print("\nLabel Percentages:\n", percent)

# --- Plot (minimal, clean) ---
plt.figure()
counts.plot(kind="bar")
plt.title("Label Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("results/label_distribution.png")
plt.show()