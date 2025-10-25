# ============================================================
# 🔍 Misclassification Analysis for Language Identification
# ============================================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
import itertools

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "../saved_models/language_pipeline.joblib"
DATA_CSV = "../../output_clean.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_MISCLASS = 10   # Number of misclassified samples to display

# -------------------------------
# LOAD MODEL AND DATA
# -------------------------------
print("📂 Loading model and dataset...")
pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_CSV)

X = df["text"]
y = df["class"]

# Split test data
_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# -------------------------------
# PREDICT ON TEST DATA
# -------------------------------
print("⚙️ Running predictions...")
y_pred = pipeline.predict(X_test)

# -------------------------------
# FIND MISCLASSIFIED CASES
# -------------------------------
misclassified_mask = y_pred != y_test
misclassified = pd.DataFrame({
    "text": X_test[misclassified_mask],
    "actual": y_test[misclassified_mask],
    "predicted": y_pred[misclassified_mask]
}).reset_index(drop=True)

print(f"\n❌ Total misclassified samples: {len(misclassified)}")
print("\n📊 Sample Misclassified Examples:")
for i in range(min(TOP_N_MISCLASS, len(misclassified))):
    row = misclassified.iloc[i]
    print(f"{i+1}. '{row['text'][:80]}...' → Predicted: {row['predicted']} | Actual: {row['actual']}")

# -------------------------------
# COMMON CONFUSION PAIRS
# -------------------------------
pairs = list(zip(misclassified["actual"], misclassified["predicted"]))
pair_counts = Counter(pairs)

print("\n🔄 Most Frequent Confusion Pairs:")
for (a, p), count in pair_counts.most_common(10):
    print(f"{a} ↔ {p} : {count} times")

# -------------------------------
# OPTIONAL: Confusion Matrix Summary
# -------------------------------
print("\n🧮 Computing confusion matrix...")
labels = sorted(list(set(y)))
cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_summary = cm_df.sum(axis=1).to_frame("Total") \
    .join(cm_df.sum(axis=0).to_frame("Predicted_Total")) \
    .assign(Correct=cm_df.values.diagonal())

print("\n✅ Confusion Summary (Top 10 Languages):")
print(cm_summary.head(10))

# -------------------------------
# SAVE RESULTS (OPTIONAL)
# -------------------------------
misclassified.to_csv("misclassified_samples.csv", index=False)
print("\n💾 Misclassified samples saved to 'misclassified_samples.csv'")
