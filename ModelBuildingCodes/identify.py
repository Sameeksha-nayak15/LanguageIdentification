"""
Train an SVM (LinearSVC) language identification model
-------------------------------------------------------
- Uses TF-IDF (char-level) + LinearSVC
- Evaluates performance on test set
- Displays accuracy, confusion matrix
- Generates a per-language table of correct vs misclassified counts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =======================================================
# CONFIGURATION
# =======================================================
DATA_CSV = "../../output_clean.csv"   # Your cleaned dataset
TEST_SIZE = 0.2
RANDOM_STATE = 42
NGRAM_RANGE = (1, 4)
MAX_FEATURES = 3000
TOP_LANGUAGES = 10

# =======================================================
# LOAD DATA
# =======================================================
df = pd.read_csv(DATA_CSV)
print(f"✅ Loaded {len(df)} samples from {DATA_CSV}")

X = df['text']
y = df['class']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# =======================================================
# TRAIN SVC MODEL
# =======================================================
print("\n🚀 Training LinearSVC model...")

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='char', ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)),
    ('clf', LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=RANDOM_STATE))
])

pipeline.fit(X_train, y_train)

# =======================================================
# EVALUATE MODEL
# =======================================================
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =======================================================
# CONFUSION MATRIX
# =======================================================
top_langs = df['class'].value_counts().head(TOP_LANGUAGES).index.tolist()
mask = y_test.isin(top_langs)
cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_langs)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_langs, yticklabels=top_langs)
plt.title(f"LinearSVC Confusion Matrix (Top {TOP_LANGUAGES} Languages)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =======================================================
# TABLE: CORRECT vs MISCLASSIFIED COUNT PER LANGUAGE
# =======================================================
print("\n📋 Per-Language Correct vs Misclassified Summary:\n")

# Create DataFrame with true & predicted labels
results_df = pd.DataFrame({
    'true': y_test,
    'pred': y_pred
})

# Compute correct & misclassified counts
summary = []
for lang in sorted(y_test.unique()):
    total = sum(results_df['true'] == lang)
    correct = sum((results_df['true'] == lang) & (results_df['pred'] == lang))
    misclassified = total - correct
    summary.append({
        'Language': lang,
        'Total Samples': total,
        'Correctly Classified': correct,
        'Misclassified': misclassified,
        'Accuracy (%)': round((correct / total) * 100, 2)
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by="Accuracy (%)", ascending=False)

print(summary_df.to_string(index=False))

# =======================================================
# (OPTIONAL) SAVE TABLE AS CSV
# =======================================================
summary_df.to_csv("svc_language_performance.csv", index=False)
print("\n📁 Saved detailed per-language performance to 'svc_language_performance.csv'")
