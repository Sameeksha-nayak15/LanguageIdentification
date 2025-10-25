"""
Language Identification - LinearSVC Training and Saving
------------------------------------------------------
- Trains a LinearSVC model using character-level TF-IDF features
- Evaluates accuracy, prints classification report & confusion matrix
- Saves the trained TF-IDF + SVC pipeline as 'language_pipeline.joblib'
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_CSV = "output_clean.csv"       # Your cleaned dataset
TEST_SIZE = 0.2
RANDOM_STATE = 42
NGRAM_RANGE = (1, 4)                # Character-level n-grams
MAX_FEATURES = 3000                 # Reduced for speed
TOP_LANGUAGES = 10
SAVE_DIR = "saved_models"
MODEL_PATH = os.path.join(SAVE_DIR, "language_pipeline.joblib")

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_CSV)
print(f"📂 Loaded {len(df)} samples from {DATA_CSV}")

X = df['text']
y = df['class']

# -------------------------------
# TRAIN/TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"✅ Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------------------
# DEFINE MODEL PIPELINE
# -------------------------------
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='char',
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES
    )),
    ('clf', LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=RANDOM_STATE))
])

# -------------------------------
# TRAIN MODEL
# -------------------------------
print("\n🚀 Training LinearSVC model...")
pipeline.fit(X_train, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ LinearSVC Accuracy: {acc*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
top_langs = df['class'].value_counts().head(TOP_LANGUAGES).index.tolist()
mask = y_test.isin(top_langs)
cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_langs)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_langs, yticklabels=top_langs)
plt.title(f"LinearSVC Confusion Matrix (Top {TOP_LANGUAGES} Languages)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------
# SAVE PIPELINE
# -------------------------------
print("\n💾 Retraining on full dataset and saving pipeline...")
pipeline.fit(X, y)  # retrain on full data
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Model pipeline saved successfully at: {MODEL_PATH}")
