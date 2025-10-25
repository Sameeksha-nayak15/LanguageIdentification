import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_CSV = "output_clean.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
NGRAM_RANGE = (1, 4)    # Character-level n-grams (reduced from 5)
MAX_FEATURES = 3000     # Reduced from 5000
TOP_LANGUAGES = 10       # For confusion matrix plotting

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} samples from {DATA_CSV}")

X = df['text']
y = df['class']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------------------
# DEFINE MODELS
# -------------------------------
models = {
    "LinearSVC": LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,           # Reduced from 200
        max_depth=20,              # Added to limit tree size
        min_samples_leaf=2,        # Added to prevent very deep branches
        max_features='sqrt',       # Use sqrt of features for splits
        random_state=RANDOM_STATE,
        n_jobs=2                   # Limit parallel jobs
    ),
    "NaiveBayes": MultinomialNB()
}

# -------------------------------
# TRAIN AND EVALUATE
# -------------------------------
results = {}

for name, clf in models.items():
    print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
    
    # Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)),
        ('clf', clf)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    
    # Classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\nClassification Report:\n{report}")
    
    # Top N languages for confusion matrix
    top_langs = df['class'].value_counts().head(TOP_LANGUAGES).index.tolist()
    mask = y_test.isin(top_langs)
    cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_langs)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_langs, yticklabels=top_langs)
    plt.title(f"{name} Confusion Matrix (Top {TOP_LANGUAGES} Languages)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Misclassified examples
    misclassified = X_test[y_test != y_pred]
    mis_labels = y_test[y_test != y_pred]
    mis_preds = y_pred[y_test != y_pred]
    print(f"\nSample Misclassified Examples ({name}):")
    for i in range(min(5, len(misclassified))):
        print(f"Text: {misclassified.iloc[i][:50]}... | True: {mis_labels.iloc[i]} | Pred: {mis_preds[i]}")
    
    # Store results
    results[name] = {
        "accuracy": acc,
        "misclassified_count": len(misclassified)
    }

# -------------------------------
# SUMMARY
# -------------------------------
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
for name, res in results.items():
    print(f"{name}: Accuracy = {res['accuracy']*100:.2f}%, Misclassified = {res['misclassified_count']}")
