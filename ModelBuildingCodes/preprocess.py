import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer if not already done
nltk.download('punkt')

# -------------------------------
# CONFIGURATION
# -------------------------------
INPUT_CSV = "output_file.csv"          # Original dataset
OUTPUT_CSV = "output_clean.csv"   # Cleaned dataset

# -------------------------------
# HELPER FUNCTION
# -------------------------------
def clean_and_tokenize(text):
    """
    Cleans the text by removing invisible/zero-width characters,
    normalizing spaces, and tokenizing using NLTK.
    Preserves all scripts and punctuation.
    """
    text = str(text)
    
    # Remove zero-width and invisible characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize using NLTK
    tokens = word_tokenize(text)
    
    # Join tokens back to a single string
    return ' '.join(tokens)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} samples from {INPUT_CSV}")

# -------------------------------
# CLEAN AND TOKENIZE
# -------------------------------
df['text'] = df['text'].apply(clean_and_tokenize)

# -------------------------------
# REMOVE EMPTY ROWS (if any)
# -------------------------------
df = df[df['text'].str.strip() != '']
print(f"After cleaning: {len(df)} samples remain")

# -------------------------------
# SAVE CLEANED DATASET
# -------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Cleaned dataset saved to {OUTPUT_CSV}")
