"""
FastAPI Backend for Language Identification
-------------------------------------------
- Loads a pre-trained SVM + TF-IDF pipeline (joblib)
- Loads a CSV mapping of language codes to full names
- Exposes an API endpoint for Streamlit frontend integration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# =====================================================
# 🔧 CONFIGURATION
# =====================================================

PIPELINE_PATH =  "../saved_models/language_pipeline.joblib"
LANGUAGE_CSV =  "../saved_models/language_codes.csv" # <-- must exist

# =====================================================
# 🚀 FastAPI App Initialization
# =====================================================
app = FastAPI(
    title="Language Identification API",
    description="Detects the language of input text using a trained SVM model.",
    version="2.0"
)

# Enable CORS for Streamlit or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # For local development; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 📦 LOAD MODEL PIPELINE
# =====================================================
if os.path.exists(PIPELINE_PATH):
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print(f"✅ Model pipeline loaded successfully from {PIPELINE_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model pipeline: {e}")
        pipeline = None
else:
    print(f"❌ Model file not found at {PIPELINE_PATH}")
    pipeline = None

# =====================================================
# 📘 LOAD LANGUAGE CODE MAPPING
# =====================================================
if os.path.exists(LANGUAGE_CSV):
    try:
        lang_df = pd.read_csv(LANGUAGE_CSV)
        # Attempt to auto-detect the code and English-name columns (support several common headings)
        cols = {c.strip().lower(): c for c in lang_df.columns}
        # candidates for code column
        code_candidates = ['code', 'label', 'wiki code', 'wiki_code', 'iso 369-3', 'iso', 'id']
        name_candidates = ['english', 'language_name', 'language', 'name', 'english_name']

        code_col = None
        name_col = None
        for cand in code_candidates:
            if cand in cols:
                code_col = cols[cand]
                break
        for cand in name_candidates:
            if cand in cols:
                name_col = cols[cand]
                break

        # Fallbacks
        if code_col is None and 'Label' in lang_df.columns:
            code_col = 'Label'
        if name_col is None and 'English' in lang_df.columns:
            name_col = 'English'

        if code_col is None or name_col is None:
            raise ValueError(f"Could not find code/name columns in {LANGUAGE_CSV}. Found columns: {list(lang_df.columns)}")

        # Build mapping: normalize codes to lower-case and strip whitespace
        lang_map = {}
        for _, row in lang_df[[code_col, name_col]].iterrows():
            code_val = str(row[code_col]).strip()
            name_val = str(row[name_col]).strip()
            if code_val and name_val and code_val.lower() not in ('nan', 'none'):
                lang_map[code_val.lower()] = name_val

        print(f"✅ Loaded {len(lang_map)} language mappings from {LANGUAGE_CSV} (code column: '{code_col}', name column: '{name_col}')")
    except Exception as e:
        print(f"⚠️ Failed to load language mapping CSV: {e}")
        lang_map = {}
else:
    print(f"⚠️ Language mapping CSV not found at {LANGUAGE_CSV}.")
    lang_map = {}

# =====================================================
# 📩 REQUEST SCHEMA
# =====================================================
class TextInput(BaseModel):
    text: str

# =====================================================
# 🔍 PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict_language(data: TextInput):
    """
    Predicts the language of the input text using the trained SVM model.
    Returns both language code and full name (if available).
    """
    if pipeline is None:
        return {"error": "Model not loaded properly on the server."}

    text = data.text.strip()
    if not text:
        return {"error": "Empty input text."}

    try:
        # Remove hidden characters
        cleaned_text = text.replace("\u200b", "")
        prediction_code_raw = pipeline.predict([cleaned_text])[0]
        prediction_code = str(prediction_code_raw).strip()

        # Normalize code for lookup (lang_map keys are lower-cased)
        code_key = prediction_code.lower()
        full_name = lang_map.get(code_key)

        # If not found, try fallback: sometimes pipeline returns ISO or wiki code
        if full_name is None:
            # try uppercase, plain fallback
            full_name = lang_map.get(prediction_code) or lang_map.get(prediction_code.upper())

        # final fallback: return the code itself as name
        if full_name is None:
            full_name = prediction_code

        return {
            "language_code": prediction_code,
            "language_name": full_name
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# =====================================================
# 🏁 ROOT ENDPOINT
# =====================================================
@app.post("/predict")
async def root():
    return {"message": "✅ Language Identification API is running!"}

# =====================================================
# ▶️ MAIN ENTRY POINT
# =====================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
