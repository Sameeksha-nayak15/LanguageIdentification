import streamlit as st
import requests
from langdetect import detect, DetectorFactory

# Fix random seed for consistent langdetect results
DetectorFactory.seed = 0

# --- Page Config ---
st.set_page_config(
    page_title="Multilingual Language Identifier",
    page_icon="🌍",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            text-align: center;
            color: #2C3E50;
            font-weight: bold;
        }
        .subtext {
            text-align: center;
            color: #7F8C8D;
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #F8F9F9;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #D5D8DC;
            color: #2C3E50;
            font-size: 18px;
            text-align: center;
        }
        textarea {
            font-size: 18px !important;
            font-family: "Noto Sans", "Noto Sans Kannada", "Noto Sans Devanagari", "Noto Sans Tamil", sans-serif;
        }
        .stButton button {
            background-color: #3498DB;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 20px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #2980B9;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 class='main-title'>🌍 Multilingual Language Identifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Enter text in any language (Kannada, Hindi, Tamil, English, etc.) to identify it.</p>", unsafe_allow_html=True)

# --- Input Area ---
user_input = st.text_area(
    "Enter text (any language):",
    height=150,
    placeholder="Type or paste your text here..."
)



# --- Predict Button ---
if st.button("🔍 Identify Language"):
    if user_input.strip():
        cleaned_input = user_input.strip().replace("\u200b", "")
        try:
            with st.spinner("Analyzing language..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict",  # 👈 Your backend API endpoint
                    json={"text": cleaned_input},
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )
                result = response.json()

            # --- Output Box ---
            st.markdown("### 🎯 Prediction Result:")
            if "language_name" in result:
              st.markdown(f"<div class='result-box'>{result['language_name']} ({result['language_code'].upper()})</div>", unsafe_allow_html=True)
            else:
                st.error(result.get("error", "Unexpected response from API."))


        except Exception as e:
            st.error(f"❌ Error connecting to API: {e}")
    else:
        st.warning("Please enter a sentence before predicting.")
