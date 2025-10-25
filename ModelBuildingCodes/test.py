import joblib

# Load your trained model
pipeline = joblib.load("../saved_models/language_pipeline.joblib")  # adjust path if needed

# Sample multilingual & code-mixed texts
test_sentences = [
    "ನಾನು school ಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ",             # Kannada-English mix
    "Main kal office nahi gaya",                 # Hindi-English mix
    "I will call you later",                     # English
    "Je vais au marché aujourd'hui",             # French
    "今日は天気がいいですね",                        # Japanese
    "Hola amigo, cómo estás?",                   # Spanish
    "நான் இன்று மிகவும் மகிழ்ச்சியாக இருக்கிறேன்",     # Tamil
    "मुझे खाना पसंद है",                          # Hindi
    "Coffee ಬೇಕು please",                        # Kannada-English mix
    "😂😊💪😍"
]

print("\n🔍 Testing multilingual and code-mixed sentences:\n")
for text in test_sentences:
    prediction = pipeline.predict([text])[0]
    print(f"Input: {text}")
    print(f"→ Predicted Language: {prediction}\n")
