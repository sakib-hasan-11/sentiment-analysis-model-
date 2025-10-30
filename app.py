import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

# ----------- Helper function -----------
def load_tokenizer(path="tokenizer.json"):
    """Load the tokenizer from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def predict_sentiment(text):
    """Preprocess input, pad it, and predict sentiment using trained model."""
    tokenizer = load_tokenizer("tokenizer.json")
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding='pre')

    # Load trained model
    model = load_model("self_trained_model.h5")

    # Predict
    pred = model.predict(padded)
    sentiment = "Positive 😊" if pred[0][0] > 0.5 else "Negative 😞"
    confidence = float(pred[0][0])
    return sentiment, confidence

# ----------- Streamlit Layout -----------
st.set_page_config(page_title="Movie Review Classifier", layout="centered")

st.title("🎬 Movie Review Sentiment App")
st.write("Welcome! Please choose one of the following options:")

# Navigation options
option = st.radio(
    "Select an option:",
    ("Use Self-Trained Model", "Use Transformer Model", "Compare Both Models")
)

# ----------- Option 1: Self-trained Model -----------
if option == "Use Self-Trained Model":
    st.subheader("🧩 Self-Trained Model Review Classification")

    review = st.text_area("Enter your movie review:")
    if st.button("Classify Review"):
        if review.strip():
            try:
                sentiment, confidence = predict_sentiment(review)
                st.success(f"Predicted Sentiment: **{sentiment}**")
                st.info(f"Model Confidence: {confidence:.2f}")
            except FileNotFoundError as e:
                st.error("❌ Missing model or tokenizer file. Please make sure both are in the same directory.")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
        else:
            st.warning("Please enter a review before classifying.")

# ----------- Option 2: Transformer Model -----------
elif option == "Use Transformer Model":
    st.subheader("🚀 Transformer Model (Coming Soon)")
    st.info("This section will use a pretrained Transformer (e.g., BERT) to classify reviews.")

# ----------- Option 3: Compare Both Models -----------
else:
    st.subheader("⚖️ Compare Both Models (Coming Soon)")
    st.info("This section will compare predictions from both models side by side.")
