# app.py

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Load Files
# =========================

@st.cache_resource
def load_all():

    # Load Models
    rnn_model = load_model("rnn_model.h5")
    lstm_model = load_model("lstm_model.h5")

    # Load Tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load max length
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return rnn_model, lstm_model, tokenizer, max_len


rnn_model, lstm_model, tokenizer, max_len = load_all()

# =========================
# Reverse Word Mapping
# =========================

index_to_word = {}

for word, index in tokenizer.word_index.items():
    index_to_word[index] = word

# =========================
# Prediction Function
# =========================

def predict_next_word(text, model):

    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text])[0]

    # Padding
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')

    # Predict
    prediction = model.predict(seq, verbose=0)

    # Get highest probability index
    predicted_index = np.argmax(prediction)

    # Convert index to word
    predicted_word = index_to_word.get(predicted_index, "")

    return predicted_word


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Next Word Prediction")
st.write("Predict the next word using RNN or LSTM")

# Select Model
model_option = st.selectbox(
    "Choose Model",
    ["RNN", "LSTM"]
)

# User Input
input_text = st.text_input(
    "Enter Text",
    placeholder="Type a sentence..."
)

# Prediction Button
if st.button("Predict Next Word"):

    if input_text.strip() == "":
        st.warning("Please enter some text")
    else:

        # Choose model
        if model_option == "RNN":
            selected_model = rnn_model
        else:
            selected_model = lstm_model

        # Predict
        next_word = predict_next_word(input_text, selected_model)

        # Display Result
        st.success(f"Predicted Next Word: {next_word}")