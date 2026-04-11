import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("text_generation_lstm.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max sequence length
with open("maxlen.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)

# Text generation function
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

# Streamlit UI
st.set_page_config(page_title="Text Generator (LSTM)", page_icon="📝")


st.title("📝 Wikipedia Text Generator using LSTM")
st.write("Enter a seed text and generate new text using trained LSTM model.")

seed_text = st.text_input("Enter Seed Text:", "the world is")

num_words = st.slider("Number of words to generate:", 5, 100, 20)

if st.button("Generate Text"):
    if seed_text.strip() == "":
        st.warning("Please enter seed text!")
    else:
        result = generate_text(seed_text.lower(), num_words)
        st.success("Generated Text:")
        st.write(result)