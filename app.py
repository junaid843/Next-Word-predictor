# Streamlit UI for Text Generation
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set page config
st.set_page_config(
    page_title="Text Generator with LSTM",
    page_icon="📝",
    layout="centered"
)

# Title and description
st.title("📝 LSTM Text Generation Model")
st.markdown("Generate text using a trained LSTM neural network")
st.markdown("---")

# Load model and tokenizer with caching
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model('textgen_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer()

if model is not None and tokenizer is not None:
    # Get max sequence length from model
    max_sequence_len = model.input_shape[1] + 1
    
    # Sidebar with info
    st.sidebar.header("Model Information")
    st.sidebar.info(f"""
    - Model: LSTM Neural Network
    - Vocabulary size: {len(tokenizer.word_index) + 1}
    - Max sequence length: {max_sequence_len}
    - Framework: TensorFlow Keras
    """)
    
    # Main input section
    st.subheader("Generate New Text")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        seed_text = st.text_input(
            "Seed Text (starting words):",
            value="the sun",
            help="Enter the starting text for generation"
        )
    
    with col2:
        num_words = st.slider(
            "Number of words to generate:",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Select how many words to generate"
        )
    
    # Temperature control for creativity
    temperature = st.slider(
        "Temperature (creativity):",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Lower = more predictable, Higher = more creative"
    )
    
    # Generate button
    generate_button = st.button("✨ Generate Text", type="primary", use_container_width=True)
    
    # Function to generate text with temperature
    def generate_text_with_temperature(seed_text, next_words, model, tokenizer, max_sequence_len, temperature):
        generated_text = seed_text
        
        for _ in range(next_words):
            # Tokenize seed text
            token_list = tokenizer.texts_to_sequences([generated_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            
            # Predict
            predictions = model.predict(token_list, verbose=0)[0]
            
            # Apply temperature
            predictions = np.log(predictions + 1e-7) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)
            
            # Sample from the distribution
            predicted_index = np.random.choice(range(len(predictions)), p=predictions)
            
            # Find the word
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    generated_text += " " + word
                    break
        
        return generated_text
    
    # Generate text when button is clicked
    if generate_button:
        if seed_text.strip():
            with st.spinner("Generating text..."):
                try:
                    generated_output = generate_text_with_temperature(
                        seed_text.strip(),
                        num_words,
                        model,
                        tokenizer,
                        max_sequence_len,
                        temperature
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Generated Text")
                    st.markdown(f"**Seed text:** `{seed_text}`")
                    st.markdown(f"**Generated words:** {num_words}")
                    st.markdown(f"**Temperature:** {temperature}")
                    st.markdown("---")
                    
                    # Display in a nice box
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                            <p style="font-size: 18px; margin: 0; line-height: 1.6;">
                                {generated_output}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Option to copy
                    st.markdown("---")
                    st.caption("💡 Tip: You can copy the generated text by selecting it above")
                    
                except Exception as e:
                    st.error(f"Error during generation: {e}")
        else:
            st.warning("Please enter a seed text")
    
    # Example suggestions
    with st.expander("💡 Example Seed Texts"):
        st.markdown("""
        Try these seed texts:
        - `the sun`
        - `in the beginning`
        - `dark night`
        - `love is`
        - `the world`
        
        Click on any example above, paste it in the input box, and generate!
        """)
    
    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ using LSTM Neural Networks and Streamlit")

else:
    st.error("""
    ❌ **Model or tokenizer not found!**
    
    Please ensure that 'textgen_model.h5' and 'tokenizer.pkl' files are in the current directory.
    
    **Steps to fix:**
    1. Train the model using the Jupyter notebook
    2. Make sure both model files are saved
    3. Place them in the same directory as this app
    4. Restart the Streamlit app
    """)
