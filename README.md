# Next Word Prediction using LSTM

This project implements a next word prediction model using Long Short-Term Memory (LSTM) neural networks. The model is trained on a Wikipedia text dataset and can generate coherent text based on a given seed phrase.

## Features

- **Model Training**: Train an LSTM model on Wikipedia text data to predict the next word in a sequence.
- **Text Generation**: Use the trained model to generate new text continuations from a seed text.
- **Web Interface**: A Streamlit-based web app for easy text generation.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to train the LSTM model:

```bash
python train_model.py
```

This will:
- Download the Wikipedia dataset using KaggleHub.
- Preprocess the text data.
- Train the LSTM model.
- Save the trained model (`text_generation_lstm.h5`), tokenizer (`tokenizer.pkl`), and maximum sequence length (`maxlen.pkl`).

**Note**: Training may take some time depending on your hardware. The script limits the data to 5 files and 2 million characters for faster training.

### Running the Web App

After training, launch the Streamlit web app:

```bash
streamlit run app.py
```

This will open a web interface where you can:
- Enter a seed text (e.g., "the world is").
- Specify the number of words to generate.
- Click "Generate Text" to see the model's prediction.

## Files

- `train_model.py`: Script to train the LSTM model.
- `train_model.ipynb`: Jupyter notebook version of the training script.
- `app.py`: Streamlit web app for text generation.
- `requirements.txt`: List of Python dependencies.
- `text_generation_lstm.h5`: Trained model (generated after training).
- `tokenizer.pkl`: Tokenizer object (generated after training).
- `maxlen.pkl`: Maximum sequence length (generated after training).

## Dependencies

- numpy: For numerical computations.
- tensorflow: For building and training the LSTM model.
- kagglehub: For downloading the dataset.
- streamlit: For the web interface.

## License

This project is for educational purposes. Please ensure you comply with the licenses of the datasets and libraries used.