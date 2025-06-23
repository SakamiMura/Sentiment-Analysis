"""
Sentiment Analysis with TensorFlow/Keras
A deep learning project comparing LSTM and GRU models for binary sentiment classification.
"""

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Configuration
VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
LSTM_UNITS = 64
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_explore_data():
    """Load Amazon review dataset and display basic information."""
    df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Class distribution:\n{df['Positive'].value_counts()}")
    return df

def preprocess_text(texts):
    """
    Clean and preprocess text data using NLTK.
    
    Args:
        texts (list): List of raw text strings
        
    Returns:
        list: Cleaned and processed text strings
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenizer_nltk = RegexpTokenizer(r'\w+')
    
    def clean_text(text):
        tokens = tokenizer_nltk.tokenize(text.lower())
        filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(filtered)
    
    return [clean_text(text) for text in texts]

def prepare_sequences(texts, tokenizer=None):
    """
    Convert texts to padded sequences for neural network input.
    
    Args:
        texts (list): Preprocessed text strings
        tokenizer (Tokenizer, optional): Pre-fitted tokenizer
        
    Returns:
        tuple: (padded_sequences, tokenizer)
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    print(f"Sequences shape: {padded_sequences.shape}")
    return padded_sequences, tokenizer

def build_lstm_model():
    """
    Build LSTM-based sentiment classification model.
    
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
        LSTM(LSTM_UNITS),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def build_gru_model():
    """
    Build GRU-based sentiment classification model with additional dense layer.
    
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
        GRU(LSTM_UNITS),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Train the model with early stopping and return training history.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_test, y_test: Validation data
        model_name (str): Name for display purposes
        
    Returns:
        History: Training history object
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    print(f"\n{model_name} Architecture:")
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def plot_training_history(history, model_name="Model"):
    """
    Plot training and validation loss with best epoch marker.
    
    Args:
        history: Keras training history
        model_name (str): Model name for plot title
    """
    best_epoch = np.argmin(history.history['val_loss']) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        model_name (str): Model name for display
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{model_name} Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

def main():
    """Main execution pipeline for sentiment analysis."""
    print("=== Sentiment Analysis Pipeline ===\n")
    
    # Data loading and preprocessing
    df = load_and_explore_data()
    texts = preprocess_text(df['reviewText'])
    labels = df['Positive'].tolist()
    
    # Sequence preparation
    padded_sequences, tokenizer = prepare_sequences(texts)
    
    # Train-test split
    X = np.array(padded_sequences)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # LSTM Model
    print("\n" + "="*50)
    print("LSTM MODEL")
    print("="*50)
    lstm_model = build_lstm_model()
    lstm_history = train_model(lstm_model, X_train, y_train, X_test, y_test, "LSTM")
    plot_training_history(lstm_history, "LSTM")
    evaluate_model(lstm_model, X_test, y_test, "LSTM")
    
    # GRU Model
    print("\n" + "="*50)
    print("GRU MODEL")
    print("="*50)
    gru_model = build_gru_model()
    gru_history = train_model(gru_model, X_train, y_train, X_test, y_test, "GRU")
    plot_training_history(gru_history, "GRU")
    evaluate_model(gru_model, X_test, y_test, "GRU")
    
    # Model comparison
    lstm_accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)[1]
    gru_accuracy = gru_model.evaluate(X_test, y_test, verbose=0)[1]
    
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"LSTM Test Accuracy: {lstm_accuracy:.4f}")
    print(f"GRU Test Accuracy: {gru_accuracy:.4f}")
    print(f"Best Model: {'LSTM' if lstm_accuracy > gru_accuracy else 'GRU'}")

if __name__ == "__main__":
    main()