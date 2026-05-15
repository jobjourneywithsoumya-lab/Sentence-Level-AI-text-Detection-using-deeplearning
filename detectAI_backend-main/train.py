import pandas as pd
import re
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import os

nltk.download('punkt')

# Load dataset
file_path = "C:\\Users\\goura\\Desktop\\detect - Copy\\detectAI_backend-main\\finalfeatures_with_perplexity_burstness.csv"
df = pd.read_csv(file_path)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    return text

df['cleaned_text'] = df['Text'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

# Padding sequences
max_length = 100
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels
y = df['Label'].values  

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_units = 64

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(hidden_units, return_sequences=True),
    LSTM(hidden_units),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save model and tokenizer
model.save("ai_detection_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")
