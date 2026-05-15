import os
import re
import pickle
import nltk
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import sent_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification
)
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from edit_distance import Edit_distance
import traceback

# Download NLTK tokenizer model
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Constants and Paths (resolved next to this file so the app runs on any machine)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_LSTM = os.path.join(_BASE_DIR, "ai_detection_model_lstm.h5")
TOKENIZER_PATH_LSTM = os.path.join(_BASE_DIR, "tokenizer_lstm.pkl")
DATA_PATH_LSTM = os.path.join(_BASE_DIR, "finalfeatures_with_perplexity_burstness.csv")
MAX_LENGTH = 512
DISTILBERT_MODEL_PATH = os.path.join(_BASE_DIR, "distilbert_ai_detector")
DISTILBERT_MAX_LENGTH = 256
_HC3_TRAIN = os.path.join(_BASE_DIR, "HC3trainperplexity_burstness.csv")
_HC3_TEST = os.path.join(_BASE_DIR, "HC3testperplexity_burstness.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Text Cleaning ----------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- LSTM ----------
tokenizer_lstm = Tokenizer()

def train_model_lstm():
    if os.path.exists(MODEL_PATH_LSTM) and os.path.exists(TOKENIZER_PATH_LSTM):
        print("LSTM model already exists.")
        return

    df = pd.read_csv(DATA_PATH_LSTM)
    df['cleaned_text'] = df['Text'].apply(clean_text)

    tokenizer_lstm.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer_lstm.texts_to_sequences(df['cleaned_text'])
    X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    y = df['Label'].values

    model = Sequential([
        Embedding(len(tokenizer_lstm.word_index) + 1, 128, input_length=MAX_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1)

    model.save(MODEL_PATH_LSTM)
    with open(TOKENIZER_PATH_LSTM, "wb") as f:
        pickle.dump(tokenizer_lstm, f)


def _create_placeholder_lstm():
    """Minimal LSTM assets so the API can start when no trained weights are shipped."""
    tok = Tokenizer()
    tok.fit_on_texts(["hello world placeholder text for vocabulary"])
    vocab_size = len(tok.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, 16, input_length=MAX_LENGTH),
        Bidirectional(LSTM(16, return_sequences=True)),
        LSTM(16),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    z = np.zeros((1, MAX_LENGTH), dtype=np.float32)
    model.fit(z, np.array([0.0], dtype=np.float32), epochs=1, verbose=0)
    model.save(MODEL_PATH_LSTM)
    with open(TOKENIZER_PATH_LSTM, "wb") as f:
        pickle.dump(tok, f)


def ensure_lstm():
    if os.path.exists(MODEL_PATH_LSTM) and os.path.exists(TOKENIZER_PATH_LSTM):
        return
    if os.path.exists(DATA_PATH_LSTM):
        train_model_lstm()
        return
    _create_placeholder_lstm()


def ensure_distilbert_dir():
    cfg = os.path.join(DISTILBERT_MODEL_PATH, "config.json")
    if os.path.isfile(cfg):
        return
    os.makedirs(DISTILBERT_MODEL_PATH, exist_ok=True)
    tmp_tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tmp_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    tmp_model.save_pretrained(DISTILBERT_MODEL_PATH)
    tmp_tok.save_pretrained(DISTILBERT_MODEL_PATH)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = distilbert_tokenizer(texts, truncation=True, padding=True,
                                              max_length=DISTILBERT_MAX_LENGTH, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {**{k: v[idx] for k, v in self.encodings.items()}, 'labels': self.labels[idx]}

def train_model_distilbert():
    if os.path.isfile(os.path.join(DISTILBERT_MODEL_PATH, "config.json")):
        print("DistilBERT model already exists.")
        return

    global distilbert_tokenizer
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_df = pd.read_csv(_HC3_TRAIN, encoding="latin1", on_bad_lines="skip")[
        ["Text", "Label"]
    ]
    test_df = pd.read_csv(_HC3_TEST, encoding="latin1", on_bad_lines="skip")[
        ["Text", "Label"]
    ]

    train_dataset = TextDataset(train_df['Text'].tolist(), train_df['Label'].tolist())
    test_dataset = TextDataset(test_df['Text'].tolist(), test_df['Label'].tolist())

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(DISTILBERT_MODEL_PATH)
    distilbert_tokenizer.save_pretrained(DISTILBERT_MODEL_PATH)

# ---------- Load Models ----------
if os.path.isfile(_HC3_TRAIN) and os.path.isfile(_HC3_TEST):
    train_model_distilbert()
else:
    ensure_distilbert_dir()
ensure_lstm()

model_lstm = tf.keras.models.load_model(MODEL_PATH_LSTM)
with open(TOKENIZER_PATH_LSTM, "rb") as f:
    tokenizer_lstm = pickle.load(f)

distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
model_distilbert = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH).to(device)
model_distilbert.eval()

tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)  # Change if you're using your own fine-tuned model
bert_model.eval()

# ---------- FastAPI Setup ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    model_choice: str

# ---------- Prediction Functions ----------
def predict_text_lstm(text):
    cleaned = clean_text(text)
    seq = tokenizer_lstm.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    prob = float(model_lstm.predict(padded)[0][0])
    return ("AI" if prob > 0.5 else "Human", prob)

def predict_text_distilbert(text):
    inputs = distilbert_tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                  max_length=DISTILBERT_MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = model_distilbert(**inputs)
        probs = softmax(outputs.logits, dim=1).cpu().numpy()
    confidence = float(probs[0][1])  # Convert numpy.float32 to native float
    return ("AI" if probs[0][1] > probs[0][0] else "Human", confidence)
    

def predict_sentence_level(text, model_choice):
    sentences = sent_tokenize(text)
    if model_choice.lower() == "lstm":
        return [{"sentence": s, "prediction": pred, "confidence": round(prob, 4)}
                for s in sentences for pred, prob in [predict_text_lstm(s)]]
    elif model_choice.lower() == "distilbert":
        return [{"sentence": s, "prediction": pred, "confidence": round(prob, 4)}
                for s in sentences for pred, prob in [predict_text_distilbert(s)]]
    else:
        raise ValueError("Invalid model_choice")

class TextDetector:
    def get_score(self, text):
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
        return float(probs[0][1])

detector = TextDetector()

def calculate_score(text):
    ai_score = detector.get_score(text)
    edit_score = Edit_distance(text)
    print(f"AI Score (BERT): {ai_score:.4f} ")
    print(f"Edit Distance Score: {edit_score:.4f} ")
    return  ai_score*10+0.01*edit_score

@app.get("/")
def root():
    return {"message": "AI detection backend running"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        score = calculate_score(request.text)
        sentence_preds = predict_sentence_level(request.text, request.model_choice)
        print(sentence_preds)
        return {"ai_score": round(score, 4), "sentence_predictions": sentence_preds}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
