import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM model for sentence-level prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.softmax(out)

# LSTM Dataset
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['Text'].values
        self.labels = df['Label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0)
        return input_ids, label

# Training LSTM model (to be implemented)
def train_lstm_model():
    pass

# Sentence-level prediction using LSTM model
def predict_sentence_lstm(sentences, model, tokenizer):
    model.eval()
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs
