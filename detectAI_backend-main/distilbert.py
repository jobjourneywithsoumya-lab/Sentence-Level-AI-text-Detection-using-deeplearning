import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import softmax
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(list(df['Text']), truncation=True, padding=True, max_length=256, return_tensors='pt')
        self.labels = torch.tensor(df['Label'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Load data
train_df = pd.read_csv('/path/to/HC3trainperplexity_burstness.csv', encoding='latin1', on_bad_lines='skip')[['Text', 'Label']]
test_df = pd.read_csv('/path/to/HC3testperplexity_burstness.csv', encoding='latin1', on_bad_lines='skip')[['Text', 'Label']]

train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

model_save_path = "/path/to/distilbert_ai_detector"

# Train model
def train_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} loss: {total_loss / len(train_loader):.4f}")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to '{model_save_path}'")

# Evaluate model
def evaluate_model():
    model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
    model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch['labels'].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

# Sentence-level prediction
def predict_sentence_probabilities(sentences):
    model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
    model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
    model.eval()
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).cpu().numpy()

    results = []
    for i, sentence in enumerate(sentences):
        ai_percent = round(probs[i][1] * 100, 2)
        human_percent = round(probs[i][0] * 100, 2)
        results.append({
            "sentence": sentence,
            "AI": f"{ai_percent}%",
            "Human": f"{human_percent}%"
        })
    return results
