# Sentence-Level-AI-text-detection

# Sentence-Level AI Text Detection

A full-stack application to detect AI-generated content in text at the sentence level, leveraging LSTM, DistilBERT, and BERT models. The backend is built with FastAPI, and the frontend is a React application.

## Table of Contents

* [Features](#features)
* [Tech Stack](#tech-stack)
* [Repository Structure](#repository-structure)
* [Prerequisites](#prerequisites)
* [Installation](#installation)

  * [Backend Setup](#backend-setup)
  * [Frontend Setup](#frontend-setup)
* [Running the Application](#running-the-application)

  * [Start Backend](#start-backend)
  * [Start Frontend](#start-frontend)
* [API Endpoints](#api-endpoints)
* [Usage](#usage)
* [Configuration](#configuration)
* [Contributing](#contributing)
* [License](#license)

## Features

* Sentence-level AI vs. human detection
* Combined scoring using BERT probability and edit distance
* Switchable models: LSTM and DistilBERT for sentence classification
* FastAPI backend with CORS support
* React frontend with live word count and colored confidence slider

## Tech Stack

| Layer        | Technology                                               |
| ------------ | -------------------------------------------------------- |
| Backend      | Python, FastAPI, TensorFlow, PyTorch, Transformers, nltk |
| Frontend     | React, Axios, CSS                                        |
| Models       | LSTM, DistilBERT, BERT                                   |
| Package Mgmt | pip, npm                                                 |

## Repository Structure

```
Sentence-Level-AI-text-detection/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── ai_detection_model_lstm.h5
│   └── distilbert_ai_detector/
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── ...
│   └── public/
│       └── index.html
└── README.md
```

## Prerequisites

* Python 3.8+
* Node.js 14+
* Git
* (Optional) CUDA-enabled GPU for faster model inference

## Installation

### Backend Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Gourabh09/Sentence-Level-AI-text-detection.git
   cd Sentence-Level-AI-text-detection/backend
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download or train models**:

   * Place `ai_detection_model_lstm.h5` and `tokenizer_lstm.pkl` in the backend folder.
   * Place your DistilBERT model directory named `distilbert_ai_detector` here.

### Frontend Setup

1. **Navigate to frontend**:

   ```bash
   cd ../frontend
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

## Running the Application

### Start Backend

From the `backend/` directory:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

From the `frontend/` directory:

```bash
npm start
```

The React app will open at `http://localhost:3000` and proxy API calls to the FastAPI backend at `http://localhost:8000`.

## API Endpoints

| Method | Endpoint   | Payload                                 | Description      |                                                 |
| ------ | ---------- | --------------------------------------- | ---------------- | ----------------------------------------------- |
| GET    | `/`        | —                                       | Health check     |                                                 |
| POST   | `/predict` | \`{ text: string, model\_choice: "lstm" | "distilbert" }\` | Returns AI score and sentence-level predictions |

### Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your input text here.", "model_choice": "distilbert"}'
```

## Usage

1. Enter or paste your text in the textarea.
2. Select the detection model (LSTM or DistilBERT).
3. Click **Upload** to see:

   * **AI Detection Score** (combined BERT + edit distance)
   * **Sentence-Level Predictions** with confidence bars.

## Configuration

* **MAX\_LENGTH** (512) and **DISTILBERT\_MAX\_LENGTH** (256) can be adjusted in `main.py`.
* Model paths and CSV data paths are defined at the top of `main.py`.

## Contributing

Contributions are welcome! Please fork the repo and open a pull request with your improvements.


