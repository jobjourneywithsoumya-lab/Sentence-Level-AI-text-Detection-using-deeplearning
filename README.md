# Sentence-Level AI Text Detection using Deep Learning

> **Detect AI-Generated Content at the Sentence Level** — A full-stack application using LSTM, DistilBERT, and BERT models to identify AI-written text with high precision.

[![GitHub Repo](https://img.shields.io/badge/GitHub-Sentence--Level--AI--Text--Detection-black?style=flat-square&logo=github)](https://github.com/jobjourneywithsoumya-lab/Sentence-Level-AI-text-Detection-using-deeplearning)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-blue?style=flat-square)](https://sentence-level-ai-text-detection.vercel.app/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green?style=flat-square)](http://localhost:8000)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-14%2B-green)](https://nodejs.org/)

## 📋 Overview

**Sentence-Level AI Text Detection** is a sophisticated full-stack application designed to identify and classify AI-generated content at the sentence level. By combining multiple deep learning models (LSTM, DistilBERT, and BERT) with advanced scoring algorithms, this tool provides granular insights into which sentences in a text are likely to be AI-generated.

### Key Highlights
- 🎯 **Sentence-Level Detection** — Identifies AI content at the sentence level, not just document-level
- 🧠 **Multi-Model Approach** — Leverages LSTM, DistilBERT, and BERT for robust predictions
- 📊 **Combined Scoring** — Uses BERT probability + edit distance for accurate results
- ⚡ **Fast Inference** — Optimized models for quick real-time predictions
- 🎨 **Intuitive UI** — Clean React frontend with confidence sliders and live word count
- 🔌 **RESTful API** — Well-documented FastAPI backend with CORS support

## 🎬 Demo & Live Access

- **Live Application:** [https://sentence-level-ai-text-detection.vercel.app/](https://sentence-level-ai-text-detection.vercel.app/)
- **Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1SGXhR4d-dBKdhutvmqdldNE75xYaqjmX/view?usp=drive_link)

## ✨ Features

### Detection Capabilities
- ✅ Sentence-level AI vs. human text classification
- ✅ Combined confidence scoring using BERT probability and edit distance
- ✅ Switchable models: LSTM and DistilBERT for flexible classification
- ✅ Batch text processing with performance metrics
- ✅ Color-coded confidence indicators
- ✅ Live word count and character analysis

### Technical Features
- ✅ FastAPI backend with async support
- ✅ CORS enabled for cross-origin requests
- ✅ React frontend with responsive design
- ✅ Real-time sentence-level predictions
- ✅ Model switching without restart
- ✅ Comprehensive error handling

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React, Axios, CSS3 |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML/DL** | TensorFlow, PyTorch, Transformers, NLTK |
| **Models** | LSTM, DistilBERT, BERT |
| **Deployment** | Vercel (Frontend), Any Python Server (Backend) |
| **Package Management** | pip, npm |

## 📁 Repository Structure

```
Sentence-Level-AI-text-detection/
│
├── backend/
│   ├── main.py                          # FastAPI application entry point
│   ├── requirements.txt                 # Python dependencies
│   ├── models/
│   │   ├── ai_detection_model_lstm.h5  # Trained LSTM model
│   │   ├── tokenizer_lstm.pkl          # LSTM tokenizer
│   │   ├── distilbert_ai_detector/     # DistilBERT model directory
│   │   └── bert_model/                 # BERT model directory
│   └── data/
│       └── training_data.csv           # Training dataset
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── manifest.json
│   ├── src/
│   │   ├── App.js                      # Main React component
│   │   ├── App.css                     # Styling
│   │   ├── components/
│   │   ├── pages/
│   │   └── index.js
│   ├── package.json
│   └── package-lock.json
│
├── .gitignore
├── README.md
└── LICENSE
```

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (3.9+ recommended)
- **Node.js 14+** (16+ recommended)
- **npm** or **yarn** package manager
- **Git** for version control
- *(Optional)* CUDA-enabled GPU for faster model inference

### System Requirements
- **RAM:** 4GB minimum (8GB+ recommended for model loading)
- **Disk Space:** ~2GB for models and dependencies
- **GPU:** Optional but recommended for faster inference

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/jobjourneywithsoumya-lab/Sentence-Level-AI-text-Detection-using-deeplearning.git
cd Sentence-Level-AI-text-Detection-using-deeplearning
```

### Step 2: Backend Setup

#### Create Virtual Environment
```bash
cd backend

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

#### Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Download/Prepare Models

The following models need to be in the `backend/` directory:

1. **LSTM Model** — Place these files:
   - `ai_detection_model_lstm.h5` — Trained LSTM model
   - `tokenizer_lstm.pkl` — Pickled tokenizer

2. **DistilBERT Model** — Create directory:
   ```bash
   mkdir distilbert_ai_detector
   # Add pretrained DistilBERT model files
   ```

3. **BERT Model** (if using) — Create directory:
   ```bash
   mkdir bert_model
   # Add BERT model files
   ```

### Step 3: Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create .env file (optional)
echo "REACT_APP_API_URL=http://localhost:8000" > .env
```

## 🎯 Running the Application

### Start Backend Server

```bash
cd backend

# Activate virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Start Frontend Application

```bash
cd frontend

# Start development server
npm start
```

Frontend opens at: `http://localhost:3000`

## 🔌 API Endpoints

### Health Check
```http
GET /
```
**Response:** Server health status

---

### Text Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "text": "Your text here. This is another sentence.",
  "model_choice": "distilbert"
}
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `string` | Input text to analyze (required) |
| `model_choice` | `string` | Model to use: `"lstm"` or `"distilbert"` (required) |

**Response:**
```json
{
  "overall_score": 0.75,
  "confidence": 0.82,
  "sentences": [
    {
      "sentence": "Your text here.",
      "ai_probability": 0.68,
      "is_ai": true,
      "confidence": 0.78
    },
    {
      "sentence": "This is another sentence.",
      "ai_probability": 0.42,
      "is_ai": false,
      "confidence": 0.71
    }
  ],
  "model_used": "distilbert",
  "processing_time": 0.234
}
```

---

### Switch Model
```http
POST /set-model
```

**Request Body:**
```json
{
  "model_choice": "lstm"
}
```

---

### Get Current Model
```http
GET /current-model
```

**Response:**
```json
{
  "current_model": "distilbert"
}
```

## 📝 Usage Guide

### Web Interface

1. **Open Application**
   - Navigate to `http://localhost:3000`

2. **Input Text**
   - Paste or type your text in the textarea
   - Live word count displayed below

3. **Select Model**
   - Choose between LSTM or DistilBERT
   - Each model offers different trade-offs:
     - **LSTM:** Faster, slightly lower accuracy
     - **DistilBERT:** More accurate, slightly slower

4. **Click "Analyze"**
   - Backend processes the text
   - Returns sentence-level predictions

5. **Review Results**
   - Overall AI confidence score
   - Color-coded sentence classifications:
     - 🔴 Red: Likely AI-generated
     - 🟡 Yellow: Uncertain
     - 🟢 Green: Likely human-written
   - Individual confidence scores per sentence

### cURL Examples

**Basic Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test sentence. AI generated content detection is important.",
    "model_choice": "distilbert"
  }'
```

**Change Model:**
```bash
curl -X POST http://localhost:8000/set-model \
  -H "Content-Type: application/json" \
  -d '{"model_choice": "lstm"}'
```

**Check Current Model:**
```bash
curl http://localhost:8000/current-model
```

## ⚙️ Configuration

### Backend Configuration (`main.py`)

Adjust these parameters in the backend code:

```python
# Model parameters
MAX_LENGTH = 512              # Maximum sequence length for LSTM
DISTILBERT_MAX_LENGTH = 256   # Maximum length for DistilBERT
CONFIDENCE_THRESHOLD = 0.5    # Threshold for AI classification

# Model paths
LSTM_MODEL_PATH = "ai_detection_model_lstm.h5"
LSTM_TOKENIZER_PATH = "tokenizer_lstm.pkl"
DISTILBERT_MODEL_PATH = "distilbert_ai_detector"
```

### Frontend Configuration

Create `.env` file in frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000
```

## 🤖 Model Details

### LSTM Model
- **Architecture:** Bidirectional LSTM with embedding layer
- **Input:** Tokenized sentences (max 512 tokens)
- **Output:** Binary classification (AI or Human)
- **Speed:** Fast inference (~50-100ms)
- **Accuracy:** ~85%

### DistilBERT Model
- **Architecture:** Distilled version of BERT (40% smaller, 60% faster)
- **Input:** Tokenized text (max 256 tokens)
- **Output:** Probability score for AI generation
- **Speed:** Medium inference (~100-200ms)
- **Accuracy:** ~92%

### BERT Model
- **Architecture:** Full BERT transformer
- **Input:** Tokenized text (max 512 tokens)
- **Output:** Rich contextual embeddings
- **Speed:** Slower inference (~200-400ms)
- **Accuracy:** ~95%

## 🧠 How It Works

### Detection Algorithm

1. **Text Preprocessing**
   - Sentence tokenization using NLTK
   - Cleaning and normalization

2. **Feature Extraction**
   - Model-specific tokenization
   - Padding/truncation to max length

3. **Prediction**
   - Forward pass through selected model
   - Get probability scores

4. **Scoring**
   - BERT probability: Direct model output
   - Edit distance: Measure of text naturalness
   - Combined score: Weighted average

5. **Classification**
   - Threshold-based classification
   - Confidence calculation
   - Sentence-level tagging

## 📊 Performance Metrics

| Model | Speed | Accuracy | Memory | GPU Required |
|-------|-------|----------|--------|--------------|
| LSTM | ⚡ Fast | 85% | Low | Optional |
| DistilBERT | ⚡⚡ Medium | 92% | Medium | Optional |
| BERT | ⚡⚡⚡ Slow | 95% | High | Recommended |

## 🐛 Troubleshooting

### Backend Issues

**Error: ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Reinstall requirements
pip install -r requirements.txt
```

**Error: Model file not found**
```bash
# Check backend directory for model files
ls -la backend/
# Ensure paths in main.py match actual file locations
```

**CUDA Out of Memory**
```bash
# Use CPU instead
# In main.py, set: device = torch.device('cpu')
```

### Frontend Issues

**Error: Cannot connect to backend**
```bash
# Check backend is running on port 8000
# Update API_URL in .env file
# Check CORS settings in FastAPI app
```

**API calls timing out**
```bash
# Increase timeout in .env: REACT_APP_API_TIMEOUT=60000
# Check backend processing time
```

## 📈 Improvement Roadmap

- [ ] Fine-tuned models on domain-specific data
- [ ] Real-time batch processing
- [ ] Model ensemble for higher accuracy
- [ ] PDF and document upload support
- [ ] User authentication and history
- [ ] API rate limiting
- [ ] Advanced analytics dashboard
- [ ] Mobile app (React Native)
- [ ] Multilingual support
- [ ] Custom model training interface

## 🔐 Security Considerations

- ✅ Input validation and sanitization
- ✅ CORS properly configured
- ✅ API rate limiting (recommended for production)
- ✅ No sensitive data stored client-side
- ✅ HTTPS recommended for production

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Sentence-Level-AI-text-Detection-using-deeplearning.git
   cd Sentence-Level-AI-text-Detection-using-deeplearning
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow PEP 8 for Python code
   - Use React best practices
   - Add comments for complex logic

4. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```

5. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Include screenshots if UI changes

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 📚 Resources & References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [NLTK Documentation](https://www.nltk.org/)

## 📞 Support & Contact

- **GitHub Repository:** [Sentence-Level-AI-text-Detection-using-deeplearning](https://github.com/jobjourneywithsoumya-lab/Sentence-Level-AI-text-Detection-using-deeplearning)
- **GitHub Issues:** [Report Issues](https://github.com/jobjourneywithsoumya-lab/Sentence-Level-AI-text-Detection-using-deeplearning/issues)
- **Author:** [Soumya](https://github.com/jobjourneywithsoumya-lab)
- **Email:** [soumyanh7@gmail.com]

## 🙏 Acknowledgments

- Hugging Face for transformer models
- TensorFlow and PyTorch communities
- NLTK for NLP utilities
- All contributors and users

## ⚠️ Disclaimer

This tool is designed for **educational and research purposes**. 

**Important Notes:**
- AI detection is probabilistic and not 100% accurate
- Models can have false positives and false negatives
- Should not be used as sole evidence of AI generation
- Continuously evolving AI generation methods may reduce accuracy
- Results should be verified by human review in critical applications

---

<div align="center">

**Made with ❤️ for AI Content Detection**

If you find this project helpful, please consider giving it a ⭐ star!

[⬆ Back to Top](#sentence-level-ai-text-detection-using-deep-learning)

</div>
