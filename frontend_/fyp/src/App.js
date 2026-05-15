import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = (process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

function App() {
  const [text, setText] = useState('');
  const [wordCount, setWordCount] = useState(0);
  const [score, setScore] = useState(null);
  const [sentenceResults, setSentenceResults] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const handleTextChange = (event) => setText(event.target.value);

  useEffect(() => {
    setWordCount(text.trim() ? text.trim().split(/\s+/).length : 0);
  }, [text]);

  const getScoreColor = (score) => {
    if (score <= 10) return '#4CAF50';
    if (score <= 20) return '#8BC34A';
    if (score <= 30) return '#CDDC39';
    if (score <= 40) return '#FFEB3B';
    if (score <= 50) return '#FFC107';
    if (score <= 60) return '#FF9800';
    if (score <= 70) return '#FF5722';
    if (score <= 80) return '#F44336';
    if (score <= 90) return '#D32F2F';
    return '#B71C1C';
  };

  const handleModelSelect = (event) => {
    setSelectedModel(event.target.value);
    setScore(null);
    setSentenceResults([]);
  };

  const handleSubmit = async () => {
    if (!selectedModel) {
      alert("Please select a model (LSTM or DistilBERT).");
      return;
    }

    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        text,
        model_choice: selectedModel,
      });

      console.log("Response from backend:", response.data);
      setScore(response.data.ai_score); // AI score from BERT
      setSentenceResults(response.data.sentence_predictions); // Sentence-level predictions from LSTM or DistilBERT
    } catch (error) {
      console.error('Error details:', error.response?.data || error.message);
      alert("Something went wrong. Check console for details.");
    }
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <h1 className="logo">AiVsHuman</h1>
        <ul className="nav-links">
          <li><a href="#">Home</a></li>
          <li><a href="#">About Us</a></li>
          <li><a href="#">Contributors</a></li>
        </ul>
      </nav>

      <header className="header">
        <h2>Detect the AI content in your text</h2>
      </header>

      <main className="main-content">
        <div className="text-area-container">
          <textarea
            className="text-area"
            placeholder="Write your text here..."
            value={text}
            onChange={handleTextChange}
          />
        </div>
        <p className="word-count">Word Count: {wordCount}</p>

        <div className="model-selection">
          <label>
            <input
              type="radio"
              value="lstm"
              checked={selectedModel === 'lstm'}
              onChange={handleModelSelect}
            /> LSTM
          </label>
          <label style={{ marginLeft: '20px' }}>
            <input
              type="radio"
              value="distilbert"
              checked={selectedModel === 'distilbert'}
              onChange={handleModelSelect}
            /> DistilBERT
          </label>
        </div>

        <button className="upload-button" onClick={handleSubmit}>
          Upload
        </button>

        {/* Show AI score (only after BERT-based score calculation) */}
        {score !== null && (
          <div className="score-container">
            <p className="score-label">AI Detection Score: {score.toFixed(2)}%</p>
            <div className="slider">
              <div
                className="slider-bar"
                style={{ width: `${score}%`, backgroundColor: getScoreColor(score) }}
              ></div>
              <span className="ai-label">AI: {score.toFixed(2)}%</span>
              <span className="human-label">Human: {(100 - score).toFixed(2)}%</span>
            </div>
          </div>
        )}

        {/* Show sentence-level results */}
        {sentenceResults.length > 0 && (
          <div className="sentence-results">
            <h3>Sentence-Level Detection:</h3>
            {sentenceResults.map((sentence, index) => (
              <div
                key={index}
                className="sentence-box"
                style={{
                  backgroundColor: getScoreColor(
                    sentence.confidence * 100
                  )
                }}
              >
                <p><strong>Sentence:</strong> {sentence.sentence}</p>
                { (
                  <p>
                    <strong>Prediction:</strong> {sentence.prediction} (
                    {(sentence.confidence * 100).toFixed(2)}%)
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
