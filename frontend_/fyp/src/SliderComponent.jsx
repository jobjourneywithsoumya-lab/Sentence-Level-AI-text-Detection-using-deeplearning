import React, { useState } from 'react';
import './SliderComponent.css'; // Import custom styles for the slider

const SliderComponent = () => {
  const [aiPercentage, setAiPercentage] = useState(50); // Default value to 50

  const handleSliderChange = (event) => {
    setAiPercentage(event.target.value); // Update AI percentage dynamically
  };

  const humanPercentage = 100 - aiPercentage; // Calculate human percentage

  return (
    <div className="slider-container">
      <h2>AI vs Human Content</h2>
      <div className="percentage-display">
        <div className="ai-percentage">
          <h3>AI: {aiPercentage}%</h3>
        </div>
        <div className="human-percentage">
          <h3>Human: {humanPercentage}%</h3>
        </div>
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={aiPercentage}
        onChange={handleSliderChange}
        className="slider"
      />
    </div>
  );
};

export default SliderComponent;
