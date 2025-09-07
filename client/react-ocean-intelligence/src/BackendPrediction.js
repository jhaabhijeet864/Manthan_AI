import React, { useEffect, useState } from 'react';
import './BackendPrediction.css';

export default function BackendPrediction() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/predict')
      .then(res => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      })
      .then(setPrediction)
      .catch(err => setError(err.message));
  }, []);

  if (error) return <div className="prediction-error">Error: {error}</div>;
  if (!prediction) return <div className="prediction-loading">Loading prediction...</div>;
  
  return (
    <div className="prediction-result">
      <div className="prediction-icon"><i className="fas fa-fish"></i></div>
      <div className="prediction-info">
        <div className="prediction-species">Species: <span>{prediction.species}</span></div>
        <div className="prediction-confidence">Confidence: <span>{(prediction.confidence * 100).toFixed(1)}%</span></div>
      </div>
      <div className="prediction-bar" style={{width: `${prediction.confidence * 100}%`}}></div>
    </div>
  );
}
