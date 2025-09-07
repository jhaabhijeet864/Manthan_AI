import React, { useEffect, useState } from 'react';

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

  if (error) return <div>Error: {error}</div>;
  if (!prediction) return <div>Loading prediction...</div>;
  return (
    <div>
      <h2>Prediction from backend:</h2>
      <pre>{JSON.stringify(prediction, null, 2)}</pre>
    </div>
  );
}
