const { runPredictiveModel } = require('../services/mlService');

exports.getPrediction = async (req, res) => {
  try {
    const result = await runPredictiveModel();
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
};
