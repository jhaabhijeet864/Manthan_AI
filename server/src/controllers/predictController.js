const { predictTrend, predictSpecies } = require('../services/mlService');

exports.postTrend = async (req, res) => {
  try {
    const result = await predictTrend(req.body || {});
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: 'Trend prediction failed', details: err.message });
  }
};

exports.postSpecies = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    const result = await predictSpecies(req.file.buffer, req.file.originalname || 'image.jpg');
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: 'Species prediction failed', details: err.message });
  }
};
