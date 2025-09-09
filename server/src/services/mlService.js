const axios = require('axios');
const FormData = require('form-data');

const CNN_BASE = process.env.CNN_URL || 'http://localhost:5001';
const RF_BASE  = process.env.RF_URL  || 'http://localhost:5002';

async function predictTrend(payload) {
  const res = await axios.post(`${RF_BASE}/predict_trend`, payload, { timeout: 15000 });
  return res.data;
}

async function predictSpecies(fileBuffer, filename) {
  const form = new FormData();
  form.append('file', fileBuffer, { filename });
  const res = await axios.post(`${CNN_BASE}/predict_species`, form, {
    headers: form.getHeaders(),
    maxBodyLength: Infinity,
    timeout: 30000,
  });
  return res.data;
}

module.exports = { predictTrend, predictSpecies };
