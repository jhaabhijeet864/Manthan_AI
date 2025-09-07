const express = require('express');
const app = express();

app.use(express.json());

// --- API Endpoints ---
// Health check
app.get('/api/health', (req, res) => {
	res.json({ status: 'ok', service: 'CMLRE Marine Platform Backend' });
});

// Prediction endpoint (calls Python script)
const { spawn } = require('child_process');
const path = require('path');

app.get('/api/predict', (req, res) => {
	const scriptPath = path.join(__dirname, '../scripts/utils/predictive_model.py');
	const py = spawn('python', [scriptPath]);
	let data = '';
	let error = '';
	py.stdout.on('data', chunk => data += chunk);
	py.stderr.on('data', chunk => error += chunk);
	py.on('close', code => {
		if (code !== 0 || error) {
			res.status(500).json({ error: error || `Python exited with code ${code}` });
		} else {
			try {
				res.json(JSON.parse(data));
			} catch (e) {
				res.status(500).json({ error: 'Invalid JSON from Python script' });
			}
		}
	});
});

// Placeholder for modular routes (future expansion)
// app.use('/api/ingest', require('./src/routes/ingestRoutes'));
// app.use('/api/taxonomy', require('./src/routes/taxonomyRoutes'));
// app.use('/api/otolith', require('./src/routes/otolithRoutes'));
// app.use('/api/edna', require('./src/routes/ednaRoutes'));

module.exports = app;
