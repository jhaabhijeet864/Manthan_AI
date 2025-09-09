const express = require('express');
const app = express();

app.use(express.json());

// --- API Endpoints ---
// Health check
app.get('/api/health', (req, res) => {
	res.json({ status: 'ok', service: 'CMLRE Marine Platform Backend' });
});

// ML routes (HTTP to Flask services)
app.use('/api/predict', require('./src/routes/predictRoutes'));

// Placeholder for modular routes (future expansion)
// app.use('/api/ingest', require('./src/routes/ingestRoutes'));
// app.use('/api/taxonomy', require('./src/routes/taxonomyRoutes'));
// app.use('/api/otolith', require('./src/routes/otolithRoutes'));
// app.use('/api/edna', require('./src/routes/ednaRoutes'));

module.exports = app;
