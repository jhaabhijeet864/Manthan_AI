const express = require('express');
const router = express.Router();
const multer = require('multer');
const upload = multer();
const { postTrend, postSpecies } = require('../controllers/predictController');

// Trend prediction (JSON body)
router.post('/trend', postTrend);

// Species prediction (multipart/form-data with file)
router.post('/species', upload.single('file'), postSpecies);

module.exports = router;
