const { spawn } = require('child_process');
const path = require('path');

exports.runPredictiveModel = () => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '../../../scripts/utils/predictive_model.py');
    const py = spawn('python', [scriptPath]);
    let data = '';
    let error = '';
    py.stdout.on('data', chunk => data += chunk);
    py.stderr.on('data', chunk => error += chunk);
    py.on('close', code => {
      if (code !== 0 || error) {
        reject(new Error(error || `Python exited with code ${code}`));
      } else {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error('Invalid JSON from Python script'));
        }
      }
    });
  });
};
