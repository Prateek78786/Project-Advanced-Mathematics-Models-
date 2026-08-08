import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import multer from 'multer';
import FormData from 'form-data';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const upload = multer();

app.use(express.static(path.join(__dirname, 'client', 'dist')));

app.post('/api/analyze', upload.array('data_files'), async (req, res) => {
  const files = req.files || [];
  if (!files.length) {
    return res.status(400).json({ success: false, message: 'Please upload one or more CSV files.' });
  }

  try {
    const formData = new FormData();
    files.forEach((file) => formData.append('data_files', file.buffer, { filename: file.originalname, contentType: file.mimetype }));
    const response = await fetch('http://localhost:8501/analyze', {
      method: 'POST',
      body: formData,
      headers: formData.getHeaders()
    });
    const payload = await response.json();
    return res.status(response.status).json(payload);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ success: false, message: 'Proxy error to Python backend.' });
  }
});

app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'client', 'dist', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`LifeTrack Node proxy server listening on http://localhost:${PORT}`);
});
