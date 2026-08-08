# LifeTrack Web

A polished Flask-based frontend for LifeTrack, designed to analyze Fitbit-style wearable datasets and present advanced lifestyle consistency analytics.

## Features

- Multi-file CSV upload support
- Automatic preprocessing and daily aggregation
- Gaussian anomaly detection and Shannon entropy scoring
- Behavioral Consistency Index (BCI) with state classification
- Markov forecast and ARIMA-supported trend forecasting
- Sleep regularity / fatigue proxy calculations
- Data-driven lifestyle prescriptions and interactive dashboards
- Responsive modern UI with Plotly visualizations

## Project structure

- `app.py` — Flask web application entrypoint
- `processing.py` — data preprocessing and analytics pipeline
- `templates/index.html` — main frontend template
- `static/css/style.css` — visual style and layout
- `static/js/main.js` — frontend interaction and Plotly charts

## Requirements

- Python 3.10+ recommended
- Flask
- pandas
- numpy
- scipy
- plotly
- statsmodels

## Install

```bash
cd "f:\COLLEGE\PROJECTS\5TH SEM\mini project\LifeTrack\lifetrack_web"
pip install flask pandas numpy scipy plotly statsmodels
```

## Run

### Python backend
```bash
cd "f:\\COLLEGE\\PROJECTS\\5TH SEM\\mini project\\LifeTrack\\lifetrack_web"
pip install -r requirements.txt
python app.py
```

### React frontend
```bash
cd "f:\\COLLEGE\\PROJECTS\\5TH SEM\\mini project\\LifeTrack\\lifetrack_web\\client"
npm install
npm run dev
```

The React app will proxy requests to the Python backend at `http://localhost:8501`.

### Production
```bash
cd "f:\\COLLEGE\\PROJECTS\\5TH SEM\\mini project\\LifeTrack\\lifetrack_web"
npm install
cd client
npm install
npm run build
cd ..
node server.js
```

Then open `http://localhost:3000`.

## Notes

- The app is intended for demonstration and research.
- Upload one or more Fitbit-style CSV files to generate daily analytics.
- The site keeps the existing Streamlit app untouched in the repository.
