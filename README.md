# AI Physiological State Monitor

**PBL Code:** B19 — FAIR-AI-19  
**Project:** AI Physiological State Monitor

## Objective
Build a personalized physiological trend and anomaly dashboard using time-series physiological signals such as heart rate (HR), heart-rate variability (HRV), and activity.

The system learns an individual's normal pattern and flags observations that deviate from that personal baseline. The output is an anomaly/deviation indicator and is **not a medical diagnosis**.

## Current version
This first version includes:
- Synthetic physiological time-series generator for development/demo
- Data preprocessing
- Rolling personalized baseline features
- Isolation Forest anomaly detection
- Anomaly score
- Streamlit interactive dashboard
- CSV upload support
- Trend and anomaly visualizations

## Project structure
```text
AI_Physiological_State_Monitor/
├── app/
│   └── dashboard.py
├── data/
│   ├── sample_physiological_data.csv
│   └── generate_data.py
├── models/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── baseline.py
│   └── anomaly_detection.py
├── main.py
├── requirements.txt
└── README.md
```

## Run
```bash
pip install -r requirements.txt
python main.py
streamlit run app/dashboard.py
```

The dashboard will open in the browser.

## CSV format
Required columns:
- timestamp
- heart_rate
- hrv
- activity

Example:
```csv
timestamp,heart_rate,hrv,activity
2026-01-01 08:00:00,72,48,0.20
```

## Future upgrade
The demo data can be replaced with a permitted public physiological dataset such as an appropriate PhysioNet/HRV dataset. Dataset licensing and usage conditions must be checked before redistribution or publication.
