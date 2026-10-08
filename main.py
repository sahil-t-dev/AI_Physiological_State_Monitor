from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.generate_data import generate
from src.preprocessing import clean_data, add_time_features, add_rolling_features
from src.baseline import create_personal_baseline, add_baseline_deviation
from src.anomaly_detection import fit_isolation_forest

def build_pipeline():
    csv_path = ROOT / "data" / "sample_physiological_data.csv"
    if not csv_path.exists():
        generate().to_csv(csv_path, index=False)

    df = clean_data(__import__("pandas").read_csv(csv_path))
    df = add_time_features(df)
    df = add_rolling_features(df)
    baseline = create_personal_baseline(df, warmup=50)
    df = add_baseline_deviation(df, baseline)
    model, result = fit_isolation_forest(df)

    print("AI Physiological State Monitor")
    print("=" * 36)
    print("Rows:", len(result))
    print("Detected anomalies:", int(result["anomaly"].sum()))
    print("Personal baseline:", baseline)
    print("\\nRun the dashboard with:")
    print("streamlit run app/dashboard.py")

if __name__ == "__main__":
    build_pipeline()
