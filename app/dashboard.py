from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ============================================================
# PROJECT MODULES
# ============================================================

from src.preprocessing import (
    clean_data,
    add_time_features,
    add_rolling_features
)

from src.baseline import (
    create_personal_baseline,
    add_baseline_deviation
)

from src.anomaly_detection import fit_isolation_forest

from src.state_classification import add_state_classification, state_summary

from real_data.physionet_loader import (
    load_subject,
    list_subjects
)

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Physiological State Monitor",
    page_icon="❤️",
    layout="wide",
)

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }

    .small-note {
        font-size: 0.85rem;
        color: #8b949e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ AI Physiological State Monitor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Personalized physiological trend and anomaly detection dashboard'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR - DATA SOURCE
# ============================================================

st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Select dataset",
    [
        "Real PhysioNet Dataset",
        "Sample Dataset",
        "Upload CSV"
    ]
)

# ============================================================
# SAMPLE DATA
# ============================================================

@st.cache_data
def load_default():

    path = ROOT / "data" / "sample_physiological_data.csv"

    return pd.read_csv(path)


# ============================================================
# REAL PHYSIONET DATA
# ============================================================

@st.cache_data
def load_real_subject(dataset_type, subject):

    return load_subject(
        dataset_type=dataset_type,
        subject=subject
    )


# ============================================================
# DATA SELECTION
# ============================================================

raw = None

if data_source == "Real PhysioNet Dataset":

    st.sidebar.markdown("---")
    st.sidebar.subheader("PhysioNet Dataset")

    dataset_type = st.sidebar.selectbox(
        "Session Type",
        [
            "STRESS",
            "AEROBIC",
            "ANAEROBIC"
        ]
    )

    try:

        subjects = list_subjects(dataset_type)

    except Exception as e:

        st.error(
            f"Unable to locate the PhysioNet dataset.\n\n{e}"
        )

        st.stop()

    if not subjects:

        st.error(
            f"No participants were found for {dataset_type}."
        )

        st.stop()

    subject = st.sidebar.selectbox(
        "Participant",
        subjects
    )

    st.sidebar.success(
        f"Loaded participant: {subject}"
    )

    try:

        raw = load_real_subject(
            dataset_type,
            subject
        )

    except Exception as e:

        st.error(
            f"Unable to load participant data:\n\n{e}"
        )

        st.stop()


elif data_source == "Sample Dataset":

    raw = load_default()

    st.sidebar.success(
        "Sample physiological dataset loaded."
    )


elif data_source == "Upload CSV":

    uploaded = st.sidebar.file_uploader(
        "Upload physiological CSV",
        type=["csv"],
        help=(
            "Required columns: "
            "timestamp, heart_rate, hrv, activity"
        )
    )

    if uploaded is None:

        st.info(
            "Please upload a CSV file containing "
            "timestamp, heart_rate, hrv and activity."
        )

        st.stop()

    try:

        raw = pd.read_csv(uploaded)

    except Exception as e:

        st.error(
            f"Unable to read CSV file:\n\n{e}"
        )

        st.stop()


# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown("### Project Information")

st.sidebar.write("**Signals:** HR, HRV, Activity")
st.sidebar.write("**ML Algorithm:** Isolation Forest")
st.sidebar.write("**Learning:** Personalized Baseline")
st.sidebar.write("**Dataset:** PhysioNet Wearable Dataset")

st.sidebar.markdown(
    '<div class="small-note">'
    'Anomaly detection is an analytical indicator and not a medical diagnosis.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# DATA PREPROCESSING
# ============================================================

try:

    df = clean_data(raw)

except Exception as e:

    st.error(str(e))

    st.stop()


if len(df) < 10:

    st.error(
        "At least 10 valid observations are required."
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df = add_time_features(df)

df = add_rolling_features(
    df,
    window=min(
        20,
        max(5, len(df) // 5)
    )
)


# ============================================================
# PERSONAL BASELINE
# ============================================================

baseline = create_personal_baseline(
    df,
    warmup=min(
        50,
        len(df)
    )
)

df = add_baseline_deviation(
    df,
    baseline
)


# ============================================================
# ISOLATION FOREST
# ============================================================

try:

    model, result = fit_isolation_forest(
        df,
        contamination=0.05
    )

    # Classify the physiological state before any dashboard
    # component tries to read the physiological_state column.
    result = add_state_classification(result)

except Exception as e:

    st.error(
        f"Machine-learning model error:\n\n{e}"
    )

    st.stop()


# ============================================================
# ANOMALY SEVERITY
# ============================================================

zmax = result[
    [
        "baseline_hr_z",
        "baseline_hrv_z",
        "baseline_activity_z"
    ]
].abs().max(axis=1)


def severity(z, is_anomaly):

    if not is_anomaly:
        return "Normal"

    if z >= 4:
        return "High"

    if z >= 2.5:
        return "Moderate"

    return "Low"


result["severity"] = [
    severity(z, anomaly)
    for z, anomaly in zip(
        zmax,
        result["anomaly"]
    )
]


# ============================================================
# LATEST OBSERVATION
# ============================================================

latest = result.iloc[-1]
current_state = latest["physiological_state"]

if bool(latest["anomaly"]):

    status = "Anomaly / significant deviation"

else:

    status = "Within personal baseline"


# ============================================================
# DATASET INFORMATION
# ============================================================

if data_source == "Real PhysioNet Dataset":

    st.info(
        f"📊 Real PhysioNet data • "
        f"{dataset_type} • Participant {subject} • "
        f"{len(result):,} observations"
    )

elif data_source == "Sample Dataset":

    st.info(
        f"📊 Sample dataset • "
        f"{len(result):,} observations"
    )

else:

    st.info(
        f"📊 Uploaded dataset • "
        f"{len(result):,} observations"
    )


# ============================================================
# TOP KPI SECTION
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Heart Rate",
    f"{latest['heart_rate']:.1f} BPM"
)

c2.metric(
    "HRV",
    f"{latest['hrv']:.1f} ms"
)

c3.metric(
    "Activity",
    f"{latest['activity']:.2f}"
)

c4.metric(
    "Detected Anomalies",
    f"{int(result['anomaly'].sum())}"
)

c5.metric(
    "Physiological State",
    current_state
)


# ============================================================
# CURRENT AI STATUS
# ============================================================

if bool(latest["anomaly"]):

    st.warning(
        f"AI Status: {status} • "
        f"Severity: {latest['severity']}. "
        "This indicates deviation from the observed "
        "personal pattern; it is not a diagnosis."
    )

else:

    st.success(
        f"AI Status: {status}."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Trends",
        "Anomaly Analysis",
        "Data"
    ]
)


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab1:

    st.subheader("Personal Baseline")

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Baseline HR",
        f"{baseline['heart_rate_mean']:.1f} BPM"
    )

    b2.metric(
        "Baseline HRV",
        f"{baseline['hrv_mean']:.1f} ms"
    )

    b3.metric(
        "Baseline Activity",
        f"{baseline['activity_mean']:.2f}"
    )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader("Dataset Summary")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Observations",
        f"{len(result):,}"
    )

    s2.metric(
        "Normal",
        f"{int((~result['anomaly']).sum()):,}"
    )

    s3.metric(
        "Anomalies",
        f"{int(result['anomaly'].sum()):,}"
    )

    s4.metric(
        "Anomaly Rate",
        f"{100 * result['anomaly'].mean():.1f}%"
    )

    # --------------------------------------------------------
    # SEVERITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("AI Physiological State")

    st.info(f"Current classified state: **{current_state}**")

    state_counts = state_summary(result)
    if not state_counts.empty:
        st.bar_chart(state_counts, use_container_width=True)

    st.subheader("Anomaly Severity Distribution")

    severity_counts = (
        result[result["anomaly"]]["severity"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High"],
            fill_value=0
        )
    )

    st.bar_chart(
        severity_counts
    )


# ============================================================
# TAB 2 - TRENDS
# ============================================================

with tab2:

    # --------------------------------------------------------
    # HEART RATE
    # --------------------------------------------------------

    st.subheader("Heart Rate Trend")

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        result["timestamp"],
        result["heart_rate"],
        label="Heart Rate"
    )

    ax.axhline(
        baseline["heart_rate_mean"],
        linestyle="--",
        label="Personal baseline"
    )

    anom = result[
        result["anomaly"]
    ]

    ax.scatter(
        anom["timestamp"],
        anom["heart_rate"],
        label="Anomaly"
    )

    ax.set_xlabel("Time")

    ax.set_ylabel(
        "Heart Rate (BPM)"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    # --------------------------------------------------------
    # HRV
    # --------------------------------------------------------

    st.subheader("HRV Trend")

    fig2, ax2 = plt.subplots(
        figsize=(12, 4)
    )

    ax2.plot(
        result["timestamp"],
        result["hrv"],
        label="HRV"
    )

    ax2.axhline(
        baseline["hrv_mean"],
        linestyle="--",
        label="Personal baseline"
    )

    ax2.set_xlabel("Time")

    ax2.set_ylabel(
        "HRV (ms)"
    )

    ax2.legend()

    ax2.grid(
        alpha=0.2
    )

    st.pyplot(
        fig2,
        clear_figure=True
    )

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    st.subheader("Activity Trend")

    st.line_chart(
        result.set_index(
            "timestamp"
        )["activity"]
    )


# ============================================================
# TAB 3 - ANOMALY ANALYSIS
# ============================================================

with tab3:

    st.subheader(
        "AI Anomaly Analysis"
    )

    st.write(
        "Isolation Forest identifies observations "
        "that differ from the learned multivariate "
        "physiological pattern."
    )

    # --------------------------------------------------------
    # ANOMALY TABLE
    # --------------------------------------------------------

    anomaly_table = result[
        result["anomaly"]
    ].copy()

    anomaly_table = anomaly_table[
        [
            "timestamp",
            "heart_rate",
            "hrv",
            "activity",
            "anomaly_score",
            "severity",
            "physiological_state"
        ]
    ].sort_values(
        "timestamp",
        ascending=False
    )

    if anomaly_table.empty:

        st.info(
            "No anomalies were detected "
            "in the current dataset."
        )

    else:

        st.dataframe(
            anomaly_table.head(50),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # ANOMALY SCORE GRAPH
    # --------------------------------------------------------

    st.subheader(
        "Anomaly Score Over Time"
    )

    fig3, ax3 = plt.subplots(
        figsize=(12, 4)
    )

    ax3.plot(
        result["timestamp"],
        result["anomaly_score"],
        label="Anomaly score"
    )

    if not anomaly_table.empty:

        ax3.scatter(
            anomaly_table["timestamp"],
            anomaly_table["anomaly_score"],
            label="Detected anomaly"
        )

    ax3.set_xlabel(
        "Time"
    )

    ax3.set_ylabel(
        "Score"
    )

    ax3.legend()

    ax3.grid(
        alpha=0.2
    )

    st.pyplot(
        fig3,
        clear_figure=True
    )


# ============================================================
# TAB 4 - DATA
# ============================================================

with tab4:

    st.subheader(
        "Processed Data"
    )

    st.dataframe(
        result.tail(100),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = result.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Processed Data",
        data=csv_data,
        file_name="physiological_analysis.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Educational PBL prototype. The system detects "
    "deviations from an individual's observed "
    "physiological baseline and must not be used "
    "for medical diagnosis."
)