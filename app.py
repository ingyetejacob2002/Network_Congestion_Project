
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NetGuard | Congestion Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_congestion_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

FEATURES = [
    "packet_count",
    "byte_count",
    "duration_total_sec",
    "log_packet_count",
    "log_byte_count",
    "packet_rate",
    "byte_rate",
    "bytes_per_packet",
]

CLASS_COLORS = {
    "Low": "#10B981",
    "Medium": "#F59E0B",
    "High": "#F43F5E",
}

# Online image: requires an internet connection to display.
BANNER_IMAGE = (
    "https://images.unsplash.com/photo-1558494949-ef010cbdcc31"
    "?auto=format&fit=crop&w=1800&q=85"
)


# ============================================================
# 3. PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #07111F;
        --panel: #101D30;
        --panel-light: #14243A;
        --border: #263951;
        --text: #F1F5F9;
        --muted: #9BAEC5;
        --cyan: #38BDF8;
        --blue: #3B82F6;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(37, 99, 235, 0.12),
                transparent 28%
            ),
            #07111F;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #0B1728;
        border-right: 1px solid #22344C;
    }

    [data-testid="stSidebar"] * {
        color: #DCE8F7;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
        letter-spacing: -0.6px;
    }

    p, label, .stMarkdown {
        color: #C5D3E4;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    }

    .brand-icon {
        background: linear-gradient(135deg, #2563EB, #06B6D4);
        width: 45px;
        height: 45px;
        border-radius: 13px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 23px;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.25);
    }

    .brand-name {
        color: #F8FAFC;
        font-size: 21px;
        font-weight: 800;
        line-height: 1.3;
    }

    .brand-subtitle {
        color: #91A8C4;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .hero {
        background: linear-gradient(
            110deg,
            rgba(13, 32, 55, 0.98) 0%,
            rgba(15, 43, 75, 0.96) 55%,
            rgba(8, 69, 100, 0.75) 100%
        );
        border: 1px solid #274968;
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hero-kicker {
        color: #67E8F9;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: clamp(26px, 3vw, 38px);
        line-height: 1.2;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #C1D7EC;
        font-size: 14px;
        max-width: 650px;
        line-height: 1.8;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6EE7B7;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-card {
        background: linear-gradient(145deg, #122239, #0D1A2D);
        border: 1px solid #263A53;
        border-radius: 17px;
        padding: 21px;
        min-height: 145px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.10);
    }

    .metric-label {
        color: #9CB0C9;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-note {
        color: #8CA3BF;
        font-size: 11px;
        margin-top: 10px;
    }

    .section-heading {
        color: #F1F5F9;
        font-size: 18px;
        font-weight: 750;
        margin: 28px 0 5px 0;
    }

    .section-description {
        color: #91A7C0;
        font-size: 12px;
        margin-bottom: 18px;
    }

    .panel {
        background: #0E1B2E;
        border: 1px solid #263951;
        border-radius: 17px;
        padding: 20px;
    }

    .prediction-card {
        border-radius: 18px;
        padding: 25px;
        background: linear-gradient(145deg, #14263D, #0C192B);
        border: 1px solid #2D4562;
        margin-top: 15px;
    }

    .prediction-label {
        color: #A9BDD4;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .prediction-value {
        font-size: 36px;
        font-weight: 800;
        margin: 10px 0;
    }

    .small-note {
        color: #94A8C0;
        font-size: 11px;
        line-height: 1.7;
    }

    .stButton > button {
        border: 0;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #2563EB, #0891B2);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border: 1px solid #67E8F9;
        color: white;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #263951;
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        border-top: 1px solid #263951;
        margin-top: 45px;
        padding-top: 20px;
        color: #8298B2;
        font-size: 11px;
        text-align: center;
        line-height: 1.8;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD THE TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()
    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as exc:
    MODEL_READY = False
    MODEL_ERROR = str(exc)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def engineer_features(data):
    """
    Convert raw traffic counters into the exact features
    expected by the trained pipeline.
    """

    required = [
        "packet_count",
        "byte_count",
        "duration_sec",
        "duration_nsec",
    ]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = data.copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[required].isna().any().any():
        raise ValueError(
            "The traffic data contains missing or non-numeric values. "
            "Please correct them and upload again."
        )

    if (df[["packet_count", "byte_count"]] < 0).any().any():
        raise ValueError(
            "Packet counts and byte counts cannot be negative."
        )

    if (
        (df["duration_sec"] < 0).any()
        or (df["duration_nsec"] < 0).any()
    ):
        raise ValueError("Duration values cannot be negative.")

    df["duration_total_sec"] = (
        df["duration_sec"] + df["duration_nsec"] / 1_000_000_000
    )

    safe_duration = df["duration_total_sec"].replace(0, np.nan)

    df["log_packet_count"] = np.log1p(df["packet_count"])
    df["log_byte_count"] = np.log1p(df["byte_count"])

    df["packet_rate"] = (
        df["packet_count"] / safe_duration
    ).replace([np.inf, -np.inf], np.nan)

    df["byte_rate"] = (
        df["byte_count"] / safe_duration
    ).replace([np.inf, -np.inf], np.nan)

    df["bytes_per_packet"] = (
        df["byte_count"]
        / df["packet_count"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Return the same feature names and order used during training.
    return df[FEATURES].replace([np.inf, -np.inf], np.nan)


def predict_traffic(raw_data):
    features = engineer_features(raw_data)

    predictions = model.predict(features)

    result = raw_data.copy()
    result["predicted_congestion"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)

        for index, class_name in enumerate(classes):
            result[f"probability_{class_name.lower()}"] = (
                probabilities[:, index] * 100
            ).round(2)

        result["prediction_score_percent"] = (
            probabilities.max(axis=1) * 100
        ).round(2)

    return result, features


def style_chart(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#C7D5E7",
            size=11,
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#C7D5E7"),
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.12)",
        ),
    )
    return fig


def metric_card(label, value, note, icon):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} &nbsp; {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🌐</div>
            <div>
                <div class="brand-name">NetGuard</div>
                <div class="brand-subtitle">Network Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("WORKSPACE")

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Single Prediction",
            "Batch Prediction",
            "Model Information",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if MODEL_READY:
        st.success("Prediction model loaded")
    else:
        st.error("Prediction model unavailable")

    st.markdown(
        """
        <div class="small-note">
        MACHINE LEARNING SYSTEM<br>
        Network Traffic Analysis<br>
        Congestion Classification
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            NETWORK ANALYTICS &nbsp; / &nbsp; INTELLIGENCE PLATFORM
        </div>
        <h1>Network Traffic<br>Congestion Intelligence</h1>
        <p>
            Analyze traffic statistics, classify congestion levels,
            inspect prediction scores, and export results from one
            centralized machine-learning workspace.
        </p>
        <span class="status-pill">● &nbsp; APPLICATION WORKSPACE</span>
    </div>
    """,
    unsafe_allow_html=True,
)


if not MODEL_READY:
    st.error(
        "The model could not be loaded. Confirm that these files exist:\n\n"
        "- models/final_congestion_model.joblib\n"
        "- models/model_metadata.json\n\n"
        f"Technical details: {MODEL_ERROR}"
    )
    st.stop()


# ============================================================
# 8. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown("## System Overview")
    st.caption(
        "A live application overview. Statistics below update when "
        "you make predictions or upload a dataset."
    )

    history = st.session_state.get("prediction_history", [])

    total_predictions = sum(item["records"] for item in history)
    high_predictions = sum(item["high"] for item in history)
    latest_prediction = (
        history[-1]["time"] if history else "No predictions yet"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Records Analyzed",
            f"{total_predictions:,}",
            "Processed in this session",
            "📊",
        )

    with col2:
        metric_card(
            "High Classifications",
            f"{high_predictions:,}",
            "Predicted High congestion",
            "🚨",
        )

    with col3:
        metric_card(
            "Available Classes",
            "03",
            "Low, Medium and High",
            "🎯",
        )

    with col4:
        metric_card(
            "Latest Analysis",
            latest_prediction if history else "Awaiting input",
            "Most recent prediction activity",
            "🕒",
        )

    st.markdown('<div class="section-heading">Explore the platform</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Choose a workspace to start analyzing network traffic.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🖥️ Network infrastructure")
        try:
            st.image(
                BANNER_IMAGE,
                use_container_width=True,
                caption="Illustrative server infrastructure",
            )
        except Exception:
            st.info(
                "The illustrative server image is unavailable. "
                "An internet connection may be required."
            )
        st.markdown(
            """
            **From raw counters to useful classifications**

            - Inspect packet and byte traffic.
            - Generate congestion predictions.
            - Review class probability scores.
            - Export predictions for further analysis.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📡 Classification overview")

        if history:
            counts = {"Low": 0, "Medium": 0, "High": 0}

            for item in history:
                for label, count in item["counts"].items():
                    counts[label] = counts.get(label, 0) + count

            chart_data = pd.DataFrame({
                "Congestion level": list(counts.keys()),
                "Records": list(counts.values()),
            })

            fig = px.pie(
                chart_data,
                names="Congestion level",
                values="Records",
                hole=0.68,
                color="Congestion level",
                color_discrete_map=CLASS_COLORS,
            )
            fig.update_traces(
                textposition="outside",
                textinfo="label+percent",
                marker=dict(line=dict(color="#0E1B2E", width=3)),
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
            )
        else:
            st.info(
                "Your chart will appear here after the first prediction."
            )

            example = go.Figure(
                data=[
                    go.Pie(
                        labels=["Low", "Medium", "High"],
                        values=[1, 1, 1],
                        hole=0.68,
                        marker_colors=[
                            CLASS_COLORS["Low"],
                            CLASS_COLORS["Medium"],
                            CLASS_COLORS["High"],
                        ],
                    )
                ]
            )
            example.update_traces(
                textinfo="label",
                hoverinfo="label",
            )
            example.update_layout(showlegend=False)
            st.plotly_chart(
                style_chart(example, 300),
                use_container_width=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Recent activity</div>',
                unsafe_allow_html=True)

    if history:
        recent = pd.DataFrame(history[::-1])
        recent = recent[
            ["time", "source", "records", "low", "medium", "high"]
        ]
        recent.columns = [
            "Time",
            "Source",
            "Records",
            "Low",
            "Medium",
            "High",
        ]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No prediction activity yet. Open Single Prediction or "
            "Batch Prediction to begin."
        )


# ============================================================
# 9. SINGLE PREDICTION
# ============================================================

elif page == "Single Prediction":

    st.markdown("## Single Traffic Prediction")
    st.caption(
        "Enter the traffic counters for one observation and classify it."
    )

    input_col, output_col = st.columns([1, 1], gap="large")

    with input_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📝 Traffic parameters")

        with st.form("single_prediction_form"):
            packet_count = st.number_input(
                "Packet count",
                min_value=0,
                value=1000,
                step=100,
                help="Number of packets recorded in the observation.",
            )

            byte_count = st.number_input(
                "Byte count",
                min_value=0,
                value=750000,
                step=10000,
                help="Total bytes recorded in the observation.",
            )

            duration_sec = st.number_input(
                "Duration (seconds)",
                min_value=0,
                value=10,
                step=1,
            )

            duration_nsec = st.number_input(
                "Duration (nanoseconds)",
                min_value=0,
                max_value=999999999,
                value=0,
                step=1000000,
            )

            submitted = st.form_submit_button(
                "⚡ Analyze Traffic",
                use_container_width=True,
            )

        st.markdown(
            '<div class="small-note">'
            'Enter non-negative traffic values. These fields should '
            'represent a consistent observation from your network data.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        raw = pd.DataFrame([{
            "packet_count": packet_count,
            "byte_count": byte_count,
            "duration_sec": duration_sec,
            "duration_nsec": duration_nsec,
        }])

        try:
            results, features = predict_traffic(raw)
            predicted = str(results.iloc[0]["predicted_congestion"])

            st.session_state["single_result"] = {
                "prediction": predicted,
                "results": results,
                "features": features,
            }

            counts = {"Low": 0, "Medium": 0, "High": 0}
            counts[predicted] = 1

            st.session_state.setdefault(
                "prediction_history", []
            ).append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Single prediction",
                "records": 1,
                "low": counts["Low"],
                "medium": counts["Medium"],
                "high": counts["High"],
                "counts": counts,
            })

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    with output_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction result")

        single_result = st.session_state.get("single_result")

        if single_result:
            predicted = single_result["prediction"]
            color = CLASS_COLORS.get(predicted, "#38BDF8")

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted congestion level</div>
                    <div class="prediction-value" style="color:{color}">
                        {predicted}
                    </div>
                    <div class="small-note">
                        This is the class predicted by the trained model,
                        not a direct measurement of actual network congestion.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            result_row = single_result["results"].iloc[0]
            features_row = single_result["features"].iloc[0]

            st.markdown("#### Derived traffic features")

            d1, d2 = st.columns(2)
            d1.metric(
                "Packet rate",
                f"{features_row['packet_rate']:,.2f}"
                if pd.notna(features_row["packet_rate"])
                else "Unavailable",
            )
            d2.metric(
                "Byte rate",
                f"{features_row['byte_rate']:,.2f}"
                if pd.notna(features_row["byte_rate"])
                else "Unavailable",
            )

            st.markdown("#### Class probability scores")

            probability_columns = [
                column for column in result_row.index
                if column.startswith("probability_")
            ]

            if probability_columns:
                probabilities = pd.DataFrame({
                    "Class": [
                        column.replace("probability_", "").title()
                        for column in probability_columns
                    ],
                    "Probability (%)": [
                        float(result_row[column])
                        for column in probability_columns
                    ],
                })

                fig = px.bar(
                    probabilities,
                    x="Class",
                    y="Probability (%)",
                    color="Class",
                    color_discrete_map=CLASS_COLORS,
                    text_auto=".1f",
                )
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )
                fig.update_yaxes(range=[0, 110])
                st.plotly_chart(
                    style_chart(fig, 300),
                    use_container_width=True,
                )
        else:
            st.info(
                "Submit the traffic parameters to see the prediction, "
                "derived features and class scores here."
            )

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 10. BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.markdown("## Batch Traffic Analysis")
    st.caption(
        "Upload a CSV file to classify many traffic observations at once."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📂 Upload your dataset")

    st.markdown(
        """
        The CSV must contain these four columns:

        - `packet_count`
        - `byte_count`
        - `duration_sec`
        - `duration_nsec`

        Additional columns are retained in the exported results.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload network traffic observations in CSV format.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.markdown("#### Dataset preview")
            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Rows: {len(uploaded_df):,} | "
                f"Columns: {len(uploaded_df.columns)}"
            )

            if st.button(
                "🚀 Run Batch Prediction",
                use_container_width=True,
            ):
                st.balloons()
                with st.spinner("Analyzing traffic records..."):
                    batch_results, _ = predict_traffic(uploaded_df)

                st.session_state["batch_results"] = batch_results

                counts_series = (
                    batch_results["predicted_congestion"]
                    .value_counts()
                )
                counts = {
                    label: int(counts_series.get(label, 0))
                    for label in ["Low", "Medium", "High"]
                }

                st.session_state.setdefault(
                    "prediction_history", []
                ).append({
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "source": uploaded_file.name,
                    "records": len(batch_results),
                    "low": counts["Low"],
                    "medium": counts["Medium"],
                    "high": counts["High"],
                    "counts": counts,
                })

                st.success("Batch prediction completed.")

        except Exception as exc:
            st.error(f"Unable to process the uploaded CSV: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)

    batch_results = st.session_state.get("batch_results")

    if batch_results is not None:
        st.markdown('<div class="section-heading">Prediction summary</div>',
                    unsafe_allow_html=True)

        counts = (
            batch_results["predicted_congestion"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        a, b, c, d = st.columns(4)

        with a:
            metric_card(
                "Total Records",
                f"{len(batch_results):,}",
                "Records processed",
                "📊",
            )

        with b:
            metric_card(
                "Low",
                f"{counts['Low']:,}",
                "Low class predictions",
                "🟢",
            )

        with c:
            metric_card(
                "Medium",
                f"{counts['Medium']:,}",
                "Medium class predictions",
                "🟠",
            )

        with d:
            metric_card(
                "High",
                f"{counts['High']:,}",
                "High class predictions",
                "🔴",
            )

        chart_col, table_col = st.columns([0.8, 1.2], gap="large")

        with chart_col:
            st.markdown("### Class distribution")

            chart_df = pd.DataFrame({
                "Congestion": counts.index,
                "Records": counts.values,
            })

            fig = px.pie(
                chart_df,
                names="Congestion",
                values="Records",
                hole=0.60,
                color="Congestion",
                color_discrete_map=CLASS_COLORS,
            )
            fig.update_traces(
                textinfo="label+percent",
                marker=dict(line=dict(color="#0E1B2E", width=3)),
            )
            fig.update_layout(showlegend=False)

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
            )

        with table_col:
            st.markdown("### Prediction records")
            st.dataframe(
                batch_results,
                use_container_width=True,
                hide_index=True,
                height=350,
            )

        st.markdown("### Export results")

        csv_bytes = batch_results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results (CSV)",
            data=csv_bytes,
            file_name="network_congestion_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Prediction scores, where available, are model outputs and "
            "should not be interpreted as independently calibrated "
            "probabilities of real-world congestion."
        )


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.markdown("## Model Information")
    st.caption(
        "Technical details about the trained classifier and its intended use."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Machine-learning pipeline")

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("**Model type**")
        st.write("Logistic Regression")

        st.markdown("**Preprocessing**")
        st.write("Median imputation and feature standardization")

        st.markdown("**Input features**")
        for feature in FEATURES:
            st.markdown(f"- `{feature}`")

    with info2:
        st.markdown("**Output classes**")
        for class_name in ["Low", "Medium", "High"]:
            st.markdown(
                f"- <span style='color:{CLASS_COLORS[class_name]}'>"
                f"●</span> {class_name}",
                unsafe_allow_html=True,
            )

        st.markdown("**Training records**")
        st.write(metadata.get("training_rows", "See training script"))

        st.markdown("**Model file**")
        st.code(str(MODEL_PATH.relative_to(BASE_DIR)))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Important evaluation notes</div>',
                unsafe_allow_html=True)

    st.markdown("### Validation reference")

    st.markdown(
        """
        The timestamp-grouped cross-validation experiment previously
        completed for this project reported the following Logistic
        Regression results:
        """
    )

    v1, v2 = st.columns(2)

    with v1:
        metric_card(
            "Grouped Accuracy",
            "33.81%",
            "Timestamp-grouped cross-validation",
            "📉",
        )

    with v2:
        metric_card(
            "Grouped Macro F1",
            "31.84%",
            "Timestamp-grouped cross-validation",
            "📐",
        )

    st.caption(
        "These figures are from the earlier validation experiment. "
        "They are not live metrics calculated from the deployed model "
        "and should be reported with the validation method and dataset "
        "limitations."
    )

    st.markdown("### Feature engineering")

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Original packet counter",
            "Original byte counter",
            "Duration converted to seconds",
            "Log-transformed packet count",
            "Log-transformed byte count",
            "Packets divided by duration",
            "Bytes divided by duration",
            "Bytes divided by packet count",
        ],
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color:#CBD9EA;">NETGUARD</strong>
        &nbsp; · &nbsp; Network Traffic Congestion Prediction System
        <br>
        Machine-learning project prototype · For academic demonstration
        <br>
        Prediction outputs depend on the training data and its limitations.
    </div>
    """,
    unsafe_allow_html=True,
)
