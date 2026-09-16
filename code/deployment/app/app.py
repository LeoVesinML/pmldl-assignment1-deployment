"""Stage 3b - web application.

A Streamlit UI that talks to the model API over HTTP: the user fills in the
physicochemical properties of a wine, presses **Predict quality**, and the
prediction returned by the API is displayed.  The app holds no model - all
inference happens in the separate API container.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "15"))

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Fallback schema (used only while the API is still starting up)
# --------------------------------------------------------------------------- #
FALLBACK_FEATURES: Dict[str, Dict[str, float]] = {
    "fixed_acidity": {"min": 3.8, "max": 15.9, "median": 7.0, "p01": 4.7, "p99": 12.5},
    "volatile_acidity": {"min": 0.08, "max": 1.58, "median": 0.29, "p01": 0.12, "p99": 0.92},
    "citric_acid": {"min": 0.0, "max": 1.66, "median": 0.31, "p01": 0.0, "p99": 0.74},
    "residual_sugar": {"min": 0.6, "max": 65.8, "median": 3.0, "p01": 0.9, "p99": 17.5},
    "chlorides": {"min": 0.009, "max": 0.611, "median": 0.047, "p01": 0.018, "p99": 0.29},
    "free_sulfur_dioxide": {"min": 1.0, "max": 289.0, "median": 29.0, "p01": 3.0, "p99": 90.0},
    "total_sulfur_dioxide": {"min": 6.0, "max": 440.0, "median": 118.0, "p01": 11.0, "p99": 260.0},
    "density": {"min": 0.987, "max": 1.039, "median": 0.9949, "p01": 0.9884, "p99": 1.0013},
    "ph": {"min": 2.72, "max": 4.01, "median": 3.21, "p01": 2.87, "p99": 3.68},
    "sulphates": {"min": 0.22, "max": 2.0, "median": 0.51, "p01": 0.3, "p99": 1.03},
    "alcohol": {"min": 8.0, "max": 14.9, "median": 10.3, "p01": 8.5, "p99": 13.4},
}

FEATURE_LABELS: Dict[str, str] = {
    "fixed_acidity": "Fixed acidity (g/dm³)",
    "volatile_acidity": "Volatile acidity (g/dm³)",
    "citric_acid": "Citric acid (g/dm³)",
    "residual_sugar": "Residual sugar (g/dm³)",
    "chlorides": "Chlorides (g/dm³)",
    "free_sulfur_dioxide": "Free SO₂ (mg/dm³)",
    "total_sulfur_dioxide": "Total SO₂ (mg/dm³)",
    "density": "Density (g/cm³)",
    "ph": "pH",
    "sulphates": "Sulphates (g/dm³)",
    "alcohol": "Alcohol (% vol)",
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "🍇 Premium red (likely good)": {
        "fixed_acidity": 8.6, "volatile_acidity": 0.28, "citric_acid": 0.49,
        "residual_sugar": 2.2, "chlorides": 0.07, "free_sulfur_dioxide": 12.0,
        "total_sulfur_dioxide": 33.0, "density": 0.9942, "ph": 3.2,
        "sulphates": 0.85, "alcohol": 12.6, "wine_type": "red",
    },
    "🥂 Crisp white (likely good)": {
        "fixed_acidity": 6.6, "volatile_acidity": 0.24, "citric_acid": 0.35,
        "residual_sugar": 5.0, "chlorides": 0.036, "free_sulfur_dioxide": 40.0,
        "total_sulfur_dioxide": 130.0, "density": 0.9912, "ph": 3.25,
        "sulphates": 0.55, "alcohol": 12.8, "wine_type": "white",
    },
    "🍷 Table red (likely standard)": {
        "fixed_acidity": 7.8, "volatile_acidity": 0.76, "citric_acid": 0.04,
        "residual_sugar": 2.3, "chlorides": 0.092, "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 54.0, "density": 0.9970, "ph": 3.26,
        "sulphates": 0.65, "alcohol": 9.8, "wine_type": "red",
    },
    "🧃 Sweet white (likely standard)": {
        "fixed_acidity": 7.0, "volatile_acidity": 0.45, "citric_acid": 0.36,
        "residual_sugar": 18.0, "chlorides": 0.06, "free_sulfur_dioxide": 50.0,
        "total_sulfur_dioxide": 190.0, "density": 1.0005, "ph": 3.05,
        "sulphates": 0.42, "alcohol": 9.0, "wine_type": "white",
    },
}


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_get(path: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_post(path: str, payload: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code >= 400:
            return None, f"API returned {response.status_code}: {response.text[:400]}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"Could not reach the API at {API_URL}: {exc}"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_schema() -> Dict[str, Any]:
    schema = api_get("/feature-schema")
    if not schema:
        return {
            "numeric_features": list(FALLBACK_FEATURES),
            "wine_types": ["red", "white"],
            "feature_ranges": FALLBACK_FEATURES,
            "source": "fallback",
        }
    schema["source"] = "api"
    return schema


@st.cache_data(ttl=30, show_spinner=False)
def fetch_health() -> Optional[Dict[str, Any]]:
    return api_get("/health")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_info() -> Optional[Dict[str, Any]]:
    return api_get("/model-info")


def slider_bounds(stats: Dict[str, float]) -> tuple[float, float, float, float]:
    low = float(stats.get("p01", stats.get("min", 0.0)))
    high = float(stats.get("p99", stats.get("max", 1.0)))
    low, high = min(low, high), max(low, high)
    if high <= low:
        high = low + 1.0
    span = high - low
    step = max(round(span / 200, 4), 0.0001)
    default = float(stats.get("median", (low + high) / 2))
    return low, high, step, min(max(default, low), high)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    st.sidebar.title("🍷 Wine Quality")
    st.sidebar.caption("PMLDL Assignment 1 — Deployment")

    health = fetch_health()
    if health and health.get("model_loaded"):
        st.sidebar.success("API: healthy · model loaded")
    elif health:
        st.sidebar.warning("API: reachable, model NOT loaded")
    else:
        st.sidebar.error("API: unreachable")

    st.sidebar.code(API_URL, language=None)
    if st.sidebar.button("🔄 Refresh API status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    info = fetch_model_info()
    if info:
        st.sidebar.markdown("### Deployed model")
        st.sidebar.markdown(
            f"**{info.get('model_name', '—')}**  \n"
            f"version `{info.get('model_version', '—')}`  \n"
            f"trained `{(info.get('trained_at') or '—')}`"
        )
        metrics = info.get("metrics") or {}
        if metrics:
            st.sidebar.markdown("### Test metrics")
            col_a, col_b = st.sidebar.columns(2)
            col_a.metric("F1", f"{metrics.get('f1', 0):.3f}")
            col_b.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
            col_c, col_d = st.sidebar.columns(2)
            col_c.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            col_d.metric("Precision", f"{metrics.get('precision', 0):.3f}")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"API docs: {API_URL}/docs")


# --------------------------------------------------------------------------- #
# Prediction display
# --------------------------------------------------------------------------- #
def render_prediction(result: Dict[str, Any]) -> None:
    probability = float(result["probability_good"])
    is_good = int(result["prediction"]) == 1

    st.markdown("### Prediction")
    if is_good:
        st.success(f"## ✅ {result['label']}")
    else:
        st.warning(f"## 🟡 {result['label']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("P(good quality)", f"{probability:.1%}")
    col2.metric("Confidence", f"{float(result['confidence']):.1%}")
    col3.metric("Inference time", f"{float(result['inference_ms']):.1f} ms")

    st.progress(min(max(probability, 0.0), 1.0), text=f"Probability of good quality: {probability:.1%}")
    st.caption(
        f"Decision threshold {float(result['threshold']):.2f} · "
        f"model `{result.get('model_name')}` version `{result.get('model_version')}`"
    )
    with st.expander("Raw API response"):
        st.json(result)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def render_single_prediction(schema: Dict[str, Any]) -> None:
    ranges: Dict[str, Dict[str, float]] = schema["feature_ranges"]
    numeric_features: List[str] = schema.get("numeric_features") or list(ranges)
    wine_types: List[str] = schema.get("wine_types") or ["red", "white"]

    st.markdown("#### 1 · Choose a starting point")
    preset_name = st.selectbox(
        "Preset sample", ["— start from the dataset median —", *PRESETS], index=0
    )
    preset = PRESETS.get(preset_name, {})

    st.markdown("#### 2 · Enter the physicochemical properties")
    with st.form("prediction_form"):
        wine_type = st.radio(
            "Wine type",
            wine_types,
            horizontal=True,
            index=wine_types.index(preset.get("wine_type", wine_types[0]))
            if preset.get("wine_type", wine_types[0]) in wine_types
            else 0,
        )

        values: Dict[str, Any] = {}
        columns = st.columns(3)
        for index, feature in enumerate(numeric_features):
            stats = ranges.get(feature, FALLBACK_FEATURES.get(feature, {"min": 0.0, "max": 1.0}))
            low, high, step, default = slider_bounds(stats)
            default = float(preset.get(feature, default))
            default = min(max(default, low), high)
            with columns[index % 3]:
                values[feature] = st.slider(
                    FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                    min_value=float(round(low, 4)),
                    max_value=float(round(high, 4)),
                    value=float(round(default, 4)),
                    step=float(step),
                    key=f"input_{feature}",
                )

        submitted = st.form_submit_button("🔮 Predict quality", type="primary", use_container_width=True)

    if submitted:
        payload = {**values, "wine_type": wine_type}
        with st.spinner("Calling the model API…"):
            result, error = api_post("/predict", payload)
        if error:
            st.error(error)
            st.info("Make sure the API container is running: `docker compose ps` in `code/deployment`.")
        else:
            render_prediction(result)
            with st.expander("Payload sent to the API"):
                st.json(payload)


def render_batch_prediction(schema: Dict[str, Any]) -> None:
    numeric_features: List[str] = schema.get("numeric_features") or list(FALLBACK_FEATURES)
    template = pd.DataFrame([{**{f: schema["feature_ranges"].get(f, {}).get("median", 0.0) for f in numeric_features}, "wine_type": "red"}])

    st.markdown("Upload a CSV with one wine per row and the columns below.")
    st.download_button(
        "⬇️ Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="wine_batch_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return

    frame = pd.read_csv(uploaded)
    st.write(f"Loaded **{len(frame)}** rows.")
    st.dataframe(frame.head(), use_container_width=True)

    missing = [c for c in [*numeric_features, "wine_type"] if c not in frame.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    if st.button("🔮 Predict for all rows", type="primary"):
        items = frame[[*numeric_features, "wine_type"]].to_dict(orient="records")
        with st.spinner(f"Scoring {len(items)} wines…"):
            result, error = api_post("/predict/batch", {"items": items})
        if error:
            st.error(error)
            return
        predictions = pd.DataFrame(result["predictions"])
        output = frame.copy()
        output["prediction"] = predictions["prediction"]
        output["probability_good"] = predictions["probability_good"].round(4)
        st.success(
            f"Scored {result['count']} wines in {result['inference_ms']:.1f} ms · "
            f"{int(output['prediction'].sum())} predicted as good quality."
        )
        st.dataframe(output, use_container_width=True)
        st.download_button(
            "⬇️ Download predictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="wine_predictions.csv",
            mime="text/csv",
        )


def render_model_tab() -> None:
    info = fetch_model_info()
    if not info:
        st.warning("Model information is not available — is the API container running?")
        return

    st.markdown("#### Champion model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", info.get("model_name", "—"))
    col2.metric("Version", info.get("model_version", "—"))
    col3.metric("Training rows", f"{info.get('train_rows', 0):,}")
    st.caption(f"Target: `{info.get('target')}` — positive class means {info.get('positive_class_meaning')}")

    metrics = info.get("metrics") or {}
    if metrics:
        st.markdown("#### Test-set metrics")
        st.dataframe(
            pd.DataFrame([{k: round(float(v), 4) for k, v in metrics.items()}]).T.rename(columns={0: "value"}),
            use_container_width=True,
        )

    leaderboard = info.get("leaderboard") or {}
    if leaderboard:
        st.markdown("#### Candidate leaderboard (cross-validation)")
        st.dataframe(pd.DataFrame(leaderboard).T, use_container_width=True)

    matrix = info.get("confusion_matrix") or []
    if matrix:
        st.markdown("#### Confusion matrix (test set)")
        st.dataframe(
            pd.DataFrame(matrix, index=["actual: standard", "actual: good"],
                         columns=["predicted: standard", "predicted: good"]),
            use_container_width=True,
        )

    st.caption(
        f"Trained with scikit-learn {info.get('sklearn_version')} · "
        f"served with {info.get('runtime_sklearn_version')} · git `{info.get('git_revision')}`"
    )


def main() -> None:
    render_sidebar()
    st.title("🍷 Wine Quality Predictor")
    st.markdown(
        "Enter the physicochemical properties of a wine and press **Predict quality**. "
        "The prediction is produced by the model API running in a separate Docker container."
    )
    schema = fetch_schema()
    if schema.get("source") == "fallback":
        st.info("Using built-in feature ranges — the API has not published its schema yet.")

    tab_single, tab_batch, tab_model = st.tabs(["🔮 Single prediction", "📦 Batch prediction", "📊 Model"])
    with tab_single:
        render_single_prediction(schema)
    with tab_batch:
        render_batch_prediction(schema)
    with tab_model:
        render_model_tab()


if __name__ == "__main__":
    main()
