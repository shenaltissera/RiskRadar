"""
app.py — RiskRadar Streamlit Dashboard
---------------------------------------
Live traffic accident risk prediction dashboard for Melbourne.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_risk
from src.weather import get_current_weather, format_weather_summary

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskRadar",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0;
        line-height: 1.1;
    }

    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        font-size: 1rem;
        color: #888;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    .risk-card {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .risk-low {
        background: linear-gradient(135deg, #0d1f0f, #1a3d1e);
        border: 1px solid #2d6a35;
    }

    .risk-moderate {
        background: linear-gradient(135deg, #1f1800, #3d3000);
        border: 1px solid #8a6800;
    }

    .risk-high {
        background: linear-gradient(135deg, #1f0000, #3d0000);
        border: 1px solid #8a0000;
    }

    .risk-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 0.3rem;
    }

    .risk-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1;
    }

    .risk-low    .risk-value { color: #4cff6e; }
    .risk-moderate .risk-value { color: #ffd84c; }
    .risk-high   .risk-value { color: #ff4c4c; }

    .risk-score-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 1rem;
        background: #333;
    }

    .weather-strip {
        background: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #aaa;
        margin-bottom: 1rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .metric-box {
        flex: 1;
        background: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }

    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #555;
        margin-bottom: 0.2rem;
    }

    .metric-val {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        color: #eee;
    }

    .stButton > button {
        width: 100%;
        background: #1a1a1a;
        border: 1px solid #333;
        color: #eee;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 0.6rem;
        border-radius: 6px;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #252525;
        border-color: #555;
    }

    .predict-btn > button {
        background: #1a3d1e !important;
        border-color: #2d6a35 !important;
        color: #4cff6e !important;
    }

    div[data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #444;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# ── Melbourne suburb grid ──────────────────────────────────────────────────────
MELBOURNE_SUBURBS = {
    "Melbourne CBD":      (-37.8136, 144.9631, 60),
    "Southbank":          (-37.8233, 144.9631, 50),
    "St Kilda":           (-37.8676, 144.9796, 50),
    "Richmond":           (-37.8230, 145.0040, 50),
    "Fitzroy":            (-37.7994, 144.9782, 50),
    "Carlton":            (-37.7963, 144.9669, 50),
    "Collingwood":        (-37.8040, 144.9849, 50),
    "South Yarra":        (-37.8390, 144.9912, 60),
    "Toorak":             (-37.8450, 145.0130, 50),
    "Hawthorn":           (-37.8224, 145.0332, 60),
    "Box Hill":           (-37.8197, 145.1238, 60),
    "Tullamarine Fwy":    (-37.7100, 144.8800, 100),
    "Eastern Fwy":        (-37.7900, 145.0600, 100),
    "Monash Fwy":         (-37.9000, 145.1200, 100),
    "West Gate Fwy":      (-37.8300, 144.8800, 100),
    "Dandenong":          (-37.9872, 145.2150, 80),
    "Frankston":          (-38.1440, 145.1260, 80),
    "Werribee":           (-37.9000, 144.6600, 80),
}


# ── Risk colour helpers ────────────────────────────────────────────────────────
def risk_colour(level: str) -> tuple:
    return {
        "low":      (76,  255, 110, 180),
        "moderate": (255, 216,  76, 180),
        "high":     (255,  76,  76, 200),
    }.get(level, (150, 150, 150, 150))


def risk_hex(level: str) -> str:
    return {"low": "#4cff6e", "moderate": "#ffd84c", "high": "#ff4c4c"}.get(level, "#aaa")


# ── Cached predictions for the suburb heatmap ─────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_weather_cached(lat: float, lon: float, dt_str: str) -> dict:
    """Fetch weather once and cache it — reused across all suburb/hour predictions."""
    from src.weather import get_current_weather, get_weather_for_datetime
    dt  = datetime.fromisoformat(dt_str)
    now = datetime.now()
    if (dt - now).total_seconds() / 3600 < -6:
        return get_weather_for_datetime(dt, lat=lat, lon=lon)
    return get_current_weather(lat=lat, lon=lon)


@st.cache_data(ttl=300, show_spinner=False)
def get_suburb_predictions(dt_str: str, weather_risk_score: int, is_adverse_weather: int):
    """Predict risk for all suburbs using pre-fetched weather. No HTTP calls."""
    import joblib, os
    from src.features import FEATURE_COLS
    from src.predict import _is_low_risk, _build_prediction_features

    dt      = datetime.fromisoformat(dt_str)
    payload = joblib.load("models/riskradar_model.joblib")
    model   = payload["model"]

    # Fake weather dict with pre-fetched scores
    weather = {"weather_risk_score": weather_risk_score,
               "is_adverse_weather": is_adverse_weather,
               "precipitation_mm":   0.0}

    results = []
    for suburb, (lat, lon, speed) in MELBOURNE_SUBURBS.items():
        try:
            if _is_low_risk(dt.hour, speed, weather):
                risk_level, risk_score = "low", 0.10
            else:
                X = _build_prediction_features(dt=dt, weather=weather, speed_zone=speed)
                X = X[FEATURE_COLS]
                risk_score = float(model.predict_proba(X)[0][1])
                risk_level = "high" if risk_score >= 0.45 else "moderate"
            results.append({
                "suburb":     suburb,
                "lat":        lat,
                "lon":        lon,
                "risk_level": risk_level,
                "risk_score": round(risk_score, 2),
                "colour":     list(risk_colour(risk_level)),
            })
        except Exception:
            pass
    return pd.DataFrame(results)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">🚦 Risk<br>Radar</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Melbourne traffic risk</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
    suburb_choice = st.selectbox(
        "Suburb or road",
        options=list(MELBOURNE_SUBURBS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    lat_default, lon_default, speed_default = MELBOURNE_SUBURBS[suburb_choice]

    use_custom = st.checkbox("Custom coordinates")
    if use_custom:
        lat   = st.number_input("Latitude",   value=lat_default,   format="%.4f", step=0.001)
        lon   = st.number_input("Longitude",  value=lon_default,   format="%.4f", step=0.001)
        speed = st.number_input("Speed zone (km/h)", value=speed_default, step=10, min_value=30, max_value=110)
    else:
        lat, lon, speed = lat_default, lon_default, speed_default

    st.markdown('<div class="section-header">Time</div>', unsafe_allow_html=True)

    time_mode = st.radio(
        "Time mode",
        options=["Now", "Custom"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if time_mode == "Now":
        predict_dt = datetime.now()
        st.caption(f"📍 {predict_dt.strftime('%a %d %b %Y, %H:%M')}")
    else:
        pred_date = st.date_input("Date", value=datetime.now().date())
        pred_hour = st.slider("Hour", min_value=0, max_value=23,
                              value=datetime.now().hour, format="%d:00")
        pred_min  = st.select_slider("Minute", options=[0, 15, 30, 45],
                                     value=0)
        predict_dt = datetime.combine(pred_date,
                                      datetime.min.time().replace(hour=pred_hour, minute=pred_min))
        st.caption(f"📍 {predict_dt.strftime('%a %d %b %Y, %H:%M')}")

    st.markdown('<div class="section-header">Road conditions</div>', unsafe_allow_html=True)
    is_intersection   = st.checkbox("Intersection")
    has_heavy_vehicle = st.checkbox("Heavy vehicles present")

    st.markdown("")
    predict_clicked = st.button("⚡  Run prediction", use_container_width=True)


# ── Main layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 2], gap="large")

with col_left:
    st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

    # Build a cache key from all inputs — rerun prediction whenever any input changes
    prediction_key = f"{lat:.4f}_{lon:.4f}_{predict_dt.isoformat()}_{speed}_{is_intersection}_{has_heavy_vehicle}"

    if predict_clicked or "last_result" not in st.session_state or st.session_state.get("last_key") != prediction_key:
        with st.spinner("Fetching live weather & running model..."):
            try:
                result = predict_risk(
                    lat=lat, lon=lon, dt=predict_dt,
                    speed_zone=int(speed),
                    is_intersection=is_intersection,
                    has_heavy_vehicle=has_heavy_vehicle,
                )
                st.session_state["last_result"] = result
                st.session_state["last_key"] = prediction_key
            except FileNotFoundError:
                st.error("Model not found. Run `python src/train.py` first.")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    result = st.session_state["last_result"]
    level  = result["risk_level"]
    score  = result["risk_score"]
    weather = result["weather"]

    # Risk card
    st.markdown(f"""
    <div class="risk-card risk-{level}">
        <div class="risk-label">accident risk</div>
        <div class="risk-value">{level.upper()}</div>
        <div style="font-family:'DM Sans';font-size:0.85rem;opacity:0.6;margin-top:0.4rem;">
            score: {score:.2f} / 1.00
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Weather strip
    st.markdown(f"""
    <div class="weather-strip">
        🌤 {format_weather_summary(weather)}
    </div>
    """, unsafe_allow_html=True)

    # Metric row
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-label">location</div>
            <div class="metric-val">{suburb_choice.split()[0]}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">speed zone</div>
            <div class="metric-val">{int(speed)} km/h</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">time</div>
            <div class="metric-val">{predict_dt.strftime('%H:%M')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk by hour chart
    st.markdown('<div class="section-header">Risk across the day</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=600, show_spinner=False)
    def hourly_risk_chart(lat, lon, speed, date_str, weather_risk_score, is_adverse_weather):
        """Batch 24 predictions using the model directly — no HTTP calls."""
        import joblib
        from src.features import FEATURE_COLS
        from src.predict import _is_low_risk, _build_prediction_features

        payload = joblib.load("models/riskradar_model.joblib")
        model   = payload["model"]
        weather = {"weather_risk_score": weather_risk_score,
                   "is_adverse_weather": is_adverse_weather,
                   "precipitation_mm":   0.0}
        scores = []
        base = datetime.fromisoformat(date_str)
        for h in range(24):
            dt_h = base.replace(hour=h, minute=0, second=0)
            try:
                if _is_low_risk(h, speed, weather):
                    scores.append(0.10)
                else:
                    X = _build_prediction_features(dt=dt_h, weather=weather, speed_zone=speed)
                    X = X[FEATURE_COLS]
                    scores.append(float(model.predict_proba(X)[0][1]))
            except Exception:
                scores.append(0.3)
        return scores

    w = st.session_state["last_result"]["weather"]
    scores = hourly_risk_chart(lat, lon, int(speed), predict_dt.strftime("%Y-%m-%d"),
                               w.get("weather_risk_score", 0), w.get("is_adverse_weather", 0))
    hours  = list(range(24))

    chart_df = pd.DataFrame({
        "hour": [f"{h:02d}:00" for h in hours],
        "risk": scores,
    }).set_index("hour")

    st.bar_chart(
        chart_df,
        color="#ff8c42",  # single orange-red color
        height=200,
        use_container_width=True,
    )
    
    # Color legend below
    st.markdown(f"""
    <div style="font-family:monospace;font-size:0.75rem;color:#555;margin-top:0.3rem;">
        <span style="color:#4cff6e">█</span> Low (&lt;0.35) ·
        <span style="color:#ffd84c">█</span> Moderate (0.35–0.55) ·
        <span style="color:#ff4c4c">█</span> High (&gt;0.55) ·
        Current: {predict_dt.strftime('%H:00')}
    </div>
    """, unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="section-header">Melbourne risk map</div>', unsafe_allow_html=True)

    with st.spinner("Loading suburb predictions..."):
        w = st.session_state["last_result"]["weather"]
        df_map = get_suburb_predictions(
            predict_dt.strftime("%Y-%m-%dT%H:00:00"),
            w.get("weather_risk_score", 0),
            w.get("is_adverse_weather", 0),
        )

    if not df_map.empty:
        # Scatterplot layer — suburb risk dots
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_fill_color="colour",
            get_radius=1200,
            pickable=True,
            opacity=0.85,
            stroked=True,
            get_line_color=[255, 255, 255, 40],
            line_width_min_pixels=1,
        )

        # Selected location marker
        selected_df = pd.DataFrame([{
            "lat": lat, "lon": lon,
            "colour": list(risk_colour(result["risk_level"])),
        }])
        selected_layer = pdk.Layer(
            "ScatterplotLayer",
            data=selected_df,
            get_position=["lon", "lat"],
            get_fill_color="colour",
            get_radius=600,
            pickable=False,
            opacity=1.0,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=2,
        )

        # Zoom level based on speed zone — freeways zoom out more
        zoom_level = 12.5 if speed <= 60 else 11.0

        view = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=zoom_level,
            pitch=0,
            transition_duration=800,
        )

        tooltip = {
            "html": """
                <div style="font-family:monospace;background:#111;border:1px solid #333;
                            padding:8px 12px;border-radius:6px;font-size:12px;">
                    <b style="color:#eee">{suburb}</b><br/>
                    <span style="color:#888">risk:</span>
                    <span style="color:#eee">{risk_level}</span>
                    &nbsp;·&nbsp;
                    <span style="color:#888">score:</span>
                    <span style="color:#eee">{risk_score}</span>
                </div>
            """,
            "style": {"background": "transparent", "border": "none"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[scatter_layer, selected_layer],
                initial_view_state=view,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            ),
            use_container_width=True,
            height=480,
        )

        # Legend
        st.markdown("""
        <div style="display:flex;gap:1.5rem;font-family:monospace;font-size:0.75rem;
                    color:#555;margin-top:0.5rem;padding-left:0.5rem;">
            <span><span style="color:#4cff6e">●</span> Low risk</span>
            <span><span style="color:#ffd84c">●</span> Moderate risk</span>
            <span><span style="color:#ff4c4c">●</span> High risk</span>
            <span style="margin-left:auto">Hover suburbs for details</span>
        </div>
        """, unsafe_allow_html=True)

        # Summary table
        st.markdown('<div class="section-header">All suburbs</div>', unsafe_allow_html=True)
        display_df = df_map[["suburb", "risk_level", "risk_score"]].copy()
        display_df["risk_score"] = display_df["risk_score"].map("{:.2f}".format)
        display_df.columns = ["Suburb / Road", "Risk Level", "Score"]
        display_df = display_df.sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=220)