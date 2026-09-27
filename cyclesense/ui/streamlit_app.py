"""CycleSense Streamlit experience for cycle estimates and pattern exploration."""

from datetime import date, datetime
import os
import time

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = DEFAULT_API_URL.rstrip("/")
if os.getenv("DOCKER_COMPOSE") == "true":
    API_URL = "http://backend:8000"


st.set_page_config(
    page_title="CycleSense | Cycle clarity, thoughtfully",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        :root {
            --ink: #183a3d;
            --muted: #637878;
            --surface: #ffffff;
            --canvas: #f5f8f6;
            --line: #dce7e4;
            --mint: #dff4ea;
            --teal: #176b67;
            --teal-dark: #0e4f4d;
            --coral: #df7468;
        }

        .stApp {
            background:
                radial-gradient(circle at 92% -8%, #dff4ea 0, rgba(223, 244, 234, 0) 28rem),
                radial-gradient(circle at 3% 38%, #f9e7e2 0, rgba(249, 231, 226, 0) 24rem),
                var(--canvas);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        #MainMenu, footer { visibility: hidden; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #123f41;
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] > div:first-child { padding: 1.25rem .85rem; }
        [data-testid="stSidebar"] * { color: #eff9f6; }
        [data-testid="stSidebar"] .stRadio > div { gap: .35rem; }
        [data-testid="stSidebar"] label {
            background: transparent;
            border-radius: 12px;
            padding: .48rem .62rem;
            transition: background .15s ease;
        }
        [data-testid="stSidebar"] label:hover { background: rgba(255,255,255,.1); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: inherit; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #bad1cc; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }

        h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
        h1 { font-family: 'Playfair Display', Georgia, serif; font-size: clamp(2.3rem, 4vw, 3.75rem); line-height: 1.03; }
        h2 { font-size: 1.4rem; margin-top: 1rem; }
        h3 { font-size: 1.02rem; }
        p, label, .stMarkdown { color: var(--muted); }
        .stButton > button, .stFormSubmitButton > button {
            border: 0;
            border-radius: 11px;
            min-height: 2.85rem;
            font-family: 'DM Sans', sans-serif;
            font-weight: 700;
            box-shadow: 0 7px 16px rgba(23, 107, 103, .16);
        }
        .stButton > button[kind="primary"], .stFormSubmitButton > button {
            background: linear-gradient(135deg, var(--teal-dark), var(--teal));
            color: white;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: transparent;
            filter: brightness(1.04);
        }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input, [data-testid="stSelectbox"] > div > div {
            background: #fff;
            border-color: #ceddda;
            border-radius: 10px;
        }
        [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus,
        [data-testid="stDateInput"] input:focus { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal); }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.8);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem 1.05rem .9rem;
            min-height: 112px;
            box-shadow: 0 5px 20px rgba(25, 58, 61, .035);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
        [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.6rem; }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(255,255,255,.7);
        }
        [data-testid="stForm"] {
            background: rgba(255,255,255,.72);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.1rem 1.15rem .45rem;
            box-shadow: 0 10px 30px rgba(26, 61, 62, .035);
        }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb="tab"] {
            color: var(--muted);
            font-size: .92rem;
            font-weight: 600;
            padding: .55rem .9rem;
        }
        .stTabs [aria-selected="true"] { color: var(--teal-dark); }
        .stTabs [data-baseweb="tab-highlight"] { background: var(--teal); }
        .stAlert { border-radius: 12px; }
        .stDivider { border-color: var(--line); }

        .brand-mark {
            width: 2.35rem; height: 2.35rem; display: inline-flex; align-items: center; justify-content: center;
            color: #123f41; background: #dff4ea; border-radius: 10px; font-size: 1.15rem; margin-right: .7rem;
        }
        .brand-title { font-size: 1.2rem; font-weight: 700; letter-spacing: -.03em; color: white; vertical-align: .16rem; }
        .brand-subtitle { color: #bad1cc; font-size: .76rem; letter-spacing: .02em; margin: .45rem 0 1.35rem 3.1rem; }
        .eyebrow { color: var(--teal); font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .65rem; }
        .hero { padding: .35rem 0 1.7rem; }
        .hero h1 { margin: 0 0 .72rem; max-width: 760px; }
        .hero p { font-size: 1.05rem; line-height: 1.65; margin: 0; max-width: 660px; }
        .status-chip { display: inline-flex; align-items: center; gap: .42rem; background: var(--mint); color: #185e51; border-radius: 999px; padding: .35rem .68rem; font-size: .78rem; font-weight: 700; }
        .status-dot { width: .45rem; height: .45rem; background: #44a477; border-radius: 50%; display: inline-block; }
        .notice-card { background: #fffaf0; border: 1px solid #f3dfb4; border-radius: 13px; padding: .82rem 1rem; color: #72561c; font-size: .88rem; line-height: 1.45; }
        .notice-card strong { color: #604411; }
        .soft-card { background: rgba(255,255,255,.82); border: 1px solid var(--line); border-radius: 18px; padding: 1.25rem; box-shadow: 0 10px 30px rgba(26, 61, 62, .045); }
        .soft-card h3 { margin: 0 0 .35rem; }
        .soft-card p { font-size: .9rem; line-height: 1.5; margin: 0; }
        .section-kicker { color: var(--teal); font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin: 1.6rem 0 .25rem; }
        .section-title { font-size: 1.35rem; font-weight: 700; letter-spacing: -.03em; color: var(--ink); margin: 0 0 .25rem; }
        .section-copy { color: var(--muted); font-size: .91rem; margin: 0 0 1rem; }
        .result-card { background: linear-gradient(135deg, #0e5150, #1b766e); color: white; border-radius: 19px; padding: 1.45rem 1.6rem; box-shadow: 0 14px 32px rgba(14, 79, 77, .18); }
        .result-card .result-label { color: #c8e7df; text-transform: uppercase; font-size: .72rem; letter-spacing: .13em; font-weight: 700; }
        .result-card .result-number { font-family: 'Playfair Display', Georgia, serif; font-size: 3rem; line-height: 1.1; color: white; margin: .25rem 0; }
        .result-card .result-copy { color: #d8f0ea; font-size: .92rem; margin: 0; }
        .insight-row { background: rgba(255,255,255,.78); border: 1px solid var(--line); border-radius: 13px; padding: .82rem 1rem; margin-bottom: .55rem; color: var(--ink); font-size: .92rem; }
        .insight-icon { color: var(--teal); font-weight: 700; margin-right: .5rem; }
        .footer-note { color: #788b89; font-size: .78rem; text-align: center; padding: 2rem 0 0; }
        @media (max-width: 768px) {
            [data-testid="stMainBlockContainer"] { padding: 1.15rem 1rem 2rem; }
            .hero { padding-bottom: 1.1rem; }
            .hero h1 { font-size: 2.45rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_api(endpoint, method="GET", data=None, max_retries=2, show_errors=True):
    """Call the CycleSense API, returning parsed JSON when it is available."""
    for attempt in range(max_retries):
        try:
            url = f"{API_URL}{endpoint}"
            response = requests.get(url, timeout=20) if method == "GET" else requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            if show_errors:
                st.error("We couldn't complete that request. Please try again in a moment.")
            return None
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1.2 * (attempt + 1))
    if show_errors:
        st.error("CycleSense can't reach the prediction service right now.")
        st.caption(f"Service address: {API_URL}")
    return None


def render_notice():
    st.markdown(
        """
        <div class="notice-card"><strong>For awareness, not medical decisions.</strong>
        CycleSense offers educational estimates and pattern exploration. It is not a diagnostic,
        contraceptive, fertility-planning, or medical-care tool.</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(eyebrow, title, description, status=None):
    status_html = ""
    if status:
        status_html = f'<div style="margin-top:1rem">{status}</div>'
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
            {status_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section(kicker, title, description):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div><p class="section-copy">{description}</p>',
        unsafe_allow_html=True,
    )


def parse_cycles(raw_value):
    """Convert a comma-separated history into a bounded list of cycle lengths."""
    try:
        values = [float(value.strip()) for value in raw_value.split(",") if value.strip()]
    except ValueError:
        return None
    if not values or any(value < 20 or value > 45 for value in values):
        return None
    return values


def variability_label(coefficient):
    if coefficient < 5:
        return "Very consistent", "Steady history"
    if coefficient < 10:
        return "Consistent", "A small amount of variation"
    if coefficient < 15:
        return "Moderately variable", "More movement than usual"
    return "Highly variable", "A wide spread in recorded cycles"


def switch_to_estimate():
    """Move to the estimate workspace from the overview call to action."""
    st.session_state.nav_page = "Estimate"


def render_dashboard():
    health = call_api("/health", max_retries=1, show_errors=False)
    is_connected = bool(health and health.get("status") == "healthy")
    status = (
        '<span class="status-chip"><span class="status-dot"></span>Prediction service connected</span>'
        if is_connected
        else '<span class="status-chip" style="background:#f9e7e2;color:#a34d45"><span class="status-dot" style="background:#df7468"></span>Service unavailable</span>'
    )
    render_hero(
        "Your cycle, in context",
        "Understand patterns.<br>Feel more prepared.",
        "A calm place to explore your recent cycle history and generate an educational estimate based on the details you choose to share.",
        status,
    )

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="soft-card" style="padding:1.55rem">
                <div class="eyebrow">Start here</div>
                <h2 style="font-family:'Playfair Display',Georgia,serif;font-size:1.85rem;margin:.1rem 0 .45rem">Get an informed estimate</h2>
                <p style="font-size:1rem;max-width:34rem">Add a few recent cycle lengths and optional lifestyle context. The model uses those inputs to estimate your next cycle length.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Create an estimate  →",
            type="primary",
            use_container_width=True,
            on_click=switch_to_estimate,
        )
    with right:
        st.markdown(
            """
            <div class="soft-card" style="background:linear-gradient(135deg,#f0eafd,#e5f5ef);min-height:100%">
                <div class="eyebrow" style="color:#6554a6">A gentler way to look at data</div>
                <h3 style="font-size:1.15rem">Patterns over perfection</h3>
                <p>Use the insights space to see your recent average, variability, and directional trend at a glance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section("Explore", "Choose what you need today", "Each space is designed to keep the useful signal visible and the rest out of your way.")
    cards = [
        ("01", "Estimate", "Build a model-based cycle estimate from your profile and current cycle."),
        ("02", "Patterns", "Turn a list of past cycle lengths into a clear visual story."),
        ("03", "Model", "See the scope, performance, and educational limits of the model."),
    ]
    cols = st.columns(3, gap="medium")
    for column, (number, name, copy) in zip(cols, cards):
        with column:
            st.markdown(
                f'<div class="soft-card"><div class="eyebrow">{number}</div><h3>{name}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )

    if not is_connected:
        st.warning("The interface is ready, but the prediction API is offline. Pattern insights remain available without it.")


def render_predict_page():
    render_hero(
        "Personal estimate",
        "Bring your recent history<br>into focus.",
        "Start with what you know. Required fields are kept to the essentials; the extra context helps tailor the model input.",
    )
    render_notice()
    render_section("Your inputs", "Tell the model a little about this cycle", "Cycle history should be entered most recent first. You can adjust any prefilled value.")

    with st.form("prediction_form", border=True):
        profile_tab, cycle_tab, wellbeing_tab = st.tabs(["Profile", "Current cycle", "Lifestyle context"])
        with profile_tab:
            left, right = st.columns(2, gap="large")
            with left:
                age = st.number_input("Age", min_value=18, max_value=50, value=28, step=1)
                bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.8, step=0.1)
                diet_quality = st.selectbox("Diet quality", ["Good", "Excellent", "Fair", "Poor"])
            with right:
                exercise_frequency = st.selectbox("Exercise frequency", ["3-4 days/week", "1-2 days/week", "5-6 days/week"])
                sleep_hours = st.number_input("Usual sleep (hours)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
                stress_baseline = st.slider("Usual stress level", 1.0, 10.0, 5.5, 0.5, help="1 is low and 10 is high.")
        with cycle_tab:
            left, right = st.columns(2, gap="large")
            with left:
                cycle_length = st.number_input("Current cycle length (days)", min_value=20.0, max_value=45.0, value=28.0, step=0.1)
                prev_cycle_length = st.number_input("Previous cycle length (days)", min_value=20.0, max_value=45.0, value=28.0, step=0.1)
                cycle_phase = st.selectbox("Current cycle phase", ["Follicular", "Luteal", "Ovulation"])
                flow_level = st.selectbox("Flow level", ["Medium", "Light", "Heavy"])
            with right:
                pain_level = st.slider("Pain level", 1, 10, 5)
                pms_symptoms = st.selectbox("PMS symptoms", ["No", "Yes"])
                mood_score = st.slider("Mood score", 1, 10, 7)
                stress_cycle = st.slider("Current stress level", 1.0, 10.0, 5.5, 0.5)
                sleep_cycle = st.number_input("Current sleep (hours)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
        with wellbeing_tab:
            left, right = st.columns(2, gap="large")
            with left:
                caffeine_intake = st.number_input("Caffeine (cups/day)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
                water_intake = st.number_input("Water (litres/day)", min_value=0.5, max_value=5.0, value=2.3, step=0.1)
                alcohol_consumption = st.selectbox("Alcohol consumption", ["Occasionally", "Never", "Weekly"])
            with right:
                smoking_status = st.selectbox("Smoking status", ["No", "Yes"])
                birth_control = st.selectbox("Birth control use", [0, 1], format_func=lambda value: "Yes" if value else "No")
                pcos_diagnosed = st.selectbox("PCOS diagnosis", [0, 1], format_func=lambda value: "Yes" if value else "No")

        st.divider()
        history_column, date_column = st.columns([1.6, 1], gap="large")
        with history_column:
            historical_raw = st.text_input(
                "Recent cycle lengths",
                "28, 29, 27, 28, 30",
                help="Use commas between values. Enter the most recent cycle first.",
            )
        with date_column:
            start_date = st.date_input("Current cycle start", value=date.today())
        submitted = st.form_submit_button("Generate my estimate", type="primary", use_container_width=True)

    if submitted:
        historical_cycles = parse_cycles(historical_raw)
        if not historical_cycles:
            st.error("Enter cycle lengths between 20 and 45 days, separated by commas.")
            return
        request_data = {
            "profile": {
                "age": age, "bmi": bmi, "diet_quality": diet_quality,
                "exercise_frequency": exercise_frequency, "sleep_hours": sleep_hours,
                "caffeine_intake": caffeine_intake, "water_intake_liters": water_intake,
                "alcohol_consumption": alcohol_consumption, "smoking_status": smoking_status,
                "birth_control_use": birth_control, "pcos_diagnosed": pcos_diagnosed,
                "stress_score_baseline": stress_baseline,
            },
            "cycle_history": {
                "cycle_length_days": cycle_length, "prev_cycle_length": prev_cycle_length,
                "cycle_phase": cycle_phase, "flow_level": flow_level, "pain_level": pain_level,
                "pms_symptoms": pms_symptoms, "mood_score": mood_score,
                "stress_score_cycle": stress_cycle, "sleep_hours_cycle": sleep_cycle,
                "energy_level": 7, "concentration_score": 7, "work_hours_lost": 0.0,
                "start_date": start_date.isoformat(),
            },
            "historical_cycles": historical_cycles,
        }
        with st.spinner("Creating your educational estimate…"):
            response = call_api("/predict", method="POST", data=request_data)
        if response:
            st.session_state.latest_prediction = response

    response = st.session_state.get("latest_prediction")
    if response:
        prediction = response["predicted_next_cycle_length_days"]
        next_date = response.get("predicted_next_period_date")
        next_date_text = "Not available"
        if next_date:
            try:
                next_date_text = datetime.fromisoformat(next_date).strftime("%d %b %Y")
            except ValueError:
                next_date_text = next_date
        render_section("Your result", "A model-based starting point", "Treat this as a helpful reference point, not a certainty or medical recommendation.")
        result_col, details_col = st.columns([1.15, 1], gap="large")
        with result_col:
            st.markdown(
                f'<div class="result-card"><div class="result-label">Estimated next cycle length</div><div class="result-number">{prediction:.1f} days</div><p class="result-copy">Generated from your recent history and the context you shared.</p></div>',
                unsafe_allow_html=True,
            )
        with details_col:
            st.metric("Estimated next period date", next_date_text)
            st.metric("Model version", response.get("model_version", "1.0"), "Educational estimate")
        render_notice()


def render_pattern_insights():
    render_hero(
        "Pattern explorer",
        "See the rhythm<br>in your history.",
        "Enter recent cycle lengths to make variability, trend, and a practical reference estimate easier to see.",
    )
    render_notice()
    render_section("Cycle history", "Start with recent lengths", "Enter at least three values, most recent first. Nothing is saved by this page.")
    historical_raw = st.text_input("Cycle lengths (days)", "28, 29, 27, 28, 30, 27, 26, 28, 29, 28", label_visibility="collapsed")
    historical_cycles = parse_cycles(historical_raw)
    if not historical_cycles:
        st.error("Enter valid cycle lengths between 20 and 45 days, separated by commas.")
        return
    if len(historical_cycles) < 3:
        st.info("Add at least three cycle lengths to reveal a meaningful pattern.")
        return

    cycle_array = np.array(historical_cycles)
    average = float(np.mean(cycle_array))
    standard_deviation = float(np.std(cycle_array))
    coefficient = (standard_deviation / average) * 100 if average else 0
    recent_average = float(np.mean(cycle_array[:3]))
    trend = float(np.polyfit(np.arange(len(cycle_array)), cycle_array, 1)[0])
    regularity, regularity_copy = variability_label(coefficient)

    render_section("At a glance", "Your recent cycle profile", "A few simple reference points from the values you entered.")
    metric_columns = st.columns(4, gap="medium")
    metrics = [
        ("Average length", f"{average:.1f} days", "All recorded cycles"),
        ("Recent average", f"{recent_average:.1f} days", "Most recent 3"),
        ("Variation", f"{coefficient:.1f}%", regularity),
        ("Range", f"{np.min(cycle_array):.0f}–{np.max(cycle_array):.0f} days", f"± {standard_deviation:.1f} days"),
    ]
    for column, (label, value, delta) in zip(metric_columns, metrics):
        with column:
            st.metric(label, value, delta)

    chart_column, reading_column = st.columns([1.55, 1], gap="large")
    with chart_column:
        render_section("Trend", "Cycle length over time", "Most recent cycle appears at the left of the chart.")
        labels = ["Most recent" if index == 0 else f"{index} cycles ago" for index in range(len(cycle_array))]
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=labels, y=cycle_array, mode="lines+markers", name="Cycle length",
            line=dict(color="#176b67", width=3), marker=dict(size=8, color="#176b67", line=dict(color="#ffffff", width=2)),
            fill="tozeroy", fillcolor="rgba(223,244,234,.55)", hovertemplate="%{x}<br><b>%{y:.1f} days</b><extra></extra>",
        ))
        figure.add_hline(y=average, line_dash="dot", line_color="#df7468", annotation_text=f"Average {average:.1f} days", annotation_font_color="#a75148")
        figure.update_layout(
            height=350, margin=dict(l=0, r=5, t=18, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.62)",
            showlegend=False, xaxis=dict(showgrid=False, tickfont=dict(color="#637878", size=11)),
            yaxis=dict(title="Days", gridcolor="#e7efec", zeroline=False, tickfont=dict(color="#637878"), range=[max(0, np.min(cycle_array) - 3), np.max(cycle_array) + 3]),
            font=dict(family="DM Sans, sans-serif", color="#183a3d"),
        )
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    with reading_column:
        render_section("Reading", "What stands out", "Plain-language context from the numbers—not a diagnosis.")
        if trend > 0.1:
            trend_label, trend_copy = "Lengthening", "Recent entries lean longer over time."
        elif trend < -0.1:
            trend_label, trend_copy = "Shortening", "Recent entries lean shorter over time."
        else:
            trend_label, trend_copy = "Steady", "There is no meaningful upward or downward trend."
        delta = recent_average - average
        recent_label = "above" if delta > 1 else "below" if delta < -1 else "near"
        st.markdown(f'<div class="insight-row"><span class="insight-icon">01</span><strong>{regularity}</strong> · {regularity_copy}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-row"><span class="insight-icon">02</span><strong>{trend_label}</strong> · {trend_copy}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-row"><span class="insight-icon">03</span>Your recent average is <strong>{recent_label}</strong> your overall average by {abs(delta):.1f} days.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        weighted_estimate = recent_average * 0.6 + average * 0.4
        st.markdown(
            f'<div class="soft-card" style="background:#e6f4ee"><div class="eyebrow">History-based reference</div><h2 style="margin:.2rem 0">{weighted_estimate:.1f} days</h2><p>This simple reference weights your three most recent entries alongside your full history.</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Everyday context that can be useful to track"):
        st.markdown("Consistent sleep, stress changes, activity, illness, medication, and major routine shifts can all be useful context to discuss with a qualified clinician if you have concerns. This app does not determine causes or provide medical guidance.")


def render_model_insights():
    render_hero(
        "Model transparency",
        "Useful context,<br>clearly stated.",
        "CycleSense is an educational machine-learning demonstration. Here is what informs its estimate—and where its limits begin.",
    )
    render_notice()
    model_info = call_api("/model-info", max_retries=1, show_errors=False)
    metrics = model_info.get("metrics", {}) if model_info else {}

    render_section("Performance", "How the model is evaluated", "Scores are reported on the project’s held-out evaluation data.")
    metric_columns = st.columns(3, gap="medium")
    metric_values = [
        ("Mean absolute error", f"{metrics.get('mae', 1.70):.2f} days", "Average distance from observed length"),
        ("Root mean squared error", f"{metrics.get('rmse', 2.18):.2f} days", "Weights larger misses more heavily"),
        ("Explained variance (R²)", f"{metrics.get('r2', .306):.2f}", "A limited, educational benchmark"),
    ]
    for column, (label, value, detail) in zip(metric_columns, metric_values):
        with column:
            st.metric(label, value, detail)

    left, right = st.columns(2, gap="large")
    with left:
        render_section("Inputs", "What it considers", "The model combines the context you provide with recent cycle history.")
        st.markdown(
            """
            <div class="soft-card"><h3>Signals in scope</h3><p>Recent cycle lengths, current-cycle details, sleep and stress measures, profile information, and selected lifestyle context.</p></div>
            <div class="soft-card" style="margin-top:.7rem"><h3>Designed with availability in mind</h3><p>Features are selected to avoid using information that would not realistically be known when an estimate is made.</p></div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        render_section("Details", "Model card", "A compact view of the current service configuration.")
        model_name = model_info.get("model_name", "CycleSenseCycleLengthModel") if model_info else "CycleSenseCycleLengthModel"
        model_type = model_info.get("model_type", "HistGradientBoostingRegressor") if model_info else "HistGradientBoostingRegressor"
        feature_set = model_info.get("feature_set", "strict") if model_info else "strict"
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="eyebrow">Current model</div>
                <h3>{model_name}</h3>
                <p><strong>Method:</strong> {model_type}</p>
                <p><strong>Feature set:</strong> {feature_set}</p>
                <p><strong>Version:</strong> {(model_info or {}).get('model_version', '1.0.0')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Service details are shown when the API is available; otherwise the default project model card is shown.")


def render_about():
    render_hero(
        "About CycleSense",
        "A small tool for<br>more thoughtful reflection.",
        "CycleSense makes a machine-learning project feel human: understandable inputs, legible outputs, and clear boundaries around what an estimate can mean.",
    )
    render_notice()
    left, right = st.columns(2, gap="large")
    with left:
        render_section("Purpose", "Designed for exploration", "This is an educational demonstration, not a health-care product.")
        st.markdown(
            """
            <div class="soft-card"><h3>What it does</h3><p>It uses historical cycle data and selected self-reported context to estimate a next-cycle length and visualize simple patterns.</p></div>
            <div class="soft-card" style="margin-top:.7rem"><h3>What it does not do</h3><p>It does not diagnose conditions, replace clinical advice, estimate fertility, or provide contraception guidance.</p></div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        render_section("Build", "Made with an explainable stack", "The app is intentionally simple to inspect and discuss.")
        st.markdown(
            """
            <div class="soft-card"><h3>Data & model</h3><p>Feature engineering and a gradient-boosting model are used to learn from the project’s educational dataset.</p></div>
            <div class="soft-card" style="margin-top:.7rem"><h3>Delivery</h3><p>FastAPI serves estimates and Streamlit provides this interface, with DVC and MLflow supporting the project workflow.</p></div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar():
    st.sidebar.markdown(
        '<span class="brand-mark">✦</span><span class="brand-title">CycleSense</span><div class="brand-subtitle">cycle clarity, thoughtfully</div>',
        unsafe_allow_html=True,
    )
    pages = {"Overview": "Dashboard", "Estimate": "Estimate", "Patterns": "Patterns", "Model": "Model", "About": "About"}
    labels = list(pages.keys())
    if "nav_page" not in st.session_state or st.session_state.nav_page not in labels:
        st.session_state.nav_page = "Overview"
    page = st.sidebar.radio("Navigation", labels, key="nav_page", label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("EDUCATIONAL USE ONLY")
    st.sidebar.caption("Estimates support reflection; they are not medical advice or a clinical tool.")
    return pages[page]


def main():
    page = render_sidebar()
    if page == "Dashboard":
        render_dashboard()
    elif page == "Estimate":
        render_predict_page()
    elif page == "Patterns":
        render_pattern_insights()
    elif page == "Model":
        render_model_insights()
    else:
        render_about()
    st.markdown('<div class="footer-note">CycleSense · Built for educational pattern exploration</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
