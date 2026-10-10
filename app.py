import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Rubber Production Analytics | Bunawan, Agusan del Sur",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CUSTOM CSS  (clean, distinct UI - light body, dark sidebar)
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; font-size: 0.72rem; letter-spacing: 0.08em; }

h1, h2, h3, h4 { color: #0f172a; font-weight: 700; }

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 22px;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0d9488;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.metric-label {
    color: #64748b; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.metric-value {
    color: #0f172a; font-size: 1.7rem; font-weight: 700; margin-top: 6px;
}
.metric-sub { color: #0d9488; font-size: 0.8rem; font-weight: 500; margin-top: 4px; }

.section-header {
    color: #0f172a; font-size: 1.15rem; font-weight: 700;
    border-bottom: 2px solid #0d9488; padding-bottom: 8px;
    margin: 28px 0 16px 0;
}
.info-box {
    background: #f0fdfa; border: 1px solid #99f6e4;
    border-radius: 10px; padding: 16px 18px;
    color: #134e4a; font-size: 0.9rem; line-height: 1.55;
}
.warn-box {
    background: #fefce8; border: 1px solid #fde68a;
    border-radius: 10px; padding: 16px 18px;
    color: #713f12; font-size: 0.9rem; line-height: 1.55;
}
.sidebar-title { color: #5eead4; margin: 0; font-size: 1.15rem; font-weight: 700; }
.sidebar-sub   { color: #94a3b8; font-size: 0.78rem; margin-top: 4px; }
div[data-testid="stMetric"] {
    background: #fff; border-radius: 12px; padding: 14px 16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DATA LOADING + CLEANING
# ------------------------------------------------------------------
DATA_FILE = "Dataset  of Rubber_ 2024 - 2025.xlsx"

def _clean_text(val):
    """Normalize a text cell: strip, collapse spaces, convert junk to None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = " ".join(s.split())                    # collapse multiple spaces
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s

def _clean_number(val):
    """Convert a cell to a float where possible."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).replace(",", "").strip()
    if s == "" or s.lower() in {"nan", "none", "/", "-"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

SKIP_KEYWORDS = [
    "Province", "Municipality", "Commodity", "No.", "Prepared", "Noted",
    "MELYDEN", "ENDRENIE", "ENGR", "Last Name", "Sexual", "Asexual",
    "Intercrop", "Monocrop", "Complete Name",
]

@st.cache_data(show_spinner=False)
def load_and_clean_data(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    records = []

    for sheet_name in xls.sheet_names:
        df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        current_period = None

        for _, row in df_raw.iterrows():
            first = row.iloc[0]

            if pd.isna(first):
                continue

            first_str = str(first).strip()

            # Capture the period marker
            if first_str.startswith("Period"):
                period = first_str.split(":")[-1].strip()
                current_period = " ".join(period.split())   # collapse spaces
                continue

            # Skip all the meta / footer / header rows
            if any(kw.lower() in first_str.lower() for kw in SKIP_KEYWORDS):
                continue
            if first_str.startswith("="):                    # =SUM(...) rows
                continue

            # Data rows start with an integer in column A
            try:
                row_no = int(float(first_str))
            except ValueError:
                continue

            if not current_period:
                continue

            records.append({
                "sheet":             sheet_name,
                "period":            current_period,
                "row_no":            row_no,
                "last_name":         _clean_text(row.iloc[2]),
                "given_name":        _clean_text(row.iloc[3]),
                "middle_name":       _clean_text(row.iloc[4]),
                "home_address":      _clean_text(row.iloc[6]),
                "farm_barangay":     _clean_text(row.iloc[7]),
                "variety":           _clean_text(row.iloc[10]),
                "no_of_trees":       _clean_number(row.iloc[13]),
                "annual_volume":     _clean_number(row.iloc[15]),
                "intercrop":         _clean_text(row.iloc[17]),
                "monocrop":          _clean_text(row.iloc[18]),
                "cropping_density":  _clean_number(row.iloc[20]),
                "buyer_name":        _clean_text(row.iloc[21]),
                "buyer_location":    _clean_text(row.iloc[22]),
                "market_volume":     _clean_number(row.iloc[23]),
            })

    df = pd.DataFrame(records)

    # --- additional cleaning ---
    df["farmer_name"] = (
        df["last_name"].fillna("") + ", " +
        df["given_name"].fillna("") + " " +
        df["middle_name"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip(", ")

    df["cropping_system"] = np.where(df["monocrop"].notna(), "Monocrop",
                              np.where(df["intercrop"].notna(), "Intercrop", "Unknown"))

    # Production proxy: prefer market_volume, fall back to annual_volume
    df["production_kg"] = df["market_volume"].fillna(df["annual_volume"])

    # Drop rows without any production data (still keep for reference if needed)
    df = df.dropna(subset=["no_of_trees"], how="all")

    # Clean variety labels
    df["variety"] = df["variety"].fillna("Unknown")

    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
try:
    df = load_and_clean_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_FILE}. Place the Excel file in the same folder as app.py.")
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 8px 0 20px 0;">
    <p class="sidebar-title">Rubber Analytics</p>
    <p class="sidebar-sub">Bunawan, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Data Explorer", "Trends & Patterns", "Model Comparison", "Forecast"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.6;">
Periods in dataset: {df['period'].nunique()}<br>
Records cleaned: {len(df):,}<br>
Data source: Department of Agriculture
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
C_PRIMARY   = "#0d9488"
C_ACCENT    = "#0f766e"
C_SECONDARY = "#f59e0b"
C_DARK      = "#0f172a"
C_GRID      = "#e2e8f0"

def style_plot(fig, height=420):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", color=C_DARK, size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=C_GRID, linecolor=C_GRID),
        yaxis=dict(gridcolor=C_GRID, linecolor=C_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# ------------------------------------------------------------------
# PAGE: OVERVIEW
# ------------------------------------------------------------------
if page == "Overview":
    st.markdown("## Rubber Production Analytics")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px;'>A data mining based analysis and forecasting of rubber "
        "production in Bunawan, Agusan del Sur.</p>",
        unsafe_allow_html=True,
    )

    periods   = sorted(df["period"].dropna().unique())
    tot_farm  = len(df)
    tot_trees = int(df["no_of_trees"].sum(skipna=True))
    tot_prod  = float(df["production_kg"].sum(skipna=True))
    barangays = df["farm_barangay"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Total Farmers",     f"{tot_farm:,}",     "Records in cleaned dataset"),
        (c2, "Total Rubber Trees", f"{tot_trees:,}",   "Across all farms"),
        (c3, "Total Production",   f"{tot_prod:,.0f} kg", "Sum of market volume"),
        (c4, "Barangays Covered",  f"{barangays}",     "Active rubber areas"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Production by Period</div>", unsafe_allow_html=True)

    period_agg = (
        df.groupby("period", as_index=False)
          .agg(total_production=("production_kg", "sum"),
               total_trees=("no_of_trees", "sum"),
               farmers=("farmer_name", "nunique"))
    )

    c_left, c_right = st.columns([1.4, 1])
    with c_left:
        fig = px.bar(
            period_agg, x="period", y="total_production",
            color_discrete_sequence=[C_PRIMARY],
            text_auto=".2s",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Total Production per Period (kg)")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with c_right:
        fig = px.pie(
            df[df["variety"] != "Unknown"],
            names="variety",
            color_discrete_sequence=[C_PRIMARY, C_SECONDARY, "#6366f1", "#ec4899"],
            hole=0.55,
        )
        fig.update_layout(title="Rubber Variety Distribution")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown("<div class='section-header'>Production by Barangay</div>", unsafe_allow_html=True)
    brgy = (
        df.groupby("farm_barangay", as_index=False)
          .agg(total_production=("production_kg", "sum"),
               farmers=("farmer_name", "nunique"))
          .sort_values("total_production", ascending=False)
    )
    fig = px.bar(
        brgy, x="farm_barangay", y="total_production",
        color="total_production",
        color_continuous_scale=["#99f6e4", C_PRIMARY, C_ACCENT],
        text_auto=".2s",
    )
    fig.update_layout(
        title="Production by Barangay",
        xaxis_title="Barangay",
        yaxis_title="Production (kg)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)

# ------------------------------------------------------------------
# PAGE: DATA EXPLORER
# ------------------------------------------------------------------
elif page == "Data Explorer":
    st.markdown("## Cleaned Data Explorer")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px;'>Raw Excel rows are parsed, "
        "whitespace is collapsed, blank rows and footer rows are removed, and "
        "'nan'/'none' strings are converted to nulls.</p>",
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        period_filter = st.multiselect("Period", sorted(df["period"].unique()),
                                       default=sorted(df["period"].unique()))
    with f2:
        brgy_filter = st.multiselect("Barangay", sorted(df["farm_barangay"].dropna().unique()),
                                     default=sorted(df["farm_barangay"].dropna().unique()))
    with f3:
        var_filter = st.multiselect("Variety", sorted(df["variety"].dropna().unique()),
                                    default=sorted(df["variety"].dropna().unique()))

    view = df[df["period"].isin(period_filter)
              & df["farm_barangay"].isin(brgy_filter)
              & df["variety"].isin(var_filter)]

    display_cols = ["period", "farmer_name", "farm_barangay", "variety",
                    "no_of_trees", "cropping_system", "production_kg"]
    st.dataframe(
        view[display_cols].rename(columns={
            "period": "Period", "farmer_name": "Farmer", "farm_barangay": "Barangay",
            "variety": "Variety", "no_of_trees": "No. of Trees",
            "cropping_system": "Cropping System", "production_kg": "Production (kg)",
        }),
        use_container_width=True, height=520,
    )

    st.download_button(
        "Download Cleaned Dataset (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_rubber_production.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------
# PAGE: TRENDS & PATTERNS
# ------------------------------------------------------------------
elif page == "Trends & Patterns":
    st.markdown("## Trends & Seasonal Patterns")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px;'>Visual exploration of production "
        "movement across periods, barangays, cropping systems, and tree counts.</p>",
        unsafe_allow_html=True,
    )

    period_agg = (
        df.groupby("period", as_index=False)
          .agg(production=("production_kg", "sum"),
               trees=("no_of_trees", "sum"),
               farmers=("farmer_name", "nunique"))
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=period_agg["period"], y=period_agg["production"],
        mode="lines+markers",
        line=dict(color=C_PRIMARY, width=3),
        marker=dict(size=12, color=C_ACCENT),
        fill="tozeroy", fillcolor="rgba(13,148,136,0.12)",
        name="Production (kg)",
    ))
    fig.update_layout(title="Production Trend Across Periods",
                      xaxis_title="Period", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=380), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cs = df.groupby("cropping_system", as_index=False)["production_kg"].sum()
        fig = px.bar(cs, x="cropping_system", y="production_kg",
                     color="cropping_system",
                     color_discrete_map={"Monocrop": C_PRIMARY,
                                         "Intercrop": C_SECONDARY,
                                         "Unknown": "#94a3b8"})
        fig.update_layout(title="Production by Cropping System",
                          xaxis_title="", yaxis_title="Production (kg)", showlegend=False)
        st.plotly_chart(style_plot(fig, height=360), use_container_width=True)

    with c2:
        scatter_df = df.dropna(subset=["no_of_trees", "production_kg"])
        fig = px.scatter(
            scatter_df, x="no_of_trees", y="production_kg",
            color="period",
            color_discrete_sequence=[C_PRIMARY, C_SECONDARY],
            opacity=0.7,
        )
        fig.update_layout(title="No. of Trees vs Production",
                          xaxis_title="No. of Trees", yaxis_title="Production (kg)")
        st.plotly_chart(style_plot(fig, height=360), use_container_width=True)

    st.markdown("<div class='section-header'>Top 10 Barangays by Production</div>",
                unsafe_allow_html=True)
    top = (df.groupby("farm_barangay", as_index=False)["production_kg"].sum()
             .sort_values("production_kg", ascending=False).head(10))
    fig = px.bar(top, x="production_kg", y="farm_barangay", orientation="h",
                 color="production_kg",
                 color_continuous_scale=["#99f6e4", C_PRIMARY, C_ACCENT])
    fig.update_layout(xaxis_title="Production (kg)", yaxis_title="",
                      coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_plot(fig, height=440), use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>On seasonality:</b> the dataset only provides two distinct periods
    (July 2024 and January 2025), so a formal decomposition of seasonal
    variation is limited. The dashboard reports the direction of change
    between the two periods as the observable trend, and this limitation
    is acknowledged in the study's scope.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE: MODEL COMPARISON
# ------------------------------------------------------------------
elif page == "Model Comparison":
    st.markdown("## Model Comparison: ARIMA vs Random Forest")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px;'>Both models are evaluated with "
        "k-fold cross-validation and compared using RMSE, MAE, and MAPE.</p>",
        unsafe_allow_html=True,
    )

    # ---- Build feature matrix for Random Forest ----
    feat = df.dropna(subset=["production_kg", "no_of_trees"]).copy()
    feat = feat[feat["production_kg"] > 0]

    if len(feat) < 20:
        st.warning("Not enough cleaned rows to train the models reliably.")
        st.stop()

    le_var = LabelEncoder()
    le_brg = LabelEncoder()
    le_cs  = LabelEncoder()

    feat["variety_enc"]  = le_var.fit_transform(feat["variety"].fillna("Unknown"))
    feat["brgy_enc"]     = le_brg.fit_transform(feat["farm_barangay"].fillna("Unknown"))
    feat["cs_enc"]       = le_cs.fit_transform(feat["cropping_system"])

    X = feat[["no_of_trees", "variety_enc", "brgy_enc", "cs_enc", "cropping_density"]].fillna(0)
    y = feat["production_kg"].values

    # ---- Random Forest with 5-fold CV ----
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_pred = cross_val_predict(rf, X, y, cv=kf)

    rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))
    rf_mae  = mean_absolute_error(y, rf_pred)
    rf_mape = mean_absolute_percentage_error(y, rf_pred) * 100

    # ---- ARIMA on aggregated period totals ----
    period_series = (
        df.groupby("period", as_index=False)["production_kg"].sum()
          .sort_values("period")
    )
    # Fit ARIMA on the aggregated series (2 points). Use simple ARIMA(0,1,0).
    try:
        arima_model = ARIMA(period_series["production_kg"].values, order=(0, 1, 0))
        arima_fit = arima_model.fit()
        arima_fitted = arima_fit.fittedvalues
        # Align fitted values to actual (first diff drops one obs)
        actual_aligned = period_series["production_kg"].values[-len(arima_fitted):]
        arima_rmse = np.sqrt(mean_squared_error(actual_aligned, arima_fitted))
        arima_mae  = mean_absolute_error(actual_aligned, arima_fitted)
        arima_mape = mean_absolute_percentage_error(actual_aligned, arima_fitted) * 100
    except Exception:
        arima_rmse = arima_mae = arima_mape = np.nan

    # ---- Metrics cards ----
    c1, c2, c3 = st.columns(3)
    for col, label, rf_val, ar_val in [
        (c1, "RMSE", rf_rmse, arima_rmse),
        (c2, "MAE",  rf_mae,  arima_mae),
        (c3, "MAPE (%)", rf_mape, arima_mape),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{rf_val:,.2f}</div>
                <div class="metric-sub">Random Forest</div>
                <div style="margin-top:8px; color:#64748b; font-size:0.8rem;">
                  ARIMA: {ar_val:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Comparison table ----
    st.markdown("<div class='section-header'>Metric Summary</div>", unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "ARIMA"],
        "RMSE":  [rf_rmse, arima_rmse],
        "MAE":   [rf_mae,  arima_mae],
        "MAPE (%)": [rf_mape, arima_mape],
    })
    st.dataframe(metrics_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.2f}"}),
                 use_container_width=True)

    # ---- Bar comparison ----
    long_metrics = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(long_metrics, x="Metric", y="Value", color="Model", barmode="group",
                 color_discrete_map={"Random Forest": C_PRIMARY, "ARIMA": C_SECONDARY},
                 text_auto=".2f")
    fig.update_layout(title="Model Performance Comparison")
    st.plotly_chart(style_plot(fig, height=400), use_container_width=True)

    # ---- Feature importance ----
    rf.fit(X, y)
    imp = pd.DataFrame({
        "Feature": ["No. of Trees", "Variety", "Barangay", "Cropping System", "Cropping Density"],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#99f6e4", C_PRIMARY, C_ACCENT])
    fig.update_layout(title="Random Forest Feature Importance", coloraxis_showscale=False,
                      yaxis_title="")
    st.plotly_chart(style_plot(fig, height=380), use_container_width=True)

    best = "Random Forest" if rf_mape < arima_mape else "ARIMA"
    st.markdown(f"""
    <div class="info-box">
    <b>Best performing model:</b> {best} (lower error across the reported metrics).
    Random Forest captures multi-variable non-linear relationships in the
    farmer-level data, while ARIMA captures the direction of change across
    periods in the aggregated time series.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE: FORECAST
# ------------------------------------------------------------------
elif page == "Forecast":
    st.markdown("## Forecasted Rubber Yield")
    st.markdown(
        "<p style='color:#64748b; margin-top:-8px;'>Forward projection of rubber "
        "production for the next period, based on the fitted models.</p>",
        unsafe_allow_html=True,
    )

    period_series = (
        df.groupby("period", as_index=False)["production_kg"].sum()
          .sort_values("period")
    )

    # ARIMA forecast of next period
    try:
        arima_model = ARIMA(period_series["production_kg"].values, order=(0, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = float(arima_fit.forecast(steps=1)[0])
    except Exception:
        arima_forecast = float(period_series["production_kg"].iloc[-1])

    # RF-based "next period" estimate: aggregate predictions
    feat = df.dropna(subset=["production_kg", "no_of_trees"]).copy()
    feat = feat[feat["production_kg"] > 0]

    le_var = LabelEncoder(); le_brg = LabelEncoder(); le_cs = LabelEncoder()
    feat["variety_enc"] = le_var.fit_transform(feat["variety"].fillna("Unknown"))
    feat["brgy_enc"]    = le_brg.fit_transform(feat["farm_barangay"].fillna("Unknown"))
    feat["cs_enc"]      = le_cs.fit_transform(feat["cropping_system"])

    X = feat[["no_of_trees", "variety_enc", "brgy_enc", "cs_enc", "cropping_density"]].fillna(0)
    y = feat["production_kg"].values

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_forecast = float(rf.predict(X).sum())

    # Build forecast chart
    last_period = period_series["period"].iloc[-1]
    next_period = "2026 (Projected)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=period_series["period"], y=period_series["production_kg"],
        mode="lines+markers", name="Actual",
        line=dict(color=C_PRIMARY, width=3), marker=dict(size=12),
    ))
    fig.add_trace(go.Scatter(
        x=[last_period, next_period],
        y=[period_series["production_kg"].iloc[-1], arima_forecast],
        mode="lines+markers", name="ARIMA Forecast",
        line=dict(color=C_SECONDARY, width=3, dash="dash"),
        marker=dict(size=12, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=[last_period, next_period],
        y=[period_series["production_kg"].iloc[-1], rf_forecast],
        mode="lines+markers", name="Random Forest Forecast",
        line=dict(color="#6366f1", width=3, dash="dot"),
        marker=dict(size=12, symbol="square"),
    ))
    fig.update_layout(title="Historical Production and Projected Yield",
                      xaxis_title="Period", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=460), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ARIMA Projected Yield</div>
        <div class="metric-value">{arima_forecast:,.0f} kg</div>
        <div class="metric-sub">Next period</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Random Forest Projected Yield</div>
        <div class="metric-value">{rf_forecast:,.0f} kg</div>
        <div class="metric-sub">Aggregated over current farms</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box" style="margin-top:20px;">
    <b>Note on interpretation:</b> the ARIMA projection is derived from a two-point
    aggregated series, so it behaves as a random-walk forecast anchored on the
    most recent period. The Random Forest estimate is derived from the
    feature-based model applied to the current farm profile. Both figures
    should be read as directional signals, not as precise point forecasts.
    </div>
    """, unsafe_allow_html=True)