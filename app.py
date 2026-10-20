import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Rubber Production Results", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f8fafc; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #64748b !important; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stSidebar"] hr { border-color: #1e293b; }
h1, h2, h3, h4 { color: #0f172a; font-weight: 700; }
.metric-card { background: #ffffff; border-radius: 10px; padding: 18px 20px; border: 1px solid #e2e8f0; border-left: 3px solid #0d9488; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
.metric-label { color: #64748b; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #0f172a; font-size: 1.55rem; font-weight: 700; margin-top: 4px; }
.metric-sub { color: #0d9488; font-size: 0.78rem; font-weight: 500; margin-top: 2px; }
.note { background: #f1f5f9; border-left: 3px solid #64748b; padding: 12px 16px; border-radius: 6px; color: #334155; font-size: 0.86rem; line-height: 1.55; margin-top: 12px; }
.sidebar-title { color: #5eead4; margin: 0; font-size: 1rem; font-weight: 700; }
.sidebar-sub { color: #94a3b8; font-size: 0.72rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

CONFIG = {
    "data_file": "Dataset  of Rubber_ 2024 - 2025.xlsx",
    "sheet_scan": None,
    "col_last_name": 2,
    "col_given_name": 3,
    "col_middle_name": 4,
    "col_home_address": 6,
    "col_farm_barangay": 7,
    "col_variety": 10,
    "col_no_of_trees": 13,
    "col_annual_volume": 15,
    "col_intercrop": 17,
    "col_monocrop": 18,
    "col_cropping_density": 20,
    "col_buyer_name": 21,
    "col_buyer_location": 22,
    "col_market_volume": 23,
    "skip_keywords": [
        "Province", "Municipality", "Commodity", "No.", "Prepared", "Noted",
        "MELYDEN", "ENDRENIE", "ENGR", "Last Name", "Sexual", "Asexual",
        "Intercrop", "Monocrop", "Complete Name",
    ],
    "random_state": 42,
    "n_estimators": 300,
    "cv_folds": 5,
}

def _clean_text(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch.isprintable())
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s

def _clean_number(val):
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

@st.cache_data(show_spinner=False)
def load_and_clean_data(path, cfg):
    xls = pd.ExcelFile(path)
    rows = []
    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        period = None
        for _, row in raw.iterrows():
            first = row.iloc[0]
            if pd.isna(first):
                continue
            fs = str(first).strip()
            if fs.startswith("Period"):
                p = fs.split(":")[-1].strip()
                period = " ".join(p.split())
                continue
            if any(k.lower() in fs.lower() for k in cfg["skip_keywords"]):
                continue
            if fs.startswith("="):
                continue
            try:
                int(float(fs))
            except ValueError:
                continue
            if not period:
                continue
            rows.append({
                "sheet": sheet,
                "period": period,
                "row_no": int(float(fs)),
                "last_name": _clean_text(row.iloc[cfg["col_last_name"]]),
                "given_name": _clean_text(row.iloc[cfg["col_given_name"]]),
                "middle_name": _clean_text(row.iloc[cfg["col_middle_name"]]),
                "home_address": _clean_text(row.iloc[cfg["col_home_address"]]),
                "farm_barangay": _clean_text(row.iloc[cfg["col_farm_barangay"]]),
                "variety": _clean_text(row.iloc[cfg["col_variety"]]),
                "no_of_trees": _clean_number(row.iloc[cfg["col_no_of_trees"]]),
                "annual_volume": _clean_number(row.iloc[cfg["col_annual_volume"]]),
                "intercrop": _clean_text(row.iloc[cfg["col_intercrop"]]),
                "monocrop": _clean_text(row.iloc[cfg["col_monocrop"]]),
                "cropping_density": _clean_number(row.iloc[cfg["col_cropping_density"]]),
                "buyer_name": _clean_text(row.iloc[cfg["col_buyer_name"]]),
                "buyer_location": _clean_text(row.iloc[cfg["col_buyer_location"]]),
                "market_volume": _clean_number(row.iloc[cfg["col_market_volume"]]),
            })
    df = pd.DataFrame(rows)
    df["farmer_name"] = (df["last_name"].fillna("") + ", " + df["given_name"].fillna("") + " " + df["middle_name"].fillna("")).str.replace(r"\s+", " ", regex=True).str.strip(", ")
    df["cropping_system"] = np.where(df["monocrop"].notna(), "Monocrop", np.where(df["intercrop"].notna(), "Intercrop", "Unknown"))
    df["production_kg"] = df["market_volume"].fillna(df["annual_volume"])
    df = df.dropna(subset=["no_of_trees"], how="all").reset_index(drop=True)
    df["variety"] = df["variety"].fillna("Unknown")
    return df

try:
    df = load_and_clean_data(CONFIG["data_file"], CONFIG)
except FileNotFoundError:
    st.error("Dataset file not found. Place the Excel file in the same directory as app.py.")
    st.stop()

st.sidebar.markdown("""
<div style="padding: 6px 0 18px 0;">
    <p class="sidebar-title">Rubber Production</p>
    <p class="sidebar-sub">Bunawan, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

view = st.sidebar.radio(
    "Views",
    ["Overview", "Trends", "Variable Relationships", "Model Accuracy", "Forecast"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
periods_all = sorted(df["period"].dropna().unique())
barangays_all = sorted(df["farm_barangay"].dropna().unique())
varieties_all = sorted(df["variety"].dropna().unique())

sel_periods = st.sidebar.multiselect("Period", periods_all, default=periods_all)
sel_barangays = st.sidebar.multiselect("Barangay", barangays_all, default=barangays_all)
sel_varieties = st.sidebar.multiselect("Variety", varieties_all, default=varieties_all)

d = df[df["period"].isin(sel_periods) & df["farm_barangay"].isin(sel_barangays) & df["variety"].isin(sel_varieties)].copy()

if len(d) == 0:
    st.warning("No records match the current filters.")
    st.stop()

C_PRIMARY = "#0d9488"
C_ACCENT = "#0f766e"
C_SECONDARY = "#f59e0b"
C_TERTIARY = "#6366f1"
C_DARK = "#0f172a"
C_GRID = "#e2e8f0"

def style_plot(fig, height=400):
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

def build_features(frame):
    f = frame.dropna(subset=["production_kg", "no_of_trees"]).copy()
    f = f[f["production_kg"] > 0]
    le_v = LabelEncoder()
    le_b = LabelEncoder()
    le_c = LabelEncoder()
    f["variety_enc"] = le_v.fit_transform(f["variety"].fillna("Unknown"))
    f["brgy_enc"] = le_b.fit_transform(f["farm_barangay"].fillna("Unknown"))
    f["cs_enc"] = le_c.fit_transform(f["cropping_system"])
    X = f[["no_of_trees", "variety_enc", "brgy_enc", "cs_enc", "cropping_density"]].fillna(0)
    y = f["production_kg"].values
    return f, X, y

if view == "Overview":
    total_farmers = int(d["farmer_name"].nunique())
    total_trees = int(d["no_of_trees"].sum(skipna=True))
    total_prod = float(d["production_kg"].sum(skipna=True))
    avg_yield = float(d["production_kg"].mean(skipna=True))
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Farmers", f"{total_farmers:,}", "unique records"),
        (c2, "Total trees", f"{total_trees:,}", "across selected records"),
        (c3, "Total production", f"{total_prod:,.0f} kg", "sum of market volume"),
        (c4, "Average yield", f"{avg_yield:,.1f} kg", "per farmer record"),
    ]
    for col, label, val, sub in cards:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        p_agg = d.groupby("period", as_index=False).agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"))
        fig = px.bar(p_agg, x="period", y="production", text_auto=".2s", color_discrete_sequence=[C_PRIMARY])
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Total production per period", xaxis_title="", yaxis_title="Production (kg)")
        st.plotly_chart(style_plot(fig), use_container_width=True)
    with col2:
        var_counts = d[d["variety"] != "Unknown"]["variety"].value_counts().reset_index()
        var_counts.columns = ["variety", "count"]
        if len(var_counts) > 0:
            fig = px.pie(var_counts, names="variety", values="count", hole=0.55, color_discrete_sequence=[C_PRIMARY, C_SECONDARY, C_TERTIARY])
            fig.update_layout(title="Variety distribution")
            st.plotly_chart(style_plot(fig), use_container_width=True)

    b_agg = d.groupby("farm_barangay", as_index=False).agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique")).sort_values("production", ascending=False)
    fig = px.bar(b_agg, x="farm_barangay", y="production", color="production", color_continuous_scale=["#99f6e4", C_PRIMARY, C_ACCENT], text_auto=".2s")
    fig.update_layout(title="Production per barangay", xaxis_title="", yaxis_title="Production (kg)", coloraxis_showscale=False)
    st.plotly_chart(style_plot(fig), use_container_width=True)

    top_brgy = b_agg.iloc[0]
    st.markdown(f"<div class='note'>Highest producing barangay is {top_brgy['farm_barangay']} with {top_brgy['production']:,.0f} kg from {int(top_brgy['farmers'])} farmer records.</div>", unsafe_allow_html=True)

elif view == "Trends":
    p_agg = d.groupby("period", as_index=False).agg(production=("production_kg", "sum"), trees=("no_of_trees", "sum"), farmers=("farmer_name", "nunique")).sort_values("period")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_agg["period"], y=p_agg["production"], mode="lines+markers", line=dict(color=C_PRIMARY, width=3), marker=dict(size=12, color=C_ACCENT), fill="tozeroy", fillcolor="rgba(13,148,136,0.12)", name="Production (kg)"))
    fig.update_layout(title="Production direction across periods", xaxis_title="", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=360), use_container_width=True)

    if len(p_agg) >= 2:
        first = p_agg["production"].iloc[0]
        last = p_agg["production"].iloc[-1]
        pct = ((last - first) / first * 100) if first > 0 else np.nan
        direction = "increase" if pct > 0 else ("decrease" if pct < 0 else "no change")
        st.markdown(f"<div class='note'>Aggregated production shows a {direction} of {abs(pct):.1f}% between {p_agg['period'].iloc[0]} and {p_agg['period'].iloc[-1]}. With only two observation periods available, the seasonal component cannot be estimated; the chart reports the observable direction of change.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cs = d.groupby("cropping_system", as_index=False)["production_kg"].sum()
        fig = px.bar(cs, x="cropping_system", y="production_kg", color="cropping_system", color_discrete_map={"Monocrop": C_PRIMARY, "Intercrop": C_SECONDARY, "Unknown": "#94a3b8"})
        fig.update_layout(title="Production by cropping system", xaxis_title="", yaxis_title="Production (kg)", showlegend=False)
        st.plotly_chart(style_plot(fig, height=340), use_container_width=True)
    with col2:
        sc = d.dropna(subset=["no_of_trees", "production_kg"])
        if len(sc) > 0:
            fig = px.scatter(sc, x="no_of_trees", y="production_kg", color="period", color_discrete_sequence=[C_PRIMARY, C_SECONDARY], opacity=0.7)
            fig.update_layout(title="Trees vs production", xaxis_title="Number of trees", yaxis_title="Production (kg)")
            st.plotly_chart(style_plot(fig, height=340), use_container_width=True)

    top = d.groupby("farm_barangay", as_index=False)["production_kg"].sum().sort_values("production_kg", ascending=False).head(10)
    fig = px.bar(top, x="production_kg", y="farm_barangay", orientation="h", color="production_kg", color_continuous_scale=["#99f6e4", C_PRIMARY, C_ACCENT])
    fig.update_layout(title="Top barangays by production", xaxis_title="Production (kg)", yaxis_title="", coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_plot(fig, height=400), use_container_width=True)

elif view == "Variable Relationships":
    st.markdown("### Production by variety")
    var_agg = d.groupby("variety", as_index=False).agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"), avg_yield=("production_kg", "mean")).sort_values("production", ascending=False)
    st.dataframe(var_agg.rename(columns={"variety": "Variety", "production": "Total production (kg)", "farmers": "Farmers", "avg_yield": "Average yield (kg)"}).style.format({"Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}"}), use_container_width=True, hide_index=True)

    st.markdown("### Production by cropping system")
    cs_agg = d.groupby("cropping_system", as_index=False).agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"), avg_yield=("production_kg", "mean"))
    st.dataframe(cs_agg.rename(columns={"cropping_system": "Cropping system", "production": "Total production (kg)", "farmers": "Farmers", "avg_yield": "Average yield (kg)"}).style.format({"Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}"}), use_container_width=True, hide_index=True)

    st.markdown("### Production by barangay")
    brgy_agg = d.groupby("farm_barangay", as_index=False).agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"), avg_yield=("production_kg", "mean")).sort_values("production", ascending=False)
    st.dataframe(brgy_agg.rename(columns={"farm_barangay": "Barangay", "production": "Total production (kg)", "farmers": "Farmers", "avg_yield": "Average yield (kg)"}).style.format({"Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}"}), use_container_width=True, hide_index=True)

    if d["no_of_trees"].notna().sum() > 2 and d["production_kg"].notna().sum() > 2:
        corr_frame = d.dropna(subset=["no_of_trees", "production_kg", "cropping_density"])
        if len(corr_frame) > 2:
            corr_trees = corr_frame["no_of_trees"].corr(corr_frame["production_kg"])
            corr_dens = corr_frame["cropping_density"].corr(corr_frame["production_kg"]) if corr_frame["cropping_density"].notna().sum() > 2 else np.nan
            st.markdown(f"<div class='note'>Pearson correlation between number of trees and production is {corr_trees:.3f}. Correlation between cropping density and production is {corr_dens:.3f}.</div>", unsafe_allow_html=True)

    crosstab = pd.crosstab(d["farm_barangay"], d["cropping_system"])
    if crosstab.size > 0:
        st.markdown("### Barangay by cropping system")
        st.dataframe(crosstab, use_container_width=True)

elif view == "Model Accuracy":
    feat, X, y = build_features(d)
    if len(feat) < 20:
        st.warning("Not enough records after filtering to fit the models.")
        st.stop()

    kf = KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"])

    rf = RandomForestRegressor(n_estimators=CONFIG["n_estimators"], random_state=CONFIG["random_state"], n_jobs=-1)
    rf_pred = cross_val_predict(rf, X, y, cv=kf)
    rf_rmse = float(np.sqrt(mean_squared_error(y, rf_pred)))
    rf_mae = float(mean_absolute_error(y, rf_pred))
    rf_mape = float(mean_absolute_percentage_error(y, rf_pred) * 100)

    lr = LinearRegression()
    lr_pred = cross_val_predict(lr, X, y, cv=kf)
    lr_rmse = float(np.sqrt(mean_squared_error(y, lr_pred)))
    lr_mae = float(mean_absolute_error(y, lr_pred))
    lr_mape = float(mean_absolute_percentage_error(y, lr_pred) * 100)

    period_series = d.groupby("period", as_index=False)["production_kg"].sum().sort_values("period")
    arima_rmse = arima_mae = arima_mape = np.nan
    arima_forecast_value = np.nan
    if len(period_series) >= 2:
        try:
            am = ARIMA(period_series["production_kg"].values, order=(0, 1, 0))
            af = am.fit()
            fitted = af.fittedvalues
            actual_aligned = period_series["production_kg"].values[-len(fitted):]
            arima_rmse = float(np.sqrt(mean_squared_error(actual_aligned, fitted)))
            arima_mae = float(mean_absolute_error(actual_aligned, fitted))
            arima_mape = float(mean_absolute_percentage_error(actual_aligned, fitted) * 100)
            arima_forecast_value = float(af.forecast(steps=1)[0])
        except Exception:
            pass

    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-card'><div class='metric-label'>RMSE</div><div class='metric-value'>{rf_rmse:,.2f}</div><div class='metric-sub'>Random Forest</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-label'>MAE</div><div class='metric-value'>{rf_mae:,.2f}</div><div class='metric-sub'>Random Forest</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-label'>MAPE</div><div class='metric-value'>{rf_mape:,.2f}%</div><div class='metric-sub'>Random Forest</div></div>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression", "ARIMA"],
        "RMSE": [rf_rmse, lr_rmse, arima_rmse],
        "MAE": [rf_mae, lr_mae, arima_mae],
        "MAPE (%)": [rf_mape, lr_mape, arima_mape],
    })
    st.markdown("### Performance summary")
    st.dataframe(metrics_df.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}", "MAPE (%)": "{:,.2f}"}), use_container_width=True, hide_index=True)

    long_metrics = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(long_metrics, x="Metric", y="Value", color="Model", barmode="group", text_auto=".2f", color_discrete_map={"Random Forest": C_PRIMARY, "Linear Regression": C_TERTIARY, "ARIMA": C_SECONDARY})
    fig.update_layout(title="Metric comparison across models")
    st.plotly_chart(style_plot(fig, height=380), use_container_width=True)

    rf.fit(X, y)
    imp = pd.DataFrame({
        "Feature": ["No. of trees", "Variety", "Barangay", "Cropping system", "Cropping density"],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#99f6e4", C_PRIMARY, C_ACCENT])
    fig.update_layout(title="Random Forest feature importance", coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(style_plot(fig, height=340), use_container_width=True)

    best_label = "Random Forest"
    if not np.isnan(arima_mape) and arima_mape < rf_mape:
        best_label = "ARIMA"
    st.markdown(f"<div class='note'>Lowest MAPE is achieved by {best_label}. Random Forest is fit on farm-level features with 5-fold cross-validation; ARIMA is fit on the aggregated period totals and its residual error reflects the two-point series limitation.</div>", unsafe_allow_html=True)

elif view == "Forecast":
    period_series = d.groupby("period", as_index=False)["production_kg"].sum().sort_values("period")
    if len(period_series) < 2:
        st.warning("At least two periods are required to produce a forecast.")
        st.stop()

    try:
        am = ARIMA(period_series["production_kg"].values, order=(0, 1, 0))
        af = am.fit()
        arima_next = float(af.forecast(steps=1)[0])
    except Exception:
        arima_next = float(period_series["production_kg"].iloc[-1])

    feat, X, y = build_features(d)
    rf_next = np.nan
    if len(feat) >= 20:
        rf = RandomForestRegressor(n_estimators=CONFIG["n_estimators"], random_state=CONFIG["random_state"], n_jobs=-1)
        rf.fit(X, y)
        rf_next = float(rf.predict(X).sum())

    last_label = period_series["period"].iloc[-1]
    next_label = "Projected next period"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=period_series["period"], y=period_series["production_kg"], mode="lines+markers", name="Actual", line=dict(color=C_PRIMARY, width=3), marker=dict(size=12)))
    fig.add_trace(go.Scatter(x=[last_label, next_label], y=[period_series["production_kg"].iloc[-1], arima_next], mode="lines+markers", name="ARIMA projection", line=dict(color=C_SECONDARY, width=3, dash="dash"), marker=dict(size=12, symbol="diamond")))
    if not np.isnan(rf_next):
        fig.add_trace(go.Scatter(x=[last_label, next_label], y=[period_series["production_kg"].iloc[-1], rf_next], mode="lines+markers", name="Random Forest projection", line=dict(color=C_TERTIARY, width=3, dash="dot"), marker=dict(size=12, symbol="square")))
    fig.update_layout(title="Production trajectory and next-period projection", xaxis_title="", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=440), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>ARIMA projected production</div><div class='metric-value'>{arima_next:,.0f} kg</div><div class='metric-sub'>next period</div></div>", unsafe_allow_html=True)
    if not np.isnan(rf_next):
        c2.markdown(f"<div class='metric-card'><div class='metric-label'>Random Forest projected production</div><div class='metric-value'>{rf_next:,.0f} kg</div><div class='metric-sub'>aggregated over current farm profile</div></div>", unsafe_allow_html=True)

    last_actual = float(period_series["production_kg"].iloc[-1])
    delta_arima = arima_next - last_actual
    direction_word = "increase" if delta_arima > 0 else ("decrease" if delta_arima < 0 else "no change")
    st.markdown(f"<div class='note'>ARIMA projection indicates a {direction_word} of {abs(delta_arima):,.0f} kg relative to the latest observed period. The random walk formulation reflects the two-point series; the projection is a directional signal, not a precise point estimate.</div>", unsafe_allow_html=True)
