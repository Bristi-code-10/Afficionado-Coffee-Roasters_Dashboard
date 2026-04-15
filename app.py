import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Afficionado Coffee Roasters – Analytics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #1a0f00; }
    section[data-testid="stSidebar"] { background-color: #2c1a00; }
 
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #3d2200, #5c3300);
        border: 1px solid #a0522d;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }
    .kpi-label {
        color: #d2a679;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #fff8f0;
        font-size: 2rem;
        font-weight: 700;
    }
    .kpi-sub {
        color: #a07050;
        font-size: 0.78rem;
        margin-top: 4px;
    }
 
    /* Section headers */
    h1, h2, h3 { color: #e8c49a !important; }
    .section-divider {
        border-top: 1px solid #4a2800;
        margin: 30px 0 20px 0;
    }
 
    /* Streamlit widget labels */
    label { color: #d2a679 !important; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("transactions.csv")
 
    # Parse time
    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"], format="%H:%M:%S", errors="coerce"
    )
 
    # Derived temporal features
    df["hour"]        = df["transaction_time"].dt.hour
    df["minute"]      = df["transaction_time"].dt.minute
    df["revenue"]     = df["transaction_qty"] * df["unit_price"]
 
    # Day of week proxy using transaction_id modulo (since only year is given)
    # We use a synthetic date seeded from transaction_id for realistic day spread
    base_date = pd.Timestamp("2025-01-06")           # Monday
    df["date"] = base_date + pd.to_timedelta(
        (df["transaction_id"] - 1) // 300, unit="D"  # ~300 txns/day
    )
    df["day_of_week"] = df["date"].dt.day_name()
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]       = df["date"].dt.month_name()
    df["month_num"]   = df["date"].dt.month
 
    # Time bucket
    def time_bucket(h):
        if 6 <= h <= 11:   return "Morning (6–11)"
        elif 12 <= h <= 16: return "Afternoon (12–16)"
        elif 17 <= h <= 21: return "Evening (17–21)"
        else:               return "Late/Early (22–5)"
 
    df["time_bucket"] = df["hour"].apply(time_bucket)
    return df
 
df = load_data()
 
# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☕ Afficionado Coffee")
    st.markdown("---")
    st.markdown("### 🔍 Filters")
 
    locations = ["All Locations"] + sorted(df["store_location"].unique().tolist())
    selected_location = st.selectbox("Store Location", locations)
 
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_days = st.multiselect(
        "Days of Week",
        options=days_order,
        default=days_order,
    )
 
    hour_range = st.slider("Hour Range", 0, 23, (6, 21))
 
    metric_toggle = st.radio(
        "Primary Metric",
        ["Revenue ($)", "Transaction Count", "Quantity Sold"],
        index=0,
    )
 
    st.markdown("---")
    st.markdown("### 📊 Dashboard Sections")
    show_overview   = st.checkbox("Sales Overview",           True)
    show_weekly     = st.checkbox("Weekly Trend",             True)
    show_dow        = st.checkbox("Day-of-Week Performance",  True)
    show_hourly     = st.checkbox("Hourly Demand",            True)
    show_heatmap    = st.checkbox("Heatmap",                  True)
    show_location   = st.checkbox("Location Comparison",      True)
    show_category   = st.checkbox("Category Insights",        True)
 
# ─────────────────────────────────────────────
# FILTERED DATA
# ─────────────────────────────────────────────
dff = df.copy()
if selected_location != "All Locations":
    dff = dff[dff["store_location"] == selected_location]
if selected_days:
    dff = dff[dff["day_of_week"].isin(selected_days)]
dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]
 
metric_col = {
    "Revenue ($)":        "revenue",
    "Transaction Count":  "transaction_id",
    "Quantity Sold":      "transaction_qty",
}[metric_toggle]
 
metric_label = metric_toggle
 
# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>☕ Afficionado Coffee Roasters</h1>"
    "<p style='text-align:center; color:#a07050; font-size:1.1rem;'>"
    "Sales Trend & Time-Based Performance Dashboard · 2025</p>",
    unsafe_allow_html=True,
)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total_rev     = dff["revenue"].sum()
total_txn     = dff["transaction_id"].nunique()
total_qty     = dff["transaction_qty"].sum()
avg_txn_val   = total_rev / total_txn if total_txn else 0
peak_hour_val = dff.groupby("hour")["transaction_id"].count().idxmax() if not dff.empty else "N/A"
busiest_day   = dff.groupby("day_of_week")["transaction_id"].count().idxmax() if not dff.empty else "N/A"
 
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "Total Revenue",       f"${total_rev:,.0f}",    "All transactions"),
    (k2, "Transactions",        f"{total_txn:,}",         "Unique orders"),
    (k3, "Items Sold",          f"{total_qty:,}",         "Total quantity"),
    (k4, "Avg Order Value",     f"${avg_txn_val:.2f}",   "Revenue / txn"),
    (k5, "Peak Hour",           f"{peak_hour_val}:00",   "Most transactions"),
    (k6, "Busiest Day",         str(busiest_day)[:3],    "Highest traffic"),
]
for col, label, value, sub in kpis:
    with col:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
 
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────
COFFEE_PALETTE = [
    "#c68642", "#a0522d", "#8B4513", "#d2691e",
    "#f4a460", "#deb887", "#cd853f", "#6b3a2a",
]
 
def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d2a679", family="Arial"),
        legend=dict(bgcolor="rgba(40,20,0,0.7)", bordercolor="#4a2800", borderwidth=1),
        xaxis=dict(gridcolor="#2c1a00", linecolor="#4a2800"),
        yaxis=dict(gridcolor="#2c1a00", linecolor="#4a2800"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig
 
# ─────────────────────────────────────────────
# 1. SALES OVERVIEW – WEEKLY TREND
# ─────────────────────────────────────────────
if show_overview or show_weekly:
    st.markdown("## 📈 Sales Trend Overview")
 
    weekly = (
        dff.groupby(["week_number", "store_location"])
        .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "count"), qty=("transaction_qty", "sum"))
        .reset_index()
    )
    weekly_total = weekly.groupby("week_number").agg(
        revenue=("revenue", "sum"),
        transactions=("transactions", "sum"),
        qty=("qty", "sum"),
    ).reset_index()
 
    y_col = {"Revenue ($)": "revenue", "Transaction Count": "transactions", "Quantity Sold": "qty"}[metric_toggle]
 
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=weekly_total["week_number"],
        y=weekly_total[y_col],
        mode="lines+markers",
        name="All Locations",
        line=dict(color="#c68642", width=3),
        marker=dict(size=6, color="#f4a460"),
        fill="tozeroy",
        fillcolor="rgba(198,134,66,0.15)",
    ))
    for loc, color in zip(dff["store_location"].unique(), COFFEE_PALETTE[1:]):
        sub = weekly[weekly["store_location"] == loc]
        fig_trend.add_trace(go.Scatter(
            x=sub["week_number"], y=sub[y_col],
            mode="lines", name=loc,
            line=dict(color=color, width=1.5, dash="dot"),
        ))
    fig_trend.update_layout(
        title=f"Weekly {metric_label} Trend by Store Location",
        xaxis_title="Week Number",
        yaxis_title=metric_label,
    )
    apply_theme(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 2. DAY-OF-WEEK PERFORMANCE
# ─────────────────────────────────────────────
if show_dow:
    st.markdown("## 📅 Day-of-Week Performance")
    col1, col2 = st.columns(2)
 
    dow = (
        dff.groupby("day_of_week")
        .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "count"), qty=("transaction_qty", "sum"))
        .reindex(days_order)
        .reset_index()
    )
    y_col = {"Revenue ($)": "revenue", "Transaction Count": "transactions", "Quantity Sold": "qty"}[metric_toggle]
 
    with col1:
        fig_dow = px.bar(
            dow, x="day_of_week", y=y_col,
            title=f"Total {metric_label} by Day of Week",
            color=y_col,
            color_continuous_scale=["#3d1a00", "#c68642", "#f4a460"],
            labels={"day_of_week": "Day", y_col: metric_label},
        )
        fig_dow.update_layout(coloraxis_showscale=False)
        apply_theme(fig_dow)
        st.plotly_chart(fig_dow, use_container_width=True)
 
    with col2:
        # Weekend vs Weekday
        dow_avg = dff.copy()
        dow_avg["is_weekend"] = dow_avg["day_of_week"].isin(["Saturday", "Sunday"])
        wknd = (
            dow_avg.groupby("is_weekend")
            .agg(revenue=("revenue", "mean"), transactions=("transaction_id", "count"), qty=("transaction_qty", "mean"))
            .reset_index()
        )
        wknd["Type"] = wknd["is_weekend"].map({True: "Weekend", False: "Weekday"})
 
        fig_wknd = px.bar(
            wknd, x="Type", y=y_col,
            title=f"Weekday vs Weekend – Avg {metric_label}",
            color="Type",
            color_discrete_map={"Weekday": "#c68642", "Weekend": "#a0522d"},
            labels={y_col: metric_label},
        )
        apply_theme(fig_wknd)
        st.plotly_chart(fig_wknd, use_container_width=True)
 
    # Radar chart
    radar_vals = dow[y_col].fillna(0).tolist()
    radar_vals.append(radar_vals[0])
    radar_days = days_order + [days_order[0]]
    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals, theta=radar_days,
        fill="toself", fillcolor="rgba(198,134,66,0.25)",
        line=dict(color="#c68642", width=2),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color="#4a2800", gridcolor="#2c1a00"),
            angularaxis=dict(color="#d2a679"),
        ),
        title=f"Day-of-Week Radar – {metric_label}",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d2a679"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 3. HOURLY DEMAND
# ─────────────────────────────────────────────
if show_hourly:
    st.markdown("## ⏰ Hourly Demand Analysis")
 
    hourly = (
        dff.groupby("hour")
        .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "count"), qty=("transaction_qty", "sum"))
        .reset_index()
    )
    y_col = {"Revenue ($)": "revenue", "Transaction Count": "transactions", "Quantity Sold": "qty"}[metric_toggle]
 
    col1, col2 = st.columns(2)
 
    with col1:
        fig_hour = px.area(
            hourly, x="hour", y=y_col,
            title=f"Hourly {metric_label} Distribution",
            labels={"hour": "Hour of Day", y_col: metric_label},
            color_discrete_sequence=["#c68642"],
        )
        fig_hour.update_traces(fillcolor="rgba(198,134,66,0.3)")
        apply_theme(fig_hour)
        st.plotly_chart(fig_hour, use_container_width=True)
 
    with col2:
        # Time bucket donut
        bucket = (
            dff.groupby("time_bucket")
            .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "count"), qty=("transaction_qty", "sum"))
            .reset_index()
        )
        fig_bucket = px.pie(
            bucket, names="time_bucket", values=y_col,
            title=f"{metric_label} Share by Time Bucket",
            hole=0.5,
            color_discrete_sequence=COFFEE_PALETTE,
        )
        fig_bucket.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d2a679"))
        st.plotly_chart(fig_bucket, use_container_width=True)
 
    # Hourly by location – line
    hourly_loc = (
        dff.groupby(["hour", "store_location"])
        .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "count"), qty=("transaction_qty", "sum"))
        .reset_index()
    )
    fig_hloc = px.line(
        hourly_loc, x="hour", y=y_col, color="store_location",
        title=f"Hourly {metric_label} per Store Location",
        labels={"hour": "Hour of Day", y_col: metric_label, "store_location": "Location"},
        color_discrete_sequence=COFFEE_PALETTE,
        markers=True,
    )
    apply_theme(fig_hloc)
    st.plotly_chart(fig_hloc, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 4. HEATMAP – DAY × HOUR
# ─────────────────────────────────────────────
if show_heatmap:
    st.markdown("## 🔥 Demand Heatmap (Day × Hour)")
 
    y_col = {"Revenue ($)": "revenue", "Transaction Count": "transaction_id", "Quantity Sold": "transaction_qty"}[metric_toggle]
    agg_fn = "sum" if metric_toggle != "Transaction Count" else "count"
 
    pivot = (
        dff.groupby(["day_of_week", "hour"])[y_col]
        .agg(agg_fn)
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex([d for d in days_order if d in pivot.index])
 
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#1a0f00"],
            [0.3, "#5c2a00"],
            [0.6, "#a0522d"],
            [0.85, "#c68642"],
            [1.0, "#f4a460"],
        ],
        hoverongaps=False,
        colorbar=dict(title=metric_label, tickfont=dict(color="#d2a679")),
    ))
    fig_heat.update_layout(
        title=f"{metric_label} Heatmap – Day of Week × Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d2a679"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 5. LOCATION COMPARISON
# ─────────────────────────────────────────────
if show_location:
    st.markdown("## 🗺️ Location Comparison")
 
    loc_sum = (
        dff.groupby("store_location")
        .agg(
            Revenue=("revenue", "sum"),
            Transactions=("transaction_id", "count"),
            Qty=("transaction_qty", "sum"),
            AvgOrder=("revenue", "mean"),
        )
        .reset_index()
    )
 
    col1, col2 = st.columns(2)
    with col1:
        fig_loc_rev = px.bar(
            loc_sum, x="store_location", y="Revenue",
            title="Total Revenue by Location",
            color="store_location",
            color_discrete_sequence=COFFEE_PALETTE,
            labels={"store_location": "Location", "Revenue": "Revenue ($)"},
        )
        apply_theme(fig_loc_rev)
        st.plotly_chart(fig_loc_rev, use_container_width=True)
 
    with col2:
        fig_loc_txn = px.bar(
            loc_sum, x="store_location", y="Transactions",
            title="Transaction Count by Location",
            color="store_location",
            color_discrete_sequence=COFFEE_PALETTE[2:],
            labels={"store_location": "Location", "Transactions": "Transactions"},
        )
        apply_theme(fig_loc_txn)
        st.plotly_chart(fig_loc_txn, use_container_width=True)
 
    # Hourly heatmap per location
    st.markdown("### Hourly Heatmap per Location")
    locations_list = dff["store_location"].unique().tolist()
    fig_loc_heat = make_subplots(
        rows=1, cols=len(locations_list),
        subplot_titles=locations_list,
        shared_yaxes=True,
    )
    y_col_map = {"Revenue ($)": "revenue", "Transaction Count": "transaction_id", "Quantity Sold": "transaction_qty"}
    y_col = y_col_map[metric_toggle]
 
    for i, loc in enumerate(locations_list, start=1):
        sub = dff[dff["store_location"] == loc]
        pvt = (
            sub.groupby(["day_of_week", "hour"])[y_col]
            .sum()
            .unstack(fill_value=0)
            .reindex([d for d in days_order if d in sub["day_of_week"].unique()])
        )
        fig_loc_heat.add_trace(
            go.Heatmap(
                z=pvt.values,
                x=[f"{h:02d}:00" for h in pvt.columns],
                y=pvt.index.tolist(),
                colorscale=[
                    [0.0, "#1a0f00"], [0.5, "#a0522d"], [1.0, "#f4a460"]
                ],
                showscale=(i == len(locations_list)),
                name=loc,
            ),
            row=1, col=i,
        )
    fig_loc_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d2a679"),
        height=350,
        title=f"{metric_label} Heatmap by Location",
    )
    st.plotly_chart(fig_loc_heat, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 6. CATEGORY INSIGHTS
# ─────────────────────────────────────────────
if show_category:
    st.markdown("## ☕ Product Category Insights")
 
    cat = (
        dff.groupby("product_category")
        .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "count"), qty=("transaction_qty", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
 
    col1, col2 = st.columns(2)
    with col1:
        fig_cat = px.bar(
            cat, x="revenue", y="product_category", orientation="h",
            title="Revenue by Product Category",
            color="revenue",
            color_continuous_scale=["#3d1a00", "#c68642"],
            labels={"revenue": "Revenue ($)", "product_category": "Category"},
        )
        fig_cat.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        apply_theme(fig_cat)
        st.plotly_chart(fig_cat, use_container_width=True)
 
    with col2:
        top_types = (
            dff.groupby("product_type")["revenue"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        fig_type = px.bar(
            top_types, x="revenue", y="product_type", orientation="h",
            title="Top 10 Product Types by Revenue",
            color="revenue",
            color_continuous_scale=["#5c2a00", "#f4a460"],
            labels={"revenue": "Revenue ($)", "product_type": "Product Type"},
        )
        fig_type.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        apply_theme(fig_type)
        st.plotly_chart(fig_type, use_container_width=True)
 
    # Category × Hour heatmap
    cat_hour = (
        dff.groupby(["product_category", "hour"])["revenue"]
        .sum()
        .unstack(fill_value=0)
    )
    fig_cat_h = go.Figure(go.Heatmap(
        z=cat_hour.values,
        x=[f"{h:02d}:00" for h in cat_hour.columns],
        y=cat_hour.index.tolist(),
        colorscale=[
            [0.0, "#1a0f00"], [0.4, "#5c2a00"],
            [0.7, "#a0522d"], [1.0, "#f4a460"],
        ],
        colorbar=dict(title="Revenue ($)", tickfont=dict(color="#d2a679")),
    ))
    fig_cat_h.update_layout(
        title="Revenue Heatmap – Product Category × Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Product Category",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d2a679"),
    )
    st.plotly_chart(fig_cat_h, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#4a2800; font-size:0.8rem; margin-top:40px;'>"
    "Afficionado Coffee Roasters · Sales Analytics Dashboard · 2025 · Powered by Streamlit + Plotly"
    "</p>",
    unsafe_allow_html=True,
)