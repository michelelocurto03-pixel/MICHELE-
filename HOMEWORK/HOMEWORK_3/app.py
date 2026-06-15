import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Financial Transactions Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. DESIGN TOKENS — CVD-friendly palette (blue / orange, avoids red / green
#    so the charts remain readable for users with colour vision deficiency)
# ==============================================================================
BUY_COLOR  = "#2E75B6"   # blue  — safe for protanopes & deuteranopes
SELL_COLOR = "#E87722"   # orange
DIV_COLOR  = "#6C757D"   # neutral grey for dividend entries
ACCENT     = "#1F4E79"   # dark blue used for headers and KPI values

CAT_PALETTE = [
    "#2E75B6", "#E87722", "#5BA4CF", "#BF6900",
    "#7FAECC", "#F4A952", "#1F4E79",
]

COLOR_MAP = {"BUY": BUY_COLOR, "SELL": SELL_COLOR}

# Shared axis style applied to every figure to keep text consistently black
AXIS_STYLE = dict(
    title_font=dict(color="black"),
    tickfont=dict(color="black"),
    tickcolor="black",
)

# Base layout applied to every figure for a clean white background
LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black"),
)


def clean_layout(fig, height=300, margin=None, legend_bottom=False):
    """Apply consistent layout defaults to any Plotly figure."""
    if margin is None:
        margin = dict(l=0, r=0, t=10, b=0)
    updates = dict(
        height=height,
        margin=margin,
        **LAYOUT_BASE,
    )
    if legend_bottom:
        updates["legend"] = dict(orientation="h", y=-0.3)
    fig.update_layout(**updates)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ==============================================================================
# 3. GLOBAL CSS — metric cards and section title style
# ==============================================================================
st.markdown(
    """
    <style>
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #f0f6ff 0%, #dceeff 100%);
            border: 1px solid #2E75B6;
            border-radius: 10px;
            padding: 18px 22px;
        }
        [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #444; font-weight: 600; }
        [data-testid="stMetricValue"] { font-size: 1.8rem;  font-weight: 800; color: #1F4E79; }
        .section-title {
            font-size: 1.05rem; font-weight: 700; color: #1F4E79;
            border-left: 4px solid #2E75B6; padding-left: 10px;
            margin: 20px 0 10px 0;
        }
        div[data-testid="stTabs"] button { font-weight: 600; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str) -> None:
    """Render a styled section heading using a left-border accent."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ==============================================================================
# 4. DATA LOADING
# Decorated with @st.cache_data so the ETL pipeline runs only once per session.
# The function replicates the same transformation logic used in the notebook
# (Cell 1–3) to keep the dashboard fully self-contained.
# ==============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    base = os.path.dirname(os.path.abspath(__file__))

    df_trans = pd.read_csv(
        os.path.join(base, "account-statement-1-1-2024-12-31-2024.csv"), sep=";"
    )
    df_sym = pd.read_csv(os.path.join(base, "symbols.csv"), sep=";")
    df_cnt = pd.read_csv(os.path.join(base, "country.csv"), sep=",")

    # Normalise column names across all three sources
    for df in [df_trans, df_sym, df_cnt]:
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        df.drop(
            columns=[c for c in df.columns if "unnamed" in c],
            errors="ignore",
            inplace=True,
        )

    # Uppercase all string columns in the transaction log except the raw date field
    for col in df_trans.columns:
        if df_trans[col].dtype == "object" and col != "date":
            df_trans[col] = df_trans[col].astype(str).str.strip().str.upper()

    df_trans["unit"] = pd.to_numeric(df_trans["unit"], errors="coerce")
    df_trans["date_clean"] = pd.to_datetime(
        df_trans["date"].str[:10], format="%d/%m/%Y", errors="coerce"
    )

    # Fix the two country name mismatches between symbols.csv and country.csv
    df_sym["country"] = df_sym["country"].replace(
        {"Taiwan": "Taiwan, Province of China", "Turkey": "Türkiye"}
    )
    # Taiwan is also missing region/sub_region data in country.csv — patch it here
    tw_mask = df_cnt["name"].str.contains("Taiwan", case=False, na=False)
    df_cnt.loc[tw_mask, "region"] = "Asia"
    df_cnt.loc[tw_mask, "sub_region"] = "Eastern Asia"

    # Enrich the symbols master data with geographic attributes from country.csv
    df_sym_geo = pd.merge(
        df_sym,
        df_cnt[["name", "alpha_2", "alpha_3", "region", "sub_region"]].rename(
            columns={
                "name": "country",
                "alpha_2": "iso_code",
                "alpha_3": "iso3_code",
            }
        ),
        on="country",
        how="left",
    )

    # Join the enriched symbol data onto the transaction log
    df = pd.merge(
        df_trans,
        df_sym_geo[
            [
                "symbol", "company_name", "sector", "industry",
                "country", "iso_code", "iso3_code", "region", "sub_region",
            ]
        ],
        on="symbol",
        how="left",
    )

    # Retain only valid transaction types (BUY, SELL, DIVIDENT)
    df = df[df["transactiontype"].isin(["BUY", "SELL", "DIVIDENT"])].copy()

    # Replace NaN categorical values with readable placeholders so charts
    # don't silently exclude unmatched records
    df["sector"]    = df["sector"].fillna("Unmapped")
    df["industry"]  = df["industry"].fillna("Unmapped")
    df["region"]    = df["region"].fillna("Unassigned")
    df["country"]   = df["country"].fillna("Unknown")
    df["iso_code"]  = df["iso_code"].fillna("")
    df["iso3_code"] = df["iso3_code"].fillna("")

    df.rename(columns={"unit": "quantity"}, inplace=True)

    # Derive additional time attributes needed by the Time Analysis tab
    df["month"]     = df["date_clean"].dt.month
    df["quarter"]   = df["date_clean"].dt.quarter
    df["dayofweek"] = df["date_clean"].dt.day_name()
    df["week"]      = df["date_clean"].dt.isocalendar().week.astype(int)

    return df


df_all = load_data()

# ==============================================================================
# 5. SIDEBAR — Date range and additional filters
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=56)
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("📅 Date Range")
from datetime import date

min_d = df_all["date_clean"].min().date()
max_d = df_all["date_clean"].max().date()

DEFAULT_START = min_d
DEFAULT_END   = max_d

start_date = st.sidebar.date_input("From", value=DEFAULT_START, min_value=min_d, max_value=max_d)
end_date   = st.sidebar.date_input("To",   value=DEFAULT_END,   min_value=min_d, max_value=max_d)

# Apply the date filter; all downstream charts use the filtered `df` variable
date_mask = (
    (df_all["date_clean"].dt.date >= start_date)
    & (df_all["date_clean"].dt.date <= end_date)
)
df = df_all[date_mask].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Additional Filters")

# Optional sector filter — selecting nothing keeps all sectors visible
all_sectors = sorted(s for s in df["sector"].unique() if s != "Unmapped")
sel_sectors = st.sidebar.multiselect("Sectors", options=all_sectors, default=[])
if sel_sectors:
    df = df[df["sector"].isin(sel_sectors)]

# Optional region filter — selecting nothing keeps all regions visible
all_regions = sorted(r for r in df["region"].unique() if r != "Unassigned")
sel_regions = st.sidebar.multiselect("Regions", options=all_regions, default=[])
if sel_regions:
    df = df[df["region"].isin(sel_regions)]

# Sidebar summary KPIs — computed on BUY/SELL only (dividends excluded from trade counts)
st.sidebar.markdown("---")
df_trade = df[df["transactiontype"].isin(["BUY", "SELL"])]
total = len(df_trade)
buys  = (df_trade["transactiontype"] == "BUY").sum()
sells = (df_trade["transactiontype"] == "SELL").sum()

st.sidebar.metric("Total Trades", f"{total:,}")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("🔵 BUY",  f"{buys:,}")
col_b.metric("🟠 SELL", f"{sells:,}")

# ==============================================================================
# 6. PAGE HEADER
# ==============================================================================
st.title("📊 Financial Transactions Analytics — 2024")

# Build a human-readable caption summarising active filters
period_label = (
    f"**{start_date.strftime('%d %b %Y')}** → **{end_date.strftime('%d %b %Y')}**"
)
filter_parts = []
if sel_sectors:
    filter_parts.append(f"Sectors: {', '.join(sel_sectors)}")
if sel_regions:
    filter_parts.append(f"Regions: {', '.join(sel_regions)}")
filter_label = (" · " + " · ".join(filter_parts)) if filter_parts else ""
st.caption(f"Period: {period_label}{filter_label}")

# ==============================================================================
# 7. TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(
    ["⏱️ Time Analysis", "🌍 Geography Analysis", "📋 Symbol & Sector Deep Dive"]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — TIME ANALYSIS
# Required by the homework spec: date range filter + four charts.
# ──────────────────────────────────────────────────────────────────────────────
with tab1:

    # Top-level KPI row gives an at-a-glance summary of the selected period
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades",      f"{len(df_trade):,}")
    c2.metric("BUY",               f"{buys:,}")
    c3.metric("SELL",              f"{sells:,}")
    c4.metric("Total Units",       f"{df_trade['quantity'].sum():,.0f}")
    c5.metric("Avg Units / Trade", f"{df_trade['quantity'].mean():,.1f}")

    st.markdown("---")

    # ── Daily BUY vs SELL line chart with 7-day moving averages ──────────────
    # The raw daily series is drawn at low opacity; the smoothed MA lines
    # sit on top at full opacity so the trend is easy to read at a glance.
    section("📈 Daily Transactions — BUY vs SELL")

    daily = (
        df_trade.groupby(["date_clean", "transactiontype"])
        .size()
        .reset_index(name="n")
        .pivot(index="date_clean", columns="transactiontype", values="n")
        .fillna(0)
        .reset_index()
        .sort_values("date_clean")
    )
    for col in ["BUY", "SELL"]:
        if col not in daily.columns:
            daily[col] = 0
    daily["BUY_MA7"]  = daily["BUY"].rolling(7,  min_periods=1).mean()
    daily["SELL_MA7"] = daily["SELL"].rolling(7, min_periods=1).mean()

    fig_line = go.Figure()

    # Daily raw series — thin lines at 35% opacity to show volatility without noise
    fig_line.add_trace(go.Scatter(
        x=daily["date_clean"], y=daily["BUY"],
        name="BUY (daily)",
        mode="lines",
        line=dict(color=BUY_COLOR, width=1),
        opacity=0.35,
    ))
    fig_line.add_trace(go.Scatter(
        x=daily["date_clean"], y=daily["SELL"],
        name="SELL (daily)",
        mode="lines",
        line=dict(color=SELL_COLOR, width=1),
        opacity=0.35,
    ))

    # 7-day moving averages — thicker lines at full opacity highlight the trend
    fig_line.add_trace(go.Scatter(
        x=daily["date_clean"], y=daily["BUY_MA7"],
        name="BUY 7d MA",
        mode="lines",
        line=dict(color=BUY_COLOR, width=2.5),
        opacity=1.0,
    ))
    fig_line.add_trace(go.Scatter(
        x=daily["date_clean"], y=daily["SELL_MA7"],
        name="SELL 7d MA",
        mode="lines",
        line=dict(color=SELL_COLOR, width=2.5),
        opacity=1.0,
    ))

    fig_line.update_layout(
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="# Transactions",
        xaxis=dict(showgrid=False, **AXIS_STYLE),
        yaxis=dict(gridcolor="#f0f0f0", **AXIS_STYLE),
        legend=dict(
            orientation="h",
            y=-0.3,
            itemsizing="constant",   # legend swatches stay full-size regardless of trace opacity
            font=dict(color="black"),
        ),
        **LAYOUT_BASE,
    )
    clean_layout(fig_line, height=320, legend_bottom=True)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ── Quarterly and Day-of-Week breakdowns side by side ────────────────────
    col_q, col_w = st.columns(2)

    with col_q:
        section("📆 Transactions by Quarter")
        qtr = (
            df_trade.groupby(["quarter", "transactiontype"])
            .size()
            .reset_index(name="n")
            .assign(label=lambda x: "Q" + x["quarter"].astype(str))
        )
        fig_q = px.bar(
            qtr, x="label", y="n", color="transactiontype",
            barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={"n": "Transactions", "label": "Quarter", "transactiontype": "Type"},
        )
        fig_q.update_layout(legend_title=None, xaxis_title=None,
                            yaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_q, height=300)
        st.plotly_chart(fig_q, use_container_width=True)

    with col_w:
        section("📅 Transactions by Day of Week")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        dow = df_trade.groupby(["dayofweek", "transactiontype"]).size().reset_index(name="n")
        # Use an ordered Categorical so bars appear in calendar order, not alphabetical
        dow["dayofweek"] = pd.Categorical(
            dow["dayofweek"], categories=days_order, ordered=True
        )
        dow = dow.sort_values("dayofweek")
        fig_dow = px.bar(
            dow, x="dayofweek", y="n", color="transactiontype",
            barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={"n": "Transactions", "dayofweek": "Day", "transactiontype": "Type"},
        )
        fig_dow.update_layout(legend_title=None, xaxis_title=None,
                              yaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_dow, height=300)
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("---")

    # ── Top rankings — symbols, sectors, industries ───────────────────────────
    # Three horizontal bar charts displayed in a single row.
    section("🏆 Top Rankings")
    col_s, col_sec, col_ind = st.columns(3)

    with col_s:
        top_sym = (
            df_trade.groupby("symbol").size()
            .sort_values(ascending=True).tail(3)
            .reset_index(name="n")
        )
        fig_sym = px.bar(
            top_sym, x="n", y="symbol", orientation="h",
            color_discrete_sequence=[BUY_COLOR],
            labels={"n": "Transactions", "symbol": "Symbol"},
            title="Top 3 Symbols",
        )
        fig_sym.update_layout(showlegend=False, title_font_size=13,
                              xaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_sym, height=220, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sym, use_container_width=True)

    with col_sec:
        # Exclude unmapped symbols so the chart reflects only classified sectors
        top_sec = (
            df_trade[df_trade["sector"] != "Unmapped"]
            .groupby("sector").size()
            .sort_values(ascending=True).tail(5)
            .reset_index(name="n")
        )
        fig_sec = px.bar(
            top_sec, x="n", y="sector", orientation="h",
            color_discrete_sequence=[SELL_COLOR],
            labels={"n": "Transactions", "sector": "Sector"},
            title="Top 5 Sectors",
        )
        fig_sec.update_layout(showlegend=False, title_font_size=13,
                              xaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_sec, height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sec, use_container_width=True)

    with col_ind:
        top_ind = (
            df_trade[df_trade["industry"] != "Unmapped"]
            .groupby("industry").size()
            .sort_values(ascending=True).tail(5)
            .reset_index(name="n")
        )
        fig_ind = px.bar(
            top_ind, x="n", y="industry", orientation="h",
            color_discrete_sequence=[ACCENT],
            labels={"n": "Transactions", "industry": "Industry"},
            title="Top 5 Industries",
        )
        fig_ind.update_layout(showlegend=False, title_font_size=13,
                              xaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_ind, height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ind, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — GEOGRAPHY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:

    # Work only on BUY/SELL rows that have a valid ISO code so the map can plot them
    df_geo = df[df["transactiontype"].isin(["BUY", "SELL"]) & (df["iso_code"] != "")].copy()

    # ── Country Deep Dive ─────────────────────────────────────────────────────
    section("🔎 Country Deep Dive")
    all_countries = sorted(df_geo["country"].dropna().unique())
    default_idx = (
        all_countries.index("United States of America")
        if "United States of America" in all_countries
        else 0
    )
    sel_country = st.selectbox("Select a country", options=all_countries, index=default_idx)
    df_country  = df_geo[df_geo["country"] == sel_country]

    if df_country.empty:
        st.info(f"No BUY/SELL transactions found for **{sel_country}** in the selected period.")
    else:
        ka, kb, kc, kd = st.columns(4)
        ka.metric("Total Trades", f"{len(df_country):,}")
        kb.metric("BUY",  f"{(df_country['transactiontype'] == 'BUY').sum():,}")
        kc.metric("SELL", f"{(df_country['transactiontype'] == 'SELL').sum():,}")
        kd.metric("Total Units", f"{df_country['quantity'].sum():,.0f}")

        # Daily transaction trend with a 7-day moving average overlay
        ct = (
            df_country.groupby("date_clean").size()
            .reset_index(name="n").sort_values("date_clean")
        )
        ct["MA7"] = ct["n"].rolling(7, min_periods=1).mean()

        fig_ct = go.Figure()
        fig_ct.add_trace(go.Bar(
            x=ct["date_clean"], y=ct["n"],
            name="Daily", marker_color=BUY_COLOR, opacity=0.4,
        ))
        fig_ct.add_trace(go.Scatter(
            x=ct["date_clean"], y=ct["MA7"],
            name="7d MA", line=dict(color=ACCENT, width=2.5),
        ))
        fig_ct.update_layout(
            hovermode="x unified",
            title=f"Transaction trend — {sel_country}",
            title_font_size=13,
            xaxis=dict(showgrid=False, **AXIS_STYLE),
            yaxis=dict(gridcolor="#f0f0f0", **AXIS_STYLE),
            legend=dict(orientation="h", y=-0.3),
            **LAYOUT_BASE,
        )
        clean_layout(fig_ct, height=280, legend_bottom=True)
        st.plotly_chart(fig_ct, use_container_width=True)

        # Top industries split by BUY and SELL, shown side by side
        col_buy, col_sell = st.columns(2)
        for side, color, label in [
            ("BUY", BUY_COLOR, "BUY"),
            ("SELL", SELL_COLOR, "SELL"),
        ]:
            col = col_buy if side == "BUY" else col_sell
            with col:
                top = (
                    df_country[df_country["transactiontype"] == side]
                    .groupby("industry").size()
                    .sort_values(ascending=True).tail(8)
                    .reset_index(name="n")
                )
                if top.empty:
                    st.info(f"No {label} transactions.")
                else:
                    fig = px.bar(
                        top, x="n", y="industry", orientation="h",
                        color_discrete_sequence=[color],
                        title=f"Top Industries — {label}",
                        labels={"n": "Transactions", "industry": ""},
                    )
                    fig.update_layout(xaxis=dict(gridcolor="#f0f0f0"))
                    clean_layout(fig, height=320, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Choropleth World Map ──────────────────────────────────────────────────
    section("🗺️ Global Transaction Map")
    map_metric = st.radio("Show", ["# Transactions", "Total Units"], horizontal=True)
    map_type   = st.radio("Type", ["BUY + SELL", "BUY only", "SELL only"], horizontal=True)

    # Filter to rows that have a valid ISO-3 code (required by Plotly choropleth)
    df_map = df_geo[df_geo["iso3_code"] != ""].copy()
    if map_type == "BUY only":
        df_map = df_map[df_map["transactiontype"] == "BUY"]
    elif map_type == "SELL only":
        df_map = df_map[df_map["transactiontype"] == "SELL"]

    if map_metric == "# Transactions":
        geo_agg = df_map.groupby(["country", "iso3_code"]).size().reset_index(name="value")
        map_label = "Transactions"
    else:
        geo_agg = (
            df_map.groupby(["country", "iso3_code"])["quantity"]
            .sum().reset_index(name="value")
        )
        map_label = "Units"

    fig_map = px.choropleth(
        geo_agg, locations="iso3_code", locationmode="ISO-3",
        color="value", hover_name="country",
        color_continuous_scale="Blues",
        labels={"value": map_label},
    )
    fig_map.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor="#cccccc",
            bgcolor="white", landcolor="#f5f5f5",
            oceancolor="#e8f4fd", showocean=True,
        ),
        coloraxis_colorbar=dict(title=map_label, thickness=12, len=0.6),
        **LAYOUT_BASE,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # ── Region-level breakdown ────────────────────────────────────────────────
    section("🌐 Region Breakdown")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        reg = df_geo.groupby(["region", "transactiontype"]).size().reset_index(name="n")
        fig_reg = px.bar(
            reg, x="region", y="n", color="transactiontype",
            barmode="stack",
            color_discrete_map=COLOR_MAP,
            labels={"n": "Transactions", "region": "Region", "transactiontype": "Type"},
            title="Transactions by Region",
        )
        fig_reg.update_layout(legend_title=None, xaxis_title=None,
                              yaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_reg, height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_r2:
        # Top sub-regions by total units traded (BUY + SELL combined)
        sub = (
            df_geo.groupby("sub_region")["quantity"].sum()
            .sort_values(ascending=True).tail(8).reset_index()
        )
        sub.columns = ["Sub-Region", "Total Units"]
        fig_sub = px.bar(
            sub, x="Total Units", y="Sub-Region", orientation="h",
            color_discrete_sequence=[ACCENT],
            title="Top Sub-Regions by Units Traded",
        )
        fig_sub.update_layout(xaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_sub, height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sub, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — SYMBOL & SECTOR DEEP DIVE
# ──────────────────────────────────────────────────────────────────────────────
with tab3:

    df_deep = df[df["transactiontype"].isin(["BUY", "SELL"])].copy()

    # ── Sector distribution — transaction count stacked bar ───────────────────
    section("🏭 Sector Distribution — Transaction Count")

    sec_dist = (
        df_deep[df_deep["sector"] != "Unmapped"]
        .groupby(["sector", "transactiontype"]).size()
        .reset_index(name="n")
    )
    # Sort sectors by total volume so the most active sector appears at the top
    sec_order = (
        sec_dist.groupby("sector")["n"].sum()
        .sort_values(ascending=True).index.tolist()
    )
    sec_dist["sector"] = pd.Categorical(
        sec_dist["sector"], categories=sec_order, ordered=True
    )
    sec_dist = sec_dist.sort_values("sector")

    fig_sec_bar = px.bar(
        sec_dist, x="n", y="sector", color="transactiontype",
        orientation="h", barmode="stack",
        color_discrete_map=COLOR_MAP,
        labels={"n": "Transactions", "sector": "Sector", "transactiontype": "Type"},
    )
    fig_sec_bar.update_layout(legend_title=None, yaxis_title=None,
                              xaxis=dict(gridcolor="#f0f0f0"))
    clean_layout(fig_sec_bar, height=380)
    st.plotly_chart(fig_sec_bar, use_container_width=True)

    st.markdown("---")

    # ── Units by sector (grouped bar) and quarterly activity heatmap ──────────
    col_sec2, col_heat = st.columns(2)

    with col_sec2:
        section("💼 Units Traded by Sector")
        sec_bs = (
            df_deep[df_deep["sector"] != "Unmapped"]
            .groupby(["sector", "transactiontype"])["quantity"].sum()
            .reset_index()
        )
        fig_sec2 = px.bar(
            sec_bs, x="quantity", y="sector", color="transactiontype",
            orientation="h", barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={"quantity": "Units", "sector": "Sector", "transactiontype": "Type"},
        )
        fig_sec2.update_layout(legend_title=None,
                               yaxis=dict(autorange="reversed"),
                               xaxis=dict(gridcolor="#f0f0f0"))
        clean_layout(fig_sec2, height=340)
        st.plotly_chart(fig_sec2, use_container_width=True)

    with col_heat:
        # Heatmap: sectors on the y-axis, quarters on the x-axis, colour = trade count
        section("🔥 Sector Activity by Quarter")
        heat = (
            df_deep[df_deep["sector"] != "Unmapped"]
            .groupby(["sector", "quarter"]).size()
            .reset_index(name="n")
            .pivot(index="sector", columns="quarter", values="n")
            .fillna(0)
        )
        heat.columns = ["Q1", "Q2", "Q3", "Q4"]
        fig_heat = px.imshow(
            heat, color_continuous_scale="Blues", aspect="auto",
            labels=dict(x="Quarter", y="Sector", color="Transactions"),
            text_auto=True,
        )
        fig_heat.update_traces(textfont_size=11)
        clean_layout(fig_heat, height=340)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Top Symbols detail table with an adjustable row count ─────────────────
    section("📋 Top Symbols — Detail Table")
    top_n = st.slider("Show top N symbols", min_value=5, max_value=30, value=10, step=5)

    sym_table = (
        df_deep.groupby(["symbol", "company_name", "sector", "industry"])
        .agg(
            Total_Trades=("quantity", "count"),
            BUY_Trades  =("transactiontype", lambda x: (x == "BUY").sum()),
            SELL_Trades =("transactiontype", lambda x: (x == "SELL").sum()),
            Total_Units =("quantity", "sum"),
            Avg_Units   =("quantity", "mean"),
        )
        .reset_index()
        .sort_values("Total_Trades", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    sym_table.index += 1
    sym_table["Avg_Units"] = sym_table["Avg_Units"].round(1)

    st.dataframe(
        sym_table,
        use_container_width=True,
        height=420,
        column_config={
            "symbol":       st.column_config.TextColumn("Symbol"),
            "company_name": st.column_config.TextColumn("Company"),
            "sector":       st.column_config.TextColumn("Sector"),
            "industry":     st.column_config.TextColumn("Industry"),
            "Total_Trades": st.column_config.ProgressColumn(
                "Total Trades", format="%d", min_value=0,
                max_value=int(sym_table["Total_Trades"].max()),
            ),
            "BUY_Trades":   st.column_config.NumberColumn("BUY",  format="%d"),
            "SELL_Trades":  st.column_config.NumberColumn("SELL", format="%d"),
            "Total_Units":  st.column_config.NumberColumn("Total Units", format="%,.0f"),
            "Avg_Units":    st.column_config.NumberColumn("Avg Units/Trade", format="%.1f"),
        },
    )

    # ── CSV export of the current filtered dataset ────────────────────────────
    st.markdown("---")
    csv_export = (
        df[
            [
                "date_clean", "symbol", "company_name", "sector", "industry",
                "region", "country", "transactiontype", "quantity",
            ]
        ]
        .copy()
        .rename(
            columns={
                "date_clean":      "Date",
                "symbol":          "Symbol",
                "company_name":    "Company",
                "sector":          "Sector",
                "industry":        "Industry",
                "region":          "Region",
                "country":         "Country",
                "transactiontype": "Type",
                "quantity":        "Quantity",
            }
        )
    )
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_export.to_csv(index=False).encode("utf-8"),
        file_name="transactions_filtered.csv",
        mime="text/csv",
    )