import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import jarque_bera, kurtosis, norm, probplot, skew

# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="Stock Comparison and Analysis App",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252
BENCHMARK = "^GSPC"


# =========================================================
# Helper functions
# =========================================================
def parse_tickers(raw_text: str) -> list[str]:
    cleaned = raw_text.replace("\n", ",")
    tickers = [t.strip().upper() for t in cleaned.split(",") if t.strip()]
    return list(dict.fromkeys(tickers))


def annualized_mean_return(series: pd.Series) -> float:
    return float(series.mean() * TRADING_DAYS)


def annualized_vol(series: pd.Series) -> float:
    return float(series.std() * np.sqrt(TRADING_DAYS))


def validate_inputs(tickers: list[str], start_date: date, end_date: date):
    if len(tickers) < 2 or len(tickers) > 5:
        return False, "Please enter between 2 and 5 ticker symbols."
    if start_date >= end_date:
        return False, "The start date must be earlier than the end date."
    if (end_date - start_date).days < 365:
        return False, "Please select a date range of at least 1 year."
    return True, ""


@st.cache_data(ttl=3600)
def download_adjusted_close(tickers: list[str], start_date: date, end_date: date):
    all_tickers = list(dict.fromkeys(tickers + [BENCHMARK]))

    try:
        df = yf.download(
            tickers=all_tickers,
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")

    if df is None or df.empty:
        raise RuntimeError("No data was returned by yfinance.")

    prices = pd.DataFrame(index=df.index)

    if isinstance(df.columns, pd.MultiIndex):
        for ticker in all_tickers:
            try:
                if (ticker, "Adj Close") in df.columns:
                    prices[ticker] = df[(ticker, "Adj Close")]
            except Exception:
                pass
    else:
        if "Adj Close" in df.columns and len(all_tickers) == 1:
            prices[all_tickers[0]] = df["Adj Close"]

    failed = []
    for ticker in all_tickers:
        if ticker not in prices.columns or prices[ticker].dropna().shape[0] < 2:
            failed.append(ticker)

    keep_cols = [c for c in prices.columns if c not in failed]
    prices = prices[keep_cols].copy()

    return prices, failed


@st.cache_data(ttl=3600)
def clean_and_align_prices(prices_raw: pd.DataFrame, selected_tickers: list[str]):
    warnings_list = []
    dropped = []

    if prices_raw.empty:
        return prices_raw, dropped, warnings_list

    candidate_cols = [t for t in selected_tickers if t in prices_raw.columns]
    if BENCHMARK in prices_raw.columns:
        candidate_cols.append(BENCHMARK)

    working = prices_raw[candidate_cols].copy()

    for ticker in selected_tickers:
        if ticker in working.columns:
            missing_pct = working[ticker].isna().mean()
            if missing_pct > 0.05:
                dropped.append(ticker)

    if dropped:
        warnings_list.append(
            f"Dropped ticker(s) with more than 5% missing values: {', '.join(dropped)}"
        )

    keep_user = [t for t in selected_tickers if t in working.columns and t not in dropped]
    final_cols = keep_user + ([BENCHMARK] if BENCHMARK in working.columns else [])
    working = working[final_cols].copy()

    if len(keep_user) >= 2:
        original_rows = len(working)
        working = working.dropna(how="any")
        if len(working) < original_rows:
            warnings_list.append(
                "Data was truncated to the overlapping date range so all remaining series align properly."
            )

    return working, dropped, warnings_list


@st.cache_data(ttl=3600)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


@st.cache_data(ttl=3600)
def compute_summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.DataFrame(index=returns.columns)
    stats_df["Annualized Mean Return"] = returns.apply(annualized_mean_return)
    stats_df["Annualized Volatility"] = returns.apply(annualized_vol)
    stats_df["Skewness"] = returns.apply(skew)
    stats_df["Kurtosis"] = returns.apply(lambda x: kurtosis(x, fisher=True))
    stats_df["Min Daily Return"] = returns.min()
    stats_df["Max Daily Return"] = returns.max()
    return stats_df


@st.cache_data(ttl=3600)
def compute_wealth_index(returns: pd.DataFrame, user_tickers: list[str]):
    wealth = (1 + returns).cumprod() * 10000
    user_returns = returns[user_tickers].copy()
    equal_weight_returns = user_returns.mean(axis=1)
    wealth["Equal Weight Portfolio"] = (1 + equal_weight_returns).cumprod() * 10000
    return wealth, equal_weight_returns


def percent_format_df(df: pd.DataFrame, percent_cols: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for col in percent_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2%}")
    return formatted


# =========================================================
# Sidebar
# =========================================================
st.title("📈 Stock Comparison and Analysis App")
st.caption("Compare stocks, analyze returns and risk, and explore diversification.")

with st.sidebar:
    st.header("Inputs")

    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)

    ticker_text = st.text_area(
        "Enter 2 to 5 stock tickers",
        value="AAPL, MSFT, NVDA",
        height=100,
        help="Separate tickers with commas or new lines.",
    )

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    with st.expander("About / Methodology"):
        st.markdown(
            """
            **What this app does**
            - Downloads adjusted close prices for 2 to 5 user-selected stocks
            - Downloads the S&P 500 benchmark (`^GSPC`) for comparison
            - Computes simple daily returns
            - Shows price, return, risk, distribution, correlation, and diversification analysis

            **Key assumptions**
            - Returns are **simple arithmetic returns** using `pct_change()`
            - Annualized mean return = mean daily return × **252**
            - Annualized volatility = daily standard deviation × **sqrt(252)**
            - Wealth index starts at **$10,000**
            - Data source: **Yahoo Finance via yfinance**
            """
        )

tickers = parse_tickers(ticker_text)
is_valid, validation_message = validate_inputs(tickers, start_date, end_date)

if not is_valid:
    st.error(validation_message)
    st.stop()

# =========================================================
# Download and preprocess
# =========================================================
with st.spinner("Downloading market data and preparing analysis..."):
    try:
        prices_raw, failed = download_adjusted_close(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Could not download data. {e}")
        st.stop()

failed_user = [t for t in failed if t != BENCHMARK]
if failed_user:
    st.error(
        f"The following ticker(s) failed to download or returned insufficient data: {', '.join(failed_user)}"
    )

benchmark_available = BENCHMARK in prices_raw.columns and BENCHMARK not in failed

if not benchmark_available:
    st.warning(
        "The S&P 500 benchmark (^GSPC) could not be downloaded right now due to a Yahoo Finance rate limit. "
        "The rest of the stock analysis will still run."
    )

aligned_prices, dropped_tickers, warnings_list = clean_and_align_prices(prices_raw, tickers)
active_user_tickers = [t for t in tickers if t in aligned_prices.columns and t != BENCHMARK]

for message in warnings_list:
    st.warning(message)

if len(active_user_tickers) < 2:
    st.error(
        "After cleaning the data, fewer than 2 valid user-selected tickers remain. "
        "Please choose different tickers or a different date range."
    )
    st.stop()

returns = compute_returns(aligned_prices)
summary_stats = compute_summary_stats(returns)
wealth_index, equal_weight_returns = compute_wealth_index(returns, active_user_tickers)

if returns.empty:
    st.error("Not enough return observations after cleaning the data.")
    st.stop()

# =========================================================
# Layout tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Price & Return Analysis",
        "Risk & Distribution",
        "Correlation & Diversification",
        "Data Preview",
    ]
)

# =========================================================
# Tab 1
# =========================================================
with tab1:
    st.header("Price and Return Analysis")

    st.subheader("Adjusted Closing Price Chart")
    selected_price_series = st.multiselect(
        "Select stocks to show on the price chart",
        options=active_user_tickers,
        default=active_user_tickers,
        key="price_multiselect",
    )

    if selected_price_series:
        fig_prices = go.Figure()
        for ticker in selected_price_series:
            fig_prices.add_trace(
                go.Scatter(
                    x=aligned_prices.index,
                    y=aligned_prices[ticker],
                    mode="lines",
                    name=ticker,
                )
            )

        fig_prices.update_layout(
            title="Adjusted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Adjusted Close Price",
            legend_title="Ticker",
            height=520,
        )
        st.plotly_chart(fig_prices, use_container_width=True)
    else:
        st.warning("Select at least one stock to display the price chart.")

    st.subheader("Summary Statistics Table")
    stats_display = summary_stats.reset_index().rename(columns={"index": "Ticker"})
    stats_display = percent_format_df(
        stats_display,
        [
            "Annualized Mean Return",
            "Annualized Volatility",
            "Min Daily Return",
            "Max Daily Return",
        ],
    )
    st.dataframe(stats_display, use_container_width=True)

    st.subheader("Cumulative Wealth Index")
    wealth_cols = active_user_tickers + [BENCHMARK, "Equal Weight Portfolio"]
    fig_wealth = go.Figure()

    for col in wealth_cols:
        fig_wealth.add_trace(
            go.Scatter(
                x=wealth_index.index,
                y=wealth_index[col],
                mode="lines",
                name=col,
            )
        )

    fig_wealth.update_layout(
        title="Growth of a $10,000 Investment",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend_title="Series",
        height=560,
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

# =========================================================
# Tab 2
# =========================================================
with tab2:
    st.header("Risk and Distribution Analysis")

    st.subheader("Rolling Annualized Volatility")
    rolling_vol_window = st.selectbox(
        "Select rolling volatility window (days)",
        [30, 60, 90],
        index=1,
        key="rolling_vol_window",
    )

    rolling_vol = returns[active_user_tickers].rolling(rolling_vol_window).std() * np.sqrt(TRADING_DAYS)

    fig_rolling_vol = go.Figure()
    for ticker in active_user_tickers:
        fig_rolling_vol.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[ticker],
                mode="lines",
                name=ticker,
            )
        )

    fig_rolling_vol.update_layout(
        title=f"Rolling Annualized Volatility ({rolling_vol_window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        legend_title="Ticker",
        height=520,
    )
    st.plotly_chart(fig_rolling_vol, use_container_width=True)

    st.subheader("Distribution Analysis")
    dist_ticker = st.selectbox(
        "Select a stock for distribution analysis",
        active_user_tickers,
        key="dist_ticker",
    )

    distribution_view = st.radio(
        "Choose view",
        ["Histogram + Normal Curve", "Q-Q Plot"],
        horizontal=True,
        key="dist_view",
    )

    dist_returns = returns[dist_ticker].dropna()

    jb_stat, jb_pvalue = jarque_bera(dist_returns)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Jarque-Bera Statistic", f"{jb_stat:.4f}")
    with col_b:
        st.metric("p-value", f"{jb_pvalue:.6f}")

    if jb_pvalue < 0.05:
        st.error("Rejects normality (p < 0.05)")
    else:
        st.success("Fails to reject normality (p >= 0.05)")

    if distribution_view == "Histogram + Normal Curve":
        mu, sigma = norm.fit(dist_returns)
        x_vals = np.linspace(dist_returns.min(), dist_returns.max(), 300)
        y_vals = norm.pdf(x_vals, mu, sigma)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=dist_returns,
                histnorm="probability density",
                name="Daily Returns",
                opacity=0.75,
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Fitted Normal Curve",
            )
        )

        fig_hist.update_layout(
            title=f"Histogram of Daily Returns with Fitted Normal Curve: {dist_ticker}",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            height=520,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        qq_data = probplot(dist_returns, dist="norm")
        theoretical_quantiles = qq_data[0][0]
        sample_quantiles = qq_data[0][1]
        slope, intercept, _ = qq_data[1]

        line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        line_y = slope * line_x + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                name="Sample Quantiles",
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Reference Line",
            )
        )

        fig_qq.update_layout(
            title=f"Q-Q Plot: {dist_ticker} Daily Returns vs Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=520,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    st.subheader("Box Plot of Daily Return Distributions")
    box_df = returns[active_user_tickers].melt(var_name="Ticker", value_name="Daily Return")
    fig_box = px.box(
        box_df,
        x="Ticker",
        y="Daily Return",
        color="Ticker",
        title="Daily Return Distributions by Stock",
    )
    fig_box.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Daily Return",
        showlegend=False,
        height=520,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# Tab 3
# =========================================================
with tab3:
    st.header("Correlation and Diversification")

    st.subheader("Correlation Heatmap")
    corr_matrix = returns[active_user_tickers].corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Pairwise Correlation Matrix of Daily Returns",
    )
    fig_heatmap.update_layout(height=520)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Scatter Plot of Two Stocks")
    col1, col2 = st.columns(2)
    with col1:
        scatter_x = st.selectbox("Select Stock X", active_user_tickers, key="scatter_x")
    with col2:
        scatter_y_options = [t for t in active_user_tickers if t != scatter_x]
        scatter_y = st.selectbox("Select Stock Y", scatter_y_options, key="scatter_y")

    scatter_data = returns[[scatter_x, scatter_y]].dropna()
    fig_scatter = px.scatter(
        scatter_data,
        x=scatter_x,
        y=scatter_y,
        title=f"Daily Return Scatter Plot: {scatter_x} vs {scatter_y}",
    )
    fig_scatter.update_layout(
        xaxis_title=f"{scatter_x} Daily Return",
        yaxis_title=f"{scatter_y} Daily Return",
        height=520,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Rolling Correlation")
    col3, col4, col5 = st.columns(3)
    with col3:
        roll_a = st.selectbox("Rolling Corr Stock A", active_user_tickers, key="roll_a")
    with col4:
        roll_b_options = [t for t in active_user_tickers if t != roll_a]
        roll_b = st.selectbox("Rolling Corr Stock B", roll_b_options, key="roll_b")
    with col5:
        roll_window = st.selectbox("Rolling Window", [30, 60, 90], index=1, key="roll_window")

    rolling_corr = returns[roll_a].rolling(roll_window).corr(returns[roll_b])

    fig_roll_corr = go.Figure()
    fig_roll_corr.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode="lines",
            name="Rolling Correlation",
        )
    )
    fig_roll_corr.update_layout(
        title=f"Rolling Correlation: {roll_a} vs {roll_b} ({roll_window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        height=520,
    )
    st.plotly_chart(fig_roll_corr, use_container_width=True)

    st.subheader("Two-Asset Portfolio Explorer")

    col6, col7 = st.columns(2)
    with col6:
        port_a = st.selectbox("Portfolio Stock A", active_user_tickers, key="port_a")
    with col7:
        port_b_options = [t for t in active_user_tickers if t != port_a]
        port_b = st.selectbox("Portfolio Stock B", port_b_options, key="port_b")

    weight_a_pct = st.slider(
        f"Weight on {port_a} (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key="weight_a_pct",
    )

    w_a = weight_a_pct / 100
    w_b = 1 - w_a

    two_asset_returns = returns[[port_a, port_b]].dropna()
    annual_means = two_asset_returns.mean() * TRADING_DAYS
    annual_cov = two_asset_returns.cov() * TRADING_DAYS

    current_port_return = (w_a * annual_means[port_a]) + (w_b * annual_means[port_b])
    current_port_variance = (
        (w_a ** 2) * annual_cov.loc[port_a, port_a]
        + (w_b ** 2) * annual_cov.loc[port_b, port_b]
        + 2 * w_a * w_b * annual_cov.loc[port_a, port_b]
    )
    current_port_vol = math.sqrt(max(current_port_variance, 0))

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Portfolio Annualized Return", f"{current_port_return:.2%}")
    with m2:
        st.metric("Portfolio Annualized Volatility", f"{current_port_vol:.2%}")

    weight_grid = np.linspace(0, 1, 101)
    port_vols = []

    for wt in weight_grid:
        other_wt = 1 - wt
        variance = (
            (wt ** 2) * annual_cov.loc[port_a, port_a]
            + (other_wt ** 2) * annual_cov.loc[port_b, port_b]
            + 2 * wt * other_wt * annual_cov.loc[port_a, port_b]
        )
        port_vols.append(math.sqrt(max(variance, 0)))

    fig_port = go.Figure()
    fig_port.add_trace(
        go.Scatter(
            x=weight_grid * 100,
            y=port_vols,
            mode="lines",
            name="Portfolio Volatility Curve",
        )
    )
    fig_port.add_trace(
        go.Scatter(
            x=[weight_a_pct],
            y=[current_port_vol],
            mode="markers",
            name="Current Weight",
            marker=dict(size=10),
        )
    )

    stock_a_vol = math.sqrt(annual_cov.loc[port_a, port_a])
    stock_b_vol = math.sqrt(annual_cov.loc[port_b, port_b])

    fig_port.add_hline(
        y=stock_a_vol,
        line_dash="dot",
        annotation_text=f"{port_a} Vol",
        annotation_position="top left",
    )
    fig_port.add_hline(
        y=stock_b_vol,
        line_dash="dot",
        annotation_text=f"{port_b} Vol",
        annotation_position="bottom left",
    )

    fig_port.update_layout(
        title=f"Two-Asset Portfolio Volatility Curve: {port_a} and {port_b}",
        xaxis_title=f"Weight on {port_a} (%)",
        yaxis_title="Annualized Volatility",
        height=520,
    )
    st.plotly_chart(fig_port, use_container_width=True)

    st.info(
        "This curve demonstrates diversification. When the correlation between two stocks is less than 1, "
        "combining them can produce a portfolio with lower volatility than either stock individually. "
        "The diversification effect is stronger when the correlation is lower."
    )

# =========================================================
# Tab 4
# =========================================================
with tab4:
    st.header("Data Preview")

    st.subheader("Active User Tickers")
    st.write(", ".join(active_user_tickers))

    st.subheader("Benchmark")
    st.write(BENCHMARK)

    st.subheader("Cleaned Adjusted Close Prices")
    st.dataframe(aligned_prices.tail(20), use_container_width=True)

    st.subheader("Daily Returns")
    st.dataframe(returns.tail(20), use_container_width=True)