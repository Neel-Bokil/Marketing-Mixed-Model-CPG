
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CPG Marketing Mix Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "mmm_eda_clean.csv"

MEDIA_COLS = [
    "tv_S",
    "ooh_S",
    "print_S",
    "facebook_S",
    "search_S",
]

CHANNEL_LABELS = {
    "tv_S": "TV",
    "ooh_S": "OOH",
    "print_S": "Print",
    "facebook_S": "Facebook",
    "search_S": "Search",
}

# Parameters from Notebook 03
DECAY = 0.5
HILL_SLOPE = 1.0

HALF_SATURATION = {
    "tv_S": 19094.3911,
    "ooh_S": 71559.9820,
    "print_S": 6410.5009,
    "facebook_S": 3445.5238,
    "search_S": 8069.9394,
}

MEDIA_COEFFICIENTS = {
    "tv_S": 704265.0978,
    "ooh_S": 56687.2489,
    "print_S": 252240.9431,
    "facebook_S": 256060.4398,
    "search_S": 102314.0391,
}


# ============================================================
# 3. DATA LOADING
# ============================================================
@st.cache_data
def load_data(path):
    data = pd.read_csv(path, parse_dates=["DATE"])
    data = data.sort_values("DATE").reset_index(drop=True)

    required_columns = ["DATE", "revenue"] + MEDIA_COLS
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    if data[required_columns].isna().any().any():
        raise ValueError(
            "The dataset contains missing values in required columns."
        )

    if (data[MEDIA_COLS] < 0).any().any():
        raise ValueError("Media spend cannot be negative.")

    return data


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


# ============================================================
# 4. SHARED MODEL FUNCTIONS
# ============================================================
def calculate_adstock(spend, decay=DECAY):
    """Calculate geometric adstock in chronological order."""
    spend = np.asarray(spend, dtype=float)
    result = np.zeros_like(spend)

    for i in range(len(spend)):
        previous = result[i - 1] if i > 0 else 0.0
        result[i] = spend[i] + decay * previous

    return result


def hill_response(x, half_saturation, slope=HILL_SLOPE):
    """Apply Hill saturation to nonnegative adstock values."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)

    numerator = x ** slope
    denominator = half_saturation ** slope + numerator

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def steady_state_contribution(channel, weekly_spend, slope=HILL_SLOPE):
    """
    Estimate steady-state weekly media contribution for constant spending.
    """
    weekly_spend = max(float(weekly_spend), 0.0)

    steady_state_adstock = weekly_spend / (1.0 - DECAY)

    saturation = hill_response(
        np.array([steady_state_adstock]),
        HALF_SATURATION[channel],
        slope=slope,
    )[0]

    return MEDIA_COEFFICIENTS[channel] * float(saturation)


def total_steady_state_contribution(allocation, slope=HILL_SLOPE):
    """Sum steady-state modeled contributions across media channels."""
    return sum(
        steady_state_contribution(channel, spend, slope)
        for channel, spend in zip(MEDIA_COLS, allocation)
    )


def get_feasible_baseline(target_budget, upper_bounds):
    """
    Scale historical average spend to the target budget.

    If necessary, redistribute spending that exceeds a channel cap.
    The returned allocation must satisfy the budget and channel bounds.
    """
    historical = df[MEDIA_COLS].mean().to_numpy(dtype=float)

    historical_total = float(historical.sum())

    if historical_total <= 0:
        raise ValueError("Historical total media spend must be positive.")

    if target_budget < 0 or target_budget > upper_bounds.sum():
        raise ValueError("The selected budget is infeasible.")

    baseline = historical * (target_budget / historical_total)

    # Respect channel caps.
    baseline = np.minimum(baseline, upper_bounds)

    remaining = target_budget - baseline.sum()

    # Redistribute any unallocated budget among channels with capacity.
    for i in range(len(MEDIA_COLS)):
        if remaining <= 1e-7:
            break

        capacity = upper_bounds[i] - baseline[i]
        addition = min(max(capacity, 0.0), remaining)

        baseline[i] += addition
        remaining -= addition

    if remaining > 1e-4:
        raise ValueError(
            "Unable to construct a feasible baseline allocation."
        )

    if not np.isclose(baseline.sum(), target_budget, atol=1e-4):
        raise ValueError("Baseline does not satisfy the budget constraint.")

    return baseline



def optimize_budget(target_budget, upper_bounds, slope=HILL_SLOPE):
    """Optimize channel allocation under a fixed total budget."""
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    if target_budget < 0:
        raise ValueError("Target budget cannot be negative.")

    # Check whether the requested budget is feasible.
    if target_budget > upper_bounds.sum() + 1e-6:
        raise ValueError(
            "Target budget exceeds the combined channel spending caps."
        )

    baseline = np.asarray(
        get_feasible_baseline(target_budget, upper_bounds),
        dtype=float,
    )

    # Correct tiny floating-point differences in the initial allocation.
    difference = target_budget - baseline.sum()

    if abs(difference) > 1e-7:
        if difference > 0:
            available = upper_bounds - baseline
        else:
            available = baseline.copy()

        for i in range(len(baseline)):
            if difference > 0:
                adjustment = min(difference, available[i])
                baseline[i] += adjustment
            else:
                adjustment = min(-difference, available[i])
                baseline[i] -= adjustment

            difference = target_budget - baseline.sum()

            if abs(difference) <= 1e-7:
                break

    if not np.isclose(baseline.sum(), target_budget, atol=1e-4):
        raise RuntimeError(
            "Could not construct a feasible starting allocation."
        )

    # Scale variables to improve numerical stability.
    scale = max(float(target_budget), 1.0)
    scaled_budget = target_budget / scale
    scaled_upper_bounds = upper_bounds / scale
    scaled_baseline = baseline / scale

    def objective(scaled_allocation):
        allocation = scaled_allocation * scale
        return -float(
            total_steady_state_contribution(
                allocation,
                slope=slope,
            )
        )

    result = minimize(
        fun=objective,
        x0=scaled_baseline,
        method="SLSQP",
        bounds=[
            (0.0, float(cap))
            for cap in scaled_upper_bounds
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda allocation: allocation.sum() - scaled_budget,
                "jac": lambda allocation: np.ones_like(allocation),
            }
        ],
        options={
            "maxiter": 3000,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Budget optimization failed: {result.message}"
        )

    allocation = result.x * scale

    # Validate the final result.
    if not np.isclose(allocation.sum(), target_budget, atol=1e-3):
        raise RuntimeError(
            "Optimized allocation violates the budget constraint."
        )

    if (
        np.any(allocation < -1e-4)
        or np.any(allocation > upper_bounds + 1e-3)
    ):
        raise RuntimeError(
            "Optimized allocation violates channel bounds."
        )

    # Remove insignificant numerical deviations.
    allocation = np.clip(allocation, 0.0, upper_bounds)

    return baseline, allocation


# ============================================================
# 5. APP HEADER
# ============================================================
st.title("📊 CPG Marketing Mix Intelligence")
st.caption(
    "Channel performance | Response curves | Budget optimization"
)

st.info(
    "Portfolio demonstration using simulated data. Model-based contribution "
    "and optimization results are exploratory and are not validated causal "
    "effects or guarantees of actual revenue growth."
)


# ============================================================
# 6. SIDEBAR DATE FILTER
# ============================================================
st.sidebar.title("Dashboard Controls")

min_date = df["DATE"].min().date()
max_date = df["DATE"].max().date()

selected_dates = st.sidebar.date_input(
    "Select analysis period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.warning("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

if start_date > end_date:
    st.error("The start date must be before the end date.")
    st.stop()

filtered_df = df[
    (df["DATE"].dt.date >= start_date)
    & (df["DATE"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data is available for the selected period.")
    st.stop()


# ============================================================
# 7. EXECUTIVE OVERVIEW
# ============================================================
st.header("Executive Overview")

st.caption(
    f"Analysis period: {start_date:%d %b %Y} – {end_date:%d %b %Y}"
)

total_revenue = filtered_df["revenue"].sum()
average_weekly_revenue = filtered_df["revenue"].mean()
total_media_spend = filtered_df[MEDIA_COLS].sum().sum()
average_weekly_spend = filtered_df[MEDIA_COLS].sum(axis=1).mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"{total_revenue:,.0f}")
col2.metric("Average Weekly Revenue", f"{average_weekly_revenue:,.0f}")
col3.metric("Total Media Spend", f"{total_media_spend:,.0f}")
col4.metric("Average Weekly Media Spend", f"{average_weekly_spend:,.0f}")


# Revenue trend
st.subheader("Weekly Revenue Trend")

st.line_chart(
    filtered_df.set_index("DATE")[["revenue"]],
    y="revenue",
    x_label="Week",
    y_label="Revenue",
)


# Historical media spending
st.subheader("Marketing Investment by Channel")

channel_spend = pd.DataFrame({
    "Channel": [CHANNEL_LABELS[channel] for channel in MEDIA_COLS],
    "Total Spend": [filtered_df[channel].sum() for channel in MEDIA_COLS],
})

channel_spend["Budget Share (%)"] = (
    channel_spend["Total Spend"] / total_media_spend * 100
    if total_media_spend > 0
    else 0.0
)

channel_spend = channel_spend.sort_values("Total Spend", ascending=False)

left, right = st.columns([3, 2])

with left:
    st.bar_chart(
        channel_spend.set_index("Channel")[["Total Spend"]],
        horizontal=True,
        x_label="Total Spend",
        y_label="Channel",
    )

with right:
    st.dataframe(
        channel_spend.round(2),
        hide_index=True,
        use_container_width=True,
    )


# Correlation
st.subheader("Descriptive Association with Revenue")

correlations = (
    filtered_df[MEDIA_COLS + ["revenue"]]
    .corr(numeric_only=True)["revenue"]
    .drop("revenue")
    .rename_axis("Channel")
    .reset_index(name="Correlation with Revenue")
)

correlations["Channel"] = correlations["Channel"].map(CHANNEL_LABELS)
correlations = correlations.sort_values(
    "Correlation with Revenue",
    ascending=False,
)

st.bar_chart(
    correlations.set_index("Channel")[["Correlation with Revenue"]],
    y_label="Pearson correlation",
)

st.caption(
    "Correlation is descriptive and does not isolate advertising effects "
    "from seasonality, competitor activity, or other factors."
)


# ============================================================
# 8. CHANNEL CONTRIBUTION ANALYSIS
# ============================================================
st.divider()
st.header("Channel Contribution Analysis")

st.warning(
    "Contributions are reconstructed using the project's media coefficients, "
    "geometric adstock, and Hill saturation assumptions. These estimates are "
    "not validated causal contributions."
)
st.info(
    "The channel contributions shown here are the media terms from the "
    "fitted MMM. The model also includes an intercept, monthly seasonality, "
    "and standardized competitor sales. Therefore, total media contribution "
    "does not equal total observed revenue. Contributions are model-derived "
    "estimates and should not be interpreted as validated causal effects."
)

# Compute adstock using the complete chronological dataset before filtering.
full_contribution = pd.DataFrame({"DATE": df["DATE"]})

for channel in MEDIA_COLS:
    adstock_values = calculate_adstock(df[channel].to_numpy())

    saturated_values = hill_response(
        adstock_values,
        HALF_SATURATION[channel],
    )

    full_contribution[channel] = (
        MEDIA_COEFFICIENTS[channel] * saturated_values
    )

contribution_period = full_contribution[
    (full_contribution["DATE"].dt.date >= start_date)
    & (full_contribution["DATE"].dt.date <= end_date)
].copy()

contribution_summary = pd.DataFrame({
    "Channel": [CHANNEL_LABELS[channel] for channel in MEDIA_COLS],
    "Modeled Contribution": [
        contribution_period[channel].sum()
        for channel in MEDIA_COLS
    ],
})

contribution_total = contribution_summary["Modeled Contribution"].sum()

contribution_summary["Contribution Share (%)"] = (
    contribution_summary["Modeled Contribution"] / contribution_total * 100
    if contribution_total > 0
    else 0.0
)

contribution_summary = contribution_summary.sort_values(
    "Modeled Contribution",
    ascending=False,
)

col1, col2 = st.columns(2)

col1.metric("Total Modeled Media Contribution", f"{contribution_total:,.0f}")
col2.metric("Channels Analyzed", len(MEDIA_COLS))

left, right = st.columns([3, 2])

with left:
    st.bar_chart(
        contribution_summary.set_index("Channel")[["Modeled Contribution"]],
        horizontal=True,
    )

with right:
    st.dataframe(
        contribution_summary.round(2),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("Weekly Modeled Contribution")

weekly_contribution = (
    contribution_period[["DATE"] + MEDIA_COLS]
    .rename(columns=CHANNEL_LABELS)
    .set_index("DATE")
)

st.line_chart(
    weekly_contribution,
    y=list(CHANNEL_LABELS.values()),
    x_label="Week",
    y_label="Modeled Contribution",
)


# ============================================================
# 9. INTERACTIVE RESPONSE CURVES
# ============================================================
st.divider()
st.header("Channel Response Curves")

selected_channel = st.selectbox(
    "Select a channel",
    MEDIA_COLS,
    format_func=lambda channel: CHANNEL_LABELS[channel],
)

positive_spend = df.loc[df[selected_channel] > 0, selected_channel]

if positive_spend.empty:
    st.warning("No positive spending observations exist for this channel.")
else:
    spend_cap = float(df[selected_channel].quantile(0.95))

    if spend_cap <= 0:
        spend_cap = float(positive_spend.max())

    spend_grid = np.linspace(0.0, spend_cap, 150)

    response_values = [
        steady_state_contribution(selected_channel, spend)
        for spend in spend_grid
    ]

    response_df = pd.DataFrame({
        "Weekly Spend": spend_grid,
        "Modeled Weekly Contribution": response_values,
    })

    st.line_chart(
        response_df,
        x="Weekly Spend",
        y="Modeled Weekly Contribution",
    )

    reference_spend = st.slider(
        "Inspect weekly spending level",
        min_value=0.0,
        max_value=spend_cap,
        value=float(min(positive_spend.median(), spend_cap)),
        step=max(spend_cap / 100, 0.01),
        format="%.2f",
    )

    current_response = steady_state_contribution(
        selected_channel,
        reference_spend,
    )

    step_size = max(spend_cap * 0.001, 0.01)

    next_response = steady_state_contribution(
        selected_channel,
        reference_spend + step_size,
    )

    marginal_response = (next_response - current_response) / step_size

    col1, col2 = st.columns(2)

    col1.metric(
        "Modeled Weekly Contribution",
        f"{current_response:,.0f}",
    )

    col2.metric(
        "Approx. Marginal Response",
        f"{marginal_response:.3f}",
    )

    st.caption(
        "Marginal response estimates the change in modeled weekly contribution "
        "per additional unit of spend near the selected level. The response "
        "curve assumes constant weekly spending and steady-state adstock."
    )



# ============================================================
# 10. FAIR, FIXED-BUDGET OPTIMIZATION
# ============================================================
st.divider()
st.header("Interactive Budget Optimizer")

st.markdown(
    """
    Select a weekly marketing budget and compare the optimized allocation
    against a feasible baseline allocation using the **same total budget**.
    """
)

st.warning(
    "This is a model-based budget scenario using simulated data and fixed "
    "response parameters. The estimated improvement is not a guarantee of "
    "actual revenue growth or a validated causal effect."
)

# ------------------------------------------------------------
# A. Historical budget and channel spending limits
# ------------------------------------------------------------
historical_average_spend = (
    df[MEDIA_COLS].mean().to_numpy(dtype=float)
)

historical_total_budget = float(historical_average_spend.sum())

channel_upper_bounds = np.array(
    [
        float(df[channel].quantile(0.95))
        for channel in MEDIA_COLS
    ],
    dtype=float,
)

channel_lower_bounds = np.zeros(len(MEDIA_COLS), dtype=float)

if historical_total_budget <= 0:
    st.error("Historical average weekly media spend must be positive.")
    st.stop()

# ------------------------------------------------------------
# B. Select the total weekly budget
# ------------------------------------------------------------
budget_percentage = st.slider(
    "Weekly budget relative to historical average",
    min_value=70,
    max_value=130,
    value=100,
    step=5,
    format="%d%%",
    key="optimizer_budget_percentage",
)

target_budget = (
    historical_total_budget * budget_percentage / 100.0
)

budget_col1, budget_col2 = st.columns(2)

budget_col1.metric(
    "Historical Average Weekly Budget",
    f"{historical_total_budget:,.2f}",
)

budget_col2.metric(
    "Selected Weekly Budget",
    f"{target_budget:,.2f}",
)

st.caption(
    f"Selected budget is {budget_percentage}% of historical average weekly "
    "spend. Each channel's maximum allocation is capped at its historical "
    "95th-percentile weekly spend."
)

# Verify that the selected budget can fit within the channel caps.
if target_budget > channel_upper_bounds.sum() + 1e-6:
    st.error(
        "The selected budget exceeds the combined channel spending caps. "
        "Reduce the budget or review the channel limits."
    )
    st.stop()

# ------------------------------------------------------------
# C. Optimize the allocation
# ------------------------------------------------------------
try:
    baseline_allocation, optimized_allocation = optimize_budget(
        target_budget,
        channel_upper_bounds,
        slope=HILL_SLOPE,
    )

    baseline_contribution = total_steady_state_contribution(
        baseline_allocation,
        slope=HILL_SLOPE,
    )

    optimized_contribution = total_steady_state_contribution(
        optimized_allocation,
        slope=HILL_SLOPE,
    )

except Exception as error:
    st.error(f"Budget optimization failed: {error}")
    st.stop()

# ------------------------------------------------------------
# D. Validate both allocations
# ------------------------------------------------------------
baseline_total = float(np.sum(baseline_allocation))
optimized_total = float(np.sum(optimized_allocation))

if not np.isclose(baseline_total, target_budget, atol=1e-4):
    st.error("The baseline allocation does not satisfy the selected budget.")
    st.stop()

if not np.isclose(optimized_total, target_budget, atol=1e-4):
    st.error("The optimized allocation does not satisfy the selected budget.")
    st.stop()

if (
    np.any(baseline_allocation < channel_lower_bounds - 1e-5)
    or np.any(baseline_allocation > channel_upper_bounds + 1e-5)
):
    st.error("The baseline allocation violates a channel spending limit.")
    st.stop()

if (
    np.any(optimized_allocation < channel_lower_bounds - 1e-5)
    or np.any(optimized_allocation > channel_upper_bounds + 1e-5)
):
    st.error("The optimized allocation violates a channel spending limit.")
    st.stop()

# ------------------------------------------------------------
# E. Calculate the like-for-like improvement
# ------------------------------------------------------------
contribution_change = (
    optimized_contribution - baseline_contribution
)

improvement_pct = (
    contribution_change / baseline_contribution * 100.0
    if baseline_contribution > 0
    else 0.0
)

# ------------------------------------------------------------
# F. Display optimization KPIs
# ------------------------------------------------------------
st.subheader("Optimization Results")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Same-Budget Baseline Contribution",
    f"{baseline_contribution:,.0f}",
)

metric2.metric(
    "Optimized Modeled Contribution",
    f"{optimized_contribution:,.0f}",
)

metric3.metric(
    "Modeled Improvement",
    f"{improvement_pct:+.2f}%",
    delta=f"{contribution_change:+,.0f} contribution units",
)

st.caption(
    "Improvement = (optimized contribution − baseline contribution) "
    "÷ baseline contribution × 100. Both scenarios use the same total "
    "weekly budget, the same Hill slope, and the same channel spending caps."
)

# ------------------------------------------------------------
# G. Build the allocation comparison table
# ------------------------------------------------------------
budget_comparison = pd.DataFrame({
    "Channel": [
        CHANNEL_LABELS[channel]
        for channel in MEDIA_COLS
    ],
    "Baseline Weekly Spend": baseline_allocation,
    "Optimized Weekly Spend": optimized_allocation,
    "Spend Change": optimized_allocation - baseline_allocation,
    "Channel Spending Cap": channel_upper_bounds,
})

budget_comparison["Baseline Budget Share (%)"] = (
    budget_comparison["Baseline Weekly Spend"]
    / target_budget * 100.0
)

budget_comparison["Optimized Budget Share (%)"] = (
    budget_comparison["Optimized Weekly Spend"]
    / target_budget * 100.0
)

st.subheader("Recommended Channel Allocation")

st.dataframe(
    budget_comparison.round(2),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Channel": st.column_config.TextColumn("Channel"),
        "Baseline Weekly Spend": st.column_config.NumberColumn(
            "Baseline Weekly Spend",
            format="%,.2f",
        ),
        "Optimized Weekly Spend": st.column_config.NumberColumn(
            "Optimized Weekly Spend",
            format="%,.2f",
        ),
        "Spend Change": st.column_config.NumberColumn(
            "Spend Change",
            format="%+,.2f",
        ),
        "Channel Spending Cap": st.column_config.NumberColumn(
            "Channel Spending Cap",
            format="%,.2f",
        ),
        "Baseline Budget Share (%)": st.column_config.NumberColumn(
            "Baseline Budget Share",
            format="%.2f%%",
        ),
        "Optimized Budget Share (%)": st.column_config.NumberColumn(
            "Optimized Budget Share",
            format="%.2f%%",
        ),
    },
)

# ------------------------------------------------------------
# H. Compare baseline and optimized allocations visually
# ------------------------------------------------------------
st.subheader("Baseline vs. Optimized Spending")

allocation_chart = budget_comparison.set_index("Channel")[
    [
        "Baseline Weekly Spend",
        "Optimized Weekly Spend",
    ]
]

st.bar_chart(
    allocation_chart,
    y=[
        "Baseline Weekly Spend",
        "Optimized Weekly Spend",
    ],
    x_label="Channel",
    y_label="Weekly Spend",
)

# ------------------------------------------------------------
# I. Budget constraint verification
# ------------------------------------------------------------
with st.expander("View budget validation details"):
    validation_df = pd.DataFrame({
        "Check": [
            "Selected weekly budget",
            "Baseline allocation total",
            "Optimized allocation total",
            "Baseline budget difference",
            "Optimized budget difference",
        ],
        "Value": [
            target_budget,
            baseline_total,
            optimized_total,
            baseline_total - target_budget,
            optimized_total - target_budget,
        ],
    })

    st.dataframe(
        validation_df.round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        "Both allocations satisfy the fixed-budget constraint and "
        "the configured channel spending limits."
    )


# ============================================================
# 11. SENSITIVITY ANALYSIS AT THE SAME SELECTED BUDGET
# ============================================================
st.divider()
st.header("Optimization Sensitivity Analysis")

st.markdown(
    """
    Re-optimize the selected budget under different Hill saturation slopes.
    Each scenario compares its optimized allocation with a baseline using
    the **same budget and the same Hill slope**.
    """
)

st.info(
    "This tests sensitivity to one modeling assumption. It is not a confidence "
    "interval or independent validation of the response curves."
)

sensitivity_slopes = [0.8, 1.0, 1.2]
sensitivity_summary = []
sensitivity_allocations = []

for slope_value in sensitivity_slopes:
    try:
        scenario_baseline, scenario_optimized = optimize_budget(
            target_budget,
            channel_upper_bounds,
            slope=slope_value,
        )

        scenario_baseline_contribution = total_steady_state_contribution(
            scenario_baseline,
            slope=slope_value,
        )

        scenario_optimized_contribution = total_steady_state_contribution(
            scenario_optimized,
            slope=slope_value,
        )

        scenario_improvement = (
            (
                scenario_optimized_contribution
                - scenario_baseline_contribution
            )
            / scenario_baseline_contribution * 100
            if scenario_baseline_contribution > 0
            else np.nan
        )

        sensitivity_summary.append({
            "Hill Slope": slope_value,
            "Baseline Contribution": scenario_baseline_contribution,
            "Optimized Contribution": scenario_optimized_contribution,
            "Modeled Improvement (%)": scenario_improvement,
        })

        allocation_row = {"Hill Slope": slope_value}

        allocation_row.update({
            CHANNEL_LABELS[channel]: spend
            for channel, spend in zip(MEDIA_COLS, scenario_optimized)
        })

        sensitivity_allocations.append(allocation_row)

    except Exception as error:
        st.warning(
            f"Hill slope {slope_value} could not be optimized: {error}"
        )


if sensitivity_summary:
    sensitivity_df = pd.DataFrame(sensitivity_summary)

    st.subheader("Contribution Improvement by Hill Slope")

    st.dataframe(
        sensitivity_df.round(2),
        use_container_width=True,
        hide_index=True,
    )

    # Display all tested Hill-slope scenarios explicitly.
    st.subheader("Modeled Improvement by Hill Slope")

    sensitivity_chart = sensitivity_df[
        ["Hill Slope", "Modeled Improvement (%)"]
    ].copy()

    sensitivity_chart["Hill Slope"] = sensitivity_chart[
        "Hill Slope"
    ].map(lambda value: f"{value:.1f}")

    st.bar_chart(
        sensitivity_chart.set_index("Hill Slope"),
        y="Modeled Improvement (%)",
        x_label="Hill Slope",
        y_label="Modeled Improvement (%)",
    )

    st.caption(
        "Each bar represents the optimized improvement relative to the "
        "same-budget baseline under that Hill-slope assumption."
    )

    allocation_sensitivity_df = pd.DataFrame(sensitivity_allocations)

    st.subheader("Optimized Allocation Across Scenarios")

    # st.dataframe(
    #     allocation_sensitivity_df.round(2),
    #     use_container_width=True,
    #     hide_index=True,
    # )
    st.dataframe(
    sensitivity_df.round(2),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Hill Slope": st.column_config.NumberColumn(
            "Hill Slope",
            format="%.1f",
        ),
        "Baseline Contribution": st.column_config.NumberColumn(
            "Baseline Contribution",
            format="%.2f",
        ),
        "Optimized Contribution": st.column_config.NumberColumn(
            "Optimized Contribution",
            format="%.2f",
        ),
        "Modeled Improvement (%)": st.column_config.NumberColumn(
            "Improvement (%)",
            format="%.2f%%",
        ),
    },
)

    allocation_chart = allocation_sensitivity_df.set_index("Hill Slope")

    st.bar_chart(
        allocation_chart,
        y=list(CHANNEL_LABELS.values()),
        x_label="Hill Slope",
        y_label="Optimized Weekly Spend",
    )

    st.caption(
        "All scenarios use the selected weekly budget and the same spending "
        "caps. The modeled improvement is calculated against the corresponding "
        "same-budget baseline under each Hill-slope assumption."
    )
else:
    st.error("No sensitivity scenarios completed successfully.")


# ============================================================
# 12. DATA EXPLORER
# ============================================================
st.divider()

with st.expander("Explore the underlying data"):
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows in Selected Period", len(filtered_df))
    col2.metric("Missing Values", int(filtered_df.isna().sum().sum()))
    col3.metric(
        "Duplicate Dates",
        int(filtered_df["DATE"].duplicated().sum()),
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 13. FOOTER
# ============================================================
st.divider()

st.caption(
    "CPG Marketing Mix Intelligence | Python • Pandas • Streamlit • SciPy"
)
