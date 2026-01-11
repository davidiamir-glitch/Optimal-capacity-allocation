from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# --- Model parameters ---
INCOME_PER_UNIT = 250
COST_PER_CAPACITY_UNIT = 100
DAYS = 30
DEMAND_MIN, DEMAND_MAX = 50, 250


@dataclass
class DayResult:
    day: int
    demand: int
    capacity: int
    processed: int
    income: int
    cost: int
    profit: int


def simulate_one_run_fixed_capacity(capacity: int, seed: Optional[int]) -> List[DayResult]:
    rng = random.Random(seed)
    results: List[DayResult] = []

    for day in range(1, DAYS + 1):
        demand = rng.randint(DEMAND_MIN, DEMAND_MAX)
        processed = min(demand, capacity)

        income = processed * INCOME_PER_UNIT
        cost = capacity * COST_PER_CAPACITY_UNIT
        profit = income - cost

        results.append(DayResult(day, demand, capacity, processed, income, cost, profit))

    return results


def summarize_run(results: List[DayResult]) -> Dict[str, float]:
    total_demand = sum(r.demand for r in results)
    avg_daily_demand = total_demand / DAYS
    avg_daily_capacity = sum(r.capacity for r in results) / DAYS

    total_cost = sum(r.cost for r in results)
    total_income = sum(r.income for r in results)
    total_profit = sum(r.profit for r in results)

    return {
        "total_demand": float(total_demand),
        "avg_daily_demand": float(avg_daily_demand),
        "avg_daily_capacity": float(avg_daily_capacity),
        "total_cost": float(total_cost),
        "total_income": float(total_income),
        "total_profit": float(total_profit),
    }


def run_batch(capacity: int, runs: int, base_seed: Optional[int]) -> pd.DataFrame:
    seed_rng = random.Random(base_seed) if base_seed is not None else None
    rows = []

    for _ in range(runs):
        run_seed = seed_rng.randint(0, 2_000_000_000) if seed_rng is not None else None
        results = simulate_one_run_fixed_capacity(capacity, run_seed)
        rows.append(summarize_run(results))

    return pd.DataFrame(rows)


def format_currency(x: float) -> str:
    return f"${x:,.2f}"


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Optimal Resource Allocation Simulator", layout="wide")
st.title("Optimal Resource Allocation Simulator")
st.caption(
    f"10-day simulation • Demand per day: {DEMAND_MIN}..{DEMAND_MAX} • "
    f"Income: ${INCOME_PER_UNIT}/unit • Cost: ${COST_PER_CAPACITY_UNIT}/capacity unit/day"
)

# --- Session memory for history ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "capacity",
            "runs",
            "seed_mode",
            "base_seed",
            "avg_total_income",
            "avg_total_cost",
            "avg_total_profit",
            "avg_total_demand",
            "avg_daily_demand",
        ]
    )
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0


with st.sidebar:
    st.header("Inputs")

    capacity = st.slider("Fixed daily capacity", min_value=0, max_value=300, value=150, step=10)
    runs = st.number_input("Number of runs (per click)", min_value=1, max_value=200000, value=200, step=50)

    seed_mode = st.radio("Randomness", ["Random each time", "Reproducible (seed)"], index=0)
    base_seed: Optional[int] = None
    if seed_mode == "Reproducible (seed)":
        base_seed = int(st.number_input("Base seed", min_value=0, max_value=2_000_000_000, value=7, step=1))

    show_sample = st.toggle("Show one sample 10-day run", value=False)

    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button("Run", type="primary", use_container_width=True)
    with col_reset:
        reset_button = st.button("Reset memory", use_container_width=True)

    if reset_button:
        st.session_state.history = st.session_state.history.iloc[0:0].copy()
        st.session_state.run_counter = 0
        st.success("Memory reset. History cleared.")
        st.stop()


# --- Run simulation when clicked ---
if run_button:
    df = run_batch(capacity=capacity, runs=int(runs), base_seed=base_seed)

    avg = df.mean(numeric_only=True)

    # Store this click/run into history
    st.session_state.run_counter += 1
    new_row = {
        "run_id": st.session_state.run_counter,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "capacity": capacity,
        "runs": int(runs),
        "seed_mode": seed_mode,
        "base_seed": base_seed if base_seed is not None else "",
        "avg_total_income": float(avg["total_income"]),
        "avg_total_cost": float(avg["total_cost"]),
        "avg_total_profit": float(avg["total_profit"]),
        "avg_total_demand": float(avg["total_demand"]),
        "avg_daily_demand": float(avg["avg_daily_demand"]),
    }
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    # --- Current run KPIs ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Run ID", f"{st.session_state.run_counter}")
    k2.metric("Capacity", f"{capacity}")
    k3.metric("Runs", f"{int(runs):,}")
    k4.metric("Avg income (10d)", format_currency(new_row["avg_total_income"]))
    k5.metric("Avg cost (10d)", format_currency(new_row["avg_total_cost"]))
    k6.metric("Avg profit (10d)", format_currency(new_row["avg_total_profit"]))

    st.divider()

    st.subheader("This run: aggregated end-of-period report (averaged across runs)")
    report = pd.DataFrame(
        [
            ["Total demand (10 days)", f"{new_row['avg_total_demand']:.2f}"],
            ["**Average daily demand**",f"**{new_row['avg_daily_demand']:.2f}**",],
            ["**Average daily capacity**",f"**{capacity:.2f}**",],
            ["Total cost (10 days)", format_currency(new_row["avg_total_cost"])],
            ["Total income (10 days)", format_currency(new_row["avg_total_income"])],
            ["Total profit (10 days)", format_currency(new_row["avg_total_profit"])],
        ],
        columns=["Metric", "Value"],
    )
    st.table(report)


# --- History + comparison charts ---
hist = st.session_state.history.copy()

st.subheader("Saved run history")
if hist.empty:
    st.info("No runs saved yet. Choose capacity + runs, then click **Run**.")
    st.stop()

st.dataframe(hist, use_container_width=True, hide_index=True)

# Download CSV
csv_bytes = hist.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download history as CSV",
    data=csv_bytes,
    file_name="processing_center_history.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Comparison: Income / Cost / Profit vs Capacity (across saved runs)")

# Scatter: capacity vs values (each point = one click/run)
c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Avg Income (10d) vs Capacity**")
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["avg_total_income"], s=40)
    plt.xlabel("Capacity")
    plt.ylabel("Avg total income (10 days)")
    st.pyplot(fig, clear_figure=True)

with c2:
    st.write("**Avg Cost (10d) vs Capacity**")
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["avg_total_cost"], s=40)
    plt.xlabel("Capacity")
    plt.ylabel("Avg total cost (10 days)")
    st.pyplot(fig, clear_figure=True)

with c3:
    st.write("**Avg Profit (10d) vs Capacity**")
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["avg_total_profit"], s=40)
    plt.xlabel("Capacity")
    plt.ylabel("Avg total profit (10 days)")
    st.pyplot(fig, clear_figure=True)

st.divider()
st.subheader("Trends across run order (each click/run over time)")

t1, t2 = st.columns(2)

with t1:
    st.write("**Income / Cost / Profit per run (10 days)**")
    fig = plt.figure()
    plt.plot(hist["run_id"], hist["avg_total_income"], marker="o")
    plt.plot(hist["run_id"], hist["avg_total_cost"], marker="o")
    plt.plot(hist["run_id"], hist["avg_total_profit"], marker="o")
    plt.xlabel("Run ID")
    plt.ylabel("Value (10-day totals, averaged over runs)")
    plt.legend(["Income", "Cost", "Profit"])
    st.pyplot(fig, clear_figure=True)

with t2:
    st.write("**Capacity chosen per run**")
    fig = plt.figure()
    plt.plot(hist["run_id"], hist["capacity"], marker="o")
    plt.xlabel("Run ID")
    plt.ylabel("Capacity")
    st.pyplot(fig, clear_figure=True)


# --- Optional: sample run details (for the last chosen capacity) ---
# Only show when user toggles AND we have at least one run stored.
if show_sample:
    st.divider()
    st.subheader("Sample single 10-day run (last saved capacity)")

    last_capacity = int(hist.iloc[-1]["capacity"])
    # Deterministic sample if the last saved run had a seed; otherwise random
    last_seed = hist.iloc[-1]["base_seed"]
    sample_seed = None
    try:
from __future__ import annotations
        sample_seed = int(last_seed) + 999 if str(last_seed).strip() != "" else None
    except ValueError:
        sample_seed = None

    sample = simulate_one_run_fixed_capacity(capacity=last_capacity, seed=sample_seed)
    sample_df = pd.DataFrame([r.__dict__ for r in sample])
    sample_df["cum_profit"] = sample_df["profit"].cumsum()

    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    a, b = st.columns(2)

    with a:
        st.write("**Daily Demand / Capacity / Processed**")
        fig = plt.figure()
        plt.plot(sample_df["day"], sample_df["demand"], marker="o")
        plt.plot(sample_df["day"], sample_df["capacity"], marker="o")
        plt.plot(sample_df["day"], sample_df["processed"], marker="o")
        plt.xlabel("Day")
        plt.ylabel("Units")
        plt.legend(["Demand", "Capacity", "Processed"])
        st.pyplot(fig, clear_figure=True)

    with b:
        st.write("**Cumulative Profit**")
        fig = plt.figure()
        plt.plot(sample_df["day"], sample_df["cum_profit"], marker="o")
        plt.xlabel("Day")
        plt.ylabel("Cumulative profit")
        st.pyplot(fig, clear_figure=True)
import random
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# --- Model parameters ---
INCOME_PER_UNIT = 100
COST_PER_CAPACITY_UNIT = 25
DAYS = 10
DEMAND_MIN, DEMAND_MAX = 1, 8


@dataclass
class DayResult:
    day: int
    demand: int
    capacity: int
    processed: int
    income: int
    cost: int
    profit: int


def simulate_one_run_fixed_capacity(capacity: int, seed: Optional[int]) -> List[DayResult]:
    rng = random.Random(seed)
    results: List[DayResult] = []

    for day in range(1, DAYS + 1):
        demand = rng.randint(DEMAND_MIN, DEMAND_MAX)
        processed = min(demand, capacity)

        income = processed * INCOME_PER_UNIT
        cost = capacity * COST_PER_CAPACITY_UNIT
        profit = income - cost

        results.append(DayResult(day, demand, capacity, processed, income, cost, profit))

    return results


def summarize_run(results: List[DayResult]) -> Dict[str, float]:
    total_demand = sum(r.demand for r in results)
    avg_daily_demand = total_demand / DAYS
    avg_daily_capacity = sum(r.capacity for r in results) / DAYS

    total_cost = sum(r.cost for r in results)
    total_income = sum(r.income for r in results)
    total_profit = sum(r.profit for r in results)

    return {
        "total_demand": float(total_demand),
        "avg_daily_demand": float(avg_daily_demand),
        "avg_daily_capacity": float(avg_daily_capacity),
        "total_cost": float(total_cost),
        "total_income": float(total_income),
        "total_profit": float(total_profit),
    }


def run_batch(capacity: int, runs: int, base_seed: Optional[int]) -> pd.DataFrame:
    seed_rng = random.Random(base_seed) if base_seed is not None else None
    rows = []

    for _ in range(runs):
        run_seed = seed_rng.randint(0, 2_000_000_000) if seed_rng is not None else None
        results = simulate_one_run_fixed_capacity(capacity, run_seed)
        rows.append(summarize_run(results))

    return pd.DataFrame(rows)


def format_currency(x: float) -> str:
    return f"${x:,.2f}"


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Optimal Resource Allocation Simulator", layout="wide")
st.title("Optimal Resource Allocation Simulator")
st.caption(
    f"10-day simulation • Demand per day: {DEMAND_MIN}..{DEMAND_MAX} • "
    f"Income: ${INCOME_PER_UNIT}/unit • Cost: ${COST_PER_CAPACITY_UNIT}/capacity unit/day"
)

# --- Session memory for history ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "capacity",
            "runs",
            "seed_mode",
            "base_seed",
            "avg_total_income",
            "avg_total_cost",
            "avg_total_profit",
            "avg_total_demand",
            "avg_daily_demand",
        ]
    )
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0


with st.sidebar:
    st.header("Inputs")

    capacity = st.slider("Fixed daily capacity", min_value=0, max_value=30, value=6, step=1)
    runs = st.number_input("Number of runs (per click)", min_value=1, max_value=200000, value=200, step=50)

    seed_mode = st.radio("Randomness", ["Random each time", "Reproducible (seed)"], index=0)
    base_seed: Optional[int] = None
    if seed_mode == "Reproducible (seed)":
        base_seed = int(st.number_input("Base seed", min_value=0, max_value=2_000_000_000, value=7, step=1))

    show_sample = st.toggle("Show one sample 10-day run", value=False)

    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button("Run", type="primary", use_container_width=True)
    with col_reset:
        reset_button = st.button("Reset memory", use_container_width=True)

    if reset_button:
        st.session_state.history = st.session_state.history.iloc[0:0].copy()
        st.session_state.run_counter = 0
        st.success("Memory reset. History cleared.")
        st.stop()


# --- Run simulation when clicked ---
if run_button:
    df = run_batch(capacity=capacity, runs=int(runs), base_seed=base_seed)

    avg = df.mean(numeric_only=True)

    # Store this click/run into history
    st.session_state.run_counter += 1
    new_row = {
        "run_id": st.session_state.run_counter,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "capacity": capacity,
        "runs": int(runs),
        "seed_mode": seed_mode,
        "base_seed": base_seed if base_seed is not None else "",
        "avg_total_income": float(avg["total_income"]),
        "avg_total_cost": float(avg["total_cost"]),
        "avg_total_profit": float(avg["total_profit"]),
        "avg_total_demand": float(avg["total_demand"]),
        "avg_daily_demand": float(avg["avg_daily_demand"]),
    }
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    # --- Current run KPIs ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Run ID", f"{st.session_state.run_counter}")
    k2.metric("Capacity", f"{capacity}")
    k3.metric("Runs", f"{int(runs):,}")
    k4.metric("Avg income (10d)", format_currency(new_row["avg_total_income"]))
    k5.metric("Avg cost (10d)", format_currency(new_row["avg_total_cost"]))
    k6.metric("Avg profit (10d)", format_currency(new_row["avg_total_profit"]))

    st.divider()

    st.subheader("This run: aggregated end-of-period report (averaged across runs)")
    report = pd.DataFrame(
        [
            ["Total demand (10 days)", f"{new_row['avg_total_demand']:.2f}"],
            ["**Average daily demand**",f"**{new_row['avg_daily_demand']:.2f}**",],
            ["**Average daily capacity**",f"**{capacity:.2f}**",],
            ["Total cost (10 days)", format_currency(new_row["avg_total_cost"])],
            ["Total income (10 days)", format_currency(new_row["avg_total_income"])],
            ["Total profit (10 days)", format_currency(new_row["avg_total_profit"])],
        ],
        columns=["Metric", "Value"],
    )
    st.table(report)


# --- History + comparison charts ---
hist = st.session_state.history.copy()

st.subheader("Saved run history")
if hist.empty:
    st.info("No runs saved yet. Choose capacity + runs, then click **Run**.")
    st.stop()

st.dataframe(hist, use_container_width=True, hide_index=True)

# Download CSV
csv_bytes = hist.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download history as CSV",
    data=csv_bytes,
    file_name="processing_center_history.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Comparison: Income / Cost / Profit vs Capacity (across saved runs)")

# Scatter: capacity vs values (each point = one click/run)
c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Avg Income (10d) vs Capacity**")
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["avg_total_income"], s=40)
    plt.xlabel("Capacity")
    plt.ylabel("Avg total income (10 days)")
    st.pyplot(fig, clear_figure=True)

with c2:
    st.write("**Avg Cost (10d) vs Capacity**")
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["avg_total_cost"], s=40)
    plt.xlabel("Capacity")
    plt.ylabel("Avg total cost (10 days)")
    st.pyplot(fig, clear_figure=True)

with c3:
    st.write("**Avg Profit (10d) vs Capacity**")
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["avg_total_profit"], s=40)
    plt.xlabel("Capacity")
    plt.ylabel("Avg total profit (10 days)")
    st.pyplot(fig, clear_figure=True)

st.divider()
st.subheader("Trends across run order (each click/run over time)")

t1, t2 = st.columns(2)

with t1:
    st.write("**Income / Cost / Profit per run (10 days)**")
    fig = plt.figure()
    plt.plot(hist["run_id"], hist["avg_total_income"], marker="o")
    plt.plot(hist["run_id"], hist["avg_total_cost"], marker="o")
    plt.plot(hist["run_id"], hist["avg_total_profit"], marker="o")
    plt.xlabel("Run ID")
    plt.ylabel("Value (10-day totals, averaged over runs)")
    plt.legend(["Income", "Cost", "Profit"])
    st.pyplot(fig, clear_figure=True)

with t2:
    st.write("**Capacity chosen per run**")
    fig = plt.figure()
    plt.plot(hist["run_id"], hist["capacity"], marker="o")
    plt.xlabel("Run ID")
    plt.ylabel("Capacity")
    st.pyplot(fig, clear_figure=True)


# --- Optional: sample run details (for the last chosen capacity) ---
# Only show when user toggles AND we have at least one run stored.
if show_sample:
    st.divider()
    st.subheader("Sample single 10-day run (last saved capacity)")

    last_capacity = int(hist.iloc[-1]["capacity"])
    # Deterministic sample if the last saved run had a seed; otherwise random
    last_seed = hist.iloc[-1]["base_seed"]
    sample_seed = None
    try:
        sample_seed = int(last_seed) + 999 if str(last_seed).strip() != "" else None
    except ValueError:
        sample_seed = None

    sample = simulate_one_run_fixed_capacity(capacity=last_capacity, seed=sample_seed)
    sample_df = pd.DataFrame([r.__dict__ for r in sample])
    sample_df["cum_profit"] = sample_df["profit"].cumsum()

    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    a, b = st.columns(2)

    with a:
        st.write("**Daily Demand / Capacity / Processed**")
        fig = plt.figure()
        plt.plot(sample_df["day"], sample_df["demand"], marker="o")
        plt.plot(sample_df["day"], sample_df["capacity"], marker="o")
        plt.plot(sample_df["day"], sample_df["processed"], marker="o")
        plt.xlabel("Day")
        plt.ylabel("Units")
        plt.legend(["Demand", "Capacity", "Processed"])
        st.pyplot(fig, clear_figure=True)

    with b:
        st.write("**Cumulative Profit**")
        fig = plt.figure()
        plt.plot(sample_df["day"], sample_df["cum_profit"], marker="o")
        plt.xlabel("Day")
        plt.ylabel("Cumulative profit")
        st.pyplot(fig, clear_figure=True)
