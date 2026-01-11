from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =========================
# Model parameters
# =========================
DAYS = 30
DEMAND_MIN, DEMAND_MAX = 50, 250
INCOME_PER_UNIT = 350
COST_PER_CAPACITY_UNIT = 100


# =========================
# Data structures
# =========================
@dataclass
class DayResult:
    day: int
    demand: int
    capacity: int
    processed: int
    income: int
    cost: int
    profit: int


# =========================
# Simulation logic
# =========================
def simulate_one_run(capacity: int, seed: Optional[int]) -> List[DayResult]:
    rng = random.Random(seed)
    results: List[DayResult] = []

    for day in range(1, DAYS + 1):
        demand = rng.randint(DEMAND_MIN, DEMAND_MAX)
        processed = min(demand, capacity)

        income = processed * INCOME_PER_UNIT
        cost = capacity * COST_PER_CAPACITY_UNIT
        profit = income - cost

        results.append(
            DayResult(
                day=day,
                demand=demand,
                capacity=capacity,
                processed=processed,
                income=income,
                cost=cost,
                profit=profit,
            )
        )

    return results


def summarize_run(results: List[DayResult]) -> dict:
    return {
        "total_demand": sum(r.demand for r in results),
        "avg_daily_demand": sum(r.demand for r in results) / DAYS,
        "avg_daily_capacity": sum(r.capacity for r in results) / DAYS,
        "total_income": sum(r.income for r in results),
        "total_cost": sum(r.cost for r in results),
        "total_profit": sum(r.profit for r in results),
    }


def run_batch(capacity: int, runs: int, seed: Optional[int]) -> pd.DataFrame:
    rng = random.Random(seed) if seed is not None else None
    rows = []

    for _ in range(runs):
        s = rng.randint(0, 2_000_000_000) if rng else None
        results = simulate_one_run(capacity, s)
        rows.append(summarize_run(results))

    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"${x:,.0f}"


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Optimal Resource Allocation Simulator", layout="wide")
st.title("Optimal Resource Allocation Simulator")

st.caption(
    "30-day simulation | Demand 50–250 units/day | "
    "Income $350/unit | Cost $100/capacity unit/day"
)

# -------------------------
# Session memory
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "capacity",
            "avg_daily_demand",
            "avg_daily_capacity",
            "total_income",
            "total_cost",
            "total_profit",
        ]
    )

if "run_id" not in st.session_state:
    st.session_state.run_id = 0


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("Inputs")

    capacity = st.slider(
        "Fixed daily capacity",
        min_value=0,
        max_value=500,
        value=150,
        step=10,
    )

    runs = st.number_input(
        "Number of runs",
        min_value=1,
        max_value=100000,
        value=200,
        step=50,
    )

    seed_mode = st.radio("Randomness", ["Random", "Reproducible"], index=0)
    base_seed = None
    if seed_mode == "Reproducible":
        base_seed = int(st.number_input("Base seed", value=7))

    col1, col2 = st.columns(2)
    run_button = col1.button("Run", type="primary", use_container_width=True)
    reset_button = col2.button("Reset memory", use_container_width=True)

    if reset_button:
        st.session_state.history = st.session_state.history.iloc[0:0]
        st.session_state.run_id = 0
        st.success("History cleared.")
        st.stop()


# -------------------------
# Run simulation
# -------------------------
if run_button:
    df = run_batch(capacity, int(runs), base_seed)
    avg = df.mean()

    st.session_state.run_id += 1
    st.session_state.history.loc[len(st.session_state.history)] = {
        "run_id": st.session_state.run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "capacity": capacity,
        "avg_daily_demand": avg["avg_daily_demand"],
        "avg_daily_capacity": capacity,
        "total_income": avg["total_income"],
        "total_cost": avg["total_cost"],
        "total_profit": avg["total_profit"],
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Income (30 days)", fmt(avg["total_income"]))
    c2.metric("Avg Cost (30 days)", fmt(avg["total_cost"]))
    c3.metric("Avg Profit (30 days)", fmt(avg["total_profit"]))

    st.markdown(
        f"""
        **Average daily demand:** **{avg['avg_daily_demand']:.2f}**  
        **Average daily capacity:** **{capacity:.2f}**
        """
    )


# -------------------------
# History & charts
# -------------------------
hist = st.session_state.history

if hist.empty:
    st.info("Run the simulator to see results.")
    st.stop()

st.subheader("Run History")
st.dataframe(hist, use_container_width=True, hide_index=True)

st.subheader("Comparison vs Capacity")

c1, c2, c3 = st.columns(3)

with c1:
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["total_income"])
    plt.xlabel("Capacity")
    plt.ylabel("Avg Income (30 days)")
    st.pyplot(fig, clear_figure=True)

with c2:
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["total_cost"])
    plt.xlabel("Capacity")
    plt.ylabel("Avg Cost (30 days)")
    st.pyplot(fig, clear_figure=True)

with c3:
    fig = plt.figure()
    plt.scatter(hist["capacity"], hist["total_profit"])
    plt.xlabel("Capacity")
    plt.ylabel("Avg Profit (30 days)")
    st.pyplot(fig, clear_figure=True)

st.subheader("Profit Trend Across Runs")
fig = plt.figure()
plt.plot(hist["run_id"], hist["total_profit"], marker="o")
plt.xlabel("Run ID")
plt.ylabel("Avg Profit (10 days)")
st.pyplot(fig, clear_figure=True)
