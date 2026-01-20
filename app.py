import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# %% ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Seller Scheduling Dashboard", layout="wide")

# Optimized CSS
st.markdown("""
    <style>
    .stMetric { border: 1px solid #E0E0E0; padding: 10px; border_radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# %% ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    # Only load necessary columns to save memory and speed up processing
    order_cols = ["order_id", "order_approved_at", "order_estimated_delivery_date"]
    item_cols = ["order_id", "seller_id", "order_item_id"]
    
    orders = pd.read_csv("olist_orders_dataset.csv", usecols=order_cols,
                         parse_dates=["order_approved_at", "order_estimated_delivery_date"])
    
    order_items = pd.read_csv("olist_order_items_dataset.csv", usecols=item_cols)
    
    # Pre-filter: drop orders without approval times immediately
    orders = orders.dropna(subset=["order_approved_at"])
    return orders, order_items

orders, order_items = load_data()

# %% ---------------- SCHEDULING LOGIC ----------------
@st.cache_data
def get_top_sellers(order_items, top_n=50):
    return order_items["seller_id"].value_counts().head(top_n).reset_index(name="num_orders")

@st.cache_data
def create_scheduling_table(seller_id, _order_items, _orders):
    # Filtering first, then merging is much faster
    seller_items = _order_items[_order_items["seller_id"] == seller_id]
    
    # Vectorized aggregation
    df = (seller_items.merge(_orders, on="order_id")
          .groupby("order_id")
          .agg(
              arrival_time=("order_approved_at", "first"),
              due_date=("order_estimated_delivery_date", "first"),
              processing_time=("order_item_id", "count")
          )
          .sort_values("arrival_time") # Pre-sort for the simulation
          .reset_index())
    return df

def simulate_schedule_and_metrics(df):
    """Vectorized simulation for speed."""
    # Convert to numpy for maximum speed
    arrivals = df["arrival_time"].values
    process_durations = pd.to_timedelta(df["processing_time"].values, unit="D").values
    dues = df["due_date"].values

    n = len(df)
    starts = np.empty(n, dtype='datetime64[ns]')
    finishes = np.empty(n, dtype='datetime64[ns]')

    # The first job
    starts[0] = arrivals[0]
    finishes[0] = starts[0] + process_durations[0]

    # This loop is now very tight and only handles the 'finish-to-start' dependency
    current_finish = finishes[0]
    for i in range(1, n):
        starts[i] = max(current_finish, arrivals[i])
        finishes[i] = starts[i] + process_durations[i]
        current_finish = finishes[i]

    # Vectorized metrics
    waiting_times = (starts - arrivals).astype('timedelta64[D]').astype(int)
    tardiness = np.maximum(0, (finishes - dues).astype('timedelta64[D]').astype(int))

    return {
        "Average waiting time": np.mean(waiting_times),
        "Average tardiness": np.mean(tardiness),
        "Maximum tardiness": np.max(tardiness),
        "Late job %": (tardiness > 0).mean() * 100,
        "Makespan (days)": (finishes.max() - arrivals.min()) / np.timedelta64(1, 'D')
    }

@st.cache_data
def simulate_cached(schedule, rule):
    # Optimization: Only sort, the simulation itself is very fast now
    if rule == "SPT":
        schedule = schedule.sort_values("processing_time")
    elif rule == "EDD":
        schedule = schedule.sort_values("due_date")
    # FCFS is already the default sort from create_scheduling_table
    
    return simulate_schedule_and_metrics(schedule)

# %% ---------------- UI COMPONENTS ----------------
st.title("📦 Seller Order Sequencing Dashboard")

# Time conversion factor
time_unit = st.selectbox("Select time unit", ["days", "hours"])
multiplier = 24 if time_unit == "hours" else 1

# Seller Selection
top_sellers_df = get_top_sellers(order_items)
seller_id = st.selectbox("Select a seller", options=[""] + top_sellers_df["seller_id"].tolist())

if not seller_id:
    st.info("Select a seller to begin.")
    st.stop()

# %% ---------------- EXECUTION ----------------
schedule = create_scheduling_table(seller_id, order_items, orders)

if len(schedule) < 5:
    st.warning("Too few orders for a meaningful simulation.")
    st.stop()

# Run simulations
fcfs = simulate_cached(schedule, "FCFS")
spt = simulate_cached(schedule, "SPT")
edd = simulate_cached(schedule, "EDD")

# Display Metrics
st.subheader(f"📊 Performance Overview ({time_unit})")
cols = st.columns(3)
rules = [("FCFS", fcfs), ("SPT", spt), ("EDD", edd)]

for i, (name, metrics) in enumerate(rules):
    with cols[i]:
        st.metric(f"{name} Avg Waiting", f"{metrics['Average waiting time'] * multiplier:.2f}")
        st.metric(f"{name} Avg Tardiness", f"{metrics['Average tardiness'] * multiplier:.2f}")

# %% ---------------- VISUALIZATION ----------------
results_df = pd.DataFrame({
    "Metric": ["Avg Waiting", "Avg Tardiness", "Max Tardiness"],
    "FCFS": [fcfs["Average waiting time"], fcfs["Average tardiness"], fcfs["Maximum tardiness"]],
    "SPT": [spt["Average waiting time"], spt["Average tardiness"], spt["Maximum tardiness"]],
    "EDD": [edd["Average waiting time"], edd["Average tardiness"], edd["Maximum tardiness"]]
})

plot_df = results_df.melt(id_vars="Metric", var_name="Policy", value_name="Value")
plot_df["Value"] *= multiplier

fig = px.bar(plot_df, x="Metric", y="Value", color="Policy", barmode="group",
             text_auto=".2f", template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)