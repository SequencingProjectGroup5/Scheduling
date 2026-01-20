# %%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# %% ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Seller Scheduling Dashboard",
    layout="wide"
)

# %% ---------------- WHITE BACKGROUND STYLE ----------------
st.markdown(
    """
    <style>
    .stApp {background-color: #FFFFFF;}
    section[data-testid="stSidebar"] {background-color: #FFFFFF;}
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# %% ---------------- TITLE ----------------
st.title("📦 Seller Order Sequencing Dashboard")

# %% ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    orders = pd.read_csv(
        "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date"
        ]
    )
    order_items = pd.read_csv(
        "olist_order_items_dataset.csv",
        parse_dates=["shipping_limit_date"]
    )
    orders = orders[orders["order_approved_at"].notna()]
    order_items = order_items[order_items["order_id"].isin(orders["order_id"])]
    return orders, order_items

orders, order_items = load_data()

# %% ---------------- TOP SELLERS ----------------
@st.cache_data
def get_top_sellers(order_items, top_n=50):
    return (
        order_items
        .groupby("seller_id")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="num_orders")
    )

top_sellers_df = get_top_sellers(order_items, top_n=50)

# %% ---------------- SCHEDULING TABLE ----------------
@st.cache_data
def create_scheduling_table_cached(seller_id):
    seller_items = order_items[order_items["seller_id"] == seller_id]
    return (
        seller_items
        .merge(orders, on="order_id")
        .groupby("order_id")
        .agg(
            arrival_time=("order_approved_at", "first"),
            due_date=("order_estimated_delivery_date", "first"),
            processing_time=("order_item_id", "count")
        )
        .reset_index()
    )

# %% ---------------- SIMULATION ----------------
def simulate_schedule_and_metrics(df):
    df = df.copy()
    df["start_time"] = pd.NaT
    df["completion_time"] = pd.NaT
    df["waiting_time"] = 0
    df["tardiness"] = 0

    current_time = df.iloc[0]["arrival_time"]

    for i in range(len(df)):
        arrival = df.iloc[i]["arrival_time"]
        processing = df.iloc[i]["processing_time"]
        due = df.iloc[i]["due_date"]

        start = max(current_time, arrival)
        finish = start + pd.to_timedelta(processing, unit="D")

        df.loc[df.index[i], ["start_time", "completion_time"]] = [start, finish]
        df.loc[df.index[i], "waiting_time"] = (start - arrival).days
        df.loc[df.index[i], "tardiness"] = max(0, (finish - due).days)

        current_time = finish

    metrics = {
        "Average waiting time": df["waiting_time"].mean(),
        "Average tardiness": df["tardiness"].mean(),
        "Maximum tardiness": df["tardiness"].max(),
        "Late job %": (df["tardiness"] > 0).mean() * 100,
        "Makespan (days)": (
            df["completion_time"].max() - df["arrival_time"].min()
        ).days
    }

    return metrics

# %% ---------------- SEQUENCING RULES ----------------
def FCFS(df):
    return df.sort_values("arrival_time")

def SPT(df):
    return df.sort_values("processing_time")

def EDD(df):
    return df.sort_values("due_date")

# %% ---------------- TIME CONVERSION ----------------
def convert_time(value_in_days, unit="days"):
    if unit == "days":
        return value_in_days
    elif unit == "hours":
        return value_in_days * 24

# %% ---------------- METRIC DISPLAY ----------------
def display_metric(label, value, unit="days"):
    st.metric(label, f"{convert_time(value, unit):.2f} {unit}")

# %% ---------------- CONTROLS ----------------
time_unit = st.selectbox("Select time unit", ["days", "hours"], index=0)

# %% ---------------- SELLER SELECTION ----------------
st.subheader("🔍 Seller Selection")

selected_seller = st.selectbox(
    "Select a seller from Top 50 (by number of orders)",
    options=[""] + top_sellers_df["seller_id"].tolist(),
    format_func=lambda x: "— Select a seller —" if x == "" else x
)

manual_seller = st.text_input(
    "Or manually enter a Seller ID",
    placeholder="Paste seller_id here (optional)"
)

seller_id = selected_seller if selected_seller else manual_seller

# %% ---------------- RUN SIMULATION BUTTON ----------------
if seller_id:
    if st.button("Run Simulation"):
        schedule = create_scheduling_table_cached(seller_id)

        if len(schedule) < 10:
            st.warning("This seller has fewer than 10 orders. Results may not be meaningful.")
        else:
            # Run simulations (no animation)
            fcfs = simulate_schedule_and_metrics(FCFS(schedule))
            spt = simulate_schedule_and_metrics(SPT(schedule))
            edd = simulate_schedule_and_metrics(EDD(schedule))

            st.subheader("📊 Performance Overview")

            col1, col2, col3 = st.columns(3)
            with col1:
                display_metric("FCFS Avg Waiting", fcfs["Average waiting time"], unit=time_unit)
                display_metric("FCFS Avg Tardiness", fcfs["Average tardiness"], unit=time_unit)

            with col2:
                display_metric("SPT Avg Waiting", spt["Average waiting time"], unit=time_unit)
                display_metric("SPT Avg Tardiness", spt["Average tardiness"], unit=time_unit)

            with col3:
                display_metric("EDD Avg Waiting", edd["Average waiting time"], unit=time_unit)
                display_metric("EDD Avg Tardiness", edd["Average tardiness"], unit=time_unit)

            st.subheader("📋 Metric Comparison")
            results_df = pd.DataFrame({
                "Metric": fcfs.keys(),
                "FCFS": fcfs.values(),
                "SPT": spt.values(),
                "EDD": edd.values()
            }).round(2)

            st.dataframe(results_df, use_container_width=True)

            plot_df = results_df[results_df["Metric"].isin([
                "Average waiting time", "Average tardiness", "Maximum tardiness"
            ])].melt(
                id_vars="Metric",
                var_name="Policy",
                value_name="Value"
            )

            if time_unit == "hours":
                plot_df["Value"] *= 24

            fig = px.bar(
                plot_df,
                x="Metric",
                y="Value",
                color="Policy",
                barmode="group",
                title=f"Scheduling Policy Comparison ({time_unit})",
                text_auto=".2f",
                template="plotly_white"
            )
            fig.update_layout(
                transition_duration=1200,
                transition_easing="cubic-in-out",
                xaxis_title="Performance Metric",
                yaxis_title=f"Time ({time_unit})"
            )
            st.plotly_chart(fig, use_container_width=True)