import streamlit as st
import pulp
import itertools
import time

# ==================================================
# Page configuration
# ==================================================
st.set_page_config(layout="wide")
st.title("ILP-Based Production Sequencing with Sequence-Dependent Setup Costs")
st.markdown(
    """
    **Christoph Kaier**

    **FH JOANNEUM - University of Applied Sciences**  
    Graz, Austria

    **Supervisor:**  
    FH-Prof. Priv.-Doz. DI Dr. Joachim Schauer
    """
)


st.divider()
st.markdown(
    """
    This interactive demonstration illustrates the application of
    **Integer Linear Programming (ILP)** to a **single-machine production sequencing problem**
    with **sequence-dependent setup costs**.

    The example serves as a **didactic complement** to the mathematical formulation
    presented in the Master’s thesis *“Optimizing Single-Machine Production Sequencing
    Using Integer and Hybrid Optimization Methods”*.
    """
)

st.divider()

# ==================================================
# ILP background
# ==================================================
st.subheader("Integer Linear Programming (ILP)")

st.markdown(
    """
    **Integer Linear Programming (ILP)** is an exact optimization technique for solving
    discrete decision problems. It enables the precise modeling of sequencing,
    assignment, and scheduling problems under strict constraints.

    An ILP model consists of:
    - **Decision variables** representing discrete choices,
    - **A linear objective function** describing the optimization goal,
    - **Linear constraints** ensuring feasibility.

    In contrast to heuristic or learning-based approaches, ILP solvers provide
    **provable optimality guarantees** and full **solution transparency**.
    """
)

st.info(
    "Due to combinatorial complexity, ILP models scale poorly for large instances, "
    "but are highly valuable for benchmarking and methodological validation."
)

st.divider()

# ==================================================
# Problem description
# ==================================================
st.subheader("Problem Description")

st.markdown(
    """
    A set of production jobs must be processed sequentially on a **single machine**
    without preemption. Each job is characterized by:

    - **Vehicle type:** Truck 🚚 or Car 🚗  
    - **Color:** 🟢 Green, 🔴 Red, ⚫ Black, 🔵 Blue  

    Setup operations occur **only between consecutive jobs** and induce costs.
    The objective is to determine a job sequence that **minimizes total setup cost**.
    """
)

# ==================================================
# Job data
# ==================================================
jobs = {
    "J1": {"type": "Truck", "color": "green", "icon": "🚚", "c": "🟢"},
    "J2": {"type": "Truck", "color": "red",   "icon": "🚚", "c": "🔴"},
    "J3": {"type": "Car",   "color": "black", "icon": "🚗", "c": "⚫"},
    "J4": {"type": "Car",   "color": "red",   "icon": "🚗", "c": "🔴"},
    "J5": {"type": "Car",   "color": "black", "icon": "🚗", "c": "⚫"},
    "J6": {"type": "Car",   "color": "blue",  "icon": "🚗", "c": "🔵"},
    "J7": {"type": "Truck","color": "blue",  "icon": "🚚", "c": "🔵"},
}

J = list(jobs.keys())
P = list(range(1, len(J) + 1))

st.subheader("Job Set")
cols = st.columns(len(J))
for c, j in zip(cols, J):
    c.markdown(
        f"<div style='text-align:center;font-size:26px'>"
        f"{jobs[j]['icon']}<br>{jobs[j]['c']}<br>{j}</div>",
        unsafe_allow_html=True
    )

st.divider()

# ==================================================
# Cost model
# ==================================================
st.subheader("Sequence-Dependent Setup Cost Model")

st.markdown(
    """
    Setup costs arise **only from transitions between consecutive jobs**:

    **Color change**
    - Different colors: cost **5**
    - Same color: cost **0**

    **Type change**
    - Truck ↔ Car: additional cost **1**
    - Same type: cost **0**

    The total cost of a sequence is defined as the **sum of all transition costs**.
    """
)

# ==================================================
# Cost functions
# ==================================================
def transition_cost(i, j):
    cost = 0
    if jobs[i]["color"] != jobs[j]["color"]:
        cost += 5
    if jobs[i]["type"] != jobs[j]["type"]:
        cost += 1
    return cost

def sequence_cost(seq):
    return sum(transition_cost(seq[i], seq[i + 1]) for i in range(len(seq) - 1))

# ==================================================
# Initial sequence
# ==================================================
initial_sequence = ["J1", "J2", "J3", "J4", "J5", "J7", "J6"]

st.subheader("Initial (Non-Optimized) Sequence")
cols = st.columns(len(initial_sequence))
for c, j in zip(cols, initial_sequence):
    c.markdown(
        f"<div style='text-align:center;font-size:26px'>"
        f"{jobs[j]['icon']}<br>{jobs[j]['c']}<br>{j}</div>",
        unsafe_allow_html=True
    )

st.markdown(f"**Total setup cost:** {sequence_cost(initial_sequence)}")
st.divider()

# ==================================================
# Optimization
# ==================================================
if st.button("Run ILP Optimization"):
    start_time = time.time()

    model = pulp.LpProblem("Single_Machine_Sequencing", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", (J, P), cat="Binary")
    y = pulp.LpVariable.dicts("y", (J, J, P[:-1]), cat="Binary")

    # Assignment constraints
    for j in J:
        model += pulp.lpSum(x[j][p] for p in P) == 1

    for p in P:
        model += pulp.lpSum(x[j][p] for j in J) == 1

    # Transition consistency constraints
    for i, j, p in itertools.product(J, J, P[:-1]):
        if i != j:
            model += y[i][j][p] <= x[i][p]
            model += y[i][j][p] <= x[j][p + 1]
            model += y[i][j][p] >= x[i][p] + x[j][p + 1] - 1
        else:
            model += y[i][j][p] == 0

    # Objective function
    model += pulp.lpSum(
        transition_cost(i, j) * y[i][j][p]
        for i, j, p in itertools.product(J, J, P[:-1])
        if i != j
    )

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    runtime = time.time() - start_time

    # Extract optimal sequence
    optimal_sequence = sorted(
        J, key=lambda j: sum(p * pulp.value(x[j][p]) for p in P)
    )

    st.subheader("Optimal Sequence (ILP Solution)")
    cols = st.columns(len(optimal_sequence))
    for c, j in zip(cols, optimal_sequence):
        c.markdown(
            f"<div style='text-align:center;font-size:26px'>"
            f"{jobs[j]['icon']}<br>{jobs[j]['c']}<br>{j}</div>",
            unsafe_allow_html=True
        )

    optimal_cost = sequence_cost(optimal_sequence)

    st.markdown(f"### Optimal total setup cost: **{optimal_cost}**")

    st.divider()

    # ==================================================
    # Runtime analysis
    # ==================================================
    st.subheader("Computational Performance")

    st.markdown(
        f"""
        - **Solver:** CBC (open-source MILP solver)  
        - **Number of jobs:** {len(J)}  
        - **Binary variables:** {len(J) * len(P) + len(J) * len(J) * (len(P) - 1)}  
        - **Solve time:** **{runtime:.4f} seconds**

        The instance is solved to proven optimality within milliseconds on standard hardware.
        """
    )

    st.info(
        "The number of variables and constraints grows combinatorially with the number of jobs. "
        "This limits the practical applicability of exact ILP models to small and medium-sized instances "
        "and motivates the use of heuristic and hybrid optimization approaches."
    )
