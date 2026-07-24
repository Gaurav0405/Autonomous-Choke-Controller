import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from simulator import WellSimulator
from controller import MPCController

st.set_page_config(
    page_title="Autonomous Choke Controller Dashboard",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .kpi-title {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .kpi-val {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .status-safe {
        color: #16a34a !important;
    }
    .status-warning {
        color: #ea580c !important;
    }
    .status-danger {
        color: #dc2626 !important;
    }
    .terminal-box {
        background-color: #0f172a;
        color: #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 6px;
        height: 250px;
        overflow-y: scroll;
        border: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Autonomous Choke Controller Dashboard")
st.markdown("### Single Naturally Flowing Oil Well Optimization & Safety Envelope Control")
st.markdown("This interactive dashboard simulates the well dynamics and demonstrates the **Model Predictive Controller (MPC)** regulating the production choke. Adjust targets and safety constraints in real-time to see how the controller optimizes flow rates while ensuring operating pressures remain safe.")

# Session state initialization
if 'history' not in st.session_state:
    st.session_state.history = None
if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'step_num' not in st.session_state:
    st.session_state.step_num = 0
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Sidebar Controls
st.sidebar.header("🔧 Controller & Simulator Settings")

# Profile Presets
preset = st.sidebar.selectbox(
    "Select Operating Scenario",
    ["Startup to Target (Scenario A)", "Target Tracking (Scenario B)", "Infeasible Target (Scenario C)", "Manual Setpoint Profile"]
)

# Target Oil Rate Settings
if preset == "Startup to Target (Scenario A)":
    st.sidebar.info("Target: Constant 120 bbl/hr starting from 90 bbl/hr.")
    target_mode = "Constant"
    const_target = 120.0
    sim_steps = 40
elif preset == "Target Tracking (Scenario B)":
    st.sidebar.info("Target: 100 bbl/hr for 20 hours, then steps to 150 bbl/hr.")
    target_mode = "Tracking"
    sim_steps = 60
elif preset == "Infeasible Target (Scenario C)":
    st.sidebar.info("Target: 100 bbl/hr for 20 hours, then steps to 200 bbl/hr (infeasible).")
    target_mode = "Infeasible"
    sim_steps = 60
else:
    target_mode = st.sidebar.radio("Target Profile Mode", ["Constant Setpoint", "Custom Step Change"])
    if target_mode == "Constant Setpoint":
        const_target = st.sidebar.slider("Constant Target (bbl/hr)", 50.0, 250.0, 120.0, 5.0)
    else:
        st.sidebar.markdown("**Step Change Profile:**")
        step_val1 = st.sidebar.slider("Initial Target (bbl/hr)", 50.0, 250.0, 100.0, 5.0)
        step_time = st.sidebar.slider("Step Time (hours)", 5, 40, 20, 1)
        step_val2 = st.sidebar.slider("Step Target (bbl/hr)", 50.0, 250.0, 180.0, 5.0)
    sim_steps = st.sidebar.slider("Simulation Steps (hours)", 20, 100, 60, 5)

# Controller parameters
st.sidebar.subheader("🎛️ MPC Parameters")
Hp = st.sidebar.slider("Prediction Horizon Hp (hours)", 5, 50, 30, 1)
lambda_u = st.sidebar.slider("Choke Move Penalty (lambda)", 0.1, 2.0, 0.5, 0.1)

# Safety constraints limits
st.sidebar.subheader("⚠️ Safety Constraint Limits")
whp_limit = st.sidebar.slider("Min Wellhead Pressure (psi)", 180.0, 240.0, 220.0, 5.0)
flp_limit = st.sidebar.slider("Min Flowline Pressure (psi)", 120.0, 170.0, 150.0, 5.0)
bhp_limit = st.sidebar.slider("Min Bottom Hole Pressure (psi)", 2800.0, 3000.0, 2900.0, 10.0)

# Add noise checkbox
add_noise = st.sidebar.checkbox("Add Process Noise", value=True)

# Helper function to evaluate target profile at step t
def get_target(t):
    if preset == "Startup to Target (Scenario A)":
        return 120.0
    elif preset == "Target Tracking (Scenario B)":
        return 100.0 if t < 20 else 150.0
    elif preset == "Infeasible Target (Scenario C)":
        return 100.0 if t < 20 else 200.0
    else:
        if target_mode == "Constant Setpoint":
            return const_target
        else:
            return step_val1 if t < step_time else step_val2

# Run Simulation Button
if st.button("▶️ Run Full Simulation Scenario", type="primary"):
    sim = WellSimulator(initial_choke=30.0, add_noise=add_noise)
    controller = MPCController(Hp=Hp, lambda_u=lambda_u, whp_min=whp_limit, flp_min=flp_limit, bhp_min=bhp_limit)
    
    history = {k: [] for k in ['time', 'choke', 'Q', 'WHP', 'FLP', 'BHP', 'target', 'num_rejected', 'rejection_example', 'status']}
    
    Q, WHP, FLP, BHP = sim.Q, sim.WHP, sim.FLP, sim.BHP
    choke = sim.choke
    logs = []
    
    for t in range(sim_steps):
        target = get_target(t)
        next_choke, diag = controller.calculate_control(Q, WHP, FLP, BHP, choke, target)
        Q, WHP, FLP, BHP = sim.step(next_choke)
        choke = next_choke
        
        # Log entry
        history['time'].append(t)
        history['choke'].append(choke)
        history['Q'].append(Q)
        history['WHP'].append(WHP)
        history['FLP'].append(FLP)
        history['BHP'].append(BHP)
        history['target'].append(target)
        history['num_rejected'].append(diag['num_rejected'])
        history['rejection_example'].append(diag['rejection_example'])
        history['status'].append(diag['status'])
        
        log_str = f"Hour {t:2d} | Target: {target:5.1f} | Rec Choke: {choke:5.1f}% | Expected Flow: {diag['expected_flow']:5.1f} | Expected BHP: {diag['expected_bhp']:6.1f} | Status: {diag['status']} | Rejected: {diag['num_rejected']:3d}"
        logs.append(log_str)
        if diag['num_rejected'] > 0:
            logs.append(f"  └─ Rejection reason: {diag['rejection_example']}")
            
    st.session_state.history = pd.DataFrame(history)
    st.session_state.logs = logs
    st.session_state.step_num = sim_steps

# Display Results
if st.session_state.history is not None:
    df = st.session_state.history
    logs = st.session_state.logs
    
    # 1. KPI Columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    latest = df.iloc[-1]
    
    # Check for safety limit status
    # If any rejections occurred in the last few steps, status is Limit Reached
    is_safe = latest['num_rejected'] < 80 # If almost all candidates are rejected, it's operating on safety limits
    status_cls = "status-safe" if latest['num_rejected'] == 0 else ("status-warning" if is_safe else "status-danger")
    status_text = "SAFE" if latest['num_rejected'] == 0 else ("LIMIT ACTIVE" if is_safe else "UNSAFE / LIMIT ACTIVE")
    
    kpi1.markdown(f'<div class="kpi-card"><div class="kpi-title">Target Flow Rate</div><div class="kpi-val">{latest["target"]:.1f} bbl/hr</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="kpi-card"><div class="kpi-title">Actual Oil Rate</div><div class="kpi-val">{latest["Q"]:.1f} bbl/hr</div></div>', unsafe_allow_html=True)
    kpi3.markdown(f'<div class="kpi-card"><div class="kpi-title">Recommended Choke</div><div class="kpi-val">{latest["choke"]:.1f}%</div></div>', unsafe_allow_html=True)
    kpi4.markdown(f'<div class="kpi-card"><div class="kpi-title">Safety Status</div><div class="kpi-val {status_cls}">{status_text}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Charts and Logs (Columns)
    col_chart, col_logs = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📈 Performance Trends")
        
        # Render high-quality Matplotlib plots
        fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        time = df['time']
        
        # 1. Oil Rate
        axs[0, 0].plot(time, df['target'], 'r--', label='Target Rate', linewidth=1.2)
        axs[0, 0].plot(time, df['Q'], 'b-', label='Actual Rate', linewidth=1.8)
        axs[0, 0].set_ylabel('Flow Rate (bbl/hr)', fontsize=9)
        axs[0, 0].set_title('Oil Production Flow Rate', fontsize=10, fontweight='bold')
        axs[0, 0].legend(loc='lower right', fontsize=8)
        axs[0, 0].grid(True, linestyle='--', alpha=0.5)
        
        # 2. Choke
        axs[0, 1].plot(time, df['choke'], 'g-', label='Choke', linewidth=1.8)
        axs[0, 1].set_ylabel('Choke Position (%)', fontsize=9)
        axs[0, 1].set_title('Production Choke Opening', fontsize=10, fontweight='bold')
        axs[0, 1].grid(True, linestyle='--', alpha=0.5)
        
        # 3. WHP
        axs[1, 0].plot(time, df['WHP'], 'm-', label='WHP', linewidth=1.8)
        axs[1, 0].axhline(whp_limit, color='r', linestyle=':', label=f'Min Limit ({whp_limit} psi)')
        axs[1, 0].fill_between(time, 150, whp_limit, color='red', alpha=0.08)
        axs[1, 0].set_ylabel('Pressure (psi)', fontsize=9)
        axs[1, 0].set_title('Wellhead Pressure (WHP)', fontsize=10, fontweight='bold')
        axs[1, 0].legend(loc='lower left', fontsize=8)
        axs[1, 0].grid(True, linestyle='--', alpha=0.5)
        
        # 4. FLP
        axs[1, 1].plot(time, df['FLP'], 'c-', label='FLP', linewidth=1.8)
        axs[1, 1].axhline(flp_limit, color='r', linestyle=':', label=f'Min Limit ({flp_limit} psi)')
        axs[1, 1].fill_between(time, 100, flp_limit, color='red', alpha=0.08)
        axs[1, 1].set_ylabel('Pressure (psi)', fontsize=9)
        axs[1, 1].set_title('Flowline Pressure (FLP)', fontsize=10, fontweight='bold')
        axs[1, 1].legend(loc='lower left', fontsize=8)
        axs[1, 1].grid(True, linestyle='--', alpha=0.5)
        
        # 5. BHP
        axs[2, 0].plot(time, df['BHP'], 'b-', label='BHP', linewidth=1.8)
        axs[2, 0].axhline(bhp_limit, color='r', linestyle=':', label=f'Min Limit ({bhp_limit} psi)')
        axs[2, 0].fill_between(time, 2700, bhp_limit, color='red', alpha=0.08)
        axs[2, 0].set_ylabel('Pressure (psi)', fontsize=9)
        axs[2, 0].set_xlabel('Time (hours)', fontsize=9)
        axs[2, 0].set_title('Bottom Hole Pressure (BHP)', fontsize=10, fontweight='bold')
        axs[2, 0].legend(loc='lower left', fontsize=8)
        axs[2, 0].grid(True, linestyle='--', alpha=0.5)
        
        axs[2, 1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_logs:
        st.subheader("💻 Controller Decisions Terminal")
        st.markdown("Hourly predictive optimization log and candidate evaluation:")
        
        # Combine logs into a single scrollable HTML string
        log_html = "<div class='terminal-box'>"
        for line in logs:
            if "rejection" in line.lower() or "rejected:" in line.lower():
                # Highlight rejections in orange/red
                if "rejected:   0" in line or "rejected: 0" in line:
                    log_html += f"<div style='color: #10b981;'>{line}</div>"
                else:
                    log_html += f"<div style='color: #f97316;'>{line}</div>"
            else:
                log_html += f"<div>{line}</div>"
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)
        
        st.subheader("📘 Model identified")
        st.markdown("""
        **Fitted ARX Equations:**
        - **Flow Rate:** $Q(t) = 0.824Q(t-1) + 0.320u(t-1) + 6.93$
        - **Wellhead Pres:** $WHP(t) = 0.889WHP(t-1) - 0.176u(t-1) + 35.38$
        - **Flowline Pres:** $FLP(t) = 0.863FLP(t-1) - 0.136u(t-1) + 30.14$
        - **Bottom Hole Pres:** $BHP(t) = 0.926BHP(t-1) - 0.623u(t-1) + 253.76$
        """)
else:
    st.info("💡 Click **Run Full Simulation Scenario** to start the controller simulation.")
