import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from simulator import WellSimulator

def run_step_test():
    print("====================================================")
    print("1. RUNNING OPEN-LOOP STEP-TEST EXPERIMENT")
    print("====================================================")
    
    # Initialize simulator without noise for clean parameter identification
    sim = WellSimulator(initial_choke=30.0, add_noise=False)
    steps = 120
    history = {'time': [], 'choke': [], 'Q': [], 'WHP': [], 'FLP': [], 'BHP': []}
    
    # Apply choke steps (30 -> 40 -> 55 -> 45 -> 65)
    for t in range(steps):
        if t < 20:
            choke = 30.0
        elif t < 40:
            choke = 40.0
        elif t < 70:
            choke = 55.0
        elif t < 90:
            choke = 45.0
        else:
            choke = 65.0
            
        Q, WHP, FLP, BHP = sim.step(choke)
        history['time'].append(t)
        history['choke'].append(choke)
        history['Q'].append(Q)
        history['WHP'].append(WHP)
        history['FLP'].append(FLP)
        history['BHP'].append(BHP)
        
    df = pd.DataFrame(history)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/open_loop_step_test.csv", index=False)
    print("Saved open-loop step-test data to data/open_loop_step_test.csv")
    
    # Plot open loop response
    os.makedirs("plots", exist_ok=True)
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Open-Loop Step-Test Response (Deterministic)', fontsize=16, fontweight='bold')
    
    axs[0, 0].plot(df['time'], df['choke'], color='#1f77b4', linewidth=2)
    axs[0, 0].set_ylabel('Choke Position (%)', fontsize=10)
    axs[0, 0].set_title('Production Choke Opening', fontsize=12, fontweight='semibold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    axs[0, 1].plot(df['time'], df['Q'], color='#ff7f0e', linewidth=2)
    axs[0, 1].set_ylabel('Flow Rate (bbl/hr)', fontsize=10)
    axs[0, 1].set_title('Oil Production Flow Rate (Q)', fontsize=12, fontweight='semibold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    axs[1, 0].plot(df['time'], df['WHP'], color='#2ca02c', linewidth=2)
    axs[1, 0].set_ylabel('Pressure (psi)', fontsize=10)
    axs[1, 0].set_title('Wellhead Pressure (WHP)', fontsize=12, fontweight='semibold')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    axs[1, 1].plot(df['time'], df['FLP'], color='#d62728', linewidth=2)
    axs[1, 1].set_ylabel('Pressure (psi)', fontsize=10)
    axs[1, 1].set_title('Flowline Pressure (FLP)', fontsize=12, fontweight='semibold')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    axs[2, 0].plot(df['time'], df['BHP'], color='#9467bd', linewidth=2)
    axs[2, 0].set_ylabel('Pressure (psi)', fontsize=10)
    axs[2, 0].set_xlabel('Time (hours)', fontsize=10)
    axs[2, 0].set_title('Bottom Hole Pressure (BHP)', fontsize=12, fontweight='semibold')
    axs[2, 0].grid(True, linestyle='--', alpha=0.5)
    
    axs[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("plots/open_loop_step_test.png", dpi=300)
    plt.close()
    print("Saved step-test response plot to plots/open_loop_step_test.png")
    
    return df

def identify_models(df):
    print("\n====================================================")
    print("2. DYNAMIC MODEL IDENTIFICATION (SYSTEM IDENTIFICATION)")
    print("====================================================")
    
    # We fit first-order ARX models: y(t) = a * y(t-1) + b * u(t-1) + c
    model_params = {}
    
    for var in ['Q', 'WHP', 'FLP', 'BHP']:
        col_name = {
            'Q': 'Q',
            'WHP': 'WHP',
            'FLP': 'FLP',
            'BHP': 'BHP'
        }[var]
        
        y = df[col_name].values
        u = df['choke'].values
        
        # Prepare regression arrays
        X = []
        Y = []
        for t in range(1, len(df)):
            X.append([y[t-1], u[t-1]])
            Y.append(y[t])
        X = np.array(X)
        Y = np.array(Y)
        
        # Linear regression
        reg = LinearRegression().fit(X, Y)
        a = reg.coef_[0]
        b = reg.coef_[1]
        c = reg.intercept_
        
        model_params[col_name] = (a, b, c)
        
        # Calculate key control parameters
        ss_gain = b / (1 - a)
        time_constant = -1.0 / np.log(a) if a > 0 else 0.0
        
        print(f"\nModel identified for {col_name}:")
        print(f"  Equation: {col_name}(t) = {a:.5f} * {col_name}(t-1) + {b:.5f} * u(t-1) + {c:.5f}")
        print(f"  Steady-State Gain: {ss_gain:.3f} (units per % choke)")
        print(f"  Time Constant (tau): {time_constant:.2f} hours")
        
    return model_params

if __name__ == "__main__":
    df = run_step_test()
    identify_models(df)
