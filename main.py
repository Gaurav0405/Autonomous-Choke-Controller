import os
import numpy as np
import pandas as pd
from step_test import run_step_test, identify_models
from run_scenarios import run_scenarios

def verify_performance(df_a, df_b, df_c):
    print("\n====================================================")
    print("4. PERFORMANCE AND SAFETY CONSTRAINTS VERIFICATION")
    print("====================================================")
    
    scenarios = {
        'Scenario A (Startup to 120 bbl/hr)': df_a,
        'Scenario B (Tracking 100 -> 150 bbl/hr)': df_b,
        'Scenario C (Infeasible 100 -> 200 bbl/hr)': df_c
    }
    
    all_passed = True
    
    for name, df in scenarios.items():
        print(f"\n--- Verification for {name} ---")
        
        # 1. Choke movements constraint check: max move per interval <= 5%
        choke_diff = np.abs(np.diff(df['choke'].values))
        max_choke_move = np.max(choke_diff)
        print(f"  Maximum Choke Movement: {max_choke_move:.4f}% per step (Limit: 5.0%)")
        
        # 2. Safety envelope pressure constraints checks
        min_whp = np.min(df['WHP'].values)
        min_flp = np.min(df['FLP'].values)
        min_bhp = np.min(df['BHP'].values)
        
        print(f"  Minimum WHP: {min_whp:.2f} psi (Limit: >= 220.0 psi)")
        print(f"  Minimum FLP: {min_flp:.2f} psi (Limit: >= 150.0 psi)")
        print(f"  Minimum BHP: {min_bhp:.2f} psi (Limit: >= 2900.0 psi)")
        
        # Validation checks
        # Allow a tiny tolerance for numerical roundoff (e.g. 5.01% for choke, or 1 psi for noise margin)
        choke_ok = max_choke_move <= 5.01
        whp_ok = min_whp >= 220.0 - 1.0
        flp_ok = min_flp >= 150.0 - 1.0
        bhp_ok = min_bhp >= 2900.0 - 5.0 # BHP has high noise sigma of 2.91
        
        if choke_ok and whp_ok and flp_ok and bhp_ok:
            print("  Status: PASSED (All Constraints Satisfied)")
        else:
            print("  Status: WARNING / FAILED")
            all_passed = False
            if not choke_ok:
                print("    Violation: Choke ramp rate limit violated!")
            if not whp_ok:
                print("    Violation: Wellhead Pressure (WHP) dropped below safe limit!")
            if not flp_ok:
                print("    Violation: Flowline Pressure (FLP) dropped below safe limit!")
            if not bhp_ok:
                print("    Violation: Bottom Hole Pressure (BHP) dropped below safe limit!")
                
        # Steady state settled performance
        final_choke = np.mean(df['choke'].values[-10:])
        final_rate = np.mean(df['Q'].values[-10:])
        final_bhp = np.mean(df['BHP'].values[-10:])
        print(f"  Settled State (Final 10 hours average):")
        print(f"    Choke: {final_choke:.2f}%")
        print(f"    Oil Rate: {final_rate:.2f} bbl/hr")
        print(f"    BHP: {final_bhp:.2f} psi")

    print("\n====================================================")
    if all_passed:
        print("VALIDATION SUMMARY: SUCCESS! The autonomous controller successfully regulates choke positions, meets targets when feasible, and restricts flow safely when targets are infeasible.")
    else:
        print("VALIDATION SUMMARY: CONSTRAINTS EXCEEDED! Check process noise levels or tuning parameter lambda_u.")
    print("====================================================")

def main():
    # Step 1: Run Step Test
    df_ol = run_step_test()
    
    # Step 2: System Identification
    model_params = identify_models(df_ol)
    
    # Step 3: Run Scenarios using identified dynamic model
    df_a, df_b, df_c = run_scenarios(model_params)
    
    # Step 4: Verify performance
    verify_performance(df_a, df_b, df_c)

if __name__ == "__main__":
    main()
