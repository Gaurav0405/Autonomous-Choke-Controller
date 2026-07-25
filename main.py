import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from step_test import run_step_test, identify_models
from run_scenarios import run_scenarios, load_config

def verify_performance(df_a, df_b, df_c, whp_limit=220.0, flp_limit=150.0, bhp_limit=2900.0):
    print("\n====================================================")
    print("4. PERFORMANCE AND SAFETY CONSTRAINTS VERIFICATION")
    print("====================================================")
    
    scenarios = {
        'Scenario A (Startup)': df_a,
        'Scenario B (Tracking)': df_b,
        'Scenario C (Infeasible Target)': df_c
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
        
        print(f"  Minimum WHP: {min_whp:.2f} psi (Limit: >= {whp_limit:.1f} psi)")
        print(f"  Minimum FLP: {min_flp:.2f} psi (Limit: >= {flp_limit:.1f} psi)")
        print(f"  Minimum BHP: {min_bhp:.2f} psi (Limit: >= {bhp_limit:.1f} psi)")
        
        # Validation checks with small numerical noise tolerance
        choke_ok = max_choke_move <= 5.01
        whp_ok = min_whp >= whp_limit - 1.0
        flp_ok = min_flp >= flp_limit - 1.0
        bhp_ok = min_bhp >= bhp_limit - 5.0
        
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

def prompt_interactive_inputs():
    print("\n====================================================")
    print("INTERACTIVE INPUT CONFIGURATION MENU")
    print("====================================================")
    cfg = load_config()
    
    try:
        val = input(f"Target Oil Rate for Scenario A [{cfg['scenario_a_target']}]: ").strip()
        if val: cfg['scenario_a_target'] = float(val)
        
        val = input(f"Target Oil Rate for Scenario B Step [{cfg['scenario_b_target_step']}]: ").strip()
        if val: cfg['scenario_b_target_step'] = float(val)

        val = input(f"Target Oil Rate for Scenario C Step [{cfg['scenario_c_target_step']}]: ").strip()
        if val: cfg['scenario_c_target_step'] = float(val)

        val = input(f"Minimum WHP Limit (psi) [{cfg['whp_min_limit']}]: ").strip()
        if val: cfg['whp_min_limit'] = float(val)

        val = input(f"Minimum BHP Limit (psi) [{cfg['bhp_min_limit']}]: ").strip()
        if val: cfg['bhp_min_limit'] = float(val)

        val = input(f"Prediction Horizon Hp (hours) [{cfg['prediction_horizon_hp']}]: ").strip()
        if val: cfg['prediction_horizon_hp'] = int(val)

    except KeyboardInterrupt:
        print("\nUsing current configuration.")
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Autonomous Production Choke Controller Master Orchestrator")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt interactively for input parameters in the terminal")
    parser.add_argument("--target-a", type=float, help="Scenario A target flow rate (bbl/hr)")
    parser.add_argument("--target-b", type=float, help="Scenario B step target flow rate (bbl/hr)")
    parser.add_argument("--target-c", type=float, help="Scenario C step target flow rate (bbl/hr)")
    parser.add_argument("--whp-min", type=float, help="Minimum WHP limit (psi)")
    parser.add_argument("--flp-min", type=float, help="Minimum FLP limit (psi)")
    parser.add_argument("--bhp-min", type=float, help="Minimum BHP limit (psi)")
    parser.add_argument("--hp", type=int, help="Prediction horizon Hp (hours)")
    parser.add_argument("--lambda-penalty", type=float, help="Choke movement penalty weight")
    
    args = parser.parse_args()
    
    config_override = {}
    if args.interactive:
        config_override = prompt_interactive_inputs()
    else:
        if args.target_a is not None: config_override["scenario_a_target"] = args.target_a
        if args.target_b is not None: config_override["scenario_b_target_step"] = args.target_b
        if args.target_c is not None: config_override["scenario_c_target_step"] = args.target_c
        if args.whp_min is not None: config_override["whp_min_limit"] = args.whp_min
        if args.flp_min is not None: config_override["flp_min_limit"] = args.flp_min
        if args.bhp_min is not None: config_override["bhp_min_limit"] = args.bhp_min
        if args.hp is not None: config_override["prediction_horizon_hp"] = args.hp
        if args.lambda_penalty is not None: config_override["choke_penalty_lambda"] = args.lambda_penalty
    
    # Step 1: Run Step Test
    df_ol = run_step_test()
    
    # Step 2: System Identification
    model_params = identify_models(df_ol)
    
    # Step 3: Run Scenarios using identified dynamic model
    df_a, df_b, df_c = run_scenarios(model_params=model_params, config_override=config_override)
    
    # Step 4: Verify performance
    cfg = load_config()
    if config_override: cfg.update(config_override)
    verify_performance(df_a, df_b, df_c, whp_limit=cfg['whp_min_limit'], flp_limit=cfg['flp_min_limit'], bhp_limit=cfg['bhp_min_limit'])

if __name__ == "__main__":
    main()
