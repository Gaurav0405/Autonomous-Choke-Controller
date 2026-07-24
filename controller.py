import numpy as np

class MPCController:
    """
    Model Predictive Controller (MPC) using brute-force candidate evaluation.
    Optimizes choke position to track target flow rate while respecting safety constraints.
    Supports candidate diagnostics logging for educational/judging demonstration.
    """
    def __init__(self, Hp=30, lambda_u=0.5, whp_min=220.0, flp_min=150.0, bhp_min=2900.0, model_params=None):
        self.Hp = Hp              # Prediction horizon (hours)
        self.lambda_u = lambda_u  # Choke movement penalty weight
        
        # Active operating constraints
        self.whp_min = whp_min
        self.flp_min = flp_min
        self.bhp_min = bhp_min
        
        # Identified ARX parameters (deterministic part for predictions)
        # If None, use standard defaults identified from the reference data.
        if model_params is not None:
            self.params = model_params
        else:
            self.params = {
                'Q':   (0.82366,  0.32006,   6.93175),
                'WHP': (0.88924, -0.17564,  35.37564),
                'FLP': (0.86253, -0.13556,  30.14338),
                'BHP': (0.92574, -0.62250, 253.76057)
            }

    def predict_trajectory(self, u_cand, Q_init, WHP_init, FLP_init, BHP_init):
        """
        Predicts states over the horizon Hp, assuming choke is changed to u_cand
        and is held constant (control horizon Hc = 1).
        """
        Q_pred = np.zeros(self.Hp + 1)
        WHP_pred = np.zeros(self.Hp + 1)
        FLP_pred = np.zeros(self.Hp + 1)
        BHP_pred = np.zeros(self.Hp + 1)
        
        Q_pred[0] = Q_init
        WHP_pred[0] = WHP_init
        FLP_pred[0] = FLP_init
        BHP_pred[0] = BHP_init
        
        for j in range(1, self.Hp + 1):
            Q_pred[j] = self.params['Q'][0] * Q_pred[j-1] + self.params['Q'][1] * u_cand + self.params['Q'][2]
            WHP_pred[j] = self.params['WHP'][0] * WHP_pred[j-1] + self.params['WHP'][1] * u_cand + self.params['WHP'][2]
            FLP_pred[j] = self.params['FLP'][0] * FLP_pred[j-1] + self.params['FLP'][1] * u_cand + self.params['FLP'][2]
            BHP_pred[j] = self.params['BHP'][0] * BHP_pred[j-1] + self.params['BHP'][1] * u_cand + self.params['BHP'][2]
            
        return Q_pred[1:], WHP_pred[1:], FLP_pred[1:], BHP_pred[1:]

    def calculate_control(self, Q_meas, WHP_meas, FLP_meas, BHP_meas, current_choke, Q_target):
        """
        Calculates the optimal choke position for the next control step.
        Returns the chosen choke opening and a dictionary of control diagnostics.
        """
        # Choke ramp rate limit is +/- 5% per step
        u_min = max(0.0, current_choke - 5.0)
        u_max = min(100.0, current_choke + 5.0)
        
        # Generate candidates (steps of 0.1%)
        candidates = np.linspace(u_min, u_max, 101)
        
        best_u = current_choke
        min_cost = float('inf')
        
        # Diagnostic tracking counters
        num_safe = 0
        num_whp_viol = 0
        num_flp_viol = 0
        num_bhp_viol = 0
        rejection_example = "None"
        
        best_trajectory_info = {}
        
        # Penalties and weights
        w_Q = 1.0
        w_viol = 1e6  # Large penalty weight for constraint violations
        
        for u_cand in candidates:
            # Predict future trajectory
            Q_p, WHP_p, FLP_p, BHP_p = self.predict_trajectory(
                u_cand, Q_meas, WHP_meas, FLP_meas, BHP_meas
            )
            
            # Check constraints over the prediction horizon
            whp_viol_amount = np.maximum(0.0, self.whp_min - WHP_p)
            flp_viol_amount = np.maximum(0.0, self.flp_min - FLP_p)
            bhp_viol_amount = np.maximum(0.0, self.bhp_min - BHP_p)
            
            is_whp_viol = np.any(whp_viol_amount > 0.0)
            is_flp_viol = np.any(flp_viol_amount > 0.0)
            is_bhp_viol = np.any(bhp_viol_amount > 0.0)
            
            # Track counts and build a representative rejection reason for diagnostics
            if not (is_whp_viol or is_flp_viol or is_bhp_viol):
                num_safe += 1
            else:
                if is_whp_viol:
                    num_whp_viol += 1
                    if rejection_example == "None":
                        step_viol = np.argmax(whp_viol_amount > 0.0)
                        rejection_example = f"Choke {u_cand:.1f}% rejected: WHP predicted {WHP_p[step_viol]:.1f} psi (limit >= {self.whp_min:.1f} psi)"
                if is_flp_viol:
                    num_flp_viol += 1
                    if rejection_example == "None":
                        step_viol = np.argmax(flp_viol_amount > 0.0)
                        rejection_example = f"Choke {u_cand:.1f}% rejected: FLP predicted {FLP_p[step_viol]:.1f} psi (limit >= {self.flp_min:.1f} psi)"
                if is_bhp_viol:
                    num_bhp_viol += 1
                    if rejection_example == "None":
                        step_viol = np.argmax(bhp_viol_amount > 0.0)
                        rejection_example = f"Choke {u_cand:.1f}% rejected: BHP predicted {BHP_p[step_viol]:.1f} psi (limit >= {self.bhp_min:.1f} psi)"
            
            # Total constraint violation penalty
            total_viol = np.sum(whp_viol_amount**2) + np.sum(flp_viol_amount**2) + np.sum(bhp_viol_amount**2)
            
            # Calculate tracking error
            tracking_err = np.sum((Q_p - Q_target) ** 2)
            
            # Choke movement penalty
            choke_move_penalty = self.lambda_u * (u_cand - current_choke) ** 2
            
            # Total cost
            cost = w_Q * tracking_err + w_viol * total_viol + choke_move_penalty
            
            if cost < min_cost:
                min_cost = cost
                best_u = u_cand
                best_trajectory_info = {
                    'expected_flow': Q_p[-1],
                    'expected_whp': WHP_p[-1],
                    'expected_flp': FLP_p[-1],
                    'expected_bhp': BHP_p[-1],
                }
                
        # Return decision and diagnostics
        diagnostics = {
            'expected_flow': best_trajectory_info['expected_flow'],
            'expected_whp': best_trajectory_info['expected_whp'],
            'expected_flp': best_trajectory_info['expected_flp'],
            'expected_bhp': best_trajectory_info['expected_bhp'],
            'num_safe': num_safe,
            'num_rejected': 101 - num_safe,
            'rejections': {
                'WHP': num_whp_viol,
                'FLP': num_flp_viol,
                'BHP': num_bhp_viol
            },
            'rejection_example': rejection_example,
            'status': 'SAFE' if (101 - num_safe) < 101 else 'UNSAFE_SYSTEM_LIMIT'
        }
        
        return best_u, diagnostics
