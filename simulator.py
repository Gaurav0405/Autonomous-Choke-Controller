import numpy as np

class WellSimulator:
    """
    High-fidelity dynamic simulator for a single naturally flowing oil well.
    Implements first-order ARX models fitted from the reference dataset.
    """
    def __init__(self, initial_choke=30.0, add_noise=True):
        self.add_noise = add_noise
        self.choke = initial_choke
        
        # Initial states (matching steady-state at choke = 30)
        self.Q = 90.0
        self.WHP = 250.0
        self.FLP = 180.0
        self.BHP = 3000.0
        
        # ARX parameters: (a, b, c, sigma)
        # y(t) = a * y(t-1) + b * u(t-1) + c + N(0, sigma)
        self.params = {
            'Q':   (0.82366,  0.32006,   6.93175, 0.73),
            'WHP': (0.88924, -0.17564,  35.37564, 0.66),
            'FLP': (0.86253, -0.13556,  30.14338, 0.52),
            'BHP': (0.92574, -0.62250, 253.76057, 2.91)
        }

    def step(self, choke_position):
        """
        Executes one control step (1 hour) with the given choke position.
        """
        # Clip choke position to physical limits [0, 100]%
        choke_position = np.clip(choke_position, 0.0, 100.0)
        
        # Generate process noise
        n_q = np.random.normal(0, self.params['Q'][3]) if self.add_noise else 0.0
        n_w = np.random.normal(0, self.params['WHP'][3]) if self.add_noise else 0.0
        n_f = np.random.normal(0, self.params['FLP'][3]) if self.add_noise else 0.0
        n_b = np.random.normal(0, self.params['BHP'][3]) if self.add_noise else 0.0
        
        # Update states using the previous choke position
        self.Q = self.params['Q'][0] * self.Q + self.params['Q'][1] * self.choke + self.params['Q'][2] + n_q
        self.WHP = self.params['WHP'][0] * self.WHP + self.params['WHP'][1] * self.choke + self.params['WHP'][2] + n_w
        self.FLP = self.params['FLP'][0] * self.FLP + self.params['FLP'][1] * self.choke + self.params['FLP'][2] + n_f
        self.BHP = self.params['BHP'][0] * self.BHP + self.params['BHP'][1] * self.choke + self.params['BHP'][2] + n_b
        
        # Store current choke position for the next step
        self.choke = choke_position
        
        return self.Q, self.WHP, self.FLP, self.BHP
