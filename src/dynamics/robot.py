import numpy as np

# Prof. Jang Thesis Chapter 6.2.

class RobotDynamics:
    def __init__(self, dt=0.1):
        self.dt = dt
        # State: [px, py, vx, vy] (4x1)
        # Input: [ax, ay] (2x1)
        
        # System Matrices (Discrete-time Double Integrator)
        # x_{k+1} = A x_k + B u_k
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        # Limits (Optional but recommended)
        self.v_max = 1.5  # m/s
        self.u_max = 1.0  # m/s^2

    def step(self, state, control_input):
        """
        state: np.array shape (4,)
        control_input: np.array shape (2,)
        return: next_state
        """
        next_state = self.A @ state + self.B @ control_input
        
        # 속도 제한 (Clipping) - 현실적인 물리 거동을 위해 추가
        vel = next_state[2:]
        speed = np.linalg.norm(vel)
        if speed > self.v_max:
            next_state[2:] = vel / speed * self.v_max
            
        return next_state