import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from src.dynamics.robot import RobotDynamics

class SocialNavEnv:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.robot_dyn = RobotDynamics(dt=dt)
        
        # initialize
        self.robot_state = np.zeros(4) 
        self.human_state = np.array([5.0, 0.1, -1.0, 0.0]) 
        
        # visualize setting
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-2, 8)
        self.ax.set_ylim(-5, 5)
        self.ax.grid(True)

        self.robot_patch, = self.ax.plot([], [], 'bo', markersize=10, label='Robot')
        self.human_patch, = self.ax.plot([], [], 'rx', markersize=10, label='Human')
        self.traj_x, self.traj_y = [], []
        self.traj_patch, = self.ax.plot([], [], 'b--', alpha=0.3)
        self.ax.legend()

    def reset(self):
        self.robot_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.human_state = np.array([5.0, 0.1, -1.0, 0.0])
        self.traj_x, self.traj_y = [], []
        return self.robot_state, self.human_state

    def step(self, control_input):
        # Robot Step
        self.robot_state = self.robot_dyn.step(self.robot_state, control_input)
        
        # Human Step (Constant Velocity Model)
        self.human_state[0] += self.human_state[2] * self.dt
        self.human_state[1] += self.human_state[3] * self.dt
        
        # save traj
        self.traj_x.append(self.robot_state[0])
        self.traj_y.append(self.robot_state[1])
        
        return self.robot_state, self.human_state

    def render(self, ellipse_params=None):
        self.ax.clear()
        self.ax.set_xlim(-2, 10)
        self.ax.set_ylim(-6, 6)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        robot = Circle((self.robot_state[0], self.robot_state[1]), 0.3, color='blue', label='Robot')
        self.ax.add_patch(robot)
        
        human = Circle((self.human_state[0], self.human_state[1]), 0.3, color='green', label='Human')
        self.ax.add_patch(human)
        
        if ellipse_params is not None:
            a, b, theta = ellipse_params
            angle_deg = np.degrees(theta)
            
            safety_zone = Ellipse(
                xy=(self.human_state[0], self.human_state[1]),
                width=2*a, height=2*b, angle=angle_deg,
                edgecolor='red', facecolor='none', linestyle='--', linewidth=1, label='Safety Zone'
            )
            self.ax.add_patch(safety_zone)

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)