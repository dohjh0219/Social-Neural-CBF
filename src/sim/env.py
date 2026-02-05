import numpy as np
import matplotlib.pyplot as plt
from src.dynamics.robot import RobotDynamics

class SocialNavEnv:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.robot_dyn = RobotDynamics(dt=dt)
        
        # 초기화
        self.robot_state = np.zeros(4) # [0,0,0,0]
        self.human_state = np.array([5.0, 0.1, -1.0, 0.0]) # (5,0)에서 왼쪽으로 옴
        
        # 시각화 설정
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-2, 8)
        self.ax.set_ylim(-5, 5)
        self.ax.grid(True)
        
        # 그래픽 객체들
        self.robot_patch, = self.ax.plot([], [], 'bo', markersize=10, label='Robot')
        self.human_patch, = self.ax.plot([], [], 'rx', markersize=10, label='Human')
        self.traj_x, self.traj_y = [], []
        self.traj_patch, = self.ax.plot([], [], 'b--', alpha=0.3)
        self.ax.legend()

    def reset(self):
        self.robot_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.human_state = np.array([5.0, 0.1, -1.0, 0.0]) # 단순 등속 운동 사람
        self.traj_x, self.traj_y = [], []
        return self.robot_state, self.human_state

    def step(self, control_input):
        # 1. Robot Step
        self.robot_state = self.robot_dyn.step(self.robot_state, control_input)
        
        # 2. Human Step (Constant Velocity Model)
        # x_{k+1} = x_k + v_k * dt
        self.human_state[0] += self.human_state[2] * self.dt
        self.human_state[1] += self.human_state[3] * self.dt
        
        # 궤적 저장
        self.traj_x.append(self.robot_state[0])
        self.traj_y.append(self.robot_state[1])
        
        return self.robot_state, self.human_state

    def render(self):
        # 로봇 업데이트
        self.robot_patch.set_data([self.robot_state[0]], [self.robot_state[1]])
        
        # 사람 업데이트
        self.human_patch.set_data([self.human_state[0]], [self.human_state[1]])
        
        # 궤적 업데이트
        self.traj_patch.set_data(self.traj_x, self.traj_y)
        
        plt.pause(0.05) # 잠시 멈춰서 화면 갱신