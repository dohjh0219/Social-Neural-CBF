import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.env import SocialNavEnv
from src.cbf.analytical import EllipticalCBF
from src.control.mpc_solver import MPCSolver

def main():
    # initial setting
    dt = 0.1
    env = SocialNavEnv(dt=dt)
    robot_state, human_state = env.reset()
    
    # target state setting
    goal_state = np.array([8.0, 0.0, 0.0, 0.0])
    
    # CBF MODULE GENERATE
    # base_a=0.5, scale_a=0.5 의미: 정지시 0.5m, 속도 1m/s당 0.5m씩 커짐
    cbf = EllipticalCBF(base_a=0.5, base_b=0.4, scale_a=0.5, scale_b=0.2)
    
    # MPC SOLVER GENERATE
    mpc = MPCSolver(dt=dt, N=10) 
    
    print("Simulation Start with MPC (Adaptive Ellipse)...")
    
    # [추가] 데이터 로깅을 위한 리스트
    h_history = []
    u_history = []
    time_history = []
    
    plt.ion()
    
    for i in range(100):
        # Calculate Optimal control input using MPC
        u_opt = mpc.solve(robot_state, goal_state, human_state, cbf)
        
        # env Update
        robot_state, human_state = env.step(u_opt)
        
        # 4. 검증용 CBF 값 계산
        h_val = cbf.get_value(robot_state, human_state)
        
        # [추가] 시각화를 위해 현재 사람 속도에 맞는 타원 파라미터 가져오기
        # analytical.py의 _get_ellipse_params 메서드 활용
        a, b, theta = cbf._get_ellipse_params(human_state[2:])
        
        # [추가] 데이터 저장 (Graph Plotting용)
        h_history.append(h_val)
        u_history.append(u_opt)
        time_history.append(i * dt)
        
        # Rendering + Logging (타원 파라미터 전달)
        env.render(ellipse_params=(a, b, theta))
        
        print(f"Step {i:03d} | Robot: ({robot_state[0]:.2f}, {robot_state[1]:.2f}) | h(x): {h_val:.4f} | Action: {u_opt}")
        
        # End condition
        dist_to_goal = np.linalg.norm(robot_state[:2] - goal_state[:2])
        if dist_to_goal < 0.2:
            print(">>> GOAL REACHED! <<<")
            break
            
    print("Simulation Finished.")
    plt.ioff()
    plt.close()
    
  


    print("Plotting Result Graphs...")
    
    u_history = np.array(u_history)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    ax[0].plot(time_history, h_history, 'r-', linewidth=2)
    ax[0].axhline(y=0, color='k', linestyle='--', label='Safety Boundary (h=0)')
    ax[0].set_title("Safety Value h(x) over Time")
    ax[0].set_ylabel("h(x)")
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].plot(time_history, u_history[:, 0], 'b-', label='Acc X')
    ax[1].plot(time_history, u_history[:, 1], 'g-', label='Acc Y')
    ax[1].set_title("Control Input (Acceleration) over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Acceleration (m/s^2)")
    ax[1].grid(True)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()