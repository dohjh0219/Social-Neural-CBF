import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.env import SocialNavEnv
from src.cbf.analytical import EllipticalCBF
from src.control.mpc_solver import MPCSolver

def main():
    # 1. 초기 설정
    dt = 0.1
    env = SocialNavEnv(dt=dt)
    robot_state, human_state = env.reset()
    
    # 목표 지점 설정: (8, 0)
    goal_state = np.array([8.0, 0.0, 0.0, 0.0])
    
    # [수정됨] CBF 모듈 생성 (속도 반응형 파라미터 적용)
    # base_a=0.5, scale_a=0.5 의미: 정지시 0.5m, 속도 1m/s당 0.5m씩 커짐
    cbf = EllipticalCBF(base_a=0.5, base_b=0.4, scale_a=0.5, scale_b=0.2)
    
    # MPC Solver 생성
    mpc = MPCSolver(dt=dt, N=10) 
    
    print("Simulation Start with MPC (Adaptive Ellipse)...")
    
    plt.ion()
    
    for i in range(100):
        # 2. MPC로 최적 제어 입력 계산
        u_opt = mpc.solve(robot_state, goal_state, human_state, cbf)
        
        # 3. 환경 업데이트
        robot_state, human_state = env.step(u_opt)
        
        # 4. 검증용 CBF 값 계산
        h_val = cbf.get_value(robot_state, human_state)
        
        # 5. 렌더링 및 로그
        env.render()
        
        print(f"Step {i:03d} | Robot: ({robot_state[0]:.2f}, {robot_state[1]:.2f}) | h(x): {h_val:.4f} | Action: {u_opt}")
        
        # 종료 조건
        dist_to_goal = np.linalg.norm(robot_state[:2] - goal_state[:2])
        if dist_to_goal < 0.2:
            print(">>> GOAL REACHED! <<<")
            break
            
    print("Simulation Finished.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()