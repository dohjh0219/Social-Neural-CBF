import sys
import os
import numpy as np

# 프로젝트 루트 경로를 path에 추가하여 src 모듈을 찾을 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.env import SocialNavEnv

def main():
    # 환경 생성
    env = SocialNavEnv(dt=0.1)
    robot_state, human_state = env.reset()
    
    print("Simulation Start...")
    
    for i in range(100):
        # [Test] 임시 제어 입력: X축으로 0.5만큼 가속, Y축 0
        u_k = np.array([0.5, 0.0]) 
        
        # 시뮬레이션 스텝
        robot_state, human_state = env.step(u_k)
        
        # 렌더링
        env.render()
        
        # 간단한 로그
        if i % 10 == 0:
            print(f"Step {i}: Robot Pos ({robot_state[0]:.2f}, {robot_state[1]:.2f})")

    print("Simulation Finished.")

if __name__ == "__main__":
    main()