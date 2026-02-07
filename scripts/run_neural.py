import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.env import SocialNavEnv
from src.control.mpc_solver import MPCSolver
from src.cbf.neural_cbf import NeuralCBFNetwork

class NeuralCBFWrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        self.model = NeuralCBFNetwork()
        
        if torch.backends.mps.is_available() and device == 'cpu':
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path, map_location=device)
            
        self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval() # Model을 evaluation 모드로 전환.
        
    def get_value(self, robot_state, human_state):
        """ h(x) 값 계산 (Forward) """
        rel_pos = robot_state[:2] - human_state[:2]
        hum_vel = human_state[2:]
        
        input_data = np.hstack([rel_pos, hum_vel])
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            h_val = self.model(input_tensor)
            
        return h_val.item()

    def get_gradient(self, robot_state, human_state):
        # Input data
        rel_pos = robot_state[:2] - human_state[:2]
        hum_vel = human_state[2:]
        input_data = np.hstack([rel_pos, hum_vel])
        
        # Generate Tensor
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True 
        
        # Model pass(Forward)
        h_val = self.model(input_tensor)
        
        # Backward
        self.model.zero_grad() 
        h_val.backward()   #미분
        
        # Extract gradient 
        # input_tensor.grad : [d(rel_x), d(rel_y), d(v_x), d(v_y)]
        grads = input_tensor.grad.cpu().detach().numpy()[0]
        
     
        # dh/d(px) = dh/d(rel_x) * d(rel_x)/d(px) = grads[0] * 1
        grad_px = grads[0]
        grad_py = grads[1]
        
        # output : [grad_px, grad_py, 0, 0]
        return np.array([grad_px, grad_py, 0.0, 0.0])

    # visualization
    def _get_ellipse_params(self, hum_vel):
        speed = np.linalg.norm(hum_vel)
        base_a = 0.5; scale_a = 0.5
        base_b = 0.4; scale_b = 0.2
        a = base_a + scale_a * speed
        b = base_b + scale_b * speed
        if speed > 0.1:
            theta = np.arctan2(hum_vel[1], hum_vel[0])
        else:
            theta = 0.0
        return a, b, theta
    
    
def main():
    # initial setting
    dt = 0.1
    env = SocialNavEnv(dt=dt)
    robot_state, human_state = env.reset()
    goal_state = np.array([8.0, 0.0, 0.0, 0.0])
    
    # load Neural CBF 로드
    device = torch.device("cpu") 
    model_path = "models/cbf_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train_cbf.py first.")
        return

    print("Loading Neural CBF Model...")
    # AnalyticalCBF 대신 NeuralCBFWrapper를 사용
    cbf = NeuralCBFWrapper(model_path, device=device)
    
    mpc = MPCSolver(dt=dt, N=10) 
    
    print("Simulation Start with Neural Network Control...")
    
    # data logging
    h_history = []
    u_history = []
    time_history = []
    
    plt.ion()
    
    for i in range(100):
        # MPC Solve (여기서 cbf.get_value()가 호출됨 -> 신경망이 계산)
        u_opt = mpc.solve(robot_state, goal_state, human_state, cbf)
        
        # Env Step
        robot_state, human_state = env.step(u_opt)
        
        # 검증용 값 계산
        h_val = cbf.get_value(robot_state, human_state)
        
        # 시각화용 파라미터
        a, b, theta = cbf._get_ellipse_params(human_state[2:])
        
        # 데이터 저장
        h_history.append(h_val)
        u_history.append(u_opt)
        time_history.append(i * dt)
        
        # 렌더링
        env.render(ellipse_params=(a, b, theta))
        
        print(f"Step {i:03d} | Robot: ({robot_state[0]:.2f}, {robot_state[1]:.2f}) | Neural h(x): {h_val:.4f} | Action: {u_opt}")
        
        if np.linalg.norm(robot_state[:2] - goal_state[:2]) < 0.2:
            print(">>> GOAL REACHED! <<<")
            break
            
    print("Simulation Finished.")
    plt.ioff()
    plt.close()
    
    # 결과 그래프
    print("Plotting Neural CBF Results...")
    u_history = np.array(u_history)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    ax[0].plot(time_history, h_history, 'r-', linewidth=2)
    ax[0].axhline(y=0, color='k', linestyle='--', label='Safety Boundary (h=0)')
    ax[0].set_title("Neural Safety Value h(x) over Time")
    ax[0].set_ylabel("h(x) (Predicted by AI)")
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