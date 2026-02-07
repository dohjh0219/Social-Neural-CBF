import sys
import os
import numpy as np

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cbf.analytical import EllipticalCBF

def generate_dataset(num_samples=50000, save_path="data/cbf_data.npz"):
    print(f"Generating {num_samples} samples for training...")
    
    # 1. 선생님(Analytical CBF) 불러오기
    # run_baseline.py와 동일한 파라미터 사용 (이게 정답지 기준이 됨)
    cbf = EllipticalCBF(base_a=0.5, base_b=0.4, scale_a=0.5, scale_b=0.2)
    
    # 2. 랜덤한 입력 데이터 생성 (Features)
    # ---------------------------------------------------------
    # (1) 상대 위치 (Relative Position: Robot - Human)
    # 범위: -3m ~ 3m (충돌 위험이 있는 근거리 위주로 샘플링)
    # 너무 멀면 h 값이 무의미하게 커지므로, 학습 효율을 위해 근거리 집중
    rel_pos = np.random.uniform(-3.0, 3.0, (num_samples, 2))
    
    # (2) 사람 속도 (Human Velocity)
    # 범위: -2m/s ~ 2m/s (정지 ~ 달리기)
    # 사람 속도에 따라 타원 모양이 변하므로 다양한 속도를 학습해야 함
    hum_vel = np.random.uniform(-2.0, 2.0, (num_samples, 2))
    
    # 3. 정답 데이터 계산 (Labels: h(x))
    # ---------------------------------------------------------
    h_values = []
    
    print("Calculating h(x) values using Analytical CBF...")
    for i in range(num_samples):
        # 가상의 상태 생성
        # 사람은 원점(0,0)에 있다고 가정하고, 로봇을 상대 위치에 배치
        # Neural Network는 '상대적인 관계'만 알면 되므로 절대 좌표는 중요하지 않음
        
        # Human State: [px=0, py=0, vx, vy]
        dummy_human_state = np.array([0.0, 0.0, hum_vel[i, 0], hum_vel[i, 1]])
        
        # Robot State: [px=rel_x, py=rel_y, vx=0, vy=0]
        # h(x) 값 자체는 로봇 속도와 무관함 (위치 기하학적 관계 = State Constraint)
        dummy_robot_state = np.array([rel_pos[i, 0], rel_pos[i, 1], 0.0, 0.0])
        
        # Teacher에게 물어보기: "이 상황에서 h값이 뭐야?"
        h = cbf.get_value(dummy_robot_state, dummy_human_state)
        h_values.append(h)
        
    h_values = np.array(h_values).reshape(-1, 1) # (N, 1) 형태로 변환
    
    # 4. 데이터 저장
    # ---------------------------------------------------------
    # 입력 데이터(X) 합치기: [rel_x, rel_y, v_x, v_y] -> Shape: (N, 4)
    # 신경망은 이 4개 숫자를 보고 h를 맞추게 됨
    X_data = np.hstack([rel_pos, hum_vel])
    Y_data = h_values
    
    # data 폴더가 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 압축 저장 (.npz)
    np.savez(save_path, X=X_data, Y=Y_data)
    
    print(f"\n[Success] Data saved to {save_path}")
    print(f"- Feature Shape (X): {X_data.shape}") # (50000, 4)
    print(f"- Label Shape (Y):   {Y_data.shape}")   # (50000, 1)
    
    # 데이터 분포 확인 (너무 한쪽으로 쏠리지 않았는지 확인용)
    safe_cnt = np.sum(Y_data > 0)
    unsafe_cnt = np.sum(Y_data <= 0)
    print(f"- Safe samples (h > 0):   {safe_cnt} ({safe_cnt/num_samples*100:.1f}%)")
    print(f"- Unsafe samples (h <= 0): {unsafe_cnt} ({unsafe_cnt/num_samples*100:.1f}%)")
    print("  (Ratio around 3:1 to 4:1 is good for training)")

if __name__ == "__main__":
    generate_dataset()