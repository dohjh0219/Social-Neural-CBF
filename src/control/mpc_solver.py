import cvxpy as cp
import numpy as np

class MPCSolver:
    def __init__(self, dt=0.1, N=10):
        self.dt = dt
        self.N = N
        
        # 최적화 변수 (Horizon 길이만큼)
        self.U = cp.Variable((2, N))      # 가속도 입력 [ax, ay]
        self.X = cp.Variable((4, N + 1))  # 상태 [px, py, vx, vy]
        
        # 파라미터 (매 스텝 외부에서 꽂아주는 값)
        self.x_init_param = cp.Parameter(4)     # 현재 로봇 상태
        self.x_ref_param = cp.Parameter(4)      # 목표 상태
        
        # --- 비용 함수 (Cost) 설정 ---
        cost = 0
        Q = np.diag([10, 10, 1, 1])  # 상태 오차 가중치
        R = np.diag([1, 1])          # 입력 가중치
        
        # 제약 조건 리스트
        self.constraints = []
        
        # 1. 초기 상태 제약
        self.constraints.append(self.X[:, 0] == self.x_init_param)
        
        for k in range(N):
            # 2. 비용 누적
            state_err = self.X[:, k+1] - self.x_ref_param
            cost += cp.quad_form(state_err, Q) + cp.quad_form(self.U[:, k], R)
            
            # 3. 동역학 제약 (선형 시스템)
            # x_{k+1} = A x_k + B u_k
            self.constraints.append(self.X[0, k+1] == self.X[0, k] + self.X[2, k]*dt + 0.5*self.U[0, k]*dt**2)
            self.constraints.append(self.X[1, k+1] == self.X[1, k] + self.X[3, k]*dt + 0.5*self.U[1, k]*dt**2)
            self.constraints.append(self.X[2, k+1] == self.X[2, k] + self.U[0, k]*dt)
            self.constraints.append(self.X[3, k+1] == self.X[3, k] + self.U[1, k]*dt)
            
            # 4. 입력 제한
            self.constraints.append(self.U[:, k] >= -2.0)
            self.constraints.append(self.U[:, k] <= 2.0)

        # [수정된 부분] 5. 목적 함수 저장 (self.objective 변수에 할당)
        # 이 줄이 없어서 에러가 발생했습니다.
        self.objective = cp.Minimize(cost)
        
        # 기본 문제 정의 (일단 저장용)
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, current_state, target_state, human_state, cbf_model):
        self.x_init_param.value = current_state
        self.x_ref_param.value = target_state
        
        # 1. CBF 수치 계산
        h_val = cbf_model.get_value(current_state, human_state)
        h_grad = cbf_model.get_gradient(current_state, human_state)
        
        # 2. 슬랙 변수 생성 (위반 허용량)
        delta = cp.Variable(1) 
        
        # 3. 안전 제약 (Soft Constraint)
        # h_next >= -gamma * h + delta
        # delta가 커질수록(위반할수록) 비용이 엄청나게 커지게 설정
        gamma = 0.2  # 감쇠율 (낮을수록 더 보수적/안전하게 움직임)
        x_next = self.X[:, 1]
        
        cbf_constraint = [
            h_grad @ (x_next - current_state) >= -gamma * h_val - delta,
            delta >= 0 
        ]
        
        # 4. 전체 제약 및 목적함수 재구성
        # 기존 목적함수 + 슬랙 변수 페널티 (가중치 1e5로 매우 크게)
        total_cost = self.objective.args[0] + 100000 * cp.square(delta)
        all_constraints = self.constraints + cbf_constraint
        
        prob = cp.Problem(cp.Minimize(total_cost), all_constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if self.U.value is None:
                print("[MPC Warning] Infeasible solution found.")
                return np.array([0.0, 0.0]) # 그래도 실패하면 정지
            
            # 성공 시 제어 입력 반환
            return self.U.value[:, 0]
            
        except Exception as e:
            print(f"[MPC Error] {e}")
            return np.array([0.0, 0.0])