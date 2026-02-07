import cvxpy as cp
import numpy as np

class MPCSolver:
    def __init__(self, dt=0.1, N=10):
        self.dt = dt
        self.N = N
        
        # optimize variables
        self.U = cp.Variable((2, N))  # acc input[ax, ay]
        self.X = cp.Variable((4, N + 1))  # state [px, py, vx, vy]
        
        # params
        self.x_init_param = cp.Parameter(4)
        self.x_ref_param = cp.Parameter(4)
        
        # SETTING COST FUNCTION
        Q = np.diag([10, 10, 1, 1]) # Q: 경로 추종 가중치
        R = np.diag([1, 1]) # R: energy  efficiency 가중치    
        P = np.diag([20, 20, 5, 5]) # P: terminal 비용 가중치(Q보다 크거나 같게 설정하면 수렴성?)
        
        cost = 0
        self.constraints = []
        
        # Initial state constraints
        self.constraints.append(self.X[:, 0] == self.x_init_param)
        
        for k in range(N):
            # Stage Cost
            state_err = self.X[:, k] - self.x_ref_param
            cost += cp.quad_form(state_err, Q) + cp.quad_form(self.U[:, k], R)
            
            # Dynamics constraints
            self.constraints.append(self.X[0, k+1] == self.X[0, k] + self.X[2, k]*dt + 0.5*self.U[0, k]*dt**2)
            self.constraints.append(self.X[1, k+1] == self.X[1, k] + self.X[3, k]*dt + 0.5*self.U[1, k]*dt**2)
            self.constraints.append(self.X[2, k+1] == self.X[2, k] + self.U[0, k]*dt)
            self.constraints.append(self.X[3, k+1] == self.X[3, k] + self.U[1, k]*dt)
            
            # input cliping
            self.constraints.append(self.U[:, k] >= -2.0)
            self.constraints.append(self.U[:, k] <= 2.0)

        # Terminal Cost : 마지막 step에서의 에러
        terminal_err = self.X[:, N] - self.x_ref_param
        cost += cp.quad_form(terminal_err, P)

        self.objective = cp.Minimize(cost)
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, current_state, target_state, human_state, cbf_model):
        # 파라미터 업데이트
        self.x_init_param.value = current_state
        self.x_ref_param.value = target_state
        
        # CBF 계산
        h_val = cbf_model.get_value(current_state, human_state)
        h_grad = cbf_model.get_gradient(current_state, human_state)
        
        # 슬랙 변수 (Soft Constraint)
        delta = cp.Variable(1)
        
        # CBF 제약 조건
        gamma = 0.05
        x_next = self.X[:, 1]
        
        cbf_constraint = [
            h_grad @ (x_next - current_state) >= -gamma * h_val - delta,
            delta >= 0
        ]
        




        # 최종 cost function : 기본 cost function + Slack Variable panelty
        total_cost = self.objective.args[0] + 100000 * cp.square(delta)
        all_constraints = self.constraints + cbf_constraint
        
        prob = cp.Problem(cp.Minimize(total_cost), all_constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if self.U.value is None:
                return np.array([0.0, 0.0])
            return self.U.value[:, 0]
        except:
            return np.array([0.0, 0.0])