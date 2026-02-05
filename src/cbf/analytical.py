import numpy as np
from src.cbf.base import ControlBarrierFunction

class EllipticalCBF(ControlBarrierFunction):
    def __init__(self, base_a=0.5, base_b=0.4, scale_a=0.5, scale_b=0.2):
        """
        논문 Figure 7 반영: 속도가 0일 때의 기본 크기 + 속도 비례 증가량
        """
        self.base_a = base_a
        self.base_b = base_b
        self.scale_a = scale_a # 속도가 1m/s 늘 때마다 장축이 얼마나 커질지
        self.scale_b = scale_b 

    def _get_ellipse_params(self, human_vel):
        """
        사람(또는 로봇)의 속도에 따라 타원 크기 동적 결정
        """
        # 1. 속도 및 헤딩 계산
        speed = np.linalg.norm(human_vel)
        if speed > 0.01:
            theta = np.arctan2(human_vel[1], human_vel[0])
        else:
            theta = 0.0
            
        # 2. [핵심 수정] 속도 비례형 타원 크기 (Linear Approximation of Eq. 13)
        # 속도가 빠를수록 안전구역이 커짐
        current_a = self.base_a + self.scale_a * speed
        current_b = self.base_b + self.scale_b * speed
        
        return current_a, current_b, theta

    def get_value(self, robot_state, human_state):
        p_r = robot_state[:2]
        p_h = human_state[:2]
        v_h = human_state[2:]
        
        # 동적 파라미터 가져오기
        a, b, theta = self._get_ellipse_params(v_h)
        
        # a가 b보다 작아지는 예외 상황 방지
        if a < b: a, b = b, a + 0.01
            
        # 초점 거리 c
        c_len = np.sqrt(a**2 - b**2)
        
        # 초점 벡터 회전
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        c_vec = np.array([c_len * cos_t, c_len * sin_t])
        
        c1 = p_h + c_vec
        c2 = p_h - c_vec
        
        # 거리 합 계산
        dist_sum = np.linalg.norm(p_r - c1) + np.linalg.norm(p_r - c2)
        
        # h(x)
        h_val = (dist_sum / 2.0) - a
        
        return h_val

    def get_gradient(self, robot_state, human_state):
        # 수치 미분 (동일함)
        delta = 1e-5
        grad = np.zeros(4)
        original_val = self.get_value(robot_state, human_state)
        
        for i in range(2): 
            perturb = np.zeros(4)
            perturb[i] = delta
            val_plus = self.get_value(robot_state + perturb, human_state)
            grad[i] = (val_plus - original_val) / delta
            
        return grad