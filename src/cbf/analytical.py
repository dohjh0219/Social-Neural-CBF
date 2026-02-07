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
        Determine Ellipse parameters by person(or robot)'s velocity.
        """
        # Calculate speed and theta
        speed = np.linalg.norm(human_vel)
        if speed > 0.01:
            theta = np.arctan2(human_vel[1], human_vel[0])
        else:
            theta = 0.0
            
        # Adaptive Ellipse size by speed.(eq. 13에 이용되는 a(s_r)과 b(s_r)에 해당됨.)
        # 속도가 빠를수록 ellipse 커짐
        current_a = self.base_a + self.scale_a * speed
        current_b = self.base_b + self.scale_b * speed
        
        return current_a, current_b, theta

    def get_value(self, robot_state, human_state): # cbf value calculate
        # Eq. 18
        p_r = robot_state[:2]
        p_h = human_state[:2]
        v_h = human_state[2:]
        
        a, b, theta = self._get_ellipse_params(v_h)
        # 사람 속도에 따라 ellipse 사이즈 달라짐.
        
        # a < b ; Exceptional event X
        if a < b: a, b = b, a + 0.01
            
        # 초점 c
        c_len = np.sqrt(a**2 - b**2)
        
        # 초점 벡터 회전
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        c_vec = np.array([c_len * cos_t, c_len * sin_t])
        
        c1 = p_h + c_vec # 앞쪽 초점
        c2 = p_h - c_vec # 뒤쪽 초점

        # h(x)  : Eq 18
        dist_sum = np.linalg.norm(p_r - c1) + np.linalg.norm(p_r - c2)
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