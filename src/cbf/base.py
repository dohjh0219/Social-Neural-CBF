from abc import ABC, abstractmethod
import numpy as np

class ControlBarrierFunction(ABC):
    """
    모든 CBF 클래스(Analytical, Neural)가 상속받아야 할 기본 클래스
    """
    
    @abstractmethod
    def get_value(self, robot_state, human_state):
        """
        상태 x가 주어졌을 때 h(x) 값을 반환해야 함.
        :return: h(x) (scalar)
        """
        pass

    @abstractmethod
    def get_gradient(self, robot_state, human_state):
        """
        최적화(MPC)를 위해 h(x)의 로봇 상태에 대한 기울기(Gradient)를 반환해야 함.
        :return: dh/dx_robot (shape: 4,)
        """
        pass