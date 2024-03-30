# non_rl_policies.py >

import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, Ts):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self._integral_error = 0.0
        self._previous_error = 0.0
    
    def predict(self, observations, **kwargs):
        assert len(observations[0]) == 1, "observations should contain only one value, i.e., the error"
        fy_error = observations[0]
        self._integral_error += fy_error * self.Ts
        self._error_derivative = (fy_error - self._previous_error) / self.Ts
        self._previous_error = fy_error
        action = self.Kp * fy_error + self.Ki * self._integral_error + self.Kd * self._error_derivative
        return [action], []

class SS:
    def __init__(self, A, B, C, D, x0):
        self.x = x0
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def predict(self, observations, **kwargs):
        assert len(observations[0]) == 1, "observations should contain only one value, i.e., the error"
        u = observations[0] # u = fy_error
        action = np.matmul(self.C, self.x) + np.dot(self.D, u)
        self.x = np.matmul(self.A, self.x) + np.dot(self.B, u)
        return [[action]], []
    
class PrescribedAction:
    def __init__(self, alpha_ddot_prescribed):
        self.alpha_ddot_prescribed = alpha_ddot_prescribed
        self.step = 0
    
    def predict(self, observations, **kwargs):
        action = self.alpha_ddot_prescribed[self.step]
        self.step += 1
        return [action], []
