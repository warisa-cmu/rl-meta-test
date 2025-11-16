import numpy as np


class FixedParamAgent:
    def __init__(self, F, CR, MG):
        self.F = F
        self.CR = CR
        self.MG = MG

    def predict(self, observation, deterministic=True):
        return np.array([self.F, self.CR, self.MG]), None
