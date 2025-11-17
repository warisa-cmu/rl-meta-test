import numpy as np


class FixedParamAgent:
    def __init__(self, name, F_sc, CR_sc, MG_sc):
        self.model_name = name
        self.F_sc = F_sc
        self.CR_sc = CR_sc
        self.MG_sc = MG_sc

    def predict(self, observation, deterministic=True):
        return np.array([self.F_sc, self.CR_sc, self.MG_sc]), None
