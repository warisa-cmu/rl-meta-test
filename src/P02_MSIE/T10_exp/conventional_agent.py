import numpy as np


class ConventionalAgent:
    def __init__(self, type_agent="CA1"):
        self.type_agent = type_agent

    def predict(self, observation, deterministic=True):
        if self.type_agent == "CA1":
            return np.array([1.0, 0.0, 0], dtype=np.float64), None
        elif self.type_agent == "CA2":
            return np.array([0.0, 1.0, 0], dtype=np.float64), None
        elif self.type_agent == "CA3":
            return np.array([0.5, 0.5, 0], dtype=np.float64), None
        elif self.type_agent == "CA4":
            return np.array([0.7, 0.3, 0], dtype=np.float64), None
        elif self.type_agent == "CA5":
            return np.array([0.3, 0.7, 0], dtype=np.float64), None
        elif self.type_agent == "CA6":
            return np.array([0.5, 0.5, 0.5], dtype=np.float64), None
        else:
            raise ValueError(f"Unknown agent type: {self.type_agent}")
