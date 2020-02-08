import numpy as np
from skopt.callbacks import CheckpointSaver, EarlyStopper

class EarlyStopperNoImprovement(EarlyStopper):
    def __init__(self, iters):
        self.iters = iters

    def _criterion(self, result):
        best_index = np.argmin(result['func_vals'])

        if len(result['func_vals']) - best_index > self.iters:
            print("Early stopping.")
            return True

        return False
