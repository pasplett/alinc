class EarlyStoppingCB:

    def __init__(self, mode="min", patience=20):
        """Early stopping callback object. Keeps track of a specific metric <x>
        and returns True if the metric has not decreased for <patience> calls.

        Parameters
        ----------
        mode : str, optional
            Either "min" or "max", by default "min"
        patience: int, optional
            Number of object calls to wait for a metric decrease, by default 20
        """
        self.mode = mode
        if self.mode == "min":
            self.best = 1e8
        elif self.mode == "max":
            self.best = -1e8
        else:
            raise KeyError(f"{mode} is not a valid early stopping mode!")
        
        self.patience = patience
        self.n_calls = 0

    def __call__(self, x):
        if self.mode == "min" and x < self.best:
            self.best = x
            self.n_calls = 0
            return False
        
        if self.mode == "max" and x > self.best:
            self.best = x
            self.n_calls = 0
            return False
        
        if self.n_calls + 1 < self.patience:
            self.n_calls += 1
            return False
        
        if self.n_calls + 1 >= self.patience:
            return True
        
    def info(self):
        return f"{self.n_calls} call(s).\n" + \
            f"Current best value: {self.best}.\n" + \
            f"{self.patience - self.n_calls} call(s) until early stopping."
    

class BestModelCB:

    def __init__(self, mode="min"):
        """Best model callback object. Keeps track of a specific metric <x>
        and returns True if the metric has improved.

        Parameters
        ----------
        mode : str, optional
            Either "min" or "max", by default "min"
        """
        self.mode = mode
        if self.mode == "min":
            self.best = 1e8
        elif self.mode == "max":
            self.best = -1e8
        else:
            raise KeyError(f"{mode} is not a valid early stopping mode!")

    def __call__(self, x):
        if self.mode == "min" and x < self.best:
            self.best = x
            return True
        elif self.mode == "max" and x > self.best:
            self.best = x
            return True
        else:
            return False