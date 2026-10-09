from math import inf


class EarlyStopping:

    def __init__(self, patience: int, mode: str = 'min'):
        '''store patience, mode, best (initialized to +/-inf), and a bad-step counter'''

        self.patience = patience
        self.mode = mode
        self.best = inf if mode == 'min' else -inf
        self.cnt = 0

    def step(self, metric: float) -> bool:
        '''update best/counter, return True iff should stop'''

        if self.mode == 'min' and metric < self.best or self.mode == 'max' and metric > self.best:
            self.best = metric
            self.cnt = 0
        else:
            self.cnt += 1
        return self.cnt >= self.patience
