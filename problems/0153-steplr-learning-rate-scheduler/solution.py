class StepLRScheduler:

    def __init__(self, initial_lr: float, step_size: int, gamma: float):
        '''Initialize initial_lr, step_size, and gamma using PyTorch tensors'''

        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        '''Calculate and return the learning rate for the given epoch'''

        return round(self.initial_lr * self.gamma ** (epoch // self.step_size), 4)
