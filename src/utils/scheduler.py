import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def group_cosine_lr(optimizer, base_lrs, warmup_length, steps):
        def _lr_adjuster(step):
            lrs = [0] * len(base_lrs)
            for i, group in enumerate(optimizer.param_groups):
                base_lr = base_lrs[i]
                if step < warmup_length:
                    lr = _warmup_lr(base_lr, warmup_length, step)
                else:
                    e = step - warmup_length
                    es = steps - warmup_length
                    lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
                group['lr'] = lr
                lrs[i] = lr
            return lrs

        return _lr_adjuster

class GroupCosineLR():
    def __init__(self, optimizer, default_lr, warmup_length, steps):
        self.base_lrs =  [group.get("lr", default_lr) for group in optimizer.param_groups]
        self.scheduler = group_cosine_lr(optimizer, self.base_lrs, warmup_length, steps)

    def __call__(self, step):
        return self.scheduler(step)

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, cycle_step, cy_mult=1, lr_max=0.1, lr_min = 0.0, warmup=0, gamma=1., last_epoch=-1):
        if cycle_step <= 0 or not isinstance(cycle_step, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(cycle_step))
        # if T_mult < 1 or not isinstance(T_mult, int):
        #     raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if warmup < 0 or not isinstance(warmup, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(warmup))
        self.cycle_step = cycle_step
        self.cy_mult = cy_mult
        self.base_lr_max = lr_max
        self.lr_max = lr_max
        self.lr_min = [lr_min]
        self.warmup = warmup
        self.cy_i = cycle_step
        self.gamma = gamma
        self.cycle = 0
        self.cy_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.cy_cur == -1:
            return self.lr_min
        elif self.cy_cur < self.warmup:
            return [(self.lr_max - base_lr)*self.cy_cur / self.warmup + base_lr for base_lr in self.lr_min]
        else:
            return [base_lr + (self.lr_max - base_lr) * (1 + math.cos(math.pi * (self.cy_cur-self.warmup) / (self.cy_i - self.warmup))) / 2
                    for base_lr in self.lr_min]

    def __call__(self, step=None):
        if step is None:
            step = self.last_epoch + 1
            self.cy_cur = self.cy_cur + 1
            if self.cy_cur >= self.cy_i:
                self.cycle += 1
                self.cy_cur = self.cy_cur - self.cy_i
                self.cy_i = (self.cy_i - self.warmup) * self.cy_mult + self.warmup
        else:
            if step >= self.cycle_step:
                if self.cy_mult == 1:
                    self.cy_cur = step % self.cycle_step
                    self.cycle = step // self.cycle_step
                else:
                    n = int(math.log((step / self.cycle_step * (self.cy_mult - 1) + 1), self.cy_mult))
                    self.cycle = n
                    self.cy_cur = step - self.cycle_step * (self.cy_mult ** n - 1) / (self.cy_mult - 1)
                    self.cy_i = self.cycle_step * self.cy_mult ** (n)
            else:
                self.cy_i = self.cycle_step
                self.cy_cur = step
                
        self.lr_max = self.base_lr_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(step)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    