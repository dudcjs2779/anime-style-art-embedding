import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def group_cosine_lr(optimizer, base_lrs, min_lrs, warmup_length, steps):

    def _lr_adjuster(step):
        lrs = [0] * len(base_lrs)
        for i, group in enumerate(optimizer.param_groups):
            base_lr = base_lrs[i]
            min_lr = min_lrs[i]

            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr) + min_lr
            group['lr'] = lr
            lrs[i] = lr
        return lrs

    return _lr_adjuster

class GroupCosineLR():
    def __init__(self, optimizer, default_lr, min_lr, warmup_length, steps):
        self.base_lrs =  [group.get("lr", default_lr) for group in optimizer.param_groups]

        if not isinstance(min_lr, float):
            raise ValueError("min_lr should be a float, but got {}".format(type(min_lr)))
        
        self.min_lrs = [min_lr] * len(self.base_lrs)
        self.scheduler = group_cosine_lr(optimizer, self.base_lrs, self.min_lrs, warmup_length, steps)

    def __call__(self, step):
        return self.scheduler(step)
    
    
class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

class StepGenerator:
    """
    전반부/후반부로 갈수록 이벤트가 더 자주 발생하는 step 생성
    """
    def __init__(self, total_steps: int, num_events: int, shape: float = 0.5, start_ratio: float = 0.0, ):
        if start_ratio == 0.0:
            t = np.linspace(0, 1, num_events + 2)[1:-1]
        else:
            t = np.linspace(0, 1, num_events + 1)[:-1]

        curved = t ** shape * (1-start_ratio) + start_ratio
        steps = curved * total_steps

        self.step_list = np.unique(steps.astype(int)).tolist()
        self.steps = set(self.step_list)

    def is_update_step(self, step: int) -> bool:
        return step in self.steps
        

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


class InterpolationScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=None,
        freeze_iters=0,
        power=1.0,
    ):
        super().__init__()

        self.final_value = final_value
        self.total_iters = total_iters

        if start_warmup_value is None:
            start_warmup_value = base_value

        freeze_schedule = np.full(
            (freeze_iters,),
            start_warmup_value,
            dtype=np.float64,
        )

        warmup_schedule = np.linspace(
            start_warmup_value,
            base_value,
            warmup_iters,
        )

        schedule_iters = total_iters - warmup_iters - freeze_iters

        iters = np.arange(schedule_iters)

        # 0 ~ 1
        t = iters / max(schedule_iters - 1, 1)
        t = t ** power

        schedule = (
            base_value
            + (final_value - base_value) * t
        )

        self.schedule = np.concatenate(
            (
                freeze_schedule,
                warmup_schedule,
                schedule,
            )
        )

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]
