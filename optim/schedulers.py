import torch

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        # optimizer = Adam (Parameter Group 0
        #    amsgrad: False
        #    betas: (0.9, 0.98)
        #    eps: 1e-09
        #    lr: 0
        #    weight_decay: 0
        #)
        self._step = 0
        self.warmup = warmup # e.g., 4000 轮 热身
        self.factor = factor # e.g., 2
        self.model_size = model_size # 512
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate`(learning rate) above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, 
            betas=(0.9, 0.98), eps=1e-9))

