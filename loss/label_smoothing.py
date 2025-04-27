import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        # x: log probabilities (batch_size x vocab_size)
        assert x.size(1) == self.size

        true_dist = torch.full_like(x, self.smoothing / (self.size - 2))#36,5
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)#36,5
        true_dist[:, self.padding_idx] = 0

        # Zero out rows where target == padding_idx
        # mask = (target == self.padding_idx).nonzero(as_tuple=False)
        mask = target == self.padding_idx
        true_dist[mask] = 0
        if mask.numel() > 0:
            true_dist.index_fill_(0, mask.long(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist)
        #return :scaler

