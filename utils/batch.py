import torch
import numpy as np
#Batch(torch.tensor(src_padded, dtype=torch.long),
#                 torch.tensor(trg_padded, dtype=torch.long),
#                pad_idx)

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src #src.shape:4,10
        self.src_mask = (src != pad).unsqueeze(-2) #src_mask.shape:4,1,10
        if trg is not None:
            self.trg = trg[:, :-1] #trg=tgt.shape:4,9
            self.trg_y = trg[:, 1:] #trg_y = tgt_y.shape:4,9
            self.trg_mask = self.make_std_mask(self.trg, pad) #trg_mask = trg_mask.shape:4,9,9
            self.ntokens = (self.trg_y != pad).data.sum()#scaler
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # print('trg.shape:{}'.format(tgt.shape))
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def __getitem__(self, key):
        "Allow accessing the batch using a dictionary-like syntax"
        return getattr(self, key)  
    
    def to(self, device):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        if self.trg is not None:
            self.trg = self.trg.to(device)
            self.trg_y = self.trg_y.to(device)
            self.trg_mask = self.trg_mask.to(device)
        return self


def subsequent_mask(size):  #生成下三角矩阵，用来屏蔽未来的信息
    "Mask out subsequent positions."
    attn_shape = (1, size, size) # trg=tgt.size(-1)=9,(1,9,9)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # print('subsequent_mask:{}'.format(subsequent_mask.shape))
 
    return torch.from_numpy(subsequent_mask) == 0