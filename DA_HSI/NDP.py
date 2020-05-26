import numpy as np
import torch
from torch import nn

class LinearScheduler(nn.Module):
    def __init__(self, ndp, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.ndp = ndp
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.ndp(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.ndp.drop_prob = self.drop_values[self.i]
        self.i += 1
        #print(self.ndp.drop_prob)


class NDP(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(NDP, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.strides = (1, 1)

    def get_mask(self, mask):
        mask = torch.nn.functional.max_pool2d(input=mask[:, None, :, :],
                                kernel_size=(self.block_size, self.block_size),
                                stride=self.strides,
                                padding=self.block_size // 2)
        if self.block_size % 2 == 0: mask = mask[:, :, :-1, :-1]

        mask = 1 - mask.squeeze(1)
        return mask

    def get_gamma(self, x):
        feat_size = x.shape[1]
        return (self.drop_prob / (self.block_size ** 2)) * \
            ((feat_size ** 2) / float((feat_size - self.block_size + 1)**2))

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        if not self.training or self.drop_prob == 0.: return x
        else:
            # get gamma value
            gamma = self.get_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)
            mask = self.get_mask(mask)
            out = x * mask[:, None, :, :]
            # scale output
            out = out * mask.numel() / mask.sum()
            return out
