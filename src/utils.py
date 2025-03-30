from torch.utils.data.dataloader import default_collate

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# collate
def collate_fn(batch):
    x, y = default_collate(batch)
    return x.to(DEVICE), y.to(DEVICE)

# res net
class ResBlock(nn.Module):
    def __init__(self, *args: nn.Module):
        super().__init__()

        self.wrapped_module = nn.Sequential(*args)

    def forward(self, x):
        return self.wrapped_module.forward(x) + x
