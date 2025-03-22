from torch.utils.data.dataloader import default_collate

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# collate
def collate_fn(batch):
    x, y = default_collate(batch)
    return x.to(DEVICE), y.to(DEVICE)