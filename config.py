import torch


BATCH_SIZE = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")