import torch


BATCH_SIZE = 128
EPOCHS = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")