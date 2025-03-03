import os
from torch.utils.data import Dataset

from dataset.hme_ink import *

class HMEDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
       
        # self.max_seq_length = max_seq_length
        # build vocabulary for LaTeX tokens (this would normally be built from training data)
        # self.vocab = Vocabulary()
        
        # array of inkml file paths
        self.ink_files = sorted(os.path.join(root_dir, split, "*.inkml"))
        
        print(f"Found {len(self.ink_files)} files in {split} split")
        

    def __len__(self):
        return len(self.ink_files)

    def __getitem__(self, idx):
        ink_file = self.ink_files[idx]
        ink = read_inkml_file(ink_file) # returns file w/ strokes and annotations (contains the label)
        return ink
        