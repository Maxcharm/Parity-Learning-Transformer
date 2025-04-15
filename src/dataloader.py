import torch
from torch.utils.data import Dataset, DataLoader

class FormalLanguageDataset(Dataset):
    def __init__(
            self, 
            lang_type:str,
            data_seqs,
            file_name:str,
            from_file:bool=False,
        ):
        self.data = []
        if from_file:
            assert file_name is not None, "please enter a valid file path to create the dataset."

        for seq in data_seqs:
            x = seq[:-1]
            y = seq[1:]
            self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]