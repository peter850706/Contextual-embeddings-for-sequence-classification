import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Part2Dataset(Dataset):
    def __init__(self, data):
        self._data = [{
            'Id': d['Id'],
            'text_orig': d['text'],
            'label': int(d['label']) - 1
        } for d in tqdm(data, desc='[*] Indexizing', dynamic_ncols=True)]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        Id = [b['Id'] for b in batch]
        text_orig = [b['text_orig'] for b in batch]
        label = [b['label'] for b in batch]
        
        label = torch.tensor(label)

        return {
            'Id': Id,
            'text_orig': text_orig,
            'label': label
        }


def create_data_loader(dataset, batch_size, n_workers, shuffle=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, collate_fn=dataset.collate_fn)
    return data_loader