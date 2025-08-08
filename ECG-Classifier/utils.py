import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import struct


class TimeSeriesDataset(Dataset):

    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

# read data in binary format
def read_binary_from(ragged_array, bin_file):

    while True:
        chunk = bin_file.read(4)
        if not chunk:
            break

        sub_array_size = struct.unpack('i', chunk)[0]
        sub_array = np.array(struct.unpack(f'{sub_array_size}h', bin_file.read(sub_array_size * 2)))
        ragged_array.append(sub_array)

# read data
def load_data(filename: str, mode: str):

    ragged_array = []
    with open(filename, mode) as file:
        if mode == 'rb':
            read_binary_from(ragged_array, file)
        else:
            data = file.readlines()
            ragged_array = list(map(lambda p: int(p[0]), data))

    return ragged_array
