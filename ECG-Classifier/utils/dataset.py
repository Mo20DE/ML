from torch.utils.data import Dataset
from helper import *
import random

class TimeSeriesDataset(Dataset):

    def __init__(
        self, 
        sequences, 
        labels=None,
        augment=False
    ):
        new_sequences = []
        new_labels = []
        for i, x in enumerate(sequences):
            
            signal = x.copy()
            if labels is not None and augment:

                label = labels[i]
                should_augment = random.randint(0, 1) < 0.2

                if (label == 0 or label == 2) and should_augment:
                    noise = np.random.normal(0, scale=0.05, size=x.shape)
                    amp_scaler = np.random.uniform(0.8, 1.2)
                    augmented_signal = (signal.copy() + noise) * amp_scaler
                    augmented_signal = resample_signal(augmented_signal, dtype=torch.float32)
                    new_sequences.append(augmented_signal)
                    new_labels.append(label)
      
            signal = resample_signal(signal, dtype=torch.float32)
            new_sequences.append(signal)
            if labels is not None:
                new_labels.append(labels[i])

        self.sequences = new_sequences
        self.labels = new_labels

    def __getitem__(self, idx):
        if self.labels:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]
    
    def __len__(self):
        return len(self.sequences)
    