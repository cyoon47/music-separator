import argparse
import random
from pathlib import Path

import torch
import torch.utils.data
import torchaudio
import tqdm
import os

def load_datasets(train_path, valid_path):
    train_dataset = MixAudioDataset(train_path)
    valid_dataset = MixAudioDataset(valid_path)
    return train_dataset, valid_dataset
    
class MixAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_dir,
        sample_rate = 44100,
    ):
        self.file_dir = file_dir
        self.file_names = os.listdir(file_dir)
        for x in self.file_names:
            if x[-3:] != 'mp3':
                self.file_names.remove(x)
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        x, _ = torchaudio.load(self.file_dir + '/' + self.file_names[index])
        y = torch.zeros(50, dtype=torch.int32)
        for i in self.get_audio_indices(self.file_names[index]):
            y[i] = 1
        
        return x, y

    def __len__(self):
        return len(self.file_names)

    def get_audio_indices(self, file_name):
        indices = map(lambda x: int(x)-1, file_name.split('.')[0].split('_'))
        return indices

    def __repr__(self):
        return "Hi, I'm dataset :D"

