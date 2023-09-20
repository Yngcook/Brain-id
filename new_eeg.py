import os
import mne
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
import re

# Определение функции positional_encoding_1d
def positional_encoding_1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

class EEGDatasetWithTransformer(Dataset):
    def __init__(self, data_dir, sample_length=5, transformer_model=None):
        self.data_dir = data_dir
        self.file_list = []
        self.sample_length = sample_length
        self.transformer_model = transformer_model

        for person_folder in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_folder)
            if os.path.isdir(person_path):
                files = [f for f in os.listdir(person_path) if f.endswith('.edf')]
                self.file_list.extend([os.path.join(person_folder, f) for f in files])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        random_index = random.sample(range(len(self.file_list)), 2)
        data_list = []
        person_ids = []
        for i in random_index:
            file_name = os.path.join(self.data_dir, self.file_list[idx])
            person_id = self.extract_person_id(file_name)
            raw = mne.io.read_raw_edf(file_name, preload=True)
            data = raw.get_data()
            sample_length_in_points = int(self.sample_length * raw.info['sfreq'])
            start_time = random.randint(0, data.shape[1] - sample_length_in_points)
            data = data[:, start_time:start_time + sample_length_in_points]
            data_list.append(data)
            person_ids.append(person_id)

        # Применяем трансформер к данным
        data = torch.tensor(data_list, dtype=torch.float32)
        if self.transformer_model:
            batch_size, n_timestamps, _ = data.size()
            cls_token = self.transformer_model.cls_token.weight.unsqueeze(0).unsqueeze(0)
            cls_token = cls_token.expand(batch_size, n_timestamps, -1)
            data = self.transformer_model.input_proj(data)
            pe = positional_encoding_1d(256, n_timestamps).unsqueeze(0).expand(batch_size, -1, -1)
            data = data + pe
            data = torch.cat((data, cls_token), dim=2)
            vector = self.transformer_model.transformer(data)[:, -1]
            return {
                'data': vector,  # Возвращаем вектор
                'person_id': person_ids
            }
        else:
            return {
                'data': data,
                'person_id': person_ids
            }

    def extract_person_id(self, file_name):
        return file_name

# Определение модели трансформера
class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, num_layers=6, input_dim=64):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, batch_first=True),
            num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Embedding(1, d_model)

    def forward(self, data):
        cls_token = self.cls_token.weight.unsqueeze(0).unsqueeze(0)
        cls_token = cls_token.expand(data.size(0), data.size(1), -1)
        data = self.input_proj(data)
        pe = positional_encoding_1d(data.size(2), data.size(1)).unsqueeze(0).expand(data.size(0), -1, -1)
        data = data + pe
        data = torch.cat((data, cls_token), dim=2)
        vector = self.transformer(data)[:, -1]
        return vector

# Путь к 109 файлам EEG сигналов
data_dir = "./data/files"

# Создание экземпляра модели трансформера
transformer_model = TransformerModel()

# Создание экземпляра датасета с использованием трансформера
dataset = EEGDatasetWithTransformer(data_dir, transformer_model=transformer_model)

person_id = dataset[9]['person_id'][1]
ID=''
ID = ID.join(i for i in person_id if i.isdigit() and len(ID) < 2)
print('ID:', ID[0:3])