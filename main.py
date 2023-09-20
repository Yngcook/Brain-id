import os
import mne
import torch
import torch.nn as nn
import math
from torch.utils.data import Subset, Dataset, DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F
import json

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

class EEGDataset(Dataset):
    def __init__(self, data_dir, json_file, sample_length=5, transformer_model=None):
        self.data_dir = data_dir
        self.sample_length = sample_length
        self.transformer_model = transformer_model

        with open(json_file, 'r') as json_file:
            self.data_dict = json.load(json_file)

        self.file_list = []

        # Создаем список файлов для каждого человека
        for person_folder, person_files in self.data_dict.items():
            person_file_paths = [os.path.join(data_dir, person_folder, file_name) for file_name in person_files]
            self.file_list.extend([(person_folder, file_path) for file_path in person_file_paths])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        person_folder, file_path = self.file_list[idx]
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data = raw.get_data()

        # Находим количество точек в 5-ти секундах (умножаем частоту дискретизации на длину семпла)
        sample_length_in_points = int(self.sample_length * raw.info['sfreq'])

        # Создаем список для хранения двух кусков данных
        data_list = []

        for _ in range(2):
            start_time = random.randint(0, data.shape[1] - sample_length_in_points)

            # Срезаем 5 секунд
            data_chunk = data[:, start_time:start_time + sample_length_in_points]
            data_list.append(data_chunk)

        data_tensor = np.stack(data_list).astype(np.float32)
        person_tensor = int(person_folder[1:])

        return {
            'data': data_tensor,
            'person_id': person_tensor,
        }

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, json_file, batch_size, sample_length=5):
        super().__init__()
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size
        self.sample_length = sample_length

    def setup(self, stage=None):
        self.dataset = EEGDataset(self.data_dir, self.json_file, self.sample_length)
        self.train_dataset, self.val_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

class TransformerModel(pl.LightningModule):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, num_layers=6,
                 input_dim=64, learning_rate=2e-4, n_channels=64):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, batch_first=True),
            num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Embedding(1, d_model)


    def forward(self, data):
        data = data * 10**4
        batch_size, n_timestamps, n_channels = data.size()
        cls_token = self.cls_token.weight.unsqueeze(0)
        cls_token = cls_token.expand(batch_size, 1, -1)
        data = self.input_proj(data)
        pe = positional_encoding_1d(self.d_model, n_timestamps).unsqueeze(0).expand(batch_size, -1, -1)
        data = data + pe
        data = torch.cat((data, cls_token), dim=1)
        vector = self.transformer(data)[:, -1]

        return vector

    def cosine_similarity(self, vector):
        cosine_sim_matrix = F.cosine_similarity(vector.unsqueeze(1), vector.unsqueeze(0), dim=-1)
        return cosine_sim_matrix

    def triplet_loss(self, cosine_similarity_matrix, tensor_person, margin=0.5):
        assert cosine_similarity_matrix.shape[0] == cosine_similarity_matrix.shape[1]
        assert tensor_person.shape[0] == tensor_person.shape[1]

        tensor_person = tensor_person.to(torch.bool)

        n = cosine_similarity_matrix.shape[0]
        inf = 2.0

        # Min positive
        pos_gt = torch.logical_and(
            tensor_person,
            torch.logical_not(torch.eye(n, dtype=torch.bool)))
        min_pos = torch.where(pos_gt, cosine_similarity_matrix, inf)
        min_pos = torch.min(min_pos, dim=1).values

        # Max negative
        max_neg = torch.where(tensor_person, -inf, cosine_similarity_matrix)
        max_neg = torch.max(max_neg, dim=1).values

        loss = torch.clamp(max_neg - min_pos + margin, 0.0).mean()

        return loss

    def training_step(self, batch, batch_idx):
        data = batch['data']
        tensor_person = batch['person_id']
        tensor_person = tensor_person.view(-1, 1)
        tensor_person = torch.repeat_interleave(tensor_person, 2, dim=0)
        tensor_person = tensor_person / tensor_person.T
        tensor_person[tensor_person != 1] = 0

        data = data.view(-1, data.size(-2), data.size(-1)).permute(0, 2, 1)
        vector = self(data)

        cosine_similarity_matrix = self.cosine_similarity(vector)

        loss = self.triplet_loss(cosine_similarity_matrix, tensor_person)
        self.log("train_loss", loss, on_epoch=True, on_step=True)


        if batch_idx == 0:
            print(f'Epoch {self.current_epoch}: Loss = {loss:.4f}')

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        person = batch['person_id']
        person = [int(a) for a in person]
        tensor_person = torch.tensor(person)
        tensor_person = tensor_person.view(-1, 1)
        tensor_person = torch.repeat_interleave(tensor_person, 2, dim=0)
        tensor_person = tensor_person / tensor_person.T
        tensor_person[tensor_person != 1] = 0
        tensor_person = tensor_person.to(torch.bool)
        inf = 2.0

        data = data.view(-1, data.size(-2), data.size(-1)).permute(0, 2, 1)
        vector = self(data)

        cosine_similarity_matrix = self.cosine_similarity(vector)
        n = cosine_similarity_matrix.shape[0]

        eye_tensor = torch.eye(n, dtype=torch.bool, device=data.device)
        pos_gt = torch.logical_and(tensor_person, torch.logical_not(eye_tensor))


        max_neg = torch.where(tensor_person, -inf, cosine_similarity_matrix)
        max_neg = torch.max(max_neg, dim=1).values
        min_pos = torch.where(pos_gt, cosine_similarity_matrix, inf)
        min_pos = torch.min(min_pos, dim=1).values

        # считаем процент строк для которых самый худший положительный лучше самого лучшего отрицательного
        row_comparison = min_pos > max_neg

        num_rows_min_pos_gt_max_neg = row_comparison.sum().item()

        total_rows = row_comparison.size(0)

        bad_accuracy = (num_rows_min_pos_gt_max_neg / total_rows) * 100.0

        loss = self.triplet_loss(cosine_similarity_matrix, tensor_person)

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_accuracy", bad_accuracy, on_epoch=True, on_step=True)

        if batch_idx == 0:
            print(f"Epoch {self.current_epoch}: Validation Accuracy = {bad_accuracy:.4f}")

        return bad_accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# class Callback(pl.Callback):
#     def on_epoch_start(self, trainer, pl_module):
#         print(f"Epoch {trainer.current_epoch + 1}")
#
#     def on_epoch_end(self, trainer, pl_module):
#         current_epoch = trainer.current_epoch
#         loss_value = trainer.callback_metrics['loss']
#         print(f'Epoch {current_epoch}: Loss = {loss_value:.4f}')


# путь к 109 файлам EEG сигналов и json файлу
# data_dir = "./data/files"
# json_file = "C:/Users/kukur/PycharmProjects/ProjectEEG/data/files/data_dict.json"
# dataset = EEGDataset(data_dir, json_file)
# # model = TransformerModel(learning_rate=config.learning_rate)
# # data_module = EEGDataModule(data_dir="./data/files", batch_size=64, sample_length=5, transformer_model=model)
# batch_size = 8
# data_module = EEGDataModule(data_dir, json_file, batch_size, sample_length=5)
# margin = 0.5
# eeg_dataset = EEGDataset(data_dir=data_dir, sample_length=5, json_file=json_file)
# dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True)
# transformer_model = TransformerModel()

# for batch in dataloader:
#     data = batch['data']  # Тензор с данными размером [8, 2, 64, 800]
#     #создаем фактическую матрицу
#     person = batch['person_id']
#     person = [int(a) for a in person]
#     tensor_person = torch.tensor(person)
#     tensor_person = tensor_person.view(-1, 1)
#     tensor_person = torch.repeat_interleave(tensor_person, 2, dim=0)
#     tensor_person = tensor_person / tensor_person.T
#     tensor_person[tensor_person != 1] = 0
#
#     data = data.view(-1, data.size(-2), data.size(-1)).permute(0, 2, 1)
#     vector = transformer_model(data)
#
#     cosine_similarity_matrix = transformer_model.cosine_similarity(vector)
#
#     #создаем trepliet loss
#     triplet_loss = transformer_model.loss(cosine_similarity_matrix, tensor_person, margin)
#     break

if __name__ == "__main__":
    data_dir = "./data/files"
    json_file = "C:/Users/kukur/PycharmProjects/ProjectEEG/data/files/data_dict.json"
    dataset = EEGDataset(data_dir, json_file)
    batch_size = 8

    data_module = EEGDataModule(data_dir, json_file, batch_size, sample_length=5)

    transformer_model = TransformerModel()

    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=10,
        logger=pl.loggers.CSVLogger("logs/"),
    )

    trainer.fit(transformer_model, data_module)
