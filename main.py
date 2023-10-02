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
from torchmetrics import Accuracy
#import pandas as pd
#import seaborn as sn
# import matplotlib.pyplot as plt


class EEGDataset(Dataset):
    def __init__(self, data_dir, json_file, transformer_model=None):
        self.data_dir = data_dir

        with open(json_file, 'r') as json_file:
            self.data_dict = json.load(json_file)
            #создаем тренировочную выборку
            keys = list(self.data_dict.keys())[0:90]
            self.data_train_dict = {key: self.data_dict[key] for key in keys}

        self.file_list = []

        #Создаем список файлов для каждого человека
        for person_folder, person_files in self.data_train_dict.items():
            person_file_paths = [os.path.join(data_dir, person_folder, file_name) for file_name in person_files]
            self.file_list.extend([(person_folder, file_path) for file_path in person_file_paths])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        person_folder, file_path = self.file_list[idx]
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data = raw.get_data()

        #slidding window
        seconds_per_minute = 60
        samples_per_minute = seconds_per_minute * raw.info['sfreq']
        max_index = data.shape[1] - samples_per_minute
        start_index = random.randint(0, max_index)
        minute_data = data[:, start_index:int(start_index) + int(samples_per_minute)]
        window_size_seconds = 1
        window_size_samples = int(window_size_seconds*raw.info['sfreq'])
        window = np.lib.stride_tricks.sliding_window_view(minute_data, (64, window_size_samples))[0, ::4]

        #получаем ID человека, которому пренадлежит EEG сигнал
        person_tensor = int(person_folder[1:])

        return {
            'data': window,
            'person_id': person_tensor,
        }

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, json_file, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size

    # def setup(self, stage=None):
    #   if stage == 'fit' or stage is None:
    #     self.dataset = EEGDataset(self.data_dir, self.json_file)

    #   if stage == 'test' or stage is None:
    #     self.dataset_test = [EEGDataset(self.data_dir, self.json_file) for _ in range(3)]

    def train_dataloader(self):
        return DataLoader(EEGDataset(self.data_dir, self.json_file), batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(EEGDataset(self.data_dir, self.json_file), batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(EEGDataset(self.data_dir, self.json_file), batch_size=self.batch_size, shuffle=False, num_workers=1)

class CNNModel(pl.LightningModule):
    def __init__(self, d_model=256, input_channels=64, learning_rate=2e-4, dropout_prob=0.5, num_classes=90):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.d_model = d_model
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(p=0)
        self.fc2 = nn.Linear(d_model, num_classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=90)

    def forward(self, data):
        #CNN layers
        data = data * 10**4
        data = data.permute(0, 2, 3, 1)
        data = self.relu(self.conv1(data))
        data = self.maxpool(data)
        data = self.relu(self.conv2(data))
        data = self.maxpool(data)
        data = self.relu(self.conv3(data))

        data = F.avg_pool2d(data, kernel_size=data.size()[2:])

        data = data.view(data.size(0), -1)

        data = F.relu(self.fc1(data))

        data = self.dropout(data)

        vector = self.fc2(data)

        return vector

    def training_step(self, batch, batch_idx):
        data = batch['data'].float()
        tensor_person = batch['person_id']

        loss = F.cross_entropy(self(data), tensor_person - 1)

        self.log("train_loss", loss, on_epoch=True, on_step=True)

        if batch_idx == 0:
            print(f'Epoch {self.current_epoch}: Loss = {loss:.4f}')

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['data'].float()
        tensor_person = batch['person_id']

        # #для каждого батча
        # _, predicted = torch.argmax(outputs, 1)
        # correct = (predicted == tensor_person-1).sum().item()
        # accuracy = correct / len(predicted)

        loss = F.cross_entropy(self(data), tensor_person - 1)

        preds = torch.argmax(self(data), dim=1)
        self.val_accuracy.update(preds, tensor_person - 1)

        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            accuracy_value = self.val_accuracy.compute()
            print(f"Epoch {self.current_epoch}: Validation Accuracy = {accuracy_value}")

        return {'loss': loss, 'accuracy': self.val_accuracy}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

#training
data_dir = "/content/gdrive/MyDrive/files"
json_file = "/content/gdrive/MyDrive/files/data_dict.json"
dataset = EEGDataset(data_dir, json_file)
batch_size = 8

data_module = EEGDataModule(data_dir, json_file, batch_size)

#model = CNNModel()

trainer = pl.Trainer(
accelerator='gpu',
max_epochs=15,
logger=pl.loggers.CSVLogger("/content/gdrive/MyDrive/logs/"),
)

#trainer.fit(model, data_module)

model = CNNModel.load_from_checkpoint("/content/gdrive/MyDrive/logs/lightning_logs/version_7/checkpoints/epoch=14-step=2340.ckpt")
model.eval()
trainer.test(model, data_module)

# metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
# del metrics["step"]
# metrics.set_index("epoch", inplace=True)
# display(metrics.dropna(axis=1, how="all").head())
# sn.relplot(data=metrics, kind="line")