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

class EEGDataset(Dataset):
    def __init__(self, data_dir, json_file, transformer_model=None):
        self.data_dir = data_dir

        with open(json_file, 'r') as json_file:
            self.data_dict = json.load(json_file)
            #train sample
            keys = list(self.data_dict.keys())[0:90]
            self.data_train_dict = {key: self.data_dict[key] for key in keys}

        self.file_list = []

        #list for people
        for person_folder, person_files in self.data_train_dict.items():
            person_file_paths = [os.path.join(data_dir, person_folder, file_name) for file_name in person_files]
            self.file_list.extend([(person_folder, file_path) for file_path in person_file_paths])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        person_folder, _ = self.file_list[idx]
        files_in_same_folder = [path for folder, path in self.file_list if folder == person_folder]
        random_files_in_same_folder = random.sample(files_in_same_folder, 2)

        windows = []
        person_tensor = []

        #take files from identical folders
        for file_path in random_files_in_same_folder:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            seconds_per_minute = 1.5

            samples_per_minute = seconds_per_minute * raw.info['sfreq']
            max_index = data.shape[1] - samples_per_minute
            start_index = random.randint(0, max_index)
            minute_data = data[:, start_index:int(start_index) + int(samples_per_minute)]

            window_size_seconds = 1
            window_size_samples = int(window_size_seconds*raw.info['sfreq'])
            window = np.lib.stride_tricks.sliding_window_view(minute_data, (64, window_size_samples))[0, ::4]
            windows.append(window)
            person_tensor.append(int(person_folder[1:]))

        #take a file from another folder
        files_in_different_folder = [path for folder, path in self.file_list if folder != person_folder]
        random_files_in_different_folder = random.sample(files_in_different_folder, 1)

        folder_another_person = int((random_files_in_different_folder[0].split('/'))[3][1:])

        raw = mne.io.read_raw_edf(random_files_in_different_folder[0], preload=True, verbose=False)
        data = raw.get_data()
        seconds_per_minute = 1.5

        samples_per_minute = seconds_per_minute * raw.info['sfreq']
        max_index = data.shape[1] - samples_per_minute
        start_index = random.randint(0, max_index)
        minute_data = data[:, start_index:int(start_index) + int(samples_per_minute)]

        window_size_seconds = 1
        window_size_samples = int(window_size_seconds*raw.info['sfreq'])
        window = np.lib.stride_tricks.sliding_window_view(minute_data, (64, window_size_samples))[0, ::4]
        windows.append(window)
        person_tensor.append(folder_another_person)

        windows = np.array(windows)

        return {
            'data': windows,
            'person_id': person_tensor
        }

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cosine_similarity_positive = F.cosine_similarity(anchor, positive, dim=1)
        cosine_similarity_negative = F.cosine_similarity(anchor, negative, dim=1)
        losses = torch.relu(cosine_similarity_negative - cosine_similarity_positive + self.margin)

        return torch.mean(losses)

class RepeatDataset(EEGDataset):
    def __init__(self, dataset, n_repeats):
        self.dataset = dataset
        self.n = len(dataset)
        self.n_repeats = n_repeats

    def __len__(self):
        return self.n * self.n_repeats

    def __getitem__(self, i):
        return self.dataset[i % self.n]

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, json_file, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.json_file = json_file
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(RepeatDataset(eeg_dataset, n_repeats), batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(RepeatDataset(eeg_dataset, n_repeats), batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(RepeatDataset(eeg_dataset, n_repeats), batch_size=self.batch_size, shuffle=False, num_workers=2)

class CNNModel(pl.LightningModule):
    def __init__(self, d_model=256, input_channels=64, learning_rate=0.0001, dropout_prob=0.5, num_classes=90):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.d_model = d_model

        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(d_model, num_classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=90)

        self.learning_rate = learning_rate

    def forward(self, data):
        #CNN layers
        data = data * 10**4

        data = (data - torch.mean(data, dim=3, keepdims=True)) / (torch.std(data, dim=3, keepdims=True) + 1e-8)
        data = data.permute(0, 2, 3, 1)
        # data = data.reshape(-1, 3, 21, 160)
        data = self.relu(self.bn1(self.conv1(data)))
        data = self.maxpool(data)
        data = self.relu(self.bn2(self.conv2(data)))
        data = self.maxpool(data)
        data = self.relu(self.bn3(self.conv3(data)))

        data = F.avg_pool2d(data, kernel_size=data.size()[2:])

        data = data.view(data.size(0), -1)

        data = F.relu(self.fc1(data))

        data = self.dropout(data)

        vector = self.fc2(data)

        return vector

    def training_step(self, batch, batch_idx):
        data = batch['data'].float()
        tensor_person = batch['person_id']

        anchor = self(torch.unbind(data, dim=1)[0])
        positive = self(torch.unbind(data, dim=1)[1])
        negative = self(torch.unbind(data, dim=1)[2])

        loss = TripletLoss()(anchor, positive, negative)

        self.log("train_loss", loss, on_epoch=True, on_step=True)

        if batch_idx == 0:
            print(f'Epoch {self.current_epoch}: Loss = {loss:.4f}')

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['data'].float()
        tensor_person = batch['person_id']

        preds = torch.argmax(self(torch.unbind(data, dim=1)[2]), dim=1)
        self.val_accuracy.update(preds, tensor_person[2] - 1)

        self.log("val_acc", self.val_accuracy, prog_bar=True)
        #self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            accuracy_value = self.val_accuracy.compute()
            print(f"Epoch {self.current_epoch}: Validation Accuracy = {accuracy_value}")

        return {'accuracy': self.val_accuracy}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer

#training
data_dir = "/content/files"
json_file = "/content/gdrive/MyDrive/brain-ID/files/data_dict.json"
eeg_dataset = EEGDataset(data_dir, json_file)
n_repeats = 10
repeat_dataset = RepeatDataset(eeg_dataset, n_repeats)

batch_size = 64

data_module = EEGDataModule(data_dir, json_file, batch_size)

model = CNNModel()

trainer = pl.Trainer(check_val_every_n_epoch=1,
accelerator='gpu',
max_epochs=5,
logger=pl.loggers.CSVLogger("/content/gdrive/MyDrive/logs/tripletloss"),
)

trainer.fit(model, data_module)

#model = CNNModel.load_from_checkpoint("/content/gdrive/MyDrive/logs/lightning_logs/version_13/checkpoints/epoch=9-step=1950.ckpt")
#model.eval()
#trainer.fit(model, data_module)
