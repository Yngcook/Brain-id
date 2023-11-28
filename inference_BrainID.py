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
import faiss
import matplotlib.pyplot as plt

#Define test model & Get data

class CNNModel(pl.LightningModule):
    def __init__(self, d_model=256, input_channels=3, learning_rate=None, dropout_prob=0.5, num_classes=90):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.d_model = d_model

        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, data):
        #CNN layers
        data = data * 10**4
        data = ((data - data.mean(dim=2, keepdim=True))/(data.std(dim=2, keepdim=True)+1e-8))
        data = data.permute(0, 2, 3, 1)
        data = self.relu(self.conv1(data))
        data = self.maxpool(data)
        data = self.relu(self.conv2(data))
        data = self.maxpool(data)
        data = self.relu(self.conv3(data))

        data = F.avg_pool2d(data, kernel_size=data.size()[2:])

        data = data.view(data.size(0), -1)

        return data

def window_from_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data = raw.get_data()

    selected_channels = [10, 40, 61]
    data = data[selected_channels, :]

    seconds_per_minute = 30
    samples_per_minute = seconds_per_minute * raw.info['sfreq']
    max_index = data.shape[1] - samples_per_minute

    start_index = random.randint(0, max_index)

    minute_data = data[:, start_index:int(start_index) + int(samples_per_minute)]

    window_size_seconds = 1
    window_size_samples = int(window_size_seconds * raw.info['sfreq'])
    window = np.lib.stride_tricks.sliding_window_view(minute_data, (3, window_size_samples))[0, ::4]
    window = np.expand_dims(window, axis=0)
    window = torch.from_numpy(window).float()

    return window

def search_nearest_labels(loaded_fingerprints, loaded_number_person, window, k=1):
    l2_norms = np.linalg.norm(loaded_fingerprints, axis=1)
    loaded_fingerprints = loaded_fingerprints / l2_norms[:, np.newaxis]

    d = 512
    index = faiss.IndexFlatIP(d)

    index.add(loaded_fingerprints)

    l2_norms = np.linalg.norm(window, axis=1)
    window = window / l2_norms

    result = index.search(window, k)

    distances = result[0]
    indices = result[1]
    nearest_labels = [loaded_number_person[i] for i in indices]

    return nearest_labels, distances

#Make Fingerprints Base

model = CNNModel('.ckpt')# put here the checkpoint
model = model.to('cuda')
model.eval()

json_file = "data_base.json"
data_dir = ""

with open(json_file, 'r') as json_file:
    data_dict = json.load(json_file)
    keys = list(data_dict.keys())[90:]
    data_test_dict = {key: data_dict[key] for key in keys}

fingerprints = []
number_person = []

# make fingerprints base
for folder, files in data_test_dict.items():
    folder_path = os.path.join(data_dir, folder)

    for file in files:
        file_path = os.path.join(folder_path, file)
        for _ in range(5):
            window = window_from_file(file_path)
            window = window.to('cuda')
            with torch.no_grad():
                window = model(window).cpu().numpy()

            fingerprints.append(window)
            number_person.append(int(folder[1:]))

number_person = np.array(number_person)
fingerprints = np.concatenate(fingerprints, axis=0)

#Identity Authentication

import faiss

data_dir = "/content/files"
json_file = "/content/gdrive/MyDrive/files/data_inputs.json"

with open(json_file, 'r') as json_file:
    data_dict = json.load(json_file)
    keys = list(data_dict.keys())[90:]
    data_test_dict = {key: data_dict[key] for key in keys}

random_folder = random.choice(keys)
random_file = random.choice(data_test_dict[random_folder])
print('My number is', int(random_folder[1:]))

folder_path = os.path.join(data_dir, random_folder)
file_path = os.path.join(folder_path, random_file)

input_window = window_from_file(file_path).to('cuda')
with torch.no_grad():
    input_window = model(input_window).cpu().numpy()

nearest_labels, distance = search_nearest_labels(fingerprints, number_person, input_window, k=1)

if int(nearest_labels[0]) == int(random_folder[1:]):
    print('True, his is number', int(nearest_labels[0]))
else:
    print('False, his is number', int(nearest_labels[0]))
print(distance)

#Identification Accuracy

num_samples = 500
correct_predictions = 0

for _ in range(num_samples):

    random_folder = random.choice(keys)
    random_file = random.choice(data_test_dict[random_folder])

    folder_path = os.path.join(data_dir, random_folder)
    file_path = os.path.join(folder_path, random_file)

    input_window = window_from_file(file_path).to('cuda')
    with torch.no_grad():
        input_window = model(input_window).cpu().numpy()

    nearest_labels, distance = search_nearest_labels(fingerprints, number_person, input_window, k=1)

    if int(nearest_labels[0]) == int(random_folder[1:]):
        correct_predictions += 1

accuracy = (correct_predictions/num_samples)*100

print('Accuracy', accuracy, '%')