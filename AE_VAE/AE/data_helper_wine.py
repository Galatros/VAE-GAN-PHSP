import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer,MinMaxScaler

import numpy as np
import pandas as pd


class WineDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.x = data
        self.y = labels
        if(len(data) == len(labels)):
            self.n_samples = len(data)
        else:
            raise ValueError('data and labels have diffrent lenght')

        self.transform = transform
        self.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                        'Alcalinity of ash', 'Magnesium', 'Total phenols',
                        'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                        'Proline']

    def __getitem__(self, index):
        sample = self.x, self.y

        if self.transform:
            sample = self.transform(sample)
        return sample[0][index], sample[1][index]

    def __len__(self):
        return self.n_samples


class ToTensorFromNdarray:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


def get_dataloaders_and_standarscaler_wine(path, batch_size, test_fraction=0.2, validation_fraction=None, train_transforms=None, test_transforms=None, num_workers=0):

    if train_transforms is None:
        train_transforms = ToTensorFromNdarray()

    if test_transforms is None:
        test_transforms = ToTensorFromNdarray()

    df_wine = pd.read_csv(path, header=None, dtype=np.float32)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                       'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=0, stratify=y, shuffle=True)

    stdsc=StandardScaler()
    X_train_std=stdsc.fit_transform(X_train)
    X_test_std=stdsc.transform(X_test) #wykorzystujemy standaryzacje z danych treningowych
    X_train_std, X_test_std

    train_dataset = WineDataset(
        data=X_train_std, labels=y_train, transform=train_transforms)

    valid_dataset = WineDataset(
        data=X_train_std, labels=y_train, transform=test_transforms)

    test_dataset = WineDataset(
        data=X_test_std, labels=y_test, transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * len(X_train_std))
        train_indices = torch.arange(0, len(X_train_std) - num)
        valid_indices = torch.arange(len(X_train_std) - num, len(X_train_std))

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader, stdsc
