import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from typing import List, Tuple, Type

import numpy as np
import pandas as pd
import copy
import random


class PhotonsDataset(Dataset):
    def __init__(self, data, transform=None):
        self.x = data
        self.n_samples = len(data)
        self.transform = transform
        self.columns = ['X', 'Y', 'dX', 'dY', 'dZ', 'E']

    def __getitem__(self, index):
        sample = self.x

        if self.transform:
            sample = self.transform(sample)
        return sample[index]

    def __len__(self):
        return self.n_samples


class ToTensorFromNdarray:
    def __call__(self, sample):
        # Returns a tensor made from ndarray
        sample
        return torch.from_numpy(sample)


def get_standarized_constrains(constrains_list_min: List[float], constrains_list_max: List[float], stdcs: Type[StandardScaler], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns tensors created on a device with constraining parameters standardized with the help of a StandardScaler object
    constrains_min = np.asarray(
        constrains_list_min, dtype=np.float32).reshape(1, -1)
    constrains_max = np.asarray(
        constrains_list_max, dtype=np.float32).reshape(1, -1)
    standarized_constrains_min = torch.tensor(
        stdcs.transform(constrains_min), device=device)
    standarized_constrains_max = torch.tensor(
        stdcs.transform(constrains_max), device=device)
    return standarized_constrains_min, standarized_constrains_max

def get_photons_with_introduced_XY_symmetries(photons: np.ndarray,random_seed: int)-> np.ndarray:
    # Returns photons with entered X symmetry and Y symmetry depending on random_seed
    random.seed(random_seed)
    symetrized_photons=copy.deepcopy(photons)
    for photon in symetrized_photons:
        if random.uniform(0,1)>0.5:
            photon[1]=-photon[1]
            photon[2]=-photon[2]
            photon[3]=-photon[3]
            photon[4]=-photon[4]
    return symetrized_photons


def get_dataloaders_and_standarscaler_photons_from_numpy(tmp_X, batch_size, test_fraction=0.2, validation_fraction=None, train_transforms=None, test_transforms=None, num_workers=4):

    if train_transforms is None:
        train_transforms = ToTensorFromNdarray()

    if test_transforms is None:
        test_transforms = ToTensorFromNdarray()

    # photons = np.load(path)
    # tmp_X = np.zeros((10000001, 6), dtype=np.float32)
    # np.copyto(tmp_X, photons[:,:-1])

    df_data = pd.DataFrame(tmp_X, columns=['X', 'Y', 'dX', 'dY', 'dZ', 'E'])

    X = df_data.iloc[:, :].values
    X_train, X_test = train_test_split(
        X, test_size=test_fraction, random_state=0, shuffle=True)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    # wykorzystujemy standaryzacje z danych treningowych
    X_test_std = stdsc.transform(X_test)
    X_train_std, X_test_std

    train_dataset = PhotonsDataset(
        data=X_train_std, transform=train_transforms)

    valid_dataset = PhotonsDataset(
        data=X_train_std, transform=test_transforms)

    test_dataset = PhotonsDataset(data=X_test_std, transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * len(X_train_std))
        train_indices = torch.arange(0, len(X_train_std) - num)
        valid_indices = torch.arange(len(X_train_std) - num, len(X_train_std))

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler,
                                  pin_memory=True)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False, pin_memory=True)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader, stdsc


def get_dataloaders_and_standarscaler_photons(path, batch_size, test_fraction=0.2, validation_fraction=None, train_transforms=None, test_transforms=None, num_workers=4):

    if train_transforms is None:
        train_transforms = ToTensorFromNdarray()

    if test_transforms is None:
        test_transforms = ToTensorFromNdarray()

    photons = np.load(path)
    tmp_X = np.zeros(photons.shape, dtype=np.float32)
    np.copyto(tmp_X, photons)

    df_data = pd.DataFrame(tmp_X, columns=['X', 'Y', 'dX', 'dY', 'dZ', 'E'])

    X = df_data.iloc[:, :].values
    X_train, X_test = train_test_split(
        X, test_size=test_fraction, random_state=0, shuffle=True)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    # wykorzystujemy standaryzacje z danych treningowych
    X_test_std = stdsc.transform(X_test)
    X_train_std, X_test_std

    train_dataset = PhotonsDataset(
        data=X_train_std, transform=train_transforms)

    valid_dataset = PhotonsDataset(
        data=X_train_std, transform=test_transforms)

    test_dataset = PhotonsDataset(data=X_test_std, transform=test_transforms)

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
