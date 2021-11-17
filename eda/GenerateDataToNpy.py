#!/usr/bin/env python
# coding: utf-8

# In[1]:


import click
import json
import time
import socket
#import gatetools.phsp as phsp
import gaga
import copy
import numpy as np
from colorama import init
from colorama import Fore, Style
import torch
import datetime
import numpy as np
import os
import tokenize
from io import BytesIO
from matplotlib import pyplot as plt
import uproot
import logging
from testGenerator import HDF5DatasetGenerator


def readRootPHSP(filename,phspNum):
    f = uproot.open(filename)
    k = f.keys()
    psf = f[k[phspNum]]
    names = [k for k in psf.keys()]
    n = psf.num_entries
    print(names,n)
    a = psf.arrays(entry_stop=n, library="numpy")
    d = np.column_stack([a[k] for k in psf.keys()])
    print(d[0:10,:])
    return d,names,n

def save_root_to_npy(filename,rootData, rootKeys,keys):
    """
    Write a PHSP (Phase-Space) file in npy
    """

    dtype = []
    for k in keys:
        dtype.append((k, 'f4'))

    r = np.zeros(len(rootData), dtype=dtype)

    indices = []
    for k in keys:
        indices.append(np.where(np.asarray(read_keys)==k)[0][0])

    print(keys)
    print(indices)

    for n,k in enumerate(keys):
        r[k] = rootData[:, indices[n]]

    np.save(filename, r)

def save_batch_to_npy(Generator, work_list, output_file, batch_size,
                      normalise=False, reduce_keys=False):
    trainGen = Generator(work_list, batch_size)
    gen = trainGen.generator(normalise=normalise, reduce_keys=reduce_keys)
    batch = next(gen)
    print(type(batch))
    np.save(output_file, batch)
    return batch

class NpyGenerator:
    def __init__(self, dbPath, batchSize):
        self.batchSize = batchSize
        self.dbPath = dbPath
        self.electronBeamParameters = []
        self.numParticles = None        
    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0
        self.all_data_array = np.load(self.dbPath)
        #print(self.all_data_array.shape)
        self.numParticles = self.all_data_array.shape[0]

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            particles = []
            for n in range(self.batchSize):
                random_int = np.random.choice(self.numParticles)
                particle = self.all_data_array[random_int]
                particles.append(particle)
                #print(dbID,particleID)

            particles = np.asarray(particles,dtype = np.float32)
            yield particles

            # increment the total number of epochs
            epochs += 1


segment = save_batch_to_npy(HDF5DatasetGenerator, 
                          ['data/DISP_0.5_ANGLE_0/HDF5/a1TrainGenerator.hdf5'], 
                          'data/dataNormalized_10000000_6.npy', 10000000,
                          normalise=True, reduce_keys=True)
print(segment[1])
print("Shape is: " ,segment.shape)
trainGen = NpyGenerator('data/dataNormalized_10000000_6.npy', 4)
gen = trainGen.generator()
batch = next(gen)
print(batch)