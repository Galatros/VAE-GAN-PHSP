#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import torch
import random
import pickle
from matplotlib import pyplot as plt

from libCGAN import Generator,Discriminator,generate_samples2,init_pytorch_cuda,get_min_max_constraints

paramsFileName = 'params.pkl'
modelFileName = 'model.pth'

infile = open(paramsFileName,'rb')
params = pickle.load(infile)
infile.close()

outputFile = 'fake.txt'


params['gpu_mode'] = False
dtypef, device = init_pytorch_cuda(params['gpu_mode'], True)

print(device)

cmin, cmax = get_min_max_constraints(params)
cmin = torch.from_numpy(cmin).type(dtypef)
cmax = torch.from_numpy(cmax).type(dtypef)

loadedGan = Generator(params,cmin,cmax)
loadedGan.load_state_dict(torch.load('Gen_' + modelFileName,map_location=torch.device(device)))


batch_size = -1
n = 100000
params['current_gpu'] = False

f = open(outputFile,'wt')
for nbatch in range(100):
    if nbatch%50 == 0:
        print(nbatch)
    cond = np.zeros((n,2),dtype=np.float32)
    cond[:,0] = 6.0
    cond[:,1] = 2.0
    dum = np.asarray(generate_samples2(params, loadedGan, n, batch_size=batch_size, normalize=False,to_numpy=True,cond=cond),dtype=np.float32)
    # w dum w kolejnych kokumnach zwracane są '[Ekin X Y dX dY dZ]'
    r = np.random.randint(0,4,size=(dum.shape[0],4))
    #print(r.shape,np.max(r),np.min(r))
    for i in range(dum.shape[0]):     # zapis w kolejności X Y dX dY dZ Ekin
        if r[i,0]==0: 
            print(dum[i,1],dum[i,2],dum[i,3],dum[i,4],dum[i,5],dum[i,0],file=f)
        if r[i,1]==0:
            print(-dum[i,1],-dum[i,2],-dum[i,3],-dum[i,4],dum[i,5],dum[i,0],file=f)
        if r[i,2]==0:
            print(-dum[i,2],dum[i,1],-dum[i,4],dum[i,3],dum[i,5],dum[i,0],file=f)
        if r[i,3]==0:
            print(dum[i,2],-dum[i,1],dum[i,4],-dum[i,3],dum[i,5],dum[i,0],file=f)

f.close() 
