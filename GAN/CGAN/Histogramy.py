#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt


fakePHSP = './fake.txt'
truePHSP = '/net/scratch/people/plgztabor/primo_workdir/PHSPs_without_VR/ANGLE_0/TXT/Filtered_E6.0_s2.0.txt'

f = open(fakePHSP,'rt')
lines = f.readlines()
f.close()

fake = [r.split() for r in lines]
fake = np.asarray(fake,dtype=np.float32)

f = open(truePHSP,'rt')
lines = f.readlines()
f.close()

real = [r.split() for r in lines]
real = np.asarray(real,dtype=np.float32)

signs = np.random.randint(0,2,real.shape[0])*2-1
real[:,0] = real[:,0]*signs
real[:,1] = real[:,1]*signs
real[:,2] = real[:,2]*signs
real[:,3] = real[:,3]*signs

bins = np.linspace(0, 6, 100)
plt.figure(figsize = (10,10))
plt.hist(real[:,5],bins,alpha=0.25,label='r',density=True)
plt.hist(fake[:,5],bins,alpha=0.25,label='f',density=True)
plt.legend(loc='upper right')
#plt.show()
plt.savefig('Ekin')

bins = np.linspace(-10, 10, 100)

plt.figure(figsize = (10,10))
plt.hist(real[:,0],bins,alpha=0.25,label='r',density=True)
plt.hist(fake[:,0],bins,alpha=0.25,label='f',density=True)
plt.legend(loc='upper right')
#plt.show()
plt.savefig('X')

plt.figure(figsize = (10,10))
plt.hist(real[:,1],bins,alpha=0.25,label='r',density=True)
plt.hist(fake[:,1],bins,alpha=0.25,label='f',density=True)
plt.legend(loc='upper right')
#plt.show()
plt.savefig('Y')


bins = np.linspace(-0.8, 0.8, 100)

plt.figure(figsize = (10,10))
plt.hist(real[:,2],bins,alpha=0.25,label='r',density=True)
plt.hist(fake[:,2],bins,alpha=0.25,label='f',density=True)
plt.legend(loc='upper right')
#plt.show()
plt.savefig('dX')

plt.figure(figsize = (10,10))
plt.hist(real[:,3],bins,alpha=0.25,label='r',density=True)
plt.hist(fake[:,3],bins,alpha=0.25,label='f',density=True)
plt.legend(loc='upper right')
#plt.show()
plt.savefig('dY')


bins = np.linspace(0.8, 1.02, 100)

plt.figure(figsize = (10,10))
plt.hist(real[:,4],bins,alpha=0.25,label='r',density=True)
plt.hist(fake[:,4],bins,alpha=0.25,label='f',density=True)
plt.legend(loc='upper right')
#plt.show()
plt.savefig('dZ')


# In[ ]:




