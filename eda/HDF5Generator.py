#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math 
import matplotlib.pyplot as plt
import h5py
import os
import argparse

eDict = {'a': 5.6,'b':5.8,'c':6.0,'d':6.2,'e':6.4}
sDict = {'1': 0,'2':0.1,'3':0.2,'4':0.3,'5':0.4}
CUT = 5000000

class HDF5DatasetGenerator:
    def __init__(self, dbPaths, batchSize):

        self.batchSize = batchSize

        self.dbs = []
        self.electronBeamParameters = []
        for dbPath in dbPaths:
            self.dbs.append(h5py.File(dbPath,'r'))
            self.electronBeamParameters.append(self.decodeElectronBeamParameters(dbPath))
            print('PHSP ',dbPath)
            
        self.numParticles = []
        for i,db in enumerate(self.dbs):
            print('Analysing ',i,'-th of ',len(self.dbs),' PHSP')
            self.numParticles.append(db["particles"].shape[0]-CUT)
            #where = np.where(db["particles"][:,0]==0)[0]
            #if where.shape[0] >0:
            #    self.numParticles.append(where[0])
            #else:
            #    self.numParticles.append(db["particles"].shape[0])


    def decodeElectronBeamParameters(self,path):
        base = os.path.basename(path)[:2]
        E = float(eDict[base[0]])
        s = float(sDict[base[1]])
        pars = [l for l in path.split('/') if 'DISP' in l][0]
        d = float(pars.split('_')[1])
        a = float(pars.split('_')[3])
        return (E,d,s,a)
        
    def generator(self, passes=np.inf, normalise=False, add_beam_params=False):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            particles = []
            #for n in range(self.batchSize):
            while len(particles) < self.batchSize:
                dbID = np.random.randint(0,len(self.dbs))
                particleID = np.random.randint(0,self.numParticles[dbID])
                electronBeamParameters = np.asarray(self.electronBeamParameters[dbID])
                particle = (self.dbs[dbID])["particles"][particleID]
                #if particle[0]<1e-10:
                #    continue
                if normalise:
                    particle = (particle - (self.dbs[dbID])["stats"][0,:])/(self.dbs[dbID])["stats"][1,:]
                    print((self.dbs[dbID])["stats"][0,:])
                if add_beam_params: 
                    particle = np.concatenate((particle,electronBeamParameters))
                particles.append(particle)
                #print(dbID,particleID)


            particles = np.asarray(particles,dtype = np.float32)
            ###################################################
            ###    delete unwanted columns         ############
            particles = np.delete(particles,3,1)
            particles = np.delete(particles,4,1)
            ###################################################
            yield particles

            # increment the total number of epochs
            epochs += 1

    def close(self):
    # close the databases
        for db in self.dbs:
            db.close()

                



if __name__ == "__main__":

    outputHDFFilePathTrList = ['data/DISP_0_ANGLE_0/HDF5/d3TrainGenerator.hdf5']
# =============================================================================
#     
#     workPath = '/net/scratch/people/plgztabor/primo_workdir/TrainingPHSPs/'
#     outputHDFFilePathTrList = []
#     for D in ['0','0.5','1']:
#         for A in ['0','1','2','3']:
#             for E in ['a','b','c','d','e']:
#                 for s in ['1','2','3','4','5']:
#                     name = workPath + 'DISP_' + D + '_ANGLE_' + A + '/HDF5/' + E + s + 'TrainGenerator.hdf5'
#                     outputHDFFilePathTrList.append(name)
#     print(outputHDFFilePathTrList[-10:-1])
# =============================================================================
     
    trainGen = HDF5DatasetGenerator(outputHDFFilePathTrList,1)
    gen = trainGen.generator()
    part = next(gen)
    print(part)
    part = next(gen)
# =============================================================================
#     for _ in range(10):
#         part = next(gen)
#         print(part)
#         for particle in part:
#             batch = particle
#             print(batch)
#         break
# =============================================================================






