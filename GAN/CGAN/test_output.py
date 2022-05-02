#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch



columns_names=['epoch', 'd_loss', 'g_loss', 'd_real_loss', 'd_fake_loss']
statistics_df=pd.DataFrame(columns=columns_names)

file1 = open('/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/output.txt', 'r')
count = 0

while True:
    count += 1
 
    # Get next line from file
    line = file1.readline()
 
    # if line is empty
    # end of file is reached
    if not line:
        break
    if count> 52:
        s=line.strip()
        a=s.split()
        if a[0]=='Training':
            break
        # print("Line{}: {}".format(count, line.strip()))
        x=re.findall('[-+ ]\d*\.?\d+',s)
        x=list(map(float,x))
        # print(x)
        statistics_df.loc[len(statistics_df)]=x
 
file1.close()


# print(statistics_df.shape)
# statistics_df.head()


def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label='', path_to_save=None):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss {custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    #USTAWIENIE LIMITU NA OSI y
    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000
    
    if np.max(minibatch_losses[num_losses:])>0:
        ax1.set_ylim([
            np.min(minibatch_losses[num_losses:])*0.5, np.max(minibatch_losses[num_losses:])*1.5
            ])
    else:
        ax1.set_ylim([
            np.min(minibatch_losses[num_losses:])*1.5, 0
            ])

    #DORYSOWANIE ŚREDNIEJ KROCZĄCEJ https://doraprojects.net/questions/13728392/moving-average-or-running-mean
    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average {custom_label}')
    ax1.legend()

    #TWORZENIE OSI Z EPOKAMI
    ax2 = ax1.twiny() #TWORZENIE DRUGIEJ OSI DLA TEGO SAMEGO y
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::20])
    ax2.set_xticklabels(newlabel[::20])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    # ###################

    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.tight_layout()


plot_training_loss(statistics_df['d_loss'].tolist(),num_epochs=1,custom_label='d_loss',path_to_save='/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/d_loss')
plot_training_loss(statistics_df['g_loss'].tolist(),num_epochs=1,custom_label='g_loss',path_to_save='/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/g_loss')
plot_training_loss(statistics_df['d_real_loss'].tolist(),num_epochs=1,custom_label='d_real_loss',path_to_save='/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/d_real_loss')
plot_training_loss(statistics_df['d_fake_loss'].tolist(),num_epochs=1,custom_label='d_fake_loss',path_to_save='/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/d_fake_loss')
print('Test output finished')
