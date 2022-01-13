import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    #USTAWIENIE LIMITU NA OSI y
    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    #DORYSOWANIE ŚREDNIEJ KROCZĄCEJ https://doraprojects.net/questions/13728392/moving-average-or-running-mean
    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
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

    plt.tight_layout()