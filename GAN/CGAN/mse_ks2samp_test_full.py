#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import json
import time
import seaborn as sns
from scipy.stats import ks_2samp, chisquare
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator


# fakePHSP = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/fake.txt'
fakePHSP_list = []
fakePHSP_E56_s00 = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsFiles/fake_E5.6_s0.0_.txt'
fakePHSP_E56_s40 = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsFiles/fake_E5.6_s4.0_.txt'
fakePHSP_E60_s20 = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsFiles/fake_E6.0_s2.0_.txt'
fakePHSP_E64_s00 = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsFiles/fake_E6.4_s0.0_.txt'
fakePHSP_E64_s40 = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsFiles/fake_E5.6_s4.0_.txt'
fakePHSP_list.append(fakePHSP_E56_s00)
fakePHSP_list.append(fakePHSP_E56_s40)
fakePHSP_list.append(fakePHSP_E60_s20)
fakePHSP_list.append(fakePHSP_E64_s00)
fakePHSP_list.append(fakePHSP_E64_s40)

truePHSP_list = []
truePHSP__E56_s00 = '/data1/dose-3d-generative/data/gan-data/PHSPs_without_VR/ANGLE_0/Filtered_E5.6_s0.0.txt'
truePHSP__E56_s40 = '/data1/dose-3d-generative/data/gan-data/PHSPs_without_VR/ANGLE_0/Filtered_E5.6_s4.0.txt'
truePHSP__E60_s20 = '/data1/dose-3d-generative/data/gan-data/PHSPs_without_VR/ANGLE_0/Filtered_E6.0_s2.0.txt'
truePHSP__E64_s00 = '/data1/dose-3d-generative/data/gan-data/PHSPs_without_VR/ANGLE_0/Filtered_E6.4_s0.0.txt'
truePHSP__E64_s40 = '/data1/dose-3d-generative/data/gan-data/PHSPs_without_VR/ANGLE_0/Filtered_E6.4_s4.0.txt'
# '/net/scratch/people/plgztabor/primo_workdir/PHSPs_without_VR/ANGLE_0/TXT/Filtered_E6.0_s2.0.txt'
truePHSP_list.append(truePHSP__E56_s00)
truePHSP_list.append(truePHSP__E56_s40)
truePHSP_list.append(truePHSP__E60_s20)
truePHSP_list.append(truePHSP__E64_s00)
truePHSP_list.append(truePHSP__E64_s40)

fake_list = []
real_list = []

start_time = time.time()
for filename in fakePHSP_list:
    file_start_time = time.time()
    f = open(filename, 'rt')
    lines = f.readlines()
    f.close()

    fake = [r.split() for r in lines]
    fake = np.asarray(fake, dtype=np.float32)
    fake_list.append(fake)
    print('MSE KS2SAMP Time elapsed: %.2f min' %
          ((time.time() - file_start_time)/60))

for filename in truePHSP_list:
    file_start_time = time.time()
    f = open(filename, 'rt')
    lines = f.readlines()
    f.close()

    real = [r.split() for r in lines]
    real = np.asarray(real, dtype=np.float32)

    np.random.seed(0)
    signs = np.random.randint(0, 2, real.shape[0])*2-1
    real[:, 0] = real[:, 0]*signs
    real[:, 1] = real[:, 1]*signs
    real[:, 2] = real[:, 2]*signs
    real[:, 3] = real[:, 3]*signs
    real_list.append(real)
    print('MSE KS2SAMP  Time elapsed: %.2f min' %
          ((time.time() - file_start_time)/60))

print('MSE KS2SAMP All time elapsed: %.2f min' %
      ((time.time() - start_time)/60))


conditions_keys = ['E5.6_s0.0', 'E5.6_s4.0',
                   'E6.0_s2.0', 'E6.4_s0.0', 'E6.4_s4.0']
photons_parameters_keys = ['X', 'Y', 'dX', 'dY', 'dZ', 'Ekin']
verification_points_list = [
    (5.6, 0.0), (5.6, 4.0), (6.0, 2.0), (6.4, 0.0), (6.4, 4.0)]


evalreults_dict = {"name": 'config_001.json'}
ks_2samp_dict = {}


ks_statistics_sum_all = 0
ks_statistics_sum_all_list = [0, 0, 0, 0, 0, 0]

for condition_key_index, condition_key_name in enumerate(conditions_keys):
    start_time = time.time()
    ks_statistics_sum = 0
    ks_statistics_list = []
    ks_pvalue_sum = 0
    ks_pvalue_list = []

    ks2samp_statistics_dict = {}

    for index, parameter_name in enumerate(photons_parameters_keys):
        ks_statistic, ks_pvalue = ks_2samp(
            real_list[condition_key_index][:, index], fake_list[condition_key_index][:, index])
        ks_statistics_sum += ks_statistic
        ks_pvalue_sum += ks_pvalue
        ks_pvalue_list.append(ks_pvalue)
        ks_statistics_list.append(ks_statistic)

    ks_statistics_sum_all_list = [
        item1+item2 for item1, item2 in zip(ks_statistics_list, ks_statistics_sum_all_list)]
    ks_statistics_sum_all += ks_statistics_sum

    ks2samp_statistics_dict['statistics_list'] = ks_statistics_list
    ks2samp_statistics_dict['statistics_sum'] = ks_statistics_sum
    ks_2samp_dict[condition_key_name] = ks2samp_statistics_dict

    print('Time elapsed for ks2samp: %.2f min' %
          ((time.time() - start_time)/60))

ks2samp_statistics_dict = {}
ks2samp_statistics_dict['statistics_list'] = ks_statistics_sum_all_list
ks2samp_statistics_dict['statistics_sum'] = ks_statistics_sum_all
ks_2samp_dict['Summed'] = ks2samp_statistics_dict
evalreults_dict['ks_2samp'] = ks_2samp_dict


number_of_bins = 300
mean_square_error_dict = {"number_of_bins": 300}

mse_statistics_sum_all_list = [0, 0, 0, 0, 0, 0]
mse_statistics_sum_all = 0

for condition_key_index, condition_key_name in enumerate(conditions_keys):
    start_time = time.time()

    mse_statistics_dict = {}

    histogram_freq_real_mse = []
    histogram_freq_fake_mse = []

    mse_statistics_sum = 0
    mse_statistics_list = []

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(20, 14)
    fig.suptitle(condition_key_name)

    for index, photon_parameter in enumerate(photons_parameters_keys):
        mi = np.minimum(real_list[condition_key_index][:, index].min(
        ), fake_list[condition_key_index][:, index].min())
        ma = np.maximum(real_list[condition_key_index][:, index].max(
        ), fake_list[condition_key_index][:, index].max())

        bins = np.linspace(mi, ma, number_of_bins)
        tmp_histogram_freq_real_mse, _, _ = axs.flatten()[index].hist(
            real_list[condition_key_index][:, index], bins, alpha=.5, density=True, stacked=True, label='Real')
        tmp_histogram_freq_fake_mse, _, _ = axs.flatten()[index].hist(
            fake_list[condition_key_index][:, index], bins, alpha=.5, density=True, stacked=True, label='Fake')
        axs.flatten()[index].set_title(photon_parameter)
        axs.flatten()[index].legend()

        histogram_freq_real_mse.append(tmp_histogram_freq_real_mse)
        histogram_freq_fake_mse.append(tmp_histogram_freq_fake_mse)

        mse_statistic = mean_squared_error(
            tmp_histogram_freq_real_mse, tmp_histogram_freq_fake_mse)
        mse_statistics_sum += mse_statistic
        mse_statistics_list.append(mse_statistic)

    plt.savefig(
        f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/{condition_key_name}.png')

    mse_statistics_sum_all += mse_statistics_sum
    mse_statistics_sum_all_list = [
        item1+item2 for item1, item2 in zip(mse_statistics_list, mse_statistics_sum_all_list)]

    mse_statistics_dict['statistics_list'] = mse_statistics_list
    mse_statistics_dict['statistics_sum'] = mse_statistics_sum
    mean_square_error_dict[condition_key_name] = mse_statistics_dict
    print('Time elapsed for MSE: %.2f min' % ((time.time() - start_time)/60))

mse_statistics_dict = {}
mse_statistics_dict['statistics_list'] = mse_statistics_sum_all_list
mse_statistics_dict['statistics_sum'] = mse_statistics_sum_all
mean_square_error_dict['Summed'] = mse_statistics_dict

# print(mean_square_error_dict)

evalreults_dict['mean_square_error'] = mean_square_error_dict


with open('/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/config_001.json') as json_file:
    config_json_object = json.load(json_file)


config_json_object["evaluation_results"] = evalreults_dict


with open('/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/statistics_mse_ks2samp.json', 'w') as outfile:
    json.dump(config_json_object, outfile, indent=4)


ks2samp_statistics_matrix = np.empty((5, 6))
for condition_key_index, condition_key_name in enumerate(conditions_keys):
    ks2samp_statistics_matrix[condition_key_index] = np.asarray(
        evalreults_dict['ks_2samp'][condition_key_name]['statistics_list'])

ylabels = [name for name in conditions_keys]
xlabels = [name for name in photons_parameters_keys]

fig = plt.figure()
ax = sns.heatmap(ks2samp_statistics_matrix, xticklabels=xlabels,
                 yticklabels=ylabels, norm=LogNorm(), square=True)
ax.set_title('ks2samp_log')

plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/ks2samp_log')

# print(f'ks2samp_statistics max {ks2samp_statistics_matrix.max()}')
# print(f'ks2samp_statistics min {ks2samp_statistics_matrix.min()}')

fig = plt.figure()
ax2 = sns.heatmap(ks2samp_statistics_matrix, xticklabels=xlabels,
                  yticklabels=ylabels, vmin=0, vmax=0.035)
ax2.set_title('ks2samp')
plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/ks2samp')


ks2samp_statistics_summed = np.empty((1, 6))
ks2samp_statistics_summed[0] = np.asarray(
    evalreults_dict['ks_2samp']['Summed']['statistics_list'])

xlabels = [name for name in photons_parameters_keys]


# print(ks2samp_statistics_summed.max())
# print(ks2samp_statistics_summed.min())
fig = plt.figure()
ax = sns.heatmap(ks2samp_statistics_summed, xticklabels=xlabels,
                 norm=LogNorm(), square=True, vmin=0, vmax=0.16)
ax.set_title('ks2samp_log_summed')

plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/ks2samp_log_summed')
fig = plt.figure()
ax = sns.heatmap(ks2samp_statistics_summed, xticklabels=xlabels)
ax.set_title('ks2samp_summed')
plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/ks2samp_summed')


mse_statistics_matrix = np.empty((5, 6))
for condition_key_index, condition_key_name in enumerate(conditions_keys):
    mse_statistics_matrix[condition_key_index] = np.asarray(
        evalreults_dict['mean_square_error'][condition_key_name]['statistics_list'])

# print(mse_statistics_matrix.max())
# print(mse_statistics_matrix.min())

ylabels = [name for name in conditions_keys]
xlabels = [name for name in photons_parameters_keys]
fig = plt.figure()
ax = sns.heatmap(mse_statistics_matrix, xticklabels=xlabels,
                 yticklabels=ylabels, norm=LogNorm(), square=True)
ax.set_title('mse_log')
plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/mse_log')

fig = plt.figure()
ax = sns.heatmap(mse_statistics_matrix, xticklabels=xlabels,
                 vmin=1.5e-06, vmax=0.017)
ax.set_title('mse')
plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/mse')


mse_statistics_summed = np.empty((1, 6))
mse_statistics_summed[0] = np.asarray(
    evalreults_dict['mean_square_error']['Summed']['statistics_list'])

xlabels = [name for name in photons_parameters_keys]


# print(mse_statistics_summed.max())
# print(mse_statistics_summed.min())
fig = plt.figure()
ax = sns.heatmap(mse_statistics_summed, xticklabels=xlabels,
                 norm=LogNorm(), square=True)
ax.set_title('mse_log_summed')
plt.savefig(
    f'/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/FakePhotonsHistograms/mse_log_summed')
