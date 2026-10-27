# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:10:46 2025

@author: tvanneau
"""

# Load pickle file with corresponding power data 
import os
import pickle
import mne
import matplotlib.pyplot as plt
import numpy as np

with open('Power_RS_TD.pkl', 'rb') as f:
    [Power_TD,PSD_TD] = pickle.load(f)
    
# Indices to remove
indices_to_remove = [0, 6, 18, 31, 36]
# Create a new list excluding the elements at the specified indices
Power_TD = [item for i, item in enumerate(Power_TD) if i not in indices_to_remove]

Power_mean_TD = mne.grand_average(Power_TD)

with open('Power_RS_ASD.pkl', 'rb') as f:
    [Power_ASD,PSD_ASD] = pickle.load(f)
    
# Indices to remove
indices_to_remove = [0, 1, 3, 4, 7, 10, 12, 15, 16, 20, 28, 29, 35, 36, 37, 38, 40, 43, 50, 51]
# Create a new list excluding the elements at the specified indices
Power_ASD = [item for i, item in enumerate(Power_ASD) if i not in indices_to_remove]
Power_mean_ASD = mne.grand_average(Power_ASD)

with open('Power_RS_SIB.pkl', 'rb') as f:
    [Power_SIB,PSD_SIB] = pickle.load(f)
    
# Indices to remove
indices_to_remove = [22]
# Create a new list excluding the elements at the specified indices
Power_SIB = [item for i, item in enumerate(Power_SIB) if i not in indices_to_remove]
    
Power_mean_SIB = mne.grand_average(Power_SIB)

freqs = np.linspace(3, 20, num=100)


# Subject to remove from RS because rejected for AV-SRT
# For TD group: s=0 (10003) ; s=6 (10182) ; 10593 ; 10846 ; 10950

# For ASD group: 11038 ; 11049 ; 11055 ; 11057 ; 11078 ; 11181 ; 11215 ; 11235 ; 11242 ; 11265 ; 11405 ; 11423 ; 11548 ; 11551 ; 11576 ; 11593 
# 11613 ; 11674 ; 11814 ; 11831

# SIB group: 1588

#%%

# Theta activity for total and non-phase locked power at specific time points

fmin=7
fmax=13

tmin1 = 0
tmax1 = 55.0

#Audiovisual scale
vlim1 = 0
vlim2= 5e-9

Power_mean_TD.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), show=False)
Power_mean_ASD.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), show=False)
Power_mean_SIB.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), show=False)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%% Time-frequency maps

# Centro-frontal - Auditory stimulation

# Occipital - Visual stimulation
Electrodes = ['O1', 'O2', 'PO7', 'PO3', 'PO4', 'PO8']
Electrodes = ['Oz']

Power_mean_TD.plot(Electrodes,baseline=None,  cmap='jet',combine='mean', tmin=0, tmax=55,vmin=0,vmax=9e-9,fmin=4,fmax=20)
Power_mean_ASD.plot(Electrodes,baseline=None,  cmap='jet',combine='mean', tmin=0, tmax=55,vmin=0,vmax=9e-9,fmin=4,fmax=20)
Power_mean_SIB.plot(Electrodes,baseline=None,  cmap='jet',combine='mean', tmin=0, tmax=55,vmin=0,vmax=9e-9,fmin=4,fmax=20)

power.plot(Electrodes,baseline=None,  cmap='jet',combine='mean', tmin=0, tmax=55,vmin=0,vmax=9e-9,fmin=4,fmax=20)

#%%

#%%
from scipy.stats import sem

# Assuming PSD_TD, PSD_ASD, and PSD_SIB are lists of PSD arrays (subjects x frequencies)
# and 'freqs' is the array of frequency values

# Calculate mean and SEM for each group
mean_TD = np.mean(PSD_TD, axis=0)
sem_TD = sem(PSD_TD, axis=0)

mean_ASD = np.mean(PSD_ASD, axis=0)
sem_ASD = sem(PSD_ASD, axis=0)

mean_SIB = np.mean(PSD_SIB, axis=0)
sem_SIB = sem(PSD_SIB, axis=0)

# Plotting
plt.figure(figsize=(8, 6))

# TD group
plt.plot(freqs, mean_TD, label='TD', color='blue')
plt.fill_between(freqs, mean_TD - sem_TD, mean_TD + sem_TD, color='blue', alpha=0.3)

# ASD group
plt.plot(freqs, mean_ASD, label='ASD', color='red')
plt.fill_between(freqs, mean_ASD - sem_ASD, mean_ASD + sem_ASD, color='red', alpha=0.3)

# SIB group
plt.plot(freqs, mean_SIB, label='SIB', color='green')
plt.fill_between(freqs, mean_SIB - sem_SIB, mean_SIB + sem_SIB, color='green', alpha=0.3)

# Add labels and legend
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.title('Average PSD with SEM for Each Group')
plt.legend()
plt.tight_layout()
plt.xlim([3,20])
plt.show()
