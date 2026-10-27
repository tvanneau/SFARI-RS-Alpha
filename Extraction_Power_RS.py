# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:42:54 2024

@author: tvanneau
"""

import matplotlib as mpl

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

#============================================
#               AVSRT Analysis
#============================================

#loading needed toolboxes 
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import tkinter
import pandas as pd
from os.path import isfile
from tkinter import filedialog
import os
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
import pickle

# list_file = os.listdir('Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/Resting_state/Data/SIB')
# file_path = 'Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/Resting_state/Data/SIB'

# list_file = os.listdir('Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/ASSR/Data/SIB')
# file_path = 'Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/ASSR/Data/SIB'

# list_file = os.listdir('Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/AVSRT/Data/SIB')
# file_path = 'Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/AVSRT/Data/SIB'

list_file = os.listdir('Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/FAST/Data/SIB')
file_path = 'Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/FAST/Data/SIB'


list_files_epochs = [s for s in list_file if '_EEG_epo' in s]

list_files_epochs.sort()

# For TD - AVSRT - Too short
# del list_files_epochs [35]

# For ASD - FAST
# del list_files_epochs [39]
# del list_files_epochs [11]

Power_all = []

for i in range(0,len(list_files_epochs),1):
    
    f_name = file_path + "/" + list_files_epochs[i]
    
    Subject = list_files_epochs[i].split('_')[0]
    
    
    
    # For Resting-state data
    # epochs = mne.read_epochs(f_name)
    # epochs.set_eeg_reference('average')
    # epochs = epochs[0]
    # epochs.crop(tmin=5, tmax=55)
    
    
    
    # For Task data
    raw = mne.io.read_raw_fif(f_name, preload=True)
    # Remove 'BAD boundary' and 'EDGE' annotations
    raw.annotations.delete(
        [i for i, desc in enumerate(raw.annotations.description)
          if desc.lower() in ['bad boundary', 'edge']]
    )
    raw.crop(tmin=0, tmax=720)
    # Define artificial events for epoching (e.g., 2-second epochs across the whole block)
    events = mne.make_fixed_length_events(raw, id=1, start=0, duration=720)
    # Create epochs (adjust baseline and tmin/tmax if needed)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=720, baseline=None, preload=True)
    epochs.set_eeg_reference('average')
    
    
    
    
    # Time-frequency analysis for each quartile 
    freqs = np.linspace(*np.array([3, 20]), num=40) #num = step / bin allows a precision of the frequency resolution
    n_cycles = np.linspace(*np.array([3, 10]), num=40)  # different number of cycle per frequency
    
    # Total power for each quartile
    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=8, n_jobs=1, average=False)
    
    # Save with a pickle format directly into the current path
    with open('Power_FAST_'+ Subject + '.pkl', 'wb') as fp:
        pickle.dump([power], fp)   

#%%

# Theta activity for total and non-phase locked power at specific time points

fmin=7
fmax=13

# For RS
tmin1 = 5
tmax1 = 55

# RS Scale 
vlim1 = 0
vlim2= 5.0e-9

# For ASSR
# tmin1 = 0
# tmax1 = 420

# ASSR Scale 
# vlim1 = 0
# vlim2= 10.0e-9

power.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), show=False)