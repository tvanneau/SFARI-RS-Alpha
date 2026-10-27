# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:30:09 2026

@author: tvanneau
"""

#============================================
#              Resting-state Analysis
#============================================

#loading needed toolboxes 
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ebosc.eBOSC_wrapper import eBOSC_wrapper
import gc

list_file = os.listdir('Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/Resting_state/Data_eBOSC/ASD')
file_path = 'Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/Resting_state/Data_eBOSC/ASD'

save_dir = 'Z://Analysis/SFARI_Analysis/Block_EEG_Analysis/Resting_state/Extraction_eBOSC_V2/ASD'

list_files_epochs = [s for s in list_file if 'EEG_epo' in s]

list_files_epochs.sort()

for fname in list_files_epochs:
    
    fpath = os.path.join(file_path, fname)
    epochs = mne.read_epochs(fpath, preload=True)
    subject_id = fname.split("_")[0]
    
    epochs.set_eeg_reference('average')
    epochs_1 = epochs[0]
    epochs_1.crop(tmin=5, tmax=55)
    
    # Transform epochs into dataframe object
    # Step 1: Extract EEG data and transpose to (n_times, n_channels)
    eeg_data = epochs_1.get_data().squeeze().T  # Shape: (n_times, n_channels)

    # Step 2: Get time vector from MNE (in seconds)
    time_vector = epochs_1.times  # Shape: (n_times,)

    # Step 3: Create a DataFrame with channel names and a time column
    channel_names = epochs_1.ch_names
    eeg_df = pd.DataFrame(eeg_data, columns=channel_names)
    eeg_df['time'] = time_vector

    # Convert the EEG signal from volts to microvolts
    eeg_df[channel_names] = eeg_df[channel_names] * 1e6

    # Add a placeholder 'condition' column to the EEG DataFrame
    eeg_df['condition'] = '1'  # You can set any default value
    
    # Add placeholder 'epoch' column to the EEG DataFrame
    eeg_df['epoch'] = 0  # Since this is resting-state, all data can be marked as 'epoch 1'
    
    # Drop non-EEG channels and extraneous columns
    non_eeg_columns = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
    eeg_df = eeg_df.drop(columns=non_eeg_columns, errors='ignore')
    
    # eBOSC parameters
    cfg_eBOSC = dict()
    cfg_eBOSC['F'] = np.linspace(4, 20, num=40) # frequency sampling
    cfg_eBOSC['wavenumber'] = 5                 # wavelet parameter (time-frequency tradeoff)
    cfg_eBOSC['fsample'] = float(epochs_1.info["sfreq"])                 # current sampling frequency of EEG data
    cfg_eBOSC['pad.tfr_s'] = 1                  # padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
    cfg_eBOSC['pad.detection_s'] = .5           # padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
    cfg_eBOSC['pad.background_s'] = 1           # padding of segments for BG (only avoiding edge artifacts)

    # threshold settings
    cfg_eBOSC['threshold.excludePeak'] = np.array([[7,14]])   # lower and upper bound of frequencies to be excluded during background fit (Hz) (previously: LowFreqExcludeBG HighFreqExcludeBG)
    cfg_eBOSC['threshold.duration'] = np.kron(np.ones((1,len(cfg_eBOSC['F']))),2) # vector of duration thresholds at each frequency (previously: ncyc)
    cfg_eBOSC['threshold.percentile'] = .95    # percentile of background fit for power threshold

    # episode post-processing
    cfg_eBOSC['postproc.use'] = 'no'           # Post-processing of rhythmic eBOSC.episodes, i.e., wavelet 'deconvolution' (default = 'no')
    cfg_eBOSC['postproc.method'] = 'FWHM'       # Deconvolution method (default = 'MaxBias', FWHM: 'FWHM')
    cfg_eBOSC['postproc.edgeOnly'] = 'yes'      # Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
    cfg_eBOSC['postproc.effSignal'] = 'PT'      # Power deconvolution on whole signal or signal above power threshold? (default = 'PT')

    # % general processing settings
    cfg_eBOSC['channel'] = []                # select posterior channels (default: all)
    cfg_eBOSC['trial'] = []                  # select trials (default: all, indicate in natural trial number (not zero-starting))
    cfg_eBOSC['trial_background'] = []       # select trials for background (default: all, indicate in natural trial number (not zero-starting))
    
    # run eBOSC
    [eBOSC, cfg] = eBOSC_wrapper(cfg_eBOSC, eeg_df)
    
    # Create a DataFrame with only the necessary fields
    episodes_df = pd.DataFrame({
        'Channel': eBOSC['episodes']['Channel'],
        'FrequencyMean': eBOSC['episodes']['FrequencyMean'],
        'PowerMean': eBOSC['episodes']['PowerMean'],
        'Duration': eBOSC['episodes']['DurationS'],
        'SNRMean': eBOSC['episodes']['SNRMean'],
        'Onset' : eBOSC['episodes']['Onset'],
        'Offset' : eBOSC['episodes']['Offset'],
        'Subject': subject_id
    })
        
    save_path = os.path.join(save_dir, f"{subject_id}_episodes.csv")
    episodes_df.to_csv(save_path, index=False)
    print(f"Saved episodes for Subject {subject_id}")
    
    del eeg_df, eBOSC, cfg, episodes_df
    gc.collect()