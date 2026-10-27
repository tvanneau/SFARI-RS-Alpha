# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:57:38 2024

@author: theov
"""

#============================================
#               RS experiment
#============================================

#loading needed toolboxes 
import mne
import numpy as np
import copy
import tkinter as tk
import pandas as pd
from tkinter import filedialog
import logging
import os
from matplotlib import pyplot as plt

#%% Reading BDF files 

Subjects = os.listdir('Z://Data_new/SFARI_DATA_2024/Restingstate_eyetracking/')

#%% load files

Subject = Subjects[1] # To do ; S = 38, Beginning ASD group 
file_path = 'Z://Data_new/SFARI_DATA_2024/Restingstate_eyetracking/' + Subject
subject_path = f'Z://Data_new/SFARI_DATA_2024/Restingstate_eyetracking/{Subject}'

# List all files in the directory for the specific subject
EEG_files = os.listdir(subject_path)

# Filter out only the .bdf files
bdf_files = [file for file in EEG_files if file.endswith('.bdf') and 'rest' in file]
# bdf_files = [file for file in EEG_files if file.endswith('.bdf')]
ET_files = [file for file in EEG_files if file.endswith('.asc')]

# bdf_files = [file for file in EEG_files if file.endswith('.bdf')]

# Print the .bdf files
print(bdf_files)

preprocessing_info = {
    'Subject_ID': Subject,  # The subject ID
    'Bad_Channels_Detected': [],  # List of bad channels detected
    'Interpolated_Channels': [],  # List of channels that were interpolated
    'ICA_Components_Removed': []  # List of ICA components removed
    }

#%% Load ET
import re

raw_et = []
annotations = []
for i in range(0, len(ET_files),1):
    et = mne.io.read_raw_eyelink(file_path + "/" + ET_files[i])
    annotations.append(et.annotations)
    # Rename channels by removing '_right' and '_left'
    new_channel_names = {}
    for ch_name in et.ch_names:
        # Remove '_right' and '_left' from xpos, ypos, and pupil channels
        new_name = re.sub(r'_(right|left)', '', ch_name)
        new_channel_names[ch_name] = new_name
    
    # Apply the new names
    et.rename_channels(new_channel_names)
    
    # Append the processed raw ET data
    raw_et.append(et)

raw_ET = mne.concatenate_raws(raw_et, preload=True)

# mne.preprocessing.eyetracking.interpolate_blinks(
#     raw_ET, buffer=(0.05, 0.2))

# Step 3: Extract events based on annotations (instead of using find_events)
# Annotations can be used to create events
annotations = raw_ET.annotations
print(annotations)  # Print to check the available annotations

# Create a list to store events
events_list = []

# Loop through annotations and identify 'TRIALID 1'
for i, description in enumerate(annotations.description):
    # Use regex to match 'TRIALID 1' for the start of a block
    if re.match(r'TRIALID 1', description):
        # Get the onset time of this event
        onset_time = annotations.onset[i]
        
        # Append the event (MNE format requires [onset, 0, event_type])
        events_list.append([int(onset_time * raw_ET.info['sfreq']), 0, 1])  # 1 as the event_type for resting state start

# Convert the list to a numpy array in the MNE format
events_array = np.array(events_list)

# Now `events_array` is in the correct format and can be used to create epochs or for further analysis
print(events_array)

# Step 5: You can now use this events_array to create epochs in MNE
#%% load EEG

raw_EEG = []
Events = []
for i in range(0, len(bdf_files),1):
    raw_tmp = mne.io.read_raw_bdf(file_path + "/" + bdf_files[i], eog=None, misc = None, stim_channel='auto',
                        exclude = (), infer_types = True, preload= True)
    
    events_tmp = mne.find_events(raw_tmp, stim_channel="Status", shortest_event = 1)
    
    raw_EEG.append(raw_tmp)
    Events.append(events_tmp)
    
raw, events= mne.concatenate_raws(raw_EEG, preload=True, 
                                    events_list=Events)
    
print("Number of channels:", len(raw.ch_names))

montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage, on_missing='ignore')

raw.set_channel_types({"EXG1":"emg"})
raw.set_channel_types({"EXG2":"emg"})
raw.set_channel_types({"EXG3":"emg"})
raw.set_channel_types({"EXG4":"emg"})
raw.set_channel_types({"EXG5":"emg"})
raw.set_channel_types({"EXG6":"emg"})
raw.set_channel_types({"EXG7":"emg"})
raw.set_channel_types({"EXG8":"emg"})

#%% Test pyprep - Noisy Channels to detect bad channels in raw object

from pyprep.find_noisy_channels import NoisyChannels

nd = NoisyChannels(raw, random_state=None) 

nd.find_all_bads(ransac = True, channel_wise = True) # Call all the functions to detect bad channels

# Functions = 
# # # NOTE: Bad-by-NaN/flat is already run during init, no need to re-run here
#         self.find_bad_by_deviation()
#         self.find_bad_by_hfnoise()
#         self.find_bad_by_correlation()
#         self.find_bad_by_SNR()
#         if ransac:
#             self.find_bad_by_ransac(
#                 channel_wise=channel_wise, max_chunk_size=max_chunk_size
#             )

# Get the list of bad channels
bad_channels = nd.get_bads()

# Calculate the percentage of bad channels
total_channels = 64  # Since you're working with 64 channels
bad_channel_count = len(bad_channels)
bad_channel_percentage = (bad_channel_count / total_channels) * 100

print('Bad channels detected:',nd.get_bads())
print('Percent of bad channels detected:',bad_channel_percentage)

#%% interpolate bad channels

# Check if the percentage of bad channels exceeds 15%
if bad_channel_percentage > 15:
    print(f"Subject should be removed: {bad_channel_count} bad channels detected ({bad_channel_percentage:.2f}% of total channels).")
else:
    # If less than 15%, proceed to interpolate bad channels
    raw.info["bads"] = bad_channels
    raw = raw.interpolate_bads()
    # Fill the interpolated channels (which are the same as detected bad channels in this case)
    
    print(f"Interpolation performed: {bad_channel_count} bad channels detected ({bad_channel_percentage:.2f}% of total channels).")

#%% Filtration

lowpass_epochs = 40
highpass_epochs = 0.01

raw.filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)

        
#%% Epoching EEG data with all trials after PREP

event_dict = {
    "Start_RS": 50}

tmin_epochs = -1.0
tmax_epochs = 55

epochs_all = mne.Epochs(raw, events, tmin=tmin_epochs, tmax=tmax_epochs, reject=None, 
                        event_id=event_dict, baseline=None, detrend=1, preload=True, decim=1)

#%% Epochs visualisation

epochs_all[0].compute_psd().plot(picks="eeg", exclude="bads")
epochs_all[1].compute_psd().plot(picks="eeg", exclude="bads")
epochs_all[2].compute_psd().plot(picks="eeg", exclude="bads")
epochs_all[3].compute_psd().plot(picks="eeg", exclude="bads")
epochs_all[4].compute_psd().plot(picks="eeg", exclude="bads")
epochs_all[5].compute_psd().plot(picks="eeg", exclude="bads")

#%%

tmin_epochs = -1.0
tmax_epochs = 55

event_dict = {
    "Start_RS": 1}

epochs_all_ET = mne.Epochs(raw_ET, events_array, tmin=tmin_epochs, tmax=tmax_epochs, reject=None, 
                        event_id=event_dict, baseline=None, preload=True, decim=1)

#%%#INDEPENDENT COMPONENTS ANALYSIS (ICA) for PREP epochs files

# epochs_ica = raw_tmp.filter(l_freq=1., h_freq=None)

epochs_ica = epochs_all.filter(l_freq=1., h_freq=None)
# epochs_ica.set_eeg_reference('average')
#ICA est sensible aux dérives basse fréquence donc 1Hz + charge données

#Parameters ICA
n_components = None #0.99 # % de vraiance expliquée ou alors le nombre d'électrodes
max_pca_components = 63 # disparaitra dans une future version de MNE, nombre de PCA a faire avant ICA
random_state = 42 #* pour que l'ICA donne la même chose sur les mêmes données
method = 'fastica' # méthode de l'ICA (si 'picard' nécessite pip install python-picard)
fit_params = None # fastica_it=5 paramètre lié à la methode picard
max_iter = 1000 # nombre d'iterations de l'ICA

ica = mne.preprocessing.ICA(n_components=n_components, method = method, max_iter = max_iter, fit_params= fit_params, random_state=random_state)

ica.fit(epochs_ica)
ica.plot_sources(epochs_all)
# ica.plot_sources(raw_tmp)

ica.plot_components()

#%%APPLICATION ICA COMMUN AVERAGE REFERENCE
ica.apply(epochs_all)

#%% Baseline correction after ICA

# baseline = (-0.5,0)
# epochs_all.apply_baseline(baseline=baseline)

#%%
import json
import os

# Update preprocessing_info dictionary with the values
preprocessing_info = {
    'Subject_ID': Subject,  # The subject ID
    'Bad_Channels_Detected': nd.get_bads(),  # List of bad channels detected
    'Interpolated_Channels': bad_channels,  # List of channels that were interpolated
    'ICA_Components_Removed': ica.exclude  # List of ICA components removed
    }

# Print the updated preprocessing information
print(preprocessing_info)

file_path = 'Z://Analysis/SFARI_Analysis/Resting_State/Epochs/TD'

# Ensure the file path exists, if not, create it
if not os.path.exists(file_path):
    os.makedirs(file_path)

# Epochs file: Save epochs as a .fif file
saving_file_all = file_path + '/' + Subject + '_RS_EEG_epo.fif'
epochs_all.save(saving_file_all)

saving_file_epo_ET = file_path + '/' + Subject + '_RS_ET_epo.fif'
epochs_all_ET.save(saving_file_epo_ET)

# Function to convert NumPy data types to native Python types
def convert_to_serializable(d):
    for key, value in d.items():
        if isinstance(value, np.integer):
            d[key] = int(value)
        elif isinstance(value, np.floating):
            d[key] = float(value)
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()  # Convert arrays to lists if any
    return d

# Convert preprocessing_info dictionary to JSON serializable format
preprocessing_info_serializable = convert_to_serializable(preprocessing_info)

# Save the preprocessing information as a JSON file
json_file_path = file_path + '/' + Subject + '_preprocessing_info.json'
with open(json_file_path, 'w') as json_file:
    json.dump(preprocessing_info_serializable, json_file, indent=4)

print(f"Preprocessing info saved to {json_file_path}")

print("Files saved successfully:")
print(f"Epochs file: {saving_file_all}")


