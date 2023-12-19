import mne  
import pandas as pd  
import numpy as np
import utils
import feature_extraction
from scipy.signal import detrend

def prepare_labels_events(labels_df, labels=None):
    if labels:
        labels_df = labels_df[['Timestamp','Epochs'] + labels]
        labels_events = np.full(len(labels_df), 2, dtype=int)
        labels_events[labels_df[labels[0]] == 1] = 0  # Assign 0 when labels[0] is 1
        labels_events[labels_df[labels[1]] == 1] = 1  # Assign 1 when labels[1] is 1
        time_events = labels_df['Timestamp'].values[labels_events != 2]
        labels_events = labels_events[labels_events != 2]
        print("labels_events.shape:", labels_events.shape)
        print("time_events.shape:", time_events.shape)
        events = np.column_stack((time_events, np.full(len(time_events), 1, dtype=int), labels_events))
        return [labels_events, events]
    else:
        raise ValueError("Choose a label to include in the analysis")
    

def prepare_data(filepath_raw, filepath_labels, labels=None):
    # Loading raw EEG data and creating Raw object
    raw, raw_data, labels_df = utils.load_data(filepath_raw, filepath_labels)

    #Optional feature engineering

    # raw._data = detrend(raw.get_data())
    # raw._data = utils.vdm_raw(raw)
    # raw = raw.notch_filter(50)
    # raw.filter(11, 15)
    # raw = raw.copy().crop(tmin=start_crop, tmax=end_crop)
    # labels_df = process_eeg_events(labels_df, start_crop, end_crop, raw.info['sfreq'])

    raw_data = raw.get_data()
    labels_events, events = prepare_labels_events(labels_df, labels)

    #Creating epochs around the events
    epochs = mne.Epochs(raw,
                        events, 
                        tmin=-0.5, tmax=2,
                        baseline=None, preload=True)
    
    epochs = utils.normalize_epochs(epochs)
        
    return {'Epochs': epochs, 'Labels': labels_events}

def processed_data(raw_filepaths, label_filepaths, labels, fmin, fmax):
    all_epochs = []
    all_labels = []

    # Loop through each file path, prepare data, and collect epochs and labels
    for raw_filepath, label_filepath in zip(raw_filepaths, label_filepaths):
        data = prepare_data(raw_filepath, label_filepath, labels)
        all_epochs.append(data['Epochs'])
        all_labels.append(data['Labels'])

    # Combine epochs and labels from all datasets
    combined_epochs = mne.concatenate_epochs(all_epochs)
    combined_labels = np.concatenate(all_labels)

    # Extract features for all combined epochs
    X = feature_extraction.get_raw_feature_all(all_epochs, fmin, fmax)

    return X, combined_labels
