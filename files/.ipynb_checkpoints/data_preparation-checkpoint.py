import mne  
import pandas as pd  
import numpy as np
import utils
import feature_extraction
from scipy.signal import detrend
from typing import List

def prepare_full_recording_labels_events(labels_df, labels):
    """
    Creates an mne like ndarray event.

    ### Description:
    The event array contains the timestamps where the (one hot encoded) 0label or 1label is set to one.  
    In addition, the event contains the timestamps where the 0label and 1label are set to 0 in which case the event value is set to 2.\n
    """
    labels_df = labels_df[['Timestamp','Epochs'] + labels]
    #map_df_labels_to_event_labels = lambda row: {(1, 0): 1, (0, 1): 1, (0, 0): 2}[(row[labels[0]], row[labels[1]])]
    #labels_events = np.arange(len(labels_df.index)) % 2
    #labels_events = np.(len(labels_df.index)) % 2
    #print(labels_events)
    #print("event_labels_series.shape:", labels_events.shape)
    time_events = labels_df['Timestamp']
    events = np.column_stack((time_events, np.ones(len(time_events), dtype=int), np.ones(len(time_events), dtype=int)))
    return labels_df[labels].to_numpy(int) , events

def prepare_labels_events(labels_df, labels):
    """
    ### Description:
    Creates an mne like ndarray event.  
    The event array contains only the timestamps where the (one hot encoded) 0 or 1 label is set to one.  
    """
    if not labels:
        raise ValueError("Choose a label to include in the analysis")

    labels_df = labels_df[['Timestamp','Epochs'] + labels]
    labels_events = np.full(len(labels_df), 2, dtype=int)
    labels_events[labels_df[labels[0]] == 1] = 0  # Assign 0 when labels[0] is 1
    labels_events[labels_df[labels[1]] == 1] = 1  # Assign 1 when labels[1] is 1
    time_events = labels_df['Timestamp'].values[labels_events != 2]
    events = np.column_stack((time_events, np.full(len(time_events), 1, dtype=int), labels_events))
    return [labels_events, events]

def mk_epochs(raw: mne.io.Raw, labels_df: pd.DataFrame, labels, tmin, epoch_duration) -> mne.Epochs:
    """
    Create MNE epochs from a DataFrame with binary labels and a Raw object.

    Parameters:
    raw (mne.io.Raw): The MNE Raw object containing EEG data.
    labels_df (pd.DataFrame): DataFrame with timestamps and binary labels.
                       Assumes first column is 'timestamp' and label columns start at index 2.
    epoch_duration (float): Duration of each epoch in seconds.

    Returns:
    mne.Epochs: The epochs created from the Raw data based on DataFrame labels.
    """
    labels_df = labels_df[labels + ["Timestamp"]]

    # Prepare arrays for annotations
    onsets = []
    durations = []
    descriptions = []

    # Create Annotations
    for _, row in labels_df.iterrows():
        label_values = tuple(row[labels])
        description = '_'.join(map(str, label_values))  # Unique description for each label combination
        onsets.append(row['Timestamp'] / raw.info["sfreq"])
        durations.append(epoch_duration)
        descriptions.append(description)

    # Create an mne.Annotations object
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

    # Attach Annotations to Raw Object
    raw.set_annotations(annotations)

    # Create Epochs
    events, event_id = mne.events_from_annotations(raw)
    tmin, tmax = 0, epoch_duration  # Define epoch window based on duration
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(0, 0))

    print("epcohs.get_data().shape:", epochs.get_data().shape)
    print("raw_data.shape:", raw.get_data().shape)
    print("len(labels_df.index):", len(labels_df.index))
    
    return labels_df[labels], epochs

def prepare_data(filepath_raw, filepath_labels, labels, include_entire_recording=False):
    # Loading raw EEG data and creating Raw object
    raw, raw_data, labels_df = utils.load_data(filepath_raw, filepath_labels)

    # Optional feature engineering
    # raw._data = detrend(raw.get_data())
    # raw._data = utils.vdm_raw(raw)
    # raw = raw.notch_filter(50)
    raw.filter(8, 16)
    # raw = raw.copy().crop(tmin=start_crop, tmax=end_crop)
    # labels_df = process_eeg_events(labels_df, start_crop, end_crop, raw.info['sfreq'])

    raw_data = raw.get_data()
    if include_entire_recording:
        labels_events, epochs = mk_epochs(raw, labels_df, labels, 0, 2.5)
    else:
        labels_events, events = prepare_labels_events(labels_df, labels)
        #Creating epochs around the events
        epochs = mne.Epochs(raw,
                            events, 
                            tmin=-0.5, tmax=2,
                            baseline=None, preload=True)
    
    epochs = utils.normalize_epochs(epochs)
    return {'Epochs': epochs, 'Labels': labels_events}


def processed_data(raw_filepaths, label_filepaths, labels, fmin, fmax, include_entire_recording=False):
    all_epochs = []
    all_labels = []

    # Loop through each file path, prepare data, and collect epochs and labels
    for raw_filepath, label_filepath in zip(raw_filepaths, label_filepaths):
        data = prepare_data(raw_filepath, label_filepath, labels, include_entire_recording)
        all_epochs.append(data['Epochs'])
        all_labels.append(data['Labels'])

    # Combine epochs and labels from all datasets
    combined_epochs = mne.concatenate_epochs(all_epochs)
    combined_labels = np.concatenate(all_labels)

    # Extract features for all combined epochs
    #X = feature_extraction.get_raw_feature_all(all_epochs, fmin, fmax)

    return combined_epochs.get_data(), combined_labels
