import mne  
import pandas as pd  
import numpy as np
import utils
# import feature_extraction
# from scipy.signal import detrend
# from typing import List

# def mk_epochs(raw: mne.io.Raw, labels_df: pd.DataFrame, labels, epoch_tmin :float, epoch_duration :float) -> mne.Epochs:
#     """
#     Create MNE epochs from a DataFrame with binary labels and a Raw object.

#     ### Parameters:
#     raw (mne.io.Raw): The MNE Raw object containing EEG data.
#     labels_df (pd.DataFrame): DataFrame with timestamps and binary labels.
#                        Assumes first column is 'timestamp' and label columns start at index 2.
#     epoch_duration (float): Duration of each epoch in seconds.

#     ### Returns:
#     mne.Epochs: The epochs created from the Raw data based on DataFrame labels.
#     """
#     labels_df = labels_df[labels + ["Timestamp"]]

#     # Prepare arrays for annotations
#     onsets = []
#     durations = []
#     descriptions = []

#     # Create Annotations
#     for _, row in labels_df.iterrows():
#         label_values = tuple(row[labels])
#         description = '_'.join(map(str, label_values))  # Unique description for each label combination
#         onsets.append(row['Timestamp'] / raw.info["sfreq"])
#         durations.append(epoch_duration)
#         descriptions.append(description)

#     # Create an mne.Annotations object
#     annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

#     # Attach Annotations to Raw Object
#     raw.set_annotations(annotations)

#     # Create Epochs
#     events, event_id = mne.events_from_annotations(raw)
#     epoch_tmin, tmax = 0, epoch_duration  # Define epoch window based on duration
#     epochs = mne.Epochs(raw, events, event_id, epoch_tmin, tmax, baseline=(0, 0))

#     print("epcohs.get_data().shape:", epochs.get_data().shape)
#     print("raw_data.shape:", raw.get_data().shape)
#     print("len(labels_df.index):", len(labels_df.index))
    
#     return labels_df[labels], epochs

def prepare_data(filepath_raw, filepath_labels, labels, epoch_tmin, epoch_duration, frquency_band=None):
    # Loading raw EEG data and creating Raw object
    raw, _, labels_df = utils.load_data(filepath_raw, filepath_labels)

    # Optional feature engineering
    # raw._data = detrend(raw.get_data())
    if frquency_band is not None:
        raw.filter(frquency_band[0], frquency_band[1])

    # labels_events, epochs = mk_epochs(raw, labels_df, labels, epoch_tmin, epoch_duration)
    
    epochs = utils.normalize_epochs(epochs)
    return epochs, labels_events


def processed_data(raw_filepaths, label_filepaths, labels, epoch_tmin, epoch_duration, frquency_band=None):
    all_epochs = []
    all_labels = []

    # Loop through each file path, prepare data, and collect epochs and labels
    for raw_filepath, label_filepath in zip(raw_filepaths, label_filepaths):
        epochs, labels_events = prepare_data(raw_filepath, label_filepath, labels, epoch_tmin, epoch_duration, frquency_band)
        all_epochs.append(epochs)
        all_labels.append(labels_events)

    # Combine epochs and labels from all datasets
    combined_epochs = mne.concatenate_epochs(all_epochs)
    combined_labels = np.concatenate(all_labels)

    return combined_epochs.get_data(), combined_labels
