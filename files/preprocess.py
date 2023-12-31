import warnings
warnings.filterwarnings("ignore")
import mne
import utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend
from scipy import signal
import sklearn.preprocessing as preprocess
import vmdpy
import pickle


def load_data(file_path, labels_path):
    raw_mat = utils.load_eeg_data(file_path)
    raw = utils.mk_raw_obj(raw_mat)
    raw_data = raw.get_data()
    labels = pd.read_csv(labels_path)
    labels.sort_values("Timestamp", inplace=True)
    return raw, raw_data, labels

def detrend_array(raw_data):
    detrended_data = detrend(raw_data)
    info = mne.create_info(ch_names=['EEG0'], sfreq=250, ch_types='eeg')
    detrended_raw = mne.io.RawArray(detrended_data, info)
    return detrended_raw, detrended_data

def display_statistics(raw_data, detrend_data):
    statistics = {
        "Mean": [raw_data.mean(), detrend_data.mean()],
        "Std": [raw_data.std(), detrend_data.std()]
    }
    df = pd.DataFrame(statistics, index=["Raw", "Detrended"])
    print(df)

def plot_psd(raw, title):
    psd = raw.compute_psd()
    psd.plot()
    plt.title(title)
    plt.show()

def create_epochs(raw, duration):
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, 
                        events,
                        tmin=0, 
                        tmax=duration, 
                        baseline=None, 
                        preload=False)
    return epochs

def detect_amplitude_artifacts(epoch, threshold_factor=4):
    mean_amplitude = np.mean(epoch)
    std_dev_amplitude = np.std(epoch)
    upper_threshold = mean_amplitude + threshold_factor * std_dev_amplitude
    lower_threshold = mean_amplitude - threshold_factor * std_dev_amplitude
    artifact_indices = np.logical_or(epoch > upper_threshold, epoch < lower_threshold)

    return np.where(artifact_indices == 1)[0]

def get_artifact_times(epochs, epoch_duration, sfreq):
    artifact_times = np.empty(0, dtype=int)
    for i, epoch in enumerate(epochs):
        epoch_data = epoch[0]
        artifact_indices = detect_amplitude_artifacts(epoch_data, 5)
        adjusted_indices = artifact_indices + int(i * epoch_duration * sfreq)
        print(adjusted_indices)
        artifact_times = np.append(artifact_times, adjusted_indices)
    return artifact_times

def is_artifact_close(artifact, labels, threshold):
    for label in labels:
        if abs(artifact['onset'] - label['onset']) <= threshold:
            return True
    return False

def get_artifacts_annotations(raw, artifact_times, label_annotation, filter=False):
    artifact_annotation = mne.Annotations(onset=artifact_times,
                                          duration = [0.01] * len(artifact_times),
                                          description=['Art'] * len(artifact_times))
    if filter is True:
        filtered_artifacts = [art for art in artifact_annotation if is_artifact_close(art, label_annotation, 30)]
        artifact_annotation = mne.Annotations(onset=[ann['onset'] for ann in filtered_artifacts],
                                       duration=[ann['duration'] for ann in filtered_artifacts],
                                       description=[ann['description'] for ann in filtered_artifacts])
    return artifact_annotation

def get_annotations(labels, event_types = ['SS', 'K', 'REM', 'Son', 'Soff', 'A', 'MS']):
    annotations = mne.Annotations(onset=[], duration=[], description=[])
    
    for event in event_types:
        onsets = labels.loc[labels[event + '1'] == 1, 'Timestamp'] / 250
        durations = [2] * len(onsets)
        annotation = mne.Annotations(onset=onsets, duration=durations, description=event)
        annotations += annotation
    
    return annotations

def make_vdm(raw):
    num_modes = 3  # Number of components the raw data will be splitted into 
    alpha = 100   
    tau = 0.1    
    tol = 1e-3
    maxiter = 500
    raw.filter(4, 35, verbose=False)
    raw_split = np.array_split(raw.get_data()[0], 5)
    all_modes = []
    for k, split in enumerate(raw_split):
        modes = vmdpy.VMD(split, num_modes, alpha, tau, tol, maxiter)
        np.save(f'u_sub3_{k}.npy', modes[0])
        all_modes.append(modes)
        with open('all_u_sub3.pkl', 'wb') as f:
            pickle.dump(all_modes, f)

def plot_raw_and_psd(eeg_signal, axes, row, duration=200, label="VDM component", spectrum_label="PSD"):
    # Plot the eeg_signal
    axes[row, 0].plot(eeg_signal)
    axes[row, 0].set_title(f"{label} {row + 1}")

    # Compute and plot the power spectral density (PSD)
    freqs, psd = signal.welch(eeg_signal, 250)
    axes[row, 1].plot(freqs, psd)
    axes[row, 1].set_title(f"{spectrum_label} {row + 1}")

import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_binary_crossentropy(y_true, y_pred, weights):
    """
    Custom weighted binary cross-entropy loss function.

    Parameters:
    y_true: True labels
    y_pred: Predicted labels/probabilities
    weights: 2D array of shape (2, n_labels) containing weights
             Row 0 contains weights for false negatives,
             Row 1 contains weights for false positives.

    Returns:
    Weighted binary cross-entropy loss.
    """
    # Convert weights to a TensorFlow tensor and ensure float32 data type
    weights = tf.cast(weights, dtype=tf.float32)

    # Ensure y_true and y_pred are of float32 type
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)


    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Calculate Binary Cross Entropy
    bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    print("bce:", bce.values)
    # Prepare to apply weights
    def apply_weights(args):
        y_true_slice, bce_slice = args[0], args[1]
        return bce_slice * tf.gather(weights, tf.cast(y_true_slice, tf.int32), axis=0)

    # Apply weights using tf.map_fn
    weighted_bce = tf.map_fn(apply_weights, (y_true, bce), dtype=tf.float32)
    print("weighted_bce:", weighted_bce.values)

    # Return mean loss
    return K.mean(weighted_bce, axis=-1)

def compute_multi_label_loss_weights(labels):
    """
    Compute the weights for each class in each label using the formula:
    n_samples / (n_classes * np.bincount(y))

    Parameters:
    labels: numpy array of shape (n_samples, n_labels)
            Binary labels for each sample.

    Returns:
    weights: numpy array of shape (2, n_labels)
             Row 0 contains weights for class 0,
             Row 1 contains weights for class 1.
    """
    n_samples, n_labels = labels.shape
    n_classes = 2
    weights = np.zeros((2, n_labels), dtype=float)

    for i in range(n_labels):
        # Count occurrences of each class in the current label
        class_count = np.bincount(labels[:, i], minlength=n_classes)
        
        # Compute weights inversely proportional to class frequencies
        class_weights = n_samples / (n_classes * class_count + 1e-8)  # Adding a small value to avoid division by zero
        
        # Assign weights
        weights[:, i] = class_weights

    print(weights)
    return weights














    