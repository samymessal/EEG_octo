import numpy as np
from scipy.signal import welch
import pywt
import utils
import mne

def extract_psd_features(epochs, fmin, fmax):
    # Initialize an empty list to store PSD feature values
    features = []
    # Retrieve the sampling frequency from the epochs info structure
    sfreq = epochs.info['sfreq']
    # Iterate over each epoch within the input data
    for epoch in epochs.get_data():
        # Calculate the Power Spectral Density of the epoch using Welch's method
        f, psd = welch(epoch.squeeze(), sfreq, nperseg=int(sfreq*2))
        # Filter the frequencies to extract power in the band of interest
        idx_band = np.logical_and(f >= fmin, f <= fmax)
        psd_band = psd[idx_band]
        # Calculate the average power in the band and append it to the features list
        features.append(psd_band.mean())
    return np.array(features)

def extract_bp_feature(epochs, band1, band2):
    power_ratios = []
    sfreq = epochs.info['sfreq']

    for epoch in epochs.get_data():
        f, psd = welch(epoch.squeeze(), sfreq, nperseg=int(sfreq*2))
        # Calculate power in the designated frequency bands
        band1_power = psd[(f >= band1[0]) & (f <= band1[1])].mean()
        band2_power = psd[(f >= band2[0]) & (f <= band2[1])].mean()
        ratio = band1_power / band2_power
        power_ratios.append(ratio)
    return np.array(power_ratios)

def calculate_scales(freq_min, freq_max, fs, wavelet_center_frequency=6):
    scale_min = wavelet_center_frequency * fs / freq_max
    scale_max = wavelet_center_frequency * fs / freq_min

    scales = np.arange(scale_min, scale_max)
    return scales

def extract_cwt_features(epochs, fmin, fmax, wavelet='morl'):
    scales = calculate_scales(fmin, fmax, epochs.info['sfreq'])
    eeg_data = epochs.get_data()
    num_epochs, _, signal_length = eeg_data.shape
    num_features = len(scales)
    features = np.zeros((num_epochs, num_features, signal_length))

    for epoch_idx in range(num_epochs):
        signal = eeg_data[epoch_idx, 0, :]
        coefficients, _ = pywt.cwt(signal, scales, wavelet=wavelet)
        features[epoch_idx, :, :] = coefficients

    return features

def combined_raw_features(epochs, features):
    raw_data = epochs.get_data()
    n_epochs, n_channels, n_times = raw_data.shape

    # Ensure 'features' is a list, even if it's a single feature
    features = np.atleast_1d(features)
    total_channels = n_channels + len(features)
    combined_data = np.zeros((n_epochs, total_channels, n_times))
    combined_data[:, :n_channels, :] = raw_data

    # Add each feature, repeated across all time points
    for i, feature in enumerate(features):
        if feature.ndim == 1 and feature.size == n_epochs:
            combined_data[:, n_channels + i, :] = np.repeat(feature[:, np.newaxis], n_times, axis=1)
        else:
            msg = "Feature must be a 1D array with length equal to the number of epochs."
            raise ValueError(msg)

    return combined_data

def get_raw_feature_all(epochs, fmin, fmax):
    combined_all = None
    for subject in epochs:
        psd = extract_psd_features(subject, fmin, fmax)
        bd_power = extract_bp_feature(subject, (fmin, 13), (13, fmax))
        cwt = extract_cwt_features(subject, fmin, fmax) 
        combined = combined_raw_features(subject, [psd, bd_power])
        combined = np.concatenate([combined, cwt], axis=1)
        if combined_all is None:
            combined_all = combined
        else:
            combined_all = np.concatenate([combined_all, combined])
    return combined_all

