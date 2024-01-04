from keras.utils import timeseries_dataset_from_array
import numpy as np
import pandas as pd  
import mne  
from scipy.io import loadmat
from scipy.signal import detrend
import yasa
from scipy.signal import welch

DEFAULT_DIVIDER = 10000000


def load_eeg_data(mat_file_path):
    # Load the .mat file using scipy
    mat = loadmat(mat_file_path)
    # Extract EEG data
    return mat['EEG'][0, 0]['data']

def mk_raw_obj(eeg_data, sfreq=250):
    info = mne.create_info(
        ch_names=[f'EEG{i}' for i in range(len(eeg_data))],
        sfreq=sfreq,
        ch_types=['eeg' for _ in range(len(eeg_data))]
    )
    
    return mne.io.RawArray(eeg_data, info)

def load_data(file_path, labels_path):
    raw_mat = load_eeg_data(file_path)
    raw = mk_raw_obj(raw_mat)
    raw_data = raw.get_data()
    labels = pd.read_csv(labels_path)
    labels.sort_values("Timestamp", inplace=True)
    return raw, raw_data, labels

def preprocess_recording_data(recording_data, frequency_band=None, sampling_freq=250):
    # Detrending
    recording_data = detrend(recording_data)
    # band pass filtering
    raw_obj = mk_raw_obj(recording_data, sfreq=sampling_freq)
    bp_filter_raw_obj = raw_obj.filter(frequency_band[0], frequency_band[1])
    recording_data = bp_filter_raw_obj.get_data()

    return recording_data

def hypnogram_propas(recording_data, sampling_freq=250):
    """
    Computes the propabilites of the each sleep stages at each 30s epoch.
    Then, upsamples the probabilites to match the shape of the recording.
    ### Parameters:
    recording_data: ndarray of the recording
    ### Returns:
    Tuple of shape four, each item is a 1D array of the probability of a sleep stage at a given timestamp.
    Four for the four sleep stages: awake, REM, NREM1, NREM2, NREM3
    """
    # For some reason, yasa doesn't work properly with the unscaled data.
    scalled_raw_obj = mk_raw_obj(recording_data / DEFAULT_DIVIDER, sfreq=sampling_freq)
    sls = yasa.SleepStaging(scalled_raw_obj, eeg_name="EEG0")
    hypno_proba = sls.predict_proba()
    return (yasa.hypno_upsample_to_data(hypno_proba[column], 1/30, scalled_raw_obj, verbose=False) for column in hypno_proba.columns)

def band_psd_ratio(recording_data, band1, band2, window_size, sfreq=250):

    num_windows = len(recording_data) - window_size + 1
    power_ratios = np.empty((num_windows))

    for i in range(num_windows):
        f, psd = welch(recording_data[i:i+window_size].squeeze(), sfreq, nperseg=int(sfreq * 2))
        # Calculate power in the designated frequency bands
        band1_power = psd[(f >= band1[0]) & (f <= band1[1])].mean()
        band2_power = psd[(f >= band2[0]) & (f <= band2[1])].mean()
        ratio = band1_power / band2_power
        power_ratios[i] = ratio
    
    power_ratios = np.pad(
        power_ratios,
        pad_width=(0, len(recording_data) - num_windows),
        mode='constant',
        constant_values=(power_ratios[0], power_ratios[-1]))
    return power_ratios


def dataset_from_files(
        recording_files,
        labels_files=None,
        target_label=None,
        sampling_freq=250,
        frequency_band=None,
        include_hypno_proba=True,
        window_size=None,
        band1=None,
        band2=None,
        shuffle=False
        ):
    """
    Loads and preprocesses the EEG recordings.
    Returns a dataset keras obj.
    
    ### Parameters:
    recording_files: List of tuples of (.mat single channel eeg_recording
    labels_files: .csv recording labels
    sampling_freq: sampling frequency of the recording.
    target_label: target label
    frequency_band: tuple (min frequency, max frequency), if not None, used to band pass filter the recordings

    ### Returns:
    Timeseries dataset keras obj
    """
    time_series = []
    for recording_file in recording_files:
        recording_data = load_data(recording_file)
        preprocessed_recording_data = preprocess_recording_data(recording_data, frequency_band=frequency_band)
        hypno_propas = hypnogram_propas(recording_data, sampling_freq=sampling_freq) if include_hypno_proba else ()
        band_psd_ratio = band_psd_ratio(window_size, band1, band2) if window_size is not None else ()
        time_serie = np.column_stack((preprocessed_recording_data, *hypno_propas, *band_psd_ratio))
        time_series.append(time_serie)
    concat_time_serie = np.concatenate(time_serie)

    if labels_files is not None:
        assert target_label is None, "labels_files was set but not target_label."
        target_arrays = []
        for time_serie, labels_file in zip(time_series, labels_files):
            labels_df = pd.read_csv(labels_file)
            presence_incdices = labels_df[labels_df[target_label] == 1]['Timestamp']
            target_array = np.zeros(time_serie.shappe[0])
            target_array[presence_incdices] = 1
            target_arrays.append(target_array)

        concat_target_array = np.concatenate(target_arrays)
    else: 
        concat_target_array = None

    return timeseries_dataset_from_array(concat_time_serie, concat_target_array, window_size, shuffle=shuffle)