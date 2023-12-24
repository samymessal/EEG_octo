from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import sklearn.preprocessing as preprocess
import pywt
import vmdpy
import pickle
import seaborn as sns
import json

def load_eeg_data(mat_file_path):
    # Load the .mat file using scipy
    mat = loadmat(mat_file_path)
    # Extract EEG data
    return mat['EEG'][0, 0]['data']

def mk_raw_obj(eeg_data):
    info = mne.create_info(
        ch_names=[f'EEG{i}' for i in range(len(eeg_data))],
        sfreq=250,
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

def normalize_epochs(epochs):
    data = epochs.get_data()

    scaler = preprocess.StandardScaler()
    n_epochs, n_channels, n_times = data.shape
    data_normalized = np.zeros_like(data)
    for i in range(n_epochs):
        # Reshape each epoch for normalization and then reshape it back
        epoch_normalized = scaler.fit_transform(data[i].reshape(n_channels * n_times, 1)).reshape(n_channels, n_times)
        data_normalized[i] = epoch_normalized
    # Assign the normalized data back to the Epochs object
    epochs._data = data_normalized
    return epochs

def process_eeg_events(labels, crop_start, crop_end, sampling_freq):
    crop_start_samples = int(crop_start * sampling_freq)
    crop_end_samples = int(crop_end * sampling_freq)
    # Adjust timestamps based on cropping
    labels['Timestamp'] -= crop_start_samples
    # Filter out events outside the cropping window
    labels = labels[(labels['Timestamp'] >= 0) & (labels['Timestamp'] <= crop_end_samples - crop_start_samples)]
    return labels

    
def evaluate_model(model, X_test, y_test):
    print("X_test.shape:", X_test.shape)
    print("X_test unique:", np.unique(X_test))
    print("y_test.shape:", y_test.shape)
    print("y_test unique:", np.unique(y_test))
    
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.round(y_pred_prob).astype(int)  # Convert probabilities to binary predictions

    # Calculate metrics
    accuracy = jaccard_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Confusion matrix
    def display_cm(y_test, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    if len(y_test.shape) == 1:
        display_cm(y_test, y_pred)
    else:
        for column_i in range(y_test.shape[1]):
            display_cm(y_test[:, column_i], y_pred[:, column_i], title=f"Label{column_i} Confusion matrix")
    
    return { "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1 }   


def make_vdm(raw):
    num_modes = 3  # Start with 6 modes and adjust as needed
    alpha = 100   # A starting point, adjust for mode smoothness and separation
    tau = 0.1    # Typically 0 for most applications
    tol = 1e-3     # Standard starting point, decrease for more accuracy
    maxiter = 500  # Increase if needed for convergence
    # raw.filter(4, 35, verbose=False)
    eeg_signal = raw.get_data()[0]
    raw_split = np.array_split(raw.get_data()[0], 5)
    all_modes = []
    for k, split in enumerate(raw_split):
        if k == 0 and k == 1:
            continue
        modes = remove_artifacts_vmd(split, num_modes, alpha, tau, tol, maxiter)
        np.save(f'u{k}.npy', modes[0])
        all_modes.append(modes)
        with open('all_u.pkl', 'wb') as f:
            pickle.dump(all_modes, f)
    return all_modes

def vdm_raw(raw, idx_components):
    concatenated = None
    fname = input("Which subject: ")
    for i in range(0, 5):
        u_load = np.load(f'{fname}{i}.npy')
        if concatenated is None:
            concatenated = u_load
        else:
            concatenated = np.concatenate([concatenated, u_load], axis=1)
    reconstructed_signal = np.sum([concatenated[idx_components]], axis=0)
    # reconstructed_signal = concatenated[2]
    reconstructed_signal = reconstructed_signal.reshape(1, -1)
    return reconstructed_signal

def save_model(model, history, perf_metrics, fold_no):
    # Save the model and training history
    model.save(f"ressources/models/h5_files/SS_0Pre_1Features_LSTM_{fold_no}.h5")
    with open(f"ressources/models/history/SS_0Pre_1Features_LSTM_{fold_no}.json", 'w') as json_file:
        json.dump(history.history, json_file)

    # Evaluate and save the model's performance
    with open(f"ressources/models/metrics/addlayerSS_0Pre_1Features_LSTM_{fold_no}_metrics.json", 'w') as json_file:
        json.dump(perf_metrics, json_file)

def print_performances(fname, n_folds):
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    # Loop through each fold and gather metrics
    for fold_no in range(1, n_folds + 1):
        with open(f"ressources/models/metrics/{fname}{fold_no}_metrics.json", 'r') as json_file:
            metrics = json.load(json_file)
        accuracies.append(round(metrics['accuracy'], 2))
        precisions.append(round(metrics['precision'], 2))
        recalls.append(round(metrics['recall'], 2))
        f1_scores.append(round(metrics['f1_score'], 2))

    # Create a DataFrame from the metrics
    df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(1, n_folds + 1)],
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    # Calculate the average for each metric
    average_metrics = {
        'Fold': 'Average',
        'Accuracy': round(df['Accuracy'].mean(), 2),
        'Precision': round(df['Precision'].mean(), 2),
        'Recall': round(df['Recall'].mean(), 2),
        'F1 Score': round(df['F1 Score'].mean(), 2)
    }

    # Append the average metrics to the DataFrame
    df = df._append(average_metrics, ignore_index=True)
    return df

def plot_fold_history(fname, fold_no):
    # Load training history
    with open(f"ressources/models/history/{fname}{fold_no}.json", 'r') as json_file:
        history = json.load(json_file)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'Model Accuracy for Fold {fold_no}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model Loss for Fold {fold_no}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


def prepare_data(filepath_raw, filepath_labels, labels=None):
    raw_mat = load_eeg_data(filepath_raw)
    raw = mk_raw_obj(raw_mat)
    scaler = preprocess.StandardScaler()
    start_crop = 2
    end_crop = raw.times[-2]
    # raw._data = vdm_raw(raw)
    # raw = raw.copy().crop(tmin=start_crop, tmax=end_crop)
    # raw.filter(4, 35)
    # raw = raw.notch_filter(50)
    #Detrend optional
    # normalize = scaler.fit_transform(raw.get_data()[0,:].reshape(-1, 1)).flatten()
    raw._data = detrend(raw.get_data())
    # raw._data = clean_eeg_wavelet(raw.get_data())
    labels_df = pd.read_csv(filepath_labels)
    labels_df.sort_values("Timestamp", inplace=True)
    # labels_df = process_eeg_events(labels_df, start_crop, end_crop, raw.info['sfreq'])
    if (labels):
        labels_df = labels_df[['Timestamp','Epochs'] + labels]
        time_events = labels_df.loc[(labels_df[labels[1]] == 1) | (labels_df[labels[0]] == 1),'Timestamp']
        labels_events = labels_df.loc[(labels_df[labels[1]] == 1) | (labels_df[labels[0]] == 1),labels[1]]
        events= np.column_stack((time_events, 
                                np.full(len(time_events), 1, dtype=int), 
                                labels_events))
        epochs = mne.Epochs(raw, 
                        events, 
                        tmin=-0.5, 
                        tmax=2,
                        baseline=None, 
                        preload=True)
        num_epochs = len(time_events)
        labels_events = np.zeros(num_epochs)
        for i, epoch in enumerate(epochs):
            if epochs.events[i, 2] == 1:
                labels_events[i] = 1
        epochs = normalize_epochs(epochs)
        return {'Epochs': epochs, 'Labels': labels_events}