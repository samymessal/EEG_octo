import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import json
import utils
import feature_extraction
import data_preparation
from memory_profiler import profile
import preprocess
import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import Callback
import json
from tensorflow.keras.metrics import Metric


X, labels = data_preparation.processed_data(["../dataset/train_S002_night1_hackathon_raw.mat",
                                            #"../dataset/train_S003_night5_hackathon_raw.mat"
                                            ],
                                            ["../dataset/train_S002_labeled.csv",
                                            #"../dataset/train_S003_labeled.csv"
                                            ],
                                            labels=["SS1", "K0", "K1"],
                                            fmin=7,
                                            fmax=17,
                                            include_entire_recording=True)

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.f1_score = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        p = self.precision.result()
        r = self.recall.result()
        self.f1_score.assign(2 * ((p * r) / (p + r + tf.keras.backend.epsilon())))

    def result(self):
        return self.f1_score

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
        self.f1_score.assign(0)

kfold = KFold(n_splits=3)
for fold_no, (train, test) in enumerate(kfold.split(X, labels)):

    # Define the model architecture
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(20))
    model.add(Dropout(0.2))
    model.add(Dense(labels.shape[1], activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            F1Score(),
        ]
    )
    
    # Train the model
    history = model.fit(
        X[train],
        labels[train],
        epochs=30,
        validation_data=(X[test], labels[test]),
    )

    print("Training history:")
    print(json.dumps(history.history, indent=1))

    
    plt.plot(training_f1_scores, label='Training F1 Score')
    plt.plot(validation_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

