import matplotlib.pyplot as plt
import mne

eeg_artifact = mne.io.read_raw_fif('ressources/plot_artifact_eeg.fif', preload=True)
eeg_artifact.filter(0.5, 35)
eeg_artifact.plot(scalings='auto')
plt.show()