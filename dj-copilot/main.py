import librosa
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, sr, order=5):
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = lfilter(b, a, data)
    return y

print("Current Working Directory: ", os.getcwd())

def transient_compatability():
    pass


def get_features(filepath):
    y, sr = librosa.load(filepath)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    energy_rms = np.mean(librosa.feature.rms(y=y))
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    # threshold = 0.99
    # y_harmonic, y_percussive = librosa.effects.hpss(y)
    # onset_frames_percussive = librosa.onset.onset_detect(y=y_percussive, sr=sr)
    # onset_times_percussive = librosa.frames_to_time(onset_frames_percussive, sr=sr)
    y_bass = lowpass_filter(y, 150, sr)
    onset_frames_bass = librosa.onset.onset_detect(y=y_bass, sr=sr)
    onset_times_bass = librosa.frames_to_time(onset_frames_bass, sr=sr)
    interval_times = np.diff(onset_times_bass)
    median_bass_interval = np.median(interval_times)
    #strong_onsets = [round(time, 2) for time, strength in zip(onsets, onset_env) if strength > threshold]
    # spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    # spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    #energy = (energy_rms + spectral_centroids + spectral_bandwidth) / 3
    return y, sr, round(tempo), energy_rms, median_bass_interval


def data_preprocessing():
    df = pd.DataFrame()
    directory = './dj-copilot/test'
    filenames = [filename for filename in os.listdir(directory)]
    df['Name'] = filenames
    df[['y', 'sr', 'Tempo', 'Energy', 'Transients_Percussive', 'Transients_Bass']] = df['Name'].apply(lambda x: pd.Series(get_features(os.path.join(directory, x))))
    df.to_csv('songs.csv', index=False)

def main():

    data_preprocessing()


if __name__ == "__main__":
    main()