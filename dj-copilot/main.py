import librosa
import numpy as np
import pandas as pd
import os

print("Current Working Directory: ", os.getcwd())



def get_features(filepath):
    y, sr = librosa.load(filepath)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    energy_rms = np.mean(librosa.feature.rms(y=y))
    # spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    # spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    #energy = (energy_rms + spectral_centroids + spectral_bandwidth) / 3
    return y, sr, round(tempo), energy_rms

def data_preprocessing():
    df = pd.DataFrame()
    directory = './dj-copilot/songs'
    filenames = [filename for filename in os.listdir(directory)]
    df['Name'] = filenames
    df[['y', 'sr', 'Tempo', 'Energy']] = df['Name'].apply(lambda x: pd.Series(get_features(os.path.join(directory, x))))
    df.to_csv('songs.csv', index=False)

def main():
    data_preprocessing()


if __name__ == "__main__":
    main()