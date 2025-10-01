import re
from features import feature_extraction, FeatureNormalizer
import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile
def load_audio(filename, mono=True, fs=44100):

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        # Load audio
        audio_data, sample_rate = soundfile.read(filename)
        audio_data = audio_data.T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate

    elif file_extension == '.flac':
        audio_data, sample_rate = librosa.load(filename, sr=fs, mono=mono)

        return audio_data, sample_rate

    return None, None

def load_dataset(evaluation_setup_path, audio_folder, labels, meta_file):
    dataset = {split: {label: [] for label in labels} for split in ['train', 'validation', 'test_labels']}
    test_files = []
    
    for setup_file in glob.glob(os.path.join(evaluation_setup_path, '*.txt')):
        #if 'train' in setup_file:
        #    split = 'train'
        #elif 'evaluate' in setup_file:
        #    split = 'validation'
        if 'test' in setup_file:
            split = 'test'
            test_files.extend([line.strip().split('\t')[0] for line in open(setup_file, 'r')])
            continue
        else:
            continue
        with open(setup_file, 'r') as f:
            for line in f:
                audio_file, label = re.split(r'[\t\s]+', line.strip())
                if label in labels:
                    dataset[split][label].append(os.path.join(audio_folder, audio_file))
    
    # Load test labels from meta.txt
    test_labels = {}
    with open(meta_file, 'r') as f:
        for line in f:
            audio_file, label= line.strip().split('\t') #for 2016
            #audio_file, label, user= line.strip().split('\t')
            if audio_file in test_files and label in labels:
                test_labels[audio_file] = label
    
    dataset['test_labels'] = test_labels
    return dataset

def extract_features(dataset, params, audio_folder):
    #all_features = {split: [] for split in ['train', 'validation', 'test']}
    all_features = {split: [] for split in ['test']}
    for split, label_files in dataset.items():
        if split == 'test_labels':
            continue
        for label, files in label_files.items():
            for audio_filename in files:
                if os.path.isfile(audio_filename):
                    y, fs = load_audio(filename=audio_filename, mono=True, fs=params['fs'])
                    features = feature_extraction(y=y,
                                                  fs=fs,
                                              include_mfcc0=params['include_mfcc0'],
                                              include_delta=params['include_delta'],
                                              include_acceleration=params['include_acceleration'],
                                              mfcc_params=params['mfcc'],
                                              delta_params=params['mfcc_delta'],
                                              acceleration_params=params['mfcc_acceleration'])
                    features = np.hstack((features['stat']['mean'],features['stat']['std'], label))
                    all_features[split].append(np.hstack(([os.path.basename(audio_filename)], features)))
                else:
                    raise IOError(f"Audio file not found [{audio_filename}]")
    
    # Extract test features and add labels from meta.txt
    for audio_file, label in dataset['test_labels'].items():
        audio_filename = os.path.join(audio_folder, audio_file)
        if os.path.isfile(audio_filename):
            y, fs = load_audio(filename=audio_filename, mono=True, fs=params['fs'])
            features = feature_extraction(y=y,
                                                  fs=fs,
                                              include_mfcc0=params['include_mfcc0'],
                                              include_delta=params['include_delta'],
                                              include_acceleration=params['include_acceleration'],
                                              mfcc_params=params['mfcc'],
                                              delta_params=params['mfcc_delta'],
                                              acceleration_params=params['mfcc_acceleration'])
            features = np.hstack((features['stat']['mean'],features['stat']['std'], label))
            all_features['test'].append(np.hstack(([os.path.basename(audio_filename)], features)))
        else:
            raise IOError(f"Audio file not found [{audio_filename}]")
    
    for split in all_features:
        all_features[split] = np.vstack(all_features[split])
        df = pd.DataFrame(all_features[split])
        df.to_csv(f"dataset_fine\\{split}_features_tut2016_evaluation.csv", index=False, header=False)

def normalize_features():
    normalizer = FeatureNormalizer()
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(f"{split}_features.csv", header=None)
        features = df.iloc[:, 1:-1].values.astype(float)
        labels = df.iloc[:, -1].values
        filenames = df.iloc[:, 0].values
        
        if split == 'train':
            normalizer.accumulate(features)
            normalizer.finalize()
        
        normalized_features = normalizer.normalize(features)
        normalized_df = pd.DataFrame(np.column_stack((filenames, normalized_features, labels)))
        normalized_df.to_csv(f"{split}_features_normalized.csv", index=False, header=False)

params = {
    'fs': 44100,
    'win_length_seconds': 0.04,
    'hop_length_seconds': 0.02,
    'include_mfcc0': True,
    'include_delta': True,
    'include_acceleration': True,
    'mfcc': {
        'window': 'hamming_asymmetric',
        'n_mfcc': 20,
        'n_mels': 40,
        'n_fft': 2048,
        'fmin': 0,
        'fmax': 22050,
        'htk': False
    },
    'mfcc_delta': {'width': 9},
    'mfcc_acceleration': {'width': 9}
}
params['mfcc']['win_length'] = int(params['win_length_seconds'] * params['fs'])
params['mfcc']['hop_length'] = int(params['hop_length_seconds'] * params['fs'])
'''
Airport - airport
Indoor shopping mall - shopping_mall
Metro station - metro_station
Pedestrian street - street_pedestrian
Public square - public_square
Street with medium level of traffic - street_traffic
Travelling by a tram - tram
Travelling by a bus - bus
Travelling by an underground metro - metro
Urban park - park'
'''
labels = [
    'beach', 'cafe/restaurant', 'city_center', 'forest_path', 'metro_station',
    'tram', 'park', 'residential_area', 'home', 'bus',
    'grocery_store', 'car', 'train', 'office', 'library'
]
evaluation_setup_path = "F:\sensor\\tut2016_evaluation\TUT-acoustic-scenes-2016-evaluation\evaluation_setup"
audio_folder = "F:\sensor\\tut2016_evaluation\TUT-acoustic-scenes-2016-evaluation"
meta_file = "F:\sensor\\tut2016_evaluation\TUT-acoustic-scenes-2016-evaluation\\meta.txt"

dataset = load_dataset(evaluation_setup_path, audio_folder, labels,meta_file)
extract_features(dataset, params, audio_folder)