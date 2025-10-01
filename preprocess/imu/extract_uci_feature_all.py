import numpy as np
import pandas as pd
from scipy.stats import entropy, moment
from scipy.fftpack import fft
from scipy.signal import correlate
from scipy.signal import butter, filtfilt
from scipy.stats import entropy
def calculate_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_time_domain_features(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    third_moment = moment(data, moment=3)
    fourth_moment = moment(data, moment=4)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)

    value_entropy = entropy(np.histogram(data, bins=20, density=True)[0])

    normalized_data = data / np.sum(data) 
    normalized_data = normalized_data[normalized_data > 0] 
    time_entropy = entropy(normalized_data)

    return [
        mean_val, std_val, third_moment, fourth_moment,
        p25, p50, p75, value_entropy, time_entropy
    ]

def calculate_frequency_domain_features(data, sampling_rate=50):
    fs=50
    bands = [(0, 0.5), (0.5, 1), (1, 3), (3, 5), (5, fs//2)]
    
    def bandpass_filter(data, low_cutoff, high_cutoff):
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        if low == 0:
            b, a = butter(4, high, btype='low')
        elif high == 1:
            b, a = butter(4, low, btype='high')
        else:
            b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def log_energy(data):
        return np.log(np.sum(data**2))
    
    def spectral_entropy(data):
        spectrum = np.abs(np.fft.fft(data))[:len(data)//2]
        spectrum = spectrum / np.sum(spectrum)
        return entropy(spectrum)
    
    log_energies = [log_energy(bandpass_filter(data, band[0], band[1])) for band in bands]
    
    spec_entropy = spectral_entropy(data)
    
    return log_energies + [spec_entropy]

def calculate_autocorrelation_features(data):
    mean_value = np.mean(data)
    data_centered = data - mean_value
    fs = 50

    autocorr = np.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_normalized = autocorr / autocorr[0] 
    minima = np.where((autocorr_normalized[1:-1] < autocorr_normalized[:-2]) & 
                 (autocorr_normalized[1:-1] < autocorr_normalized[2:]))[0] + 1

    if len(minima) > 0:
        main_lobe_end = minima[0] 
        post_main_autocorr = autocorr_normalized[main_lobe_end+1:]
    
        if len(post_main_autocorr) > 0:
            max_lag = np.argmax(post_main_autocorr)
            dominant_lag = main_lobe_end + 1 + max_lag 
        
            dominant_periodicity = dominant_lag / fs
            normalized_autocorr_value = autocorr_normalized[dominant_lag]
        else:
            import pdb; pdb.set_trace()
            dominant_periodicity = None
            normalized_autocorr_value = None
    else:
        import pdb; pdb.set_trace()
        dominant_periodicity = None
        normalized_autocorr_value = None
    
    return [dominant_periodicity,normalized_autocorr_value]
def calculate_correlation_features(data_x, data_y, data_z):
    corr_xy = np.corrcoef(data_x, data_y)[0, 1]
    corr_xz = np.corrcoef(data_x, data_z)[0, 1]
    corr_yz = np.corrcoef(data_y, data_z)[0, 1]
    return [corr_xy, corr_xz, corr_yz]

def extract_features_group(sensor_data, sensor_name):
    x, y, z = sensor_data
    magnitude = calculate_magnitude(x, y, z)

    time_features = calculate_time_domain_features(magnitude)
    freq_features = calculate_frequency_domain_features(magnitude)
    autocorr_features = calculate_autocorrelation_features(magnitude)

    corr_features = calculate_correlation_features(x, y, z)

    axis_stats = [
        np.mean(x), np.mean(y), np.mean(z),
        np.std(x), np.std(y), np.std(z)
    ]
    sensor_features = time_features + freq_features + autocorr_features + axis_stats + corr_features
    return sensor_features

def extract_all_features(body_acc_files, body_gyro_files, total_acc_files, label_file, output_csv):
    body_acc_data = [pd.read_csv(file, header=None).values for file in body_acc_files]
    body_gyro_data = [pd.read_csv(file, header=None).values for file in body_gyro_files]
    total_acc_data = [pd.read_csv(file, header=None).values for file in total_acc_files]

    num_samples = len(body_acc_data[0])
    assert all(len(data) == num_samples for data in body_acc_data + body_gyro_data + total_acc_data), "inconsistent number of samples in sensor files"

    labels = pd.read_csv(label_file, header=None).values.flatten()
    assert len(labels) == num_samples, "inconsistent number of samples between labels and data"

    all_features = []
    from tqdm import tqdm
    for i in tqdm(range(num_samples)):
        body_acc_features = extract_features_group([data[i] for data in body_acc_data], "body_acc")
        body_gyro_features = extract_features_group([data[i] for data in body_gyro_data], "body_gyro")
        total_acc_features = extract_features_group([data[i] for data in total_acc_data], "total_acc")

        feature_row = body_acc_features + body_gyro_features + total_acc_features + [labels[i]]
        all_features.append(feature_row)
    columns = (
        [f"body_acc_time_mean", f"body_acc_time_std", f"body_acc_time_third_moment", f"body_acc_time_fourth_moment",
         f"body_acc_time_p25", f"body_acc_time_p50", f"body_acc_time_p75", f"body_acc_time_value_entropy", 
         f"body_acc_time_time_entropy"] +

        [f"body_acc_freq_band_{i+1}" for i in range(5)] + 
        [f"body_acc_freq_spectral_entropy"] +

        [f"body_acc_autocorr_period", f"body_acc_autocorr_max"] +

        [f"body_acc_stat_mean_x", f"body_acc_stat_mean_y", f"body_acc_stat_mean_z",
         f"body_acc_stat_std_x", f"body_acc_stat_std_y", f"body_acc_stat_std_z"] +

        [f"body_acc_corr_xy", f"body_acc_corr_xz", f"body_acc_corr_yz"] +

        [f"body_gyro_time_mean", f"body_gyro_time_std", f"body_gyro_time_third_moment", f"body_gyro_time_fourth_moment",
         f"body_gyro_time_p25", f"body_gyro_time_p50", f"body_gyro_time_p75", f"body_gyro_time_value_entropy", 
         f"body_gyro_time_time_entropy"] +

        [f"body_gyro_freq_band_{i+1}" for i in range(5)] + 
        [f"body_gyro_freq_spectral_entropy"] +

        [f"body_gyro_autocorr_period", f"body_gyro_autocorr_max"] +

        [f"body_gyro_stat_mean_x", f"body_gyro_stat_mean_y", f"body_gyro_stat_mean_z",
         f"body_gyro_stat_std_x", f"body_gyro_stat_std_y", f"body_gyro_stat_std_z"] +

        [f"body_gyro_corr_xy", f"body_gyro_corr_xz", f"body_gyro_corr_yz"] +

        [f"total_acc_time_mean", f"total_acc_time_std", f"total_acc_time_third_moment", f"total_acc_time_fourth_moment",
         f"total_acc_time_p25", f"total_acc_time_p50", f"total_acc_time_p75", f"total_acc_time_value_entropy", 
         f"total_acc_time_time_entropy"] +

        [f"total_acc_freq_band_{i+1}" for i in range(5)] + 
        [f"total_acc_freq_spectral_entropy"] +

        [f"total_acc_autocorr_period", f"total_acc_autocorr_max"] +

        [f"total_acc_stat_mean_x", f"total_acc_stat_mean_y", f"total_acc_stat_mean_z",
         f"total_acc_stat_std_x", f"total_acc_stat_std_y", f"total_acc_stat_std_z"] +

        [f"total_acc_corr_xy", f"total_acc_corr_xz", f"total_acc_corr_yz"] +

        ["label"]
    )
    import pdb; pdb.set_trace()
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
def extract_xyz(body_acc_files, body_gyro_files, total_acc_files, label_file, output_csv):

    body_acc_x = pd.read_csv(body_acc_files[0], header=None)
    body_acc_y = pd.read_csv(body_acc_files[1], header=None)
    body_acc_z = pd.read_csv(body_acc_files[2], header=None)

    body_gyro_x = pd.read_csv(body_gyro_files[0], header=None)
    body_gyro_y = pd.read_csv(body_gyro_files[1], header=None)
    body_gyro_z = pd.read_csv(body_gyro_files[2], header=None)

    total_acc_x = pd.read_csv(total_acc_files[0], header=None)
    total_acc_y = pd.read_csv(total_acc_files[1], header=None)
    total_acc_z = pd.read_csv(total_acc_files[2], header=None)

    labels = pd.read_csv(label_file, header=None)

    combined_features = []
    for i in range(len(body_acc_x)):
        body_acc = [[round(float(body_acc_x.iloc[i, j]), 4), round(float(body_acc_y.iloc[i, j]), 4), round(float(body_acc_z.iloc[i, j]), 4)] for j in range(body_acc_x.shape[1])]
        body_gyro = [[round(float(body_gyro_x.iloc[i, j]), 4), round(float(body_gyro_y.iloc[i, j]), 4), round(float(body_gyro_z.iloc[i, j]), 4)] for j in range(body_gyro_x.shape[1])]
        total_acc = [[round(float(total_acc_x.iloc[i, j]), 4), round(float(total_acc_y.iloc[i, j]), 4), round(float(total_acc_z.iloc[i, j]), 4)] for j in range(total_acc_x.shape[1])]
        combined_features.append([body_acc, body_gyro, total_acc, labels.iloc[i, 0]])

    combined_df = pd.DataFrame(combined_features, columns=['body_acc', 'body_gyro', 'total_acc', 'label'])

    combined_df.to_csv(output_csv, index=False, header=False)


extract_all_features(
        body_acc_files=[r"UCI HAR Dataset\UCI HAR Dataset\train\bodyx_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyy_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyz_train.csv"],
        body_gyro_files=[r"UCI HAR Dataset\UCI HAR Dataset\train\bodyx_gyro_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyy_gyro_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyz_gyro_train.csv"],
        total_acc_files=[r"UCI HAR Dataset\UCI HAR Dataset\train\totalx_acc_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\totaly_acc_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\totalz_acc_train.csv"],
        label_file=r"UCI HAR Dataset\UCI HAR Dataset\train\y_train.csv",
        output_csv=r"UCI HAR Dataset\UCI HAR Dataset\train\extract_all_train.csv"
    )
