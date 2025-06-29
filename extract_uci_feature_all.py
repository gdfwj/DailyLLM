import numpy as np
import pandas as pd
from scipy.stats import entropy, moment
from scipy.fftpack import fft
from scipy.signal import correlate
from scipy.signal import butter, filtfilt
from scipy.stats import entropy
def calculate_magnitude(x, y, z):
    """计算向量模值"""
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_time_domain_features(data):
    """计算时域特征"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    third_moment = moment(data, moment=3)
    fourth_moment = moment(data, moment=4)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)

    # Value-entropy: 基于直方图
    value_entropy = entropy(np.histogram(data, bins=20, density=True)[0])

    # Time-entropy: 将模值归一化后计算概率分布熵
    normalized_data = data / np.sum(data)  # 归一化
    normalized_data = normalized_data[normalized_data > 0]  # 过滤零值以避免 log 问题
    time_entropy = entropy(normalized_data)

    return [
        mean_val, std_val, third_moment, fourth_moment,
        p25, p50, p75, value_entropy, time_entropy
    ]

def calculate_frequency_domain_features(data, sampling_rate=50):
    """计算频域特征"""
    fs=50
    # 定义频段范围
    bands = [(0, 0.5), (0.5, 1), (1, 3), (3, 5), (5, fs//2)]
    
    # 计算带通滤波器
    def bandpass_filter(data, low_cutoff, high_cutoff):
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        if low == 0:
            b, a = butter(4, high, btype='low')
        elif high == 1:
            b, a = butter(4, low, btype='high')
        else:
            #import pdb; pdb.set_trace()
            b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    # 计算对数能量
    def log_energy(data):
        return np.log(np.sum(data**2))
    
    # 计算谱熵
    def spectral_entropy(data):
        spectrum = np.abs(np.fft.fft(data))[:len(data)//2]
        spectrum = spectrum / np.sum(spectrum)  # 归一化
        return entropy(spectrum)
    
    # 计算每个频段的对数能量
    log_energies = [log_energy(bandpass_filter(data, band[0], band[1])) for band in bands]
    
    # 计算整体信号的谱熵
    spec_entropy = spectral_entropy(data)
    
    # 返回所有特征
    return log_energies + [spec_entropy]

def calculate_autocorrelation_features(data):
    """计算自相关特性"""
    """计算自相关特性（最大自相关值和主导周期）"""
    '''
    fs=50
    # 标准化数据（去除直流分量并进行归一化）
    norm_data = (data - np.mean(data)) / np.std(data)
    
    # 计算自相关函数
    autocorr = correlate(norm_data, norm_data, mode='full')
    
    # 取正滞后部分
    mid_idx = len(autocorr) // 2
    autocorr = autocorr[mid_idx:]  # 只取滞后为正的部分

    # 计算最大自相关值（滞后0之后的最大值）
    max_acf_value = np.max(autocorr[1:])  # 最大值在滞后0之后
    
    # 找到主导滞后位置（最大自相关值所在的位置）
    dominant_lag = np.argmax(autocorr[1:]) + 1  # 滞后0之后最大自相关值的滞后位置
    
    # 计算主导周期（以秒为单位）
    dominant_period = dominant_lag / fs
    
    return [max_acf_value, dominant_period]
    '''
    mean_value = np.mean(data)
    data_centered = data - mean_value
    fs = 50

# 计算自相关函数并归一化
    autocorr = np.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # 只取非负滞后的部分
    autocorr_normalized = autocorr / autocorr[0]  # 归一化

# 寻找主瓣结束位置（第一个局部最小值）
    minima = np.where((autocorr_normalized[1:-1] < autocorr_normalized[:-2]) & 
                 (autocorr_normalized[1:-1] < autocorr_normalized[2:]))[0] + 1

    if len(minima) > 0:
        main_lobe_end = minima[0]  # 主瓣结束位置
    
    # 在主瓣之后寻找最大值
        post_main_autocorr = autocorr_normalized[main_lobe_end+1:]
    
        if len(post_main_autocorr) > 0:
            max_lag = np.argmax(post_main_autocorr)  # 区域内的相对位置
            dominant_lag = main_lobe_end + 1 + max_lag  # 绝对滞后位置
        
            dominant_periodicity = dominant_lag / fs  # 转换周期
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
    """计算三轴之间的相关性"""
    corr_xy = np.corrcoef(data_x, data_y)[0, 1]
    corr_xz = np.corrcoef(data_x, data_z)[0, 1]
    corr_yz = np.corrcoef(data_y, data_z)[0, 1]
    return [corr_xy, corr_xz, corr_yz]

def extract_features_group(sensor_data, sensor_name):
    """
    针对单一传感器（如 body_acc）提取特征。
    返回：指定传感器的 26 维特征。
    """
    x, y, z = sensor_data
    magnitude = calculate_magnitude(x, y, z)

    # 计算模值特征
    time_features = calculate_time_domain_features(magnitude)
    freq_features = calculate_frequency_domain_features(magnitude)
    autocorr_features = calculate_autocorrelation_features(magnitude)

    # 计算三轴相关特性
    corr_features = calculate_correlation_features(x, y, z)

    # 计算每个轴的均值和标准差
    axis_stats = [
        np.mean(x), np.mean(y), np.mean(z),
        np.std(x), np.std(y), np.std(z)
    ]
    #import pdb; pdb.set_trace()
    # 合并特征
    sensor_features = time_features + freq_features + autocorr_features + axis_stats + corr_features
    return sensor_features

def extract_all_features(body_acc_files, body_gyro_files, total_acc_files, label_file, output_csv):
    """
    提取所有传感器特征（body_acc, body_gyro, total_acc），并保存到 CSV 文件。
    """
    # 分别读取 body_acc, body_gyro, total_acc 数据
    body_acc_data = [pd.read_csv(file, header=None).values for file in body_acc_files]
    body_gyro_data = [pd.read_csv(file, header=None).values for file in body_gyro_files]
    total_acc_data = [pd.read_csv(file, header=None).values for file in total_acc_files]

    #import pdb; pdb.set_trace()
    # 检查行数是否一致
    num_samples = len(body_acc_data[0])
    assert all(len(data) == num_samples for data in body_acc_data + body_gyro_data + total_acc_data), "数据行数不一致"

    # 读取标签
    labels = pd.read_csv(label_file, header=None).values.flatten()
    assert len(labels) == num_samples, "标签行数与数据行数不一致"

    # 提取特征
    all_features = []
    from tqdm import tqdm
    for i in tqdm(range(num_samples)):
        # 提取每组传感器数据
        body_acc_features = extract_features_group([data[i] for data in body_acc_data], "body_acc")
        body_gyro_features = extract_features_group([data[i] for data in body_gyro_data], "body_gyro")
        total_acc_features = extract_features_group([data[i] for data in total_acc_data], "total_acc")

        # 合并三组传感器特征并添加标签
        feature_row = body_acc_features + body_gyro_features + total_acc_features + [labels[i]]
        all_features.append(feature_row)
        #import pdb; pdb.set_trace()
    # 保存到 CSV 文件
    columns = (
        # 时域特征
        [f"body_acc_time_mean", f"body_acc_time_std", f"body_acc_time_third_moment", f"body_acc_time_fourth_moment",
         f"body_acc_time_p25", f"body_acc_time_p50", f"body_acc_time_p75", f"body_acc_time_value_entropy", 
         f"body_acc_time_time_entropy"] +

        # 频域特征
        [f"body_acc_freq_band_{i+1}" for i in range(5)] + 
        [f"body_acc_freq_spectral_entropy"] +

        # 自相关特征
        [f"body_acc_autocorr_period", f"body_acc_autocorr_max"] +

        # 轴向统计特征
        [f"body_acc_stat_mean_x", f"body_acc_stat_mean_y", f"body_acc_stat_mean_z",
         f"body_acc_stat_std_x", f"body_acc_stat_std_y", f"body_acc_stat_std_z"] +

        # 三轴相关性特征
        [f"body_acc_corr_xy", f"body_acc_corr_xz", f"body_acc_corr_yz"] +

        # body_gyro 特征
        [f"body_gyro_time_mean", f"body_gyro_time_std", f"body_gyro_time_third_moment", f"body_gyro_time_fourth_moment",
         f"body_gyro_time_p25", f"body_gyro_time_p50", f"body_gyro_time_p75", f"body_gyro_time_value_entropy", 
         f"body_gyro_time_time_entropy"] +

        [f"body_gyro_freq_band_{i+1}" for i in range(5)] + 
        [f"body_gyro_freq_spectral_entropy"] +

        [f"body_gyro_autocorr_period", f"body_gyro_autocorr_max"] +

        [f"body_gyro_stat_mean_x", f"body_gyro_stat_mean_y", f"body_gyro_stat_mean_z",
         f"body_gyro_stat_std_x", f"body_gyro_stat_std_y", f"body_gyro_stat_std_z"] +

        [f"body_gyro_corr_xy", f"body_gyro_corr_xz", f"body_gyro_corr_yz"] +

        # total_acc 特征
        [f"total_acc_time_mean", f"total_acc_time_std", f"total_acc_time_third_moment", f"total_acc_time_fourth_moment",
         f"total_acc_time_p25", f"total_acc_time_p50", f"total_acc_time_p75", f"total_acc_time_value_entropy", 
         f"total_acc_time_time_entropy"] +

        [f"total_acc_freq_band_{i+1}" for i in range(5)] + 
        [f"total_acc_freq_spectral_entropy"] +

        [f"total_acc_autocorr_period", f"total_acc_autocorr_max"] +

        [f"total_acc_stat_mean_x", f"total_acc_stat_mean_y", f"total_acc_stat_mean_z",
         f"total_acc_stat_std_x", f"total_acc_stat_std_y", f"total_acc_stat_std_z"] +

        [f"total_acc_corr_xy", f"total_acc_corr_xz", f"total_acc_corr_yz"] +

        # 标签
        ["label"]
    )
    import pdb; pdb.set_trace()
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_csv, index=False)
def extract_xyz(body_acc_files, body_gyro_files, total_acc_files, label_file, output_csv):
    # 读取 body_acc 文件
    body_acc_x = pd.read_csv(body_acc_files[0], header=None)
    body_acc_y = pd.read_csv(body_acc_files[1], header=None)
    body_acc_z = pd.read_csv(body_acc_files[2], header=None)

    # 读取 body_gyro 文件
    body_gyro_x = pd.read_csv(body_gyro_files[0], header=None)
    body_gyro_y = pd.read_csv(body_gyro_files[1], header=None)
    body_gyro_z = pd.read_csv(body_gyro_files[2], header=None)

    # 读取 total_acc 文件
    total_acc_x = pd.read_csv(total_acc_files[0], header=None)
    total_acc_y = pd.read_csv(total_acc_files[1], header=None)
    total_acc_z = pd.read_csv(total_acc_files[2], header=None)

    # 读取标签文件
    labels = pd.read_csv(label_file, header=None)

    # 组合特征
    combined_features = []
    for i in range(len(body_acc_x)):
        body_acc = [[round(float(body_acc_x.iloc[i, j]), 4), round(float(body_acc_y.iloc[i, j]), 4), round(float(body_acc_z.iloc[i, j]), 4)] for j in range(body_acc_x.shape[1])]
        body_gyro = [[round(float(body_gyro_x.iloc[i, j]), 4), round(float(body_gyro_y.iloc[i, j]), 4), round(float(body_gyro_z.iloc[i, j]), 4)] for j in range(body_gyro_x.shape[1])]
        total_acc = [[round(float(total_acc_x.iloc[i, j]), 4), round(float(total_acc_y.iloc[i, j]), 4), round(float(total_acc_z.iloc[i, j]), 4)] for j in range(total_acc_x.shape[1])]
        combined_features.append([body_acc, body_gyro, total_acc, labels.iloc[i, 0]])
        #import pdb; pdb.set_trace()

    # 将特征和标签组合成 DataFrame
    combined_df = pd.DataFrame(combined_features, columns=['body_acc', 'body_gyro', 'total_acc', 'label'])

    # 保存到 CSV 文件
    combined_df.to_csv(output_csv, index=False, header=False)


extract_all_features(
        body_acc_files=[r"UCI HAR Dataset\UCI HAR Dataset\train\bodyx_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyy_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyz_train.csv"],
        body_gyro_files=[r"UCI HAR Dataset\UCI HAR Dataset\train\bodyx_gyro_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyy_gyro_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\bodyz_gyro_train.csv"],
        total_acc_files=[r"UCI HAR Dataset\UCI HAR Dataset\train\totalx_acc_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\totaly_acc_train.csv", r"UCI HAR Dataset\UCI HAR Dataset\train\totalz_acc_train.csv"],
        label_file=r"UCI HAR Dataset\UCI HAR Dataset\train\y_train.csv",
        output_csv=r"UCI HAR Dataset\UCI HAR Dataset\train\extract_all_train.csv"
    )
