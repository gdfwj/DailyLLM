import glob
import os
from collections import defaultdict
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple

import torch
import pandas as pd
import random

def analyze_spectrum(data: Dict[str, np.ndarray], sampling_rate: float,
                     window: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    对XYZ三轴数据进行频谱分析

    参数:
    data: 包含'x', 'y', 'z'键的字典，值为numpy数组
    sampling_rate: 采样率(Hz)
    window: 是否使用汉宁窗

    返回:
    频谱数据字典和频率数组
    """
    spectrum_data = {}

    # 获取数据长度
    n = len(data['X'])

    # 创建频率数组
    frequencies = fftfreq(n, d=1 / sampling_rate)

    # 只取正频率部分
    positive_freq_mask = frequencies >= 0
    frequencies = frequencies[positive_freq_mask]

    # 创建汉宁窗
    window_function = np.hanning(n) if window else np.ones(n)

    # 对每个轴进行FFT分析
    for axis in ['X', 'Y', 'Z']:
        if axis not in data:
            continue

        # 应用窗函数
        windowed_data = data[axis] * window_function

        # 计算FFT
        fft_result = fft(windowed_data)

        # 计算幅值谱（取正频率部分）
        magnitude = np.abs(fft_result)[positive_freq_mask]

        # 转换为dB标度 (参考值为1)
        magnitude_db = 20 * np.log10(magnitude / n)

        spectrum_data[axis] = magnitude_db

    return spectrum_data, frequencies


def plot_spectrum(path, spectrum_data: Dict[str, np.ndarray],
                  frequencies: np.ndarray,
                  max_freq: float = None,
                  min_db: float = -100) -> None:
    """
    绘制频谱图

    参数:
    spectrum_data: 频谱数据字典
    frequencies: 频率数组
    max_freq: 最大显示频率
    min_db: 最小显示分贝值
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('XYZ轴频谱分析')

    # 如果未指定最大频率，则使用奈奎斯特频率
    if max_freq is None:
        max_freq = frequencies[-1]

    # 频率掩码
    freq_mask = frequencies <= max_freq

    colors = {'x': 'b', 'y': 'g', 'z': 'r'}
    titles = {'x': 'X轴频谱', 'y': 'Y轴频谱', 'z': 'Z轴频谱'}

    for i, axis in enumerate(['x', 'y', 'z']):
        if axis not in spectrum_data:
            continue

        ax = axes[i]

        # 绘制频谱
        ax.plot(frequencies[freq_mask],
                spectrum_data[axis][freq_mask],
                color=colors[axis])

        # 设置Y轴范围
        ax.set_ylim(min_db, max(spectrum_data[axis]) + 10)

        # 添加网格
        ax.grid(True)

        # 设置标题和标签
        ax.set_title(titles[axis])
        ax.set_ylabel('幅值 (dB)')

        if i == 2:  # 最后一个子图
            ax.set_xlabel('频率 (Hz)')

    plt.tight_layout()
    plt.savefig(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path="../../SensorLLM_dataset/ExtraSensor/Rawsensor",
    label_path="../../SensorLLM_dataset/ExtraSensor/ExtraSensory.per_uuid_features_labels",
    original_hz = 40, target_hz = 5, decimals=4, base_path="../../SensorLLM_dataset/figures"):
        self.tokenized = False
        # self.raw_data_path = raw_data_path
        self.decimals = decimals
        self.label_path = label_path
        self.data = []
        self.UUID2data = defaultdict(list)
        self.prepare_out = []
        self.original_hz = original_hz
        self.target_hz = target_hz
        self.base_path = base_path
        
        # prepare geo dict
        geo = {}
        gyroscope_path = os.path.join(os.path.join(data_path, ".."), "ExtraSensory.per_uuid_absolute_location")
        for root, _, files in os.walk(gyroscope_path):
            for file in files:
                df = pd.read_csv(os.path.join(root, file), compression='gzip')
                UUID = file[:file.find(".absolute_locations.csv.gz")]
                temp_dict = {}
                for index, row in df.iterrows():
                    temp_dict[row['timestamp']] = f"latitude: {row['latitude']}, longitude: {row['longitude']}"
                geo[UUID] = temp_dict

        for root, _, files in os.walk(self.label_path):
            for file in files:
                UUID = file[:file.find(".features_labels")]
                # print(UUID)
                label_file = os.path.join(root, file)
                # columns_to_keep = [
                #     'timestamp',
                #     'label:SITTING',
                #     'label:FIX_walking',
                #     'label:FIX_running',
                #     'label:BICYCLING',
                #     'label:SLEEPING'
                # ]
                df = pd.read_csv(label_file, compression='gzip')
                df_selected = df
                for index, row in df_selected.iterrows():
                    if row['label:SITTING']==1:
                        label = 'SITTING'
                    elif row['label:FIX_walking']==1:
                        label = 'WALKING'
                    elif row['label:FIX_running']==1:
                        label = 'RUNNING'
                    elif row['label:BICYCLING']==1:
                        label = 'BICYCLING'
                    elif row['label:SLEEPING']==1:
                        label = 'SLEEPING'
                    else:
                        continue
                    timestamp = row['timestamp']
                    # acc_data_path = os.path.join(os.path.join(os.path.join(data_path, "raw_acc"), UUID), f"{int(timestamp)}.m_raw_acc.dat")
                    # magnet_data_path = os.path.join(os.path.join(os.path.join(data_path, "raw_magnet"), UUID), f"{int(timestamp)}.m_raw_magnet.dat")
                    # watch_acc_path = os.path.join(os.path.join(os.path.join(data_path, "watch_acc"), UUID), f"{int(timestamp)}.m_watch_acc.dat")
                    # watch_compass_path = os.path.join(os.path.join(os.path.join(data_path, "watch_compass"), UUID), f"{int(timestamp)}.m_watch_compass.dat")
                    # # gyroscope_path = os.path.join(os.path.join(os.path.join(data_path, ".."), "ExtraSensory.per_uuid_absolute_location"), f"{UUID}.absolute_locations.csv.gz")
                    # # 定义路径检查清单
                    # path_checks = [
                    #     ("加速度计数据路径", acc_data_path),
                    #     ("磁力计数据路径", magnet_data_path),
                    #     ("智能手表加速度路径", watch_acc_path),
                    #     ("智能手表罗盘路径", watch_compass_path),
                    #     ("陀螺仪路径", gyroscope_path)
                    # ]

                    # missing_paths = []

                    # # 遍历检查所有路径
                    # for path_name, path in path_checks:
                    #     if not os.path.exists(path):
                    #         missing_paths.append(f"{path_name}: {path}")

                    # # 处理缺失路径
                    # if missing_paths:
                    #     print("以下路径不存在：")
                    #     for item in missing_paths:
                    #         print(f"  - {item}")
                    # if not os.path.exists(acc_data_path):
                    #     continue
                    # data = {
                    #     "UUID": UUID,
                    #     "label": label,
                    #     "timestamp": timestamp,
                    #     "data": {
                    #         "Smartphone accelerometer": acc_data_path, 
                    #         "Smartphone gyroscope": geo[UUID][timestamp], 
                    #         "Smartphone magnetometer": magnet_data_path, 
                    #         "Smartwatch accelerometer": watch_acc_path,
                    #         "Smartwatch compass": watch_compass_path
                    #     }
                    # }
                    acc_data_path = []
                    magnet_data_path = []
                    watch_acc_path = []
                    watch_compass_path = []
                    geo_path = []
                    for key, item in row.items():
                        if "raw_acc" in key:
                            acc_data_path.append(float(np.around(item, decimals=4)))
                        elif "proc_gyro" in key:
                            geo_path.append(float(np.around(item, decimals=4)))
                        elif "raw_magnet" in key:
                            magnet_data_path.append(float(np.around(item, decimals=4)))
                        elif "watch_acceleration" in key:
                            watch_acc_path.append(float(np.around(item, decimals=4)))
                        elif "location" in key:
                            watch_compass_path.append(float(np.around(item, decimals=4)))
                    data = {
                        "UUID": UUID,
                        "label": label,
                        "timestamp": timestamp,
                        "data": {
                            "Smartphone accelerometer": acc_data_path, 
                            "Smartphone gyroscope": geo_path, 
                            "Smartphone magnetometer": magnet_data_path, 
                            "Smartwatch accelerometer": watch_acc_path,
                            "Smartwatch compass": watch_compass_path
                        }
                    }
                    self.data.append(data)
                    self.UUID2data[UUID].append(data)
            self.prepare_out = self.data

    def __len__(self):
        return len(self.prepare_out)

    def set_user(self, UUID):
        self.prepare_out = self.UUID2data[UUID]

    def prepare_test(self, number_pre_class, seed=114514):
        UUIDs = self.UUID2data.keys()
        label_list = ['SITTING', 'WALKING', 'RUNNING', 'BICYCLING', 'SLEEPING']
        prepare_test = []
        random.seed(seed)
        for UUID in UUIDs:
            count = np.zeros([5])
            indices = random.sample(range(len(self.UUID2data[UUID])), len(self.UUID2data[UUID]))
            for idx in indices:
                data = self.UUID2data[UUID][idx]
                label_number = label_list.index(data['label'])
                if count[label_number]<number_pre_class:
                    count[label_number]+=1
                    prepare_test.append(data)
                    if np.sum(count==number_pre_class)==5:
                        break
        self.prepare_out=prepare_test
        
    
    def set_tokenizer(self, tokenizer):
        self.tokenized = True
        self.tokenizer = tokenizer

        
    def __getitem__(self, idx):
        def read_accelerometer_data(filepath, decimals):
            """
            读取加速度计的dat文件，假设每一行包含时间戳、X、Y、Z加速度。
            """
            try:
                data = np.loadtxt(filepath)
            except:
                print(f"failed in {filepath}")
                raise Exception
            time = data[:, 0]
            # print(data[:, 1])
            x_acc = np.around(data[:, 1], decimals=decimals)
            y_acc = np.around(data[:, 2], decimals=decimals)
            z_acc = np.around(data[:, 3], decimals=decimals)
            return time, x_acc, y_acc, z_acc

        def downsample_data(orgin_data, target_hz, original_hz, decimals):
            """
            下采样数据到目标频率，假设原始频率为 original_hz。
            """
            time, x_acc, y_acc, z_acc = orgin_data
            factor = original_hz // target_hz
            time_downsampled = time[::factor]  # Use slicing to downsample the time array
            x_acc_downsampled = decimate(x_acc, factor, zero_phase=True)
            y_acc_downsampled = decimate(y_acc, factor, zero_phase=True)
            z_acc_downsampled = decimate(z_acc, factor, zero_phase=True)

            x_acc_new = np.around(x_acc_downsampled, decimals=decimals)
            y_acc_new = np.around(y_acc_downsampled, decimals=decimals)
            z_acc_new = np.around(z_acc_downsampled, decimals=decimals)
            # return time_downsampled, x_acc_downsampled, y_acc_downsampled, z_acc_downsampled
            return time_downsampled, x_acc_new, y_acc_new, z_acc_new
        data = self.prepare_out[idx]
        output_data = {}
        # for subject in data['data'].keys():
        #     if subject == "Smartphone gyroscope":
        #         continue
        #     raw_data = read_accelerometer_data(data['data'][subject], self.decimals)
        #     downsampled_data = downsample_data(raw_data, self.target_hz, self.original_hz, self.decimals)
            # output_data[subject] = downsampled_data
        # fig_path = os.path.join(self.base_path, str(data['UUID'])+str(data['timestamp']))[:-1]+".jpg"
        # data['X']=downsampled_data[1]
        # data['Y']=downsampled_data[2]
        # data['Z']=downsampled_data[3]
        # if not os.path.exists(fig_path):
        #     spectrum_result, freqs = analyze_spectrum(data, sampling_rate=self.target_hz)
        #     plot_spectrum(fig_path, spectrum_result, freqs, max_freq=100, min_db=-60)
        # if not self.tokenized:
        #     return {
        #         "UUID": data['UUID'],
        #         "timestamp": data["timestamp"],
        #         "type": "ACC",
        #         "data_frequency": 40,
        #         'data': output_data, 
        #         # 'fig_paths': fig_path,
        #         "label": data["label"],
        #     }
        # else:
        def get_prompt(data, label):
            string_data = f"""You are an expert in signal analysis. We read the sensor data of the accelerometer, gyroscope, magnetometer on the users\u2019 mobile phone and the accelerometer and compass on their smart watch. We then extract some features of these sensing data from the time and frequency domains as follows:

            "Sensors on smartphone:"

            Accelerometer, Gyroscope and magnetometer: We first calculate the vector magnitude signal as the euclidean norm of the 3-axis measurement at each point in time. We then extract (1) 9 statistics of the magnitude signal. (2) six spectral features (log energies in 5 sub-bands and spectral entropy)of the magnitude signal. (3) Two autocorrelation features from the magnitude signal: a). Dominant periodicity: The average of the magnitude signal (DC component) was subtracted and the autocorrelation function was computed and normalized such that the autocorrelation value at lag 0 will be 1. b). Normalized autocorrelation value: The highest value after the main lobe was located. The corresponding period (in seconds) was calculated as the dominant periodicity and its normalized autocorrelation value was also extracted. (4) Nine statistics of the 3-axis time series: the mean and standard deviation of each axis and the 3 inter-axis correlation coefficients.

            In short, we extract 26-dimensional features, with specific meanings as follows:[mean, standard deviation, third moment, fourth moment, 25th percentile, 50th percentile, 75th percentile, value-entropy, time-entropy (to detect peakiness in time-sudden bursts of magnitude), log energies in 5 sub-bands (0-0.5Hz, 0.5-1Hz, 1-3Hz, 3-5Hz, >5Hz), spectral entropy, dominant periodicity, normalized autocorrelation value, Mean value of X-axis, Mean value of Y-axis, Mean value of Z-axis, Standard deviation of X-axis, Standard deviation of Y-axis, Standard deviation of Z-axis, Correlation between X and Y, Correlation between X and Z, Correlation between Y and Z].

            Sensors on smartwatch:

            Accelerometer: Since the position of the watch is easier to control than that of the phone, the data on each axis of it has a strong meaning. So, based on extracting the same 26 features as the phone sensor, we added 15 features specific to each axis in the same 5 sub-bands. In addition, to evaluate the change in the direction of the watch during the process, we calculate the cosine similarity between the acceleration directions of any two time points in the time series (a value of 1 for the same direction, a value of -1 for the opposite direction, and a value of 0 for the orthogonal direction), and then we calculate the cosine similarity between the five time ranges (0-0.5s, 0.5-1s, 1-5s, 5-10s, after 10s) average these cosine similarities. In short, we extract 46-dimensional features, with specific meanings as follows: The same 26-dimensional features as a smartphone + [log energy in 5 sub-bands for X] + [log energy in 5 sub-bands for Y] + [log energy in 5 sub-bands for Z]+[cosine similarities in 0-0.5s]+[cosine similarities in 0.5-1s]+[cosine similarities in 1-5s]+ [cosine similarities in 5-10s]+[cosine similarities after 10s] (notes: a value of 1 for the same direction, a value of -1 for the opposite direction, and a value of 0 for the orthogonal direction)

            Compass: we extract 9-dimensional features of watch heading, with specific meanings as follows: [Mean of the cosine values, standard deviation of the cosine values, the cosine values of third moment, the cosine values of fourth moment, Mean of the sine values, standard deviation of the sine values, the sine values of third moment, the sine values of fourth moment, Heading Direction Entropy (8-Bin Quantized)].

            The user's motions belong to one of these categories: [lying down, walking, sitting, bicycling]. Please analyze these features and combine your professional knowledge to determine what the user's motions are. Please think step by step and give your answer.

            Here are some features we extracted from sensors on Smartphone and Smartwatch:

            Smartphone accelerometer:[{data["Smartphone accelerometer"]}],
            Smartphone gyroscope:[{data["Smartphone gyroscope"]}],
            Smartphone magnetometer:[{data["Smartphone magnetometer"]}],
            Smartwatch accelerometer:[{data["Smartwatch accelerometer"]}],
            Smartwatch compass:[{data["Smartwatch compass"]}].
            Please analyze these features and combine your professional knowledge to determine what the user's motions are.
            """
            return {
                    "instruction": f"{string_data}",
                    "input": "",
                    "output": f"{label}"
                }
        
        return get_prompt(data['data'], data["label"])

        
        
def load_data(decimals=4, target_hz=40):
    dataset = Dataset("../SensorLLM_dataset/ExtraSensor/Rawsensor",
            "../SensorLLM_dataset/ExtraSensor/ExtraSensory.per_uuid_features_labels", target_hz=target_hz, decimals=decimals)
    return dataset
        

if __name__ == '__main__':
    dataset = load_data()
    dataset.prepare_test(3)
    print(len(dataset))
    # dataset = Dataset("../../SensorLLM_dataset/ExtraSensor/Rawsensor/raw_acc",
    #         "../../SensorLLM_dataset/ExtraSensor/ExtraSensory.per_uuid_features_labels", target_hz=40)
    # # print(len(dataset))
    # # print(dataset[100]['UUID'])
    # # dataset.set_user("0BFC35E2-4817-4865-BFA7-764742302A2D")
    # # for data in dataset:
    # #     if data["timestamp"]==1445366534:
    # print(len(dataset))