from pydantic import BaseModel
from openai import OpenAI
import openai
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate
import json
from scipy import stats
from scipy.fft import fft
from SensorLLM.Data_preparation.dataset import Dataset
from tqdm import tqdm
# 打印变量 test_message
# print(test_message)
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple


def analyze_spectrum(data: Dict[str, np.ndarray], window: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    对XYZ三轴数据进行频谱分析

    参数:
    data: 包含'x', 'y', 'z'键的字典，值为numpy数组
    window: 是否使用汉宁窗

    返回:
    频谱数据字典和频率索引数组
    """
    spectrum_data = {}

    # 获取数据长度
    n = len(data['X'])  # 应该是800

    # 创建频率索引数组（0到N/2的整数）
    freq_indices = np.arange(n // 2 + 1)

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

        # 计算幅值谱（只取正频率部分）
        magnitude = np.abs(fft_result[:n // 2 + 1])

        # 转换为dB标度 (参考值为1)
        magnitude_db = 20 * np.log10(magnitude / n)

        spectrum_data[axis] = magnitude_db

    return spectrum_data, freq_indices


def plot_spectrum(path, spectrum_data: Dict[str, np.ndarray],
                  freq_indices: np.ndarray,
                  max_index: int = None,
                  min_db: float = -100) -> None:
    """
    绘制频谱图

    参数:
    spectrum_data: 频谱数据字典
    freq_indices: 频率索引数组
    max_index: 最大显示频率索引
    min_db: 最小显示分贝值
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('XYZ analyze')

    # 如果未指定最大频率索引，则使用全部范围
    if max_index is None:
        max_index = len(freq_indices) - 1

    # 频率索引掩码
    index_mask = freq_indices <= max_index

    colors = {'X': 'b', 'Y': 'g', 'Z': 'r'}
    titles = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}

    for i, axis in enumerate(['X', 'Y', 'Z']):
        # if axis not in spectrum_data:
        #     continue

        ax = axes[i]

        # 绘制频谱
        ax.plot(freq_indices[index_mask],
                spectrum_data[axis][index_mask],
                color=colors[axis])

        # 设置Y轴范围
        # print(f"min {min_db}, max {max(spectrum_data[axis]) + 10}")
        # min_db = min(spectrum_data[axis]) - 10
        ax.set_ylim(min_db, max(spectrum_data[axis]) + 10)

        # 添加网格
        ax.grid(True)

        # 设置标题和标签
        ax.set_title(titles[axis])
        ax.set_ylabel('dB')

        if i == 2:  # 最后一个子图
            ax.set_xlabel('index')

    plt.tight_layout()
    plt.savefig(path)
    plt.close("all")
    # plt.show()


json_count = 0
GPTAPI_prediction = []

# 打开 JSON 文件并读取内容

already_json = 'user_accelerometer_data.json'
with open(already_json, 'r') as f:
    already_json = json.load(f)
# # print(already_json)
#
print(len(already_json))
identifier_list = []
for data in already_json:
    identifier_list.append((data['UUID'], int(data['timestamps'])))

l = len(identifier_list)
dataset = Dataset(decimals=10, target_hz=40)
test_message = []
#
for UUID, timestamp in tqdm(identifier_list):
    dataset.set_user(UUID)
    flag=False
    for data in dataset:
        if data['timestamp']==timestamp:
            data['X'] = data['X'].tolist()
            data['Y'] = data['Y'].tolist()
            data['Z'] = data['Z'].tolist()
            test_message.append(data)
            flag=True
            break
    if flag:
        print("get", UUID, timestamp)
    else:
        print("miss", UUID, timestamp)
#
message_json_path = "message.json"
with open(message_json_path, 'w') as f:
    json.dump(test_message, f, indent=10)

with open("message.json", "r") as f:
    test_message = json.load(f)

for data in test_message:
    path = os.path.join("figure", data['UUID']+str(int(data['timestamp']))+".jpg")
    spectrum_result, freq_indices = analyze_spectrum(data)
    plot_spectrum(path, spectrum_result, freq_indices, max_index=200, min_db=-100)
