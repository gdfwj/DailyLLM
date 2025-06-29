import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def read_and_print_json_file(file_path, index):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if index < 0 or index >= len(data):
                raise IndexError("Index out of range.")
            return json.dumps(data[index], indent=4, ensure_ascii=False)
            #print(json.dumps(data[index], indent=4, ensure_ascii=False))
    except IndexError as ie:
        print(f'Error: {ie}')
    except Exception as e:
        print(f'Error reading the JSON file: {e}')

# 读取JSON文件并根据label分类，统计每类数据的数量
def count_json_entries(file_path):
    try:
        # 打开JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Initializes a default dictionary to count the amount of each type of data
            label_counts = defaultdict(int)
            
            # Iterate through the data and count the number of each label
            for entry in data:
                label = entry.get('label', 'Unknown')
                label_counts[label] += 1
            
            for label, count in label_counts.items():
                print(f"Label '{label}': {count} entries.")
    except Exception as e:
        print(f'Error reading the JSON file: {e}')

def plot_accelerometer_data(file_path):
    
    # Open the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        acc_data = json.load(file)

    # 如果 acc_data 是一个列表，获取第一个元素
    if isinstance(acc_data, list):
        acc_data = acc_data[0]

    # 第二步：提取 X、Y、Z 三轴数据
    x_data = acc_data["X"]
    y_data = acc_data["Y"]
    z_data = acc_data["Z"]

    # 第三步：创建时间轴，假设每个数据点之间间隔固定
    data_frequency = acc_data["data_frequency"]
    print(data_frequency)
    time_axis = [i / data_frequency for i in range(len(x_data))]  # 根据频率计算时间轴

    # 第四步：绘制加速度计三轴数据
    plt.figure(figsize=(10, 6))

    # 绘制 X 轴数据
    plt.plot(time_axis, x_data, label='X-Axis', linestyle='-', marker='', color='r')

    # 绘制 Y 轴数据
    plt.plot(time_axis, y_data, label='Y-Axis', linestyle='-', marker='', color='g')

    # 绘制 Z 轴数据
    plt.plot(time_axis, z_data, label='Z-Axis', linestyle='-', marker='', color='b')

    # 设置图表标题和标签
    plt.title("Accelerometer Data (X, Y, Z Axes)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()  # 显示图例

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()

# 传入文件路径，调用函数
json_file_path = '/Users/xiaomo/Desktop/SensorLLM/ExtraSensoryDataset/dataset/test_dataset/test_output_labels/JSON/user_accelerometer_data.json'
count_json_entries(json_file_path)

# 读取和打印第一个数据
data1=read_and_print_json_file(json_file_path,5)
#print(data1)

data_dict = json.loads(data1)

x_data_1 = np.around(data_dict["X"], decimals=2)
y_data_1 = np.around(data_dict["Y"], decimals=2)
z_data_1 = np.around(data_dict["Z"], decimals=2)
label_data_1 = data_dict["label"]
print(x_data_1)
print(y_data_1)
print(z_data_1)
print(label_data_1)
data_frequency = data_dict["data_frequency"]
#print(data_frequency) 


# 计算当前活动的持续时间
# 持续时间（秒） = 数据点数 / 数据频率
num_data_points = len(x_data_1)
activity_duration_seconds = num_data_points / data_frequency
#print(activity_duration_seconds)

# 绘制第一个数据
#plot_accelerometer_data(json_file_path)