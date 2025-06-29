import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate
import json



def read_accelerometer_data(filepath):
    """
    读取加速度计的dat文件，假设每一行包含时间戳、X、Y、Z加速度。
    """
    data = np.loadtxt(filepath)
    time = data[:, 0]
    x_acc = np.around(data[:, 1], decimals=2)
    y_acc = np.around(data[:, 2], decimals=2)
    z_acc = np.around(data[:, 3], decimals=2)
    return time, x_acc, y_acc, z_acc

def downsample_data(time, x_acc, y_acc, z_acc, target_hz, original_hz):
    """
    下采样数据到目标频率，假设原始频率为 original_hz。
    """
    factor = original_hz // target_hz
    time_downsampled = time[::factor]  # Use slicing to downsample the time array
    x_acc_downsampled = decimate(x_acc, factor, zero_phase=True)
    y_acc_downsampled = decimate(y_acc, factor, zero_phase=True)
    z_acc_downsampled = decimate(z_acc, factor, zero_phase=True)

    x_acc_new = np.around(x_acc_downsampled, decimals=2)
    y_acc_new = np.around(y_acc_downsampled, decimals=2)
    z_acc_new = np.around(z_acc_downsampled, decimals=2)
    #return time_downsampled, x_acc_downsampled, y_acc_downsampled, z_acc_downsampled
    return time_downsampled, x_acc_new, y_acc_new, z_acc_new

def visualize_accelerometer_data_subplots(time, x_acc, y_acc, z_acc, filename):
    """
    可视化加速计数据，将X、Y、Z加速度画在不同的子图中。
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(time, x_acc, color='r')
    axs[0].set_ylabel('X Acceleration (m/s^2)')
    axs[0].set_title(f'Accelerometer Data from {filename} - X Axis')
    axs[0].grid(True)
    
    axs[1].plot(time, y_acc, color='g')
    axs[1].set_ylabel('Y Acceleration (m/s^2)')
    axs[1].set_title(f'Accelerometer Data from {filename} - Y Axis')
    axs[1].grid(True)
    
    axs[2].plot(time, z_acc, color='b')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z Acceleration (m/s^2)')
    axs[2].set_title(f'Accelerometer Data from {filename} - Z Axis')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def read_write_raw_acc(UUID= '0BFC35E2-4817-4865-BFA7-764742302A2D', folder_path="../../SensorLLM_dataset/ExtraSensor/Rawsensor/raw_acc"):
    # UUID = '0BFC35E2-4817-4865-BFA7-764742302A2D'
    # data_UUID='1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842'

    # unseen
    # data_UUID='9DC38D04-E82E-4F29-AB52-B476535226F2'

    target_hz = 5
    original_hz = 40
    # Set folder path for traversing files
    folder_path = os.path.join(folder_path, UUID)
    # folder_path = "/Users/xiaomo/Desktop/SensorLLM/ExtraSensoryDataset/dataset/Rawsensor/raw_acc/0BFC35E2-4817-4865-BFA7-764742302A2D/"
    # folder_path = "/Users/xiaomo/Desktop/SensorLLM/ExtraSensoryDataset/dataset/Rawsensor/raw_acc/1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842/"

    # unseen
    # folder_path = "/Users/xiaomo/Desktop/SensorLLM/ExtraSensoryDataset/dataset/Rawsensor/raw_acc/9DC38D04-E82E-4F29-AB52-B476535226F2/"

    csv_file_path = "../../SensorLLM_dataset/ExtraSensor/test_dataset/test_output_labels/Sample_UUID_timestamp_and_label.csv"
    JOSN_output_path = "../../SensorLLM_dataset/ExtraSensor/test_dataset/test_output_labels/JSON"

    csv_data = pd.read_csv(csv_file_path)
    filtered_data = csv_data[csv_data['UUID'] == UUID]
    # print(filtered_data)
    # 获取 timestamp 列的前 10 位，并将其转换为字符串列表
    filtered_timestamps = filtered_data['timestamp'].astype(str).str[:10].tolist()
    #print(filtered_timestamps)
    #print("Here is printing filtered_timestamps")

    json_count = 0
    all_acc_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".dat"):
            file_prefix = filename[:10]  # 获取文件名前 10 位
            #print(file_prefix)
            # 检查文件名前10位是否在filtered_timestamps中
            if file_prefix in filtered_timestamps:
                filepath = os.path.join(folder_path, filename)
                print(f"Processing file: {filename}")
                try:
                    time, x_acc, y_acc, z_acc = read_accelerometer_data(filepath)
                    time_downsampled, x_acc_downsampled, y_acc_downsampled, z_acc_downsampled = downsample_data(time, x_acc, y_acc, z_acc,target_hz, original_hz)
                    #print(type(time_downsampled))
                    #print(type(x_acc_downsampled))

                    data_label = filtered_data[filtered_data['timestamp'].astype(str).str[:10] == file_prefix]['label'].values[0]

                    acc_data_json = {
                        "UUID": UUID,
                        "timestamps": file_prefix,
                        "type": "ACC",
                        "data_frequency": target_hz,
                        "X": x_acc_downsampled.tolist(),  # numpy.ndarray 转为 list 以便存为 JSON
                        "Y": y_acc_downsampled.tolist(),
                        "Z": z_acc_downsampled.tolist(),
                        "label": data_label
                    }
                    # 打印JSON格式变量
                    acc_data_json_str = json.dumps(acc_data_json, indent=4)
                    print(acc_data_json_str)
                    json_count=json_count+1
                    all_acc_data.append(acc_data_json)

                    # Visualize the downsampled data in a single plot
                    #visualize_accelerometer_data_subplots(time_downsampled, x_acc_downsampled, y_acc_downsampled, z_acc_downsampled, filename)

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")


    print(json_count)
    # Check if output JSON file exists and load existing data if necessary
    json_filename = "user_accelerometer_data.json"
    output_json_file = os.path.join(JOSN_output_path, json_filename)


    # Saving the final JSON output
    if os.path.exists(output_json_file):
        # If file exists, append new data
        with open(output_json_file, 'r+') as f:
            try:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_data.extend(all_acc_data)
                else:
                    existing_data = all_acc_data
                f.seek(0)
                json.dump(existing_data, f, indent=4)
            except json.JSONDecodeError:
                # If file is empty or corrupted, overwrite it with new data
                f.seek(0)
                json.dump(all_acc_data, f, indent=4)
    else:
        # If file does not exist, create it and write data
        with open(output_json_file, 'w') as f:
            json.dump(all_acc_data, f, indent=4)


if __name__ == '__main__':
    read_write_raw_acc()