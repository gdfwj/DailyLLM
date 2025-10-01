
import os
import pandas as pd
from glob import glob
def update_flag(df, file_path, flag_column, start_col, end_col):
    """ 根据时间范围更新 flag_column 值 """
    df_flag = pd.read_csv(file_path, usecols=[start_col, end_col], dtype=str)
    
    # 遍历所有时间戳，检查是否落在 start 和 end 之间
    df[flag_column] = df["timestamp"].apply(lambda ts: any(
        (ts >= row[start_col]) and (ts <= row[end_col]) for _, row in df_flag.iterrows()
    )).astype(int)  # True → 1, False → 0
    return df
# 数据集文件夹路径
dataset_dir = "dataset_location/dataset/sensing"

# A 文件所在目录（gps）
A_dir = os.path.join(dataset_dir, "gps")
# B 文件所在目录（wifi）
B_dir = os.path.join(dataset_dir, "wifi")
# C 文件所在目录（wifi_location）
C_dir = os.path.join(dataset_dir, "wifi_location")
# D 文件所在目录（audio）
D_dir = os.path.join(dataset_dir, "audio")
# location 文件所在目录（bluetooth）
location_dir = os.path.join(dataset_dir, "bluetooth")
# E 文件所在目录（conversation）
E_dir = os.path.join(dataset_dir, "conversation")
# F 文件所在目录（dark）
F_dir = os.path.join(dataset_dir, "dark")
# G 文件所在目录（phonecharge）
G_dir = os.path.join(dataset_dir, "phonecharge")
# H 文件所在目录（phonelock）
H_dir = os.path.join(dataset_dir, "phonelock")

# 获取 A 目录下的前 3 个 CSV 文件
A_files = glob(os.path.join(A_dir, "*.csv"))

# 存储所有处理后的数据
all_data = []

for A_file in A_files[:1]:
    # 提取 `uid`（从文件名中获取 `_` 后的编号）
    filename = os.path.basename(A_file)
    uid = filename.split("_")[-1].split(".")[0]  # 取 `_` 后的部分作为 UID

    # 读取 A 文件，并重命名 time 为 timestamp
    df_a = pd.read_csv(A_file, usecols=["time", "latitude", "longitude", "altitude"],dtype=str,index_col=False).rename(columns={"time": "timestamp"})
    #print(df_a)
    # 读取匹配的 B, C, D 文件
    B_file = os.path.join(B_dir, f"wifi_{uid}.csv")
    C_file = os.path.join(C_dir, f"wifi_location_{uid}.csv")
    D_file = os.path.join(D_dir, f"audio_{uid}.csv")
    location_file = os.path.join(location_dir, f"bt_{uid}.csv")

    df_b = pd.read_csv(B_file, usecols=["time", "BSSID"],dtype=str,index_col=False).rename(columns={"time": "timestamp"})
    df_b = df_b.drop_duplicates(subset=['timestamp'], keep='first')  # 只保留第一次出现的 timestamp 行
    df_c = pd.read_csv(C_file,dtype=str,index_col=False).rename(columns={"time": "timestamp","location":"wifi_location"})
    df_d = pd.read_csv(D_file, usecols=["timestamp", " audio inference"],dtype=str,index_col=False).rename(columns={" audio inference": "environmental_noise"})
    df_location = pd.read_csv(location_file, usecols=["time", "MAC", "class_id"],dtype=str,index_col=False).rename(columns={"time": "timestamp", "MAC": "Bluetooth_MAC"})
    df_location=df_location.groupby("timestamp").agg(lambda x: ','.join(map(str, x))).reset_index()
    #print(df_location)
    # 合并 ABCD 和 location 文件
    merged_df = pd.merge(df_a, df_b, on="timestamp", how="outer")
    #print(pd.merge(df_a,df_location,on="timestamp",how="inner"))
    merged_df = pd.merge(merged_df, df_c, on="timestamp", how="outer")
    merged_df = pd.merge(merged_df, df_d, on="timestamp", how="left")
    merged_df = pd.merge(merged_df, df_location, on="timestamp", how="left")
    merged_df = merged_df.dropna(subset=["latitude", "longitude", "altitude", "wifi_location"], how="all")
    # 时间排序
    merged_df["_timestamp_sort"] = pd.to_datetime(merged_df["timestamp"],unit='s')
    merged_df = merged_df.sort_values(by="_timestamp_sort")
    filtered_rows=[]
    last_row=None
    for index, row in merged_df.iterrows():
        #import pdb;pdb.set_trace()
        if last_row is None:
            filtered_rows.append(row)
            last_row=row
        else:
            time_diff=(row["_timestamp_sort"]-last_row["_timestamp_sort"]).total_seconds()
            wifi_location_diff=(row["wifi_location"]!=last_row["wifi_location"])and (pd.notna(row["wifi_location"]))
            #if time_diff<120:
                #import pdb;pdb.set_trace()
            if time_diff>=120 or wifi_location_diff:
                filtered_rows.append(row)
                last_row=row
    #清空merged_df
    merged_df=merged_df.iloc[0:0]
    #import pdb;pdb.set_trace()
    merged_df=pd.DataFrame(filtered_rows)
    #merged_df = merged_df.drop(columns=["_timestamp_sort"])

    # 添加 uid 列
    merged_df.insert(0, "uid", uid)

    # 添加新列，初始化为 0
    merged_df["conversation"] = 0
    merged_df["dark"] = 0
    merged_df["phonecharge"] = 0
    merged_df["phonelock"] = 0

    # 遍历 E, F, G, H 文件，并更新对应列
    
    for folder, column in zip([E_dir, F_dir, G_dir, H_dir], ["conversation", "dark", "phonecharge", "phonelock"]):
        event_file = os.path.join(folder, f"{column}_{uid}.csv")
        df_event = pd.read_csv(event_file,dtype=str,index_col=False)
        if column=="conversation":
            df_event["start_timestamp"] = pd.to_datetime(df_event["start_timestamp"],unit='s')
            df_event["end_timestamp"] = pd.to_datetime(df_event[" end_timestamp"],unit='s')
        else:
            df_event["start_timestamp"] = pd.to_datetime(df_event["start"],unit='s')
            df_event["end_timestamp"] = pd.to_datetime(df_event["end"],unit='s')
        # 转换时间格式
        
        merged_df["timestamp1"] = pd.to_datetime(merged_df["timestamp"],unit='s')

        # 使用向量化操作筛选时间范围
        for _, row in df_event.iterrows():
            start, end = row["start_timestamp"], row["end_timestamp"]
            merged_df.loc[(merged_df["timestamp1"] >= start) & (merged_df["timestamp1"] <= end), column] = 1
    merged_df = merged_df.drop(columns=["timestamp1"])
    
    # 存储结果
    all_data.append(merged_df)

# 合并所有数据并保存为 CSV 文件
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("dataset_fine\extracted_raw_data.csv", index=False)
#import pdb;pdb.set_trace()
import json
# 保存为 JSONL 文件
with open("dataset_fine\\dataset_studentlife_rawdata.jsonl", "w") as jsonl_file:
    for _, row in final_df.iterrows():
        # 将每一行转换为字典并写入 JSONL
        jsonl_file.write(json.dumps(row.to_dict()) + "\n")

# 打印前几行结果
#print(final_df.head())



import googlemaps
import requests
import json
import pandas as pd
from datetime import datetime
import pytz
import tqdm

save_df = pd.read_csv("dataset_fine\\extracted_raw_data.csv",dtype=str,index_col=False)

gm = googlemaps.Client(key="your key")

message_list = []
eastern = pytz.timezone("US/Eastern")




for index, row in tqdm.tqdm(save_df.iterrows(), total=save_df.shape[0]):
    
    latitude = row.get("latitude", None)
    longitude = row.get("longitude", None)
    timestamp = row.get("timestamp", None)
    #import pdb;pdb.set_trace()
    wifi_location = row.get("wifi_location", None)
    utc_time = datetime.fromtimestamp(int(timestamp))
    eastern_time = utc_time.astimezone(eastern)

    formatted_time = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if pd.isna(latitude) or pd.isna(longitude) or pd.isna(timestamp):
        osm_address = "nan"
    else:
        headers = {
            'User-Agent': 'my-application'
        }
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&accept-language=en&lat={latitude}&lon={longitude}"
        response = requests.get(url, headers=headers)
        result = response.json()
        osm_address = result.get('display_name', 'nan')

    

    message_list.append({
        "uid": row.get("uid", None),
        "timestamp": timestamp,
        "datetime": formatted_time,
        "osm_address": osm_address,
        "wifi_location": wifi_location
    })

with open("dataset_fine\\processing_data.jsonl", 'w') as f:
    for message in message_list:
        f.write(json.dumps(message) + '\n')

import json
import openai
from tqdm import tqdm

api_key = 'your key'
client = openai.OpenAI(api_key=api_key)
input_files = [
    #r"F:\sensor\error_records.jsonl"
    #"F:\\sensor\\dataset_fine\\unique_addresses_u00_googleMap.jsonl"
    "F:\\sensor\\dataset_fine\\unique_ssid_u00.jsonl"
]
output_file = "F:\sensor\processed_location_data.jsonl"

system_prompt = (
    "We get a specific address based on the user's GPS data or the name of a nearby building based on the Wi-Fi SSID. "
    "Please use the Dartmouth College map or other related materials to provide details of this place, and classify the location type, "
    "such as library, XXX college, XXX restaurant, gymnasium, hospital, residential area, etc. "
    "Answer format example: 1. Specific address: XXX; 2. Detail information: (Provide a detailed description of this place), "
    "3. Type: Main category - Subcategory. Do not output extraneous text and do not apologise."
)

def get_openai_response(system_message, user_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Failed: {e}")
        return None

with open(output_file, "a", encoding="utf-8") as outfile:
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as infile:
            for line in tqdm(infile):
                record = json.loads(line.strip())
                location_info = record.get("info_from_SSID") or record.get("reverse_geocoding_address")
                #location_info=record
                if location_info and location_info.lower() != "nan":
                    user_message = f"Location: {location_info}"
                    
                    gpt_response = get_openai_response(system_prompt, user_message)
                    print(gpt_response)
                    if gpt_response:
                        output_record = {
                            "input_data": record,
                            "output_text": gpt_response
                        }

                        # 写入结果到输出文件
                        outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                        
                    else:
                        print(f"Pass: {location_info}")






