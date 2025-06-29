import os
import ast
import numpy as np
import pandas as pd

# UCI 原始 label: 数字 -> 名称
uci_label_dict = {
    1: "WALKING",
    2: "UPSTAIRS",
    3: "DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# 用来统一不同大小写 / 同义词
# 注意: 下面的键(key)全部小写
synonyms_map = {
    "walking": 1,
    "walk": 1,
    "walking_upstairs": 2,
    "upstairs": 2,
    "climbing": 2,
    "walking_downstairs": 3,
    "downstairs": 3,
    "decending": 3,  # 有些人拼成 decending
    "sitting": 4,
    "sit": 4,
    "standing": 5,
    "stand": 5,
    "laying": 6,
    "lying": 6
}

def unify_label(original_label_str):
    """
    将任意大小写/别名的标签转换为 (数字ID, UCI标准名称)。
    未能识别的返回 (None, None)。
    """
    label_lower = original_label_str.strip().lower()
    if label_lower in synonyms_map:
        num_id = synonyms_map[label_lower]
        std_name = uci_label_dict[num_id]
        return num_id, std_name
    else:
        return None, None


def filter_and_save_features(input_csv, output_csv_numeric, output_csv_named):
    """
    从 input_csv 中读取 features 数据(最后一列是 label)，
    将其中同义词映射到 UCI 六大类(数字1~6)，剔除无法映射的行，
    最终分别写出:
      - output_csv_numeric: label 存数字
      - output_csv_named:   label 存字符串
    """
    df = pd.read_csv(input_csv)
    # 假设 df 里最后一列叫 label
    label_nums = []
    label_names = []

    for lbl in df["label"]:
        num_id, name_str = unify_label(str(lbl))
        label_nums.append(num_id)
        label_names.append(name_str)
    print(label_nums, label_names)

    df["label_num"] = label_nums
    df["label_name"] = label_names

    # 只保留 label_num 非空的行
    df_filtered = df.dropna(subset=["label_num"])

    # 做两份拷贝:
    # (1) 数字版: 仅保留所有特征 & "label" 列(数值)
    df_numeric = df_filtered.copy()
    df_numeric = df_numeric.drop(columns=["label_name"])    # 不要名字列
    df_numeric = df_numeric.drop(columns=["label"], errors="ignore")
    df_numeric = df_numeric.rename(columns={"label_num": "label"})
    # 原先的字符串 label 列可去掉，也可保留作参考

    # (2) 名称版: 仅保留特征 & "label" 列(名称)
    df_named = df_filtered.copy()
    df_named = df_named.drop(columns=["label_num"])
    df_named = df_named.drop(columns=["label"], errors="ignore")
    df_named = df_named.rename(columns={"label_name": "label"})

    # 写出
    df_numeric.to_csv(output_csv_numeric, index=False)
    df_named.to_csv(output_csv_named, index=False)

    print(f"[Features] 完成过滤: {input_csv} =>")
    print(f"  numeric: {output_csv_numeric}")
    print(f"  named  : {output_csv_named}")


def filter_and_save_xyz(input_csv, output_csv_numeric, output_csv_named):
    """
    从 input_csv 中读取原始 xyz 数据 (gyro_data, accel_data, label)，
    对 label 做同义词映射，只保留可识别的六大类。
    分别输出数值版/名称版两个 CSV。
    """
    df = pd.read_csv(input_csv)
    label_nums = []
    label_names = []

    for lbl in df["label"]:
        num_id, name_str = unify_label(str(lbl))
        label_nums.append(num_id)
        label_names.append(name_str)

    df["label_num"] = label_nums
    df["label_name"] = label_names

    df_filtered = df.dropna(subset=["label_num"])

    # 数值版
    df_numeric = df_filtered.copy()
    df_numeric = df_numeric.drop(columns=["label_name"])
    df_numeric = df_numeric.drop(columns=["label"], errors="ignore")
    df_numeric = df_numeric.rename(columns={"label_num": "label"})

    # 名称版
    df_named = df_filtered.copy()
    df_named = df_named.drop(columns=["label_num"])
    df_named = df_named.drop(columns=["label"], errors="ignore")
    df_named = df_named.rename(columns={"label_name": "label"})

    df_numeric.to_csv(output_csv_numeric, index=False)
    df_named.to_csv(output_csv_named, index=False)

    print(f"[XYZ] 完成过滤: {input_csv} =>")
    print(f"  numeric: {output_csv_numeric}")
    print(f"  named  : {output_csv_named}")


def main():
    base_dir = "data/extracted"  # 你的特征 CSV 所在目录
    output_dir = "data/extracted/uci"  # 输出目录

    datasets = ["HHAR", "motion", "pamap2", "shoaib"]
    for ds in datasets:
        # 1) 过滤 features
        in_feat  = os.path.join(base_dir, f"{ds.lower()}_features.csv")
        out_num  = os.path.join(output_dir, f"{ds.lower()}_features_numeric.csv")
        out_name = os.path.join(output_dir, f"{ds.lower()}_features_named.csv")

        filter_and_save_features(in_feat, out_num, out_name)

        # 2) 过滤 xyz
        in_xyz   = os.path.join(base_dir,  f"{ds.lower()}_xyz.csv")
        out_num2 = os.path.join(output_dir,  f"{ds.lower()}_xyz_numeric.csv")
        out_name2= os.path.join(output_dir,  f"{ds.lower()}_xyz_named.csv")

        filter_and_save_xyz(in_xyz, out_num2, out_name2)


if __name__ == "__main__":
    main()
