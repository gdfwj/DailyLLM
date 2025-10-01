import os
import ast
import numpy as np
import pandas as pd

uci_label_dict = {
    1: "WALKING",
    2: "UPSTAIRS",
    3: "DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

synonyms_map = {
    "walking": 1,
    "walk": 1,
    "walking_upstairs": 2,
    "upstairs": 2,
    "climbing": 2,
    "walking_downstairs": 3,
    "downstairs": 3,
    "decending": 3,
    "sitting": 4,
    "sit": 4,
    "standing": 5,
    "stand": 5,
    "laying": 6,
    "lying": 6
}

def unify_label(original_label_str):
    label_lower = original_label_str.strip().lower()
    if label_lower in synonyms_map:
        num_id = synonyms_map[label_lower]
        std_name = uci_label_dict[num_id]
        return num_id, std_name
    else:
        return None, None


def filter_and_save_features(input_csv, output_csv_numeric, output_csv_named):
    df = pd.read_csv(input_csv)
    label_nums = []
    label_names = []

    for lbl in df["label"]:
        num_id, name_str = unify_label(str(lbl))
        label_nums.append(num_id)
        label_names.append(name_str)
    print(label_nums, label_names)

    df["label_num"] = label_nums
    df["label_name"] = label_names

    df_filtered = df.dropna(subset=["label_num"])

    df_numeric = df_filtered.copy()
    df_numeric = df_numeric.drop(columns=["label_name"])
    df_numeric = df_numeric.drop(columns=["label"], errors="ignore")
    df_numeric = df_numeric.rename(columns={"label_num": "label"})
    df_named = df_filtered.copy()
    df_named = df_named.drop(columns=["label_num"])
    df_named = df_named.drop(columns=["label"], errors="ignore")
    df_named = df_named.rename(columns={"label_name": "label"})

    df_numeric.to_csv(output_csv_numeric, index=False)
    df_named.to_csv(output_csv_named, index=False)

    print(f"[Features] filtered: {input_csv} =>")
    print(f"  numeric: {output_csv_numeric}")
    print(f"  named  : {output_csv_named}")


def filter_and_save_xyz(input_csv, output_csv_numeric, output_csv_named):
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

    df_numeric = df_filtered.copy()
    df_numeric = df_numeric.drop(columns=["label_name"])
    df_numeric = df_numeric.drop(columns=["label"], errors="ignore")
    df_numeric = df_numeric.rename(columns={"label_num": "label"})

    df_named = df_filtered.copy()
    df_named = df_named.drop(columns=["label_num"])
    df_named = df_named.drop(columns=["label"], errors="ignore")
    df_named = df_named.rename(columns={"label_name": "label"})

    df_numeric.to_csv(output_csv_numeric, index=False)
    df_named.to_csv(output_csv_named, index=False)

    print(f"[XYZ] filtered: {input_csv} =>")
    print(f"  numeric: {output_csv_numeric}")
    print(f"  named  : {output_csv_named}")


def main():
    base_dir = "data/extracted"
    output_dir = "data/extracted/uci" 

    datasets = ["HHAR", "motion", "pamap2", "shoaib"]
    for ds in datasets:
        in_feat  = os.path.join(base_dir, f"{ds.lower()}_features.csv")
        out_num  = os.path.join(output_dir, f"{ds.lower()}_features_numeric.csv")
        out_name = os.path.join(output_dir, f"{ds.lower()}_features_named.csv")

        filter_and_save_features(in_feat, out_num, out_name)

        in_xyz   = os.path.join(base_dir,  f"{ds.lower()}_xyz.csv")
        out_num2 = os.path.join(output_dir,  f"{ds.lower()}_xyz_numeric.csv")
        out_name2= os.path.join(output_dir,  f"{ds.lower()}_xyz_named.csv")

        filter_and_save_xyz(in_xyz, out_num2, out_name2)


if __name__ == "__main__":
    main()
