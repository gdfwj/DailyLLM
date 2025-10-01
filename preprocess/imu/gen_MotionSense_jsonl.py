import pandas as pd
import json

def load_data(csv_file):
    return pd.read_csv(csv_file, header=0)

def create_jsonl_from_dataframe(df, output_jsonl_file):
    message_list = []

    for _, row in df.iterrows():
        body_acc   = pd.to_numeric(row.iloc[0:26], errors='coerce').round(4).values.tolist()
        body_gyro  = pd.to_numeric(row.iloc[26:52], errors='coerce').round(4).values.tolist()
        total_acc  = pd.to_numeric(row.iloc[52:78], errors='coerce').round(4).values.tolist()

        label_message = row['label']  

        system_content = (
            "You are an expert in signal analysis. We have collected some sensor data from volunteers. "
            "The smartphone's embedded accelerometer and gyroscope were used to capture 3-axial linear acceleration "
            "and 3-axial angular velocity at a constant rate of 50Hz. We use the body acceleration signal, "
            "the angular velocity vector and total acceleration signal.\n\n"
            
            "We then extract some features of these sensing data from the time and frequency domains as follows:\n\n"

            "For each of the body acceleration signal, the angular velocity vector and total acceleration signal, "
            "we calculate the vector magnitude signal as the euclidean norm of the 3-axis measurement at each point in time. "
            "We then extract (1) 9 statistics of the magnitude signal. "
            "(2) six spectral features (log energies in 5 sub-bands and spectral entropy) of the magnitude signal. "
            "(3) Two autocorrelation features from the magnitude signal: "
            "   a). Dominant periodicity. "
            "   b). Normalized autocorrelation value. "
            "(4) Nine statistics of the 3-axis time series: the mean and standard deviation of each axis, "
            "and the 3 inter-axis correlation coefficients.\n\n"
            
            "Hence we obtain 26-dimensional features from each sensor group, used to classify six different activities: "
            "[WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, JOGGING]. "
            "Please analyze these features and combine your professional knowledge to determine the user's motion. "
            "Please think step by step and provide your answer with one of the six activities. Please only output one word."
        )

        user_content = (
            "Here are some features we extracted from sensors on Smartphone:\n\n"
            f"body accelerometer: {body_acc}\n"
            f"body gyroscope: {body_gyro}\n"
            f"total accelerometer: {total_acc}\n"
            "Please analyze these features and combine your professional knowledge to determine what the user's motions are."
        )

        message = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": label_message}
        ]

        message_list.append(message)

    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for message in message_list:
            f.write(json.dumps({"messages": message}, ensure_ascii=False) + '\n')

def create_xyzjsonl_from_dataframe(df, output_jsonl_file):
    message_list = []

    for _, row in df.iterrows():
        x_data = row.iloc[0]
        y_data = row.iloc[1]
        z_data = row.iloc[2]
        label_message = row['label']

        system_content = (
            "You are an expert in signal analysis. We have collected data from volunteers performing six activities: "
            "[WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, JOGGING], "
            "while wearing a smartphone on the waist. The smartphone's embedded accelerometer and gyroscope "
            "were used to capture 3-axial linear acceleration and angular velocity at 50Hz.\n"
            "Please analyze these data and determine which of the six activities the subject is doing."
        )

        user_content = (
            "Here are 3-axial data captured from the smartphone sensors:\n\n"
            f"X-axis: {x_data}\n"
            f"Y-axis: {y_data}\n"
            f"Z-axis: {z_data}\n"
            "Please analyze them and combine your professional knowledge to determine the user's motion."
        )

        message = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": label_message}
        ]

        message_list.append(message)

    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for message in message_list:
            f.write(json.dumps({"messages": message}, ensure_ascii=False) + '\n')

def split_by_label(df, train_size=100, val_size=40, random_state=42):
    train_list = []
    val_list   = []
    test_list  = []

    unique_labels = df['label'].unique()

    for lbl in unique_labels:
        sub_df = df[df['label'] == lbl].copy()
        sub_df = sub_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        train_part = sub_df.iloc[:train_size]
        val_part   = sub_df.iloc[train_size:train_size + val_size]
        test_part  = sub_df.iloc[train_size + val_size:]

        train_list.append(train_part)
        val_list.append(val_part)
        test_list.append(test_part)
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df   = pd.concat(val_list,   ignore_index=True)
    test_df  = pd.concat(test_list,  ignore_index=True)

    return train_df, val_df, test_df

def main_splitting_pipeline():
    csv_file = r"data\extracted\MotionSense.csv"
    df = pd.read_csv(csv_file)

    train_df, val_df, test_df = split_by_label(df, train_size=100, val_size=100)

    create_jsonl_from_dataframe(train_df, "train.jsonl")
    create_jsonl_from_dataframe(val_df,   "val.jsonl")
    create_jsonl_from_dataframe(test_df,  "test.jsonl")

    print(f"Done. Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    main_splitting_pipeline()

