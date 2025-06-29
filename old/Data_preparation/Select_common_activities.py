from collections import defaultdict

import pandas as pd
import gzip
import sys



def process_activities_file(input_file, output_file):
    columns_to_keep = [
        'timestamp',
        'label:SITTING',
        'label:FIX_walking',
        'label:FIX_running',
        'label:BICYCLING',
        'label:SLEEPING'
    ]
    
  
    df = pd.read_csv(input_file, compression='gzip')
    df_selected = df[columns_to_keep]
    df_selected.to_csv(output_file, index=False)
    
    print(f"Process completed. New file saved as: {output_file}")
    print(f"Shape of new dataset: {df_selected.shape}")

import os
def select_activities(UUID='9DC38D04-E82E-4F29-AB52-B476535226F2', data="unseen", test_path = '../../SensorLLM_dataset/ExtraSensory/test_dataset'):
    test_path = '../../SensorLLM_dataset/ExtraSensor/test_dataset'
    root_path_input = os.path.join(test_path, "test_features_labels")
    if data=="unseen":
        root_path_input = os.path.join(root_path_input, data)
    # root_path_input='../../SensorLLM_dataset/ExtraSensory/test_dataset/test_features_labels/'
    # root_path_input = '../../SensorLLM_dataset/ExtraSensory/test_dataset/test_features_labels/unseen/'
    test_path_output = os.path.join(test_path, "test_output_labels")
    root_path_output1 = os.path.join(test_path_output, "common_activities")
    root_path_output2 = os.path.join(test_path_output, "timestamp_and_label")
    # root_path_output1 = '../../SensorLLM_dataset/ExtraSensory/test_dataset/test_output_labels/common_activities/'
    # root_path_output2 = '../../SensorLLM_dataset/ExtraSensory/test_dataset/test_output_labels/timestamp_and_label/'

    input_file = UUID + '.features_labels.csv.gz'

    Fall_path_input_file = os.path.join(root_path_input, input_file)

    output_file_name = UUID+'_selected_common_activities.csv'
    Fall_path_output_file = os.path.join(root_path_output1, output_file_name)
    process_activities_file(Fall_path_input_file, Fall_path_output_file)

    # Load the CSV file
    data = pd.read_csv(Fall_path_output_file)
    # Inspect the first few rows to understand the structure
    data.head()

    # Define a new DataFrame to store the timestamp and the derived label
    processed_data = pd.DataFrame(columns=['UUID','timestamp', 'label'])

    # Iterate through the rows and assign label based on conditions
    for index, row in data.iterrows():
        if row['label:SITTING'] == 1:
            label = 'sitting'
        elif row['label:FIX_walking'] == 1:
            label = 'walking'
        elif row['label:FIX_running'] == 1:
            label = 'running'
        elif row['label:BICYCLING'] == 1:
            label = 'bicycling'
        elif row['label:SLEEPING'] == 1:
            label = 'sleeping'
        else:
            continue  # Skip rows where all labels are NaN or 0

        # Append the timestamp and label to the processed_data DataFrame
        processed_data = processed_data._append({'UUID': UUID,'timestamp': row['timestamp'], 'label': label}, ignore_index=True)

    label_counts = processed_data['label'].value_counts().sort_index()
    print(label_counts)
    # Save the processed data to a new CSV file
    label_file_name = 'unseen'+UUID+'_timestamp_and_label.csv'
    label_file_path =  root_path_output2+label_file_name



    processed_data.to_csv(label_file_path, index=False)

import argparse
if __name__ == '__main__':
    # UUID='0A986513-7828-4D53-AA1F-E02D6DF9561B'
    # UUID='0BFC35E2-4817-4865-BFA7-764742302A2D'
    # UUID='0E6184E1-90C0-48EE-B25A-F1ECB7B9714E'
    # UUID='00EABED2-271D-49D8-B599-1D4A09240601'
    # UUID='1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842'

    # Train dataset:
    # UUID='0BFC35E2-4817-4865-BFA7-764742302A2D'
    # UUID='1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842'

    # unseen dataset:
    # UUID='2C32C23E-E30C-498A-8DD2-0EFB9150A02E'
    # UUID='4E98F91F-4654-42EF-B908-A3389443F2E7'
    # UUID='4FC32141-E888-4BFF-8804-12559A491D8C'
    # UUID='5EF64122-B513-46AE-BCF1-E62AAC285D2C'
    # UUID='7CE37510-56D0-4120-A1CF-0E23351428D2'

    # UUID='7D9BB102-A612-4E2A-8E22-3159752F55D8'
    # UUID = '9DC38D04-E82E-4F29-AB52-B476535226F2'

    ###########################################
    ##have not use yet
    # UUID='11B5EC4D-4133-4289-B475-4E737182A406'
    # UUID='24E40C4C-A349-4F9F-93AB-01D00FB994AF'
    # UUID='27E04243-B138-4F40-A164-F40B60165CF3'
    # UUID='9DC38D04-E82E-4F29-AB52-B476535226F2'
    # UUID='7D9BB102-A612-4E2A-8E22-3159752F55D8'
    parser = argparse.ArgumentParser()
    parser.add_argument("--UUID", default="9DC38D04-E82E-4F29-AB52-B476535226F2")
    parser.add_argument("--data", default="unseen")
    args = parser.parse_args()
    select_activities(UUID=args["UUID"], data=args['data'])