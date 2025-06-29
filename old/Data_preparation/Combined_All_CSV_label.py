
import glob
import pandas as pd
# Define the path pattern to find all CSV files in the directory
#file_pattern = '/Users/xiaomo/Desktop/SensorLLM/ExtraSensoryDataset/dataset/test_dataset/test_output_labels/timestamp_and_label/*.csv'


# unseen

def combine_label(file_pattern='../../SensorLLM_dataset/ExtraSensor/test_dataset/test_output_labels/timestamp_and_label/unseen/*.csv'):

    # Get all CSV files in the specified directory
    csv_files = glob.glob(file_pattern)
    print(csv_files)
    exit(0)

    # Initialize an empty list to store each DataFrame
    dataframes = []

    # Loop through each CSV file, read it, and append to the list
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_data = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    File_name1 = 'unseen_UUID_timestamp_and_label.csv'
    combined_file_path = '../../SensorLLM_dataset/ExtraSensor/test_dataset/test_output_labels/'
    combined_file=combined_file_path+File_name1
    combined_data.to_csv(combined_file, index=False)

    # Count the occurrences of each label
    label_counts = combined_data['label'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    # Display the label counts
    print(label_counts)


    # Select the first 60 samples from each label category

    sampled_data = combined_data.groupby('label').head(30)
    # Save the sampled data to a new CSV file
    File_name2 = 'unseen_Sample_UUID_timestamp_and_label.csv'
    sampled_file = combined_file_path+File_name2
    sampled_data.to_csv(sampled_file, index=False)

    # Count the occurrences of each label
    label_counts2 = sampled_data['label'].value_counts().reindex(['sitting','walking','running','bicycling','sleeping'], fill_value=0)
    # Display the label counts
    print(label_counts2)

if __name__ == '__main__':
    combine_label()