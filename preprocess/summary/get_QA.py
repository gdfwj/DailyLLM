import json
import pandas as pd
import re

from tqdm import tqdm

def create_message(system_content, user_content, label_message):
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": label_message}
    ]


def analyze_and_generate_responses(input_file, reference_file, output_file):

    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        reference_data = [json.loads(line) for line in ref_file.readlines()]

    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f.readlines()]

    messages = []

    system_content = """
You're an expert in signal analysis. Please analyze and predict the user's activity context based on the features of these sensor data. Specifically, these data include date and time; Location: reverse geocoding addresses based on GPS coordinates or building names parsed from connection information such as Wi-Fi and Bluetooth; Sensor data captured at a constant rate using the smartphone's built-in IMU; Audio clips recorded by binaural microphones at a sampling rate of 44.1 kHz.

1. Date-time: MM-DD-YYYY HH:mm:ss GMT±HHMM. 
2. Location: GPS address: XXX; Near building name: XXX.
3. IMU Sensors Features: We calculate the vector magnitude signal as the euclidean norm of the 3-axis measurement at each point in time. We then extract (1). 9 statistics of the magnitude signal. (2).6 spectral features (log energies in 5 sub-bands and spectral entropy) of the magnitude signal. (3). 2 autocorrelation features from the magnitude signal: a). Dominant periodicity.   b). Normalized autocorrelation value. (4). 9 statistics of the 3-axis time series: the mean and standard deviation of each axis, and the 3 inter-axis correlation coefficients. Therefore, we obtain 26-dimensional features from each sensor group, used to classify different activities.
4. Audio Sensors Features: We extract 120-dimensional features from raw audio signals to identify and distinguish scene types. The first 60 features are the mean values of 20 MFCC static coefficients (including the 0th), 20 delta MFCC coefficients, 20 acceleration MFCC coefficients, and the last 60 features are the standard deviations of the same coefficients.

Task Explanation:
(1) We get a specific address based on the user's GPS data or the name of a nearby building based on the Wi-Fi/bluetooth SSID. Please use the Dartmouth College map and related materials to provide details information of this place, and classify the location type, such as library, XXX college, XXX restaurant, gymnasium, hospital, residential area, etc.\nAnswer format example: Specific address: XXX;  Detail information: XXXX,  Location type: Main category - Subcategory. Do not output extraneous text. 

(2) According to the features of IMU data and its description, please predict which of the following activities is carried out by the current user: [WALKING, SITTING, STANDING, LYING, UPSTAIRS, DOWNSTAIRS]. 

(3) According to the  features of the audio data and its description, please predict which of the following type of scene the current user is in: 1-15.

Output Format: Please organize your answer in this format:\nDate-time: XXX; Location information: {Specific address: XXX; Detail information: XXXX, Location type: Main category - Subcategory. }; Activity type: XXX, Scenario information: XXX. Do not output extraneous text.
"""
    i=0
    for entry in tqdm(input_data):
        i+=1
        #if i==24:
            #import pdb
            #pdb.set_trace()
        date_time = entry["date_time"]

        location_info_str = entry["location"].get("reverse_geocoding_address", "")
        location_info = entry["location"].get("info_from_SSID", "")

        imu_data = entry["activity"]["IMU_features"]
        body_acc = [f"{x:.4f}" for x in imu_data[:26]]
        body_gyro = [f"{x:.4f}" for x in imu_data[26:52]]
        total_acc = [f"{x:.4f}" for x in imu_data[52:]]

        audio_features = [f"{x:.4f}" for x in entry["Scene"]["audio_features"]]
        activity_type = entry["activity"]["activity_label"]
        scene_info = entry["Scene"]["scene_label"]

        user_content = (
            f"Here are some features we extracted from sensors on Smartphone:\n"
            f"1. Date-time: {date_time}\n"
            f"2. Location: GPS address: {location_info_str}; Near building name: {location_info}\n"
            f"3. IMU features: body accelerometer: {body_acc}\n"
            f"   body gyroscope: {body_gyro}\n"
            f"   total accelerometer: {total_acc}\n"
            f"4. Audio features: {audio_features}\n"
            f"Please analyze these features and output your answer according to the format."
        )

        specific_addresses = []
        detail_info = []
        location_types = []

        ssid_info_raw = location_info
        reverse_address = location_info_str

        if ssid_info_raw and not pd.isna(ssid_info_raw) and ssid_info_raw.lower() != "nan":
            match = re.search(r'\[(.*?)\]', ssid_info_raw)
            ssid_info = match.group(1) if match else ""            
            places = [place.strip() for place in ssid_info.split(";") if place.strip()]

            for place in places:
                found = False
                for ref in reference_data:
                    ref_ssid = ref.get("input_data", {}).get("info_from_SSID", "")
                    if place.lower() in ref_ssid.lower():
                        #import pdb
                        #pdb.set_trace()
                        specific_address_match = ref.get("Specific address", "")
                        detail_info_match = ref.get("Detail information", "")
                        location_type_match = ref.get("Type", "")
                        if specific_address_match and detail_info_match and location_type_match:
                            specific_addresses.append(specific_address_match)
                            detail_info.append(detail_info_match)
                            location_types.append(location_type_match)
                            found = True
                            break
                if not found:
                    specific_addresses.append("N/A")
                    detail_info.append("N/A")
                    location_types.append("N/A")
        else:
            for ref in reference_data:
                ref_addr = ref.get("input_data", {}).get("reverse_geocoding_address", "")
                if reverse_address and reverse_address.lower() in ref_addr.lower():
                    specific_address_match = ref.get("Specific address", "")
                    detail_info_match = ref.get("Detail information", "")
                    location_type_match = ref.get("Type", "")
                    if specific_address_match and detail_info_match and location_type_match:
                        specific_addresses.append(specific_address_match)
                        detail_info.append(detail_info_match)
                        location_types.append(location_type_match)
                        break

        label_message = (
            f'Date-time: {date_time}; '
            f'Location information: {{"Specific address": "{"; ".join(specific_addresses)}", '
            f'"Detail information": "{"; ".join(detail_info)}", '
            f'"Location type": {"; ".join(location_types)}}}; '
            f'Activity type: {activity_type}, Scenario information: {scene_info}.'
        )

        message = create_message(system_content, user_content, label_message)
        messages.append(message)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for msg in messages:
            json.dump(msg, out_f, ensure_ascii=False)
            out_f.write('\n')

# 示例调用
input_file = 'dataset_fine\\Updated_DailyLLM_processed_data.jsonl'
reference_file = 'updated_processed_location_data.jsonl' 
output_file = 'final_dataset1.jsonl' 

analyze_and_generate_responses(input_file, reference_file, output_file)
