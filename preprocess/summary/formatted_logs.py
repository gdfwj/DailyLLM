import json
import pandas as pd
import re

def extract_summary_info(input_file, reference_file, output_file):
    light_map = {1: "Extremely dark", 2: "Dim", 3: "Moderate brightness", 4: "Bright", 5: "Harsh light"}
    temperature_map = {1: "Cold", 2: "Cool", 3: "Comfortable", 4: "Warm", 5: "Hot"}
    noise_map = {1: "Very quiet", 2: "Soft sound", 3: "Normal sound", 4: "Noisy", 5: "Very noisy"}

    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        reference_data = [json.loads(line) for line in ref_file]

    output_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            date_time = entry.get("date_time", "")
            ssid_info_raw = entry.get("location", {}).get("info_from_SSID", "")
            reverse_address = entry.get("location", {}).get("reverse_geocoding_address", "")

            specific_addresses = []
            detail_info = []
            location_types = []

            if ssid_info_raw and not pd.isna(ssid_info_raw) and str(ssid_info_raw).lower() != "nan":
                match = re.search(r'\[(.*?)\]', ssid_info_raw)
                ssid_info = match.group(1) if match else ""
                places = [p.strip() for p in ssid_info.split(";") if p.strip()]
                for place in places:
                    for ref in reference_data:
                        ref_ssid = ref.get("input_data", {}).get("info_from_SSID", "")
                        if place.lower() in ref_ssid.lower():
                            specific_addresses.append(ref.get("Specific address", ""))
                            detail_info.append(ref.get("Detail information", ""))
                            location_types.append(ref.get("Type", ""))
                            break
            else:
                for ref in reference_data:
                    ref_addr = ref.get("input_data", {}).get("reverse_geocoding_address", "")
                    if reverse_address and reverse_address.lower() in ref_addr.lower():
                        specific_addresses.append(ref.get("Specific address", ""))
                        detail_info.append(ref.get("Detail information", ""))
                        location_types.append(ref.get("Type", ""))
                        break

            location_info = {
                "Specific address": "; ".join(specific_addresses) or reverse_address,
                "Detail information": "; ".join(detail_info),
                "Location type": "; ".join(location_types)
            }

            activity_type = entry.get("activity", {}).get("activity_label", "")
            scenario_info = str(entry.get("Scene", {}).get("scene_label", ""))

            env = entry.get("enviromental_parameter", {})
            light = light_map.get(env.get("light"), "Unknown")
            temperature = temperature_map.get(env.get("temperature"), "Unknown")
            noise = noise_map.get(env.get("noise"), "Unknown")

            physio = entry.get("physiological_indicators", {})

            output_entry = {
                "Date_time": date_time,
                "location information": location_info,
                "Activity type": activity_type,
                "Scenario information": scenario_info,
                "enviromental_parameter": {
                    "light": light,
                    "temperature": temperature,
                    "noise": noise
                },
                "physiological indicators": {
                    "EDA": physio.get("EDA"),
                    "HR": physio.get("HR"),
                    "IBI": physio.get("IBI"),
                    "TEMP": physio.get("TEMP")
                }
            }

            output_data.append(output_entry)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in output_data:
            json.dump(item, out_f, ensure_ascii=False)
            out_f.write('\n')
extract_summary_info(
    input_file="dataset_fine\\Updated_DailyLLM_processed_data.jsonl",
    reference_file="updated_processed_location_data.jsonl",
    output_file="summary_output.jsonl"
)