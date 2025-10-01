import os
import json
import ast
import pandas as pd

def jsonl_to_csv_raw_xyz(jsonl_file, output_csv):
    results = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                continue

            messages = data.get("messages", [])
            user_content = None
            for m in messages:
                if m.get("role") == "user":
                    user_content = m["content"]
                    break
            if not user_content:
                continue

            # find summary / gyroscope / accelerometer
            lines = user_content.split("\n")
            label = None
            gyro_str = None
            accel_str = None

            for ln in lines:
                ln_str = ln.strip()
                if ln_str.lower().startswith("summary:"):
                    # e.g. "Summary: standing"
                    parts = ln_str.split(":", 1)
                    summary_val = parts[1].strip() if len(parts)>1 else ""
                    label = summary_val.split()[0] if summary_val else "UNKNOWN"

                elif ln_str.lower().startswith("gyroscope:"):
                    # e.g. "Gyroscope: [[x,y,z], [x2,y2,z2], ...]"
                    parts = ln_str.split(":", 1)
                    if len(parts)>1:
                        gyro_str = parts[1].strip()

                elif ln_str.lower().startswith("accelerometer:"):
                    parts = ln_str.split(":", 1)
                    if len(parts)>1:
                        accel_str = parts[1].strip()

            # skip if any of these are missing
            if (not label) or (not gyro_str) or (not accel_str):
                continue

            row = [gyro_str, accel_str, label]
            results.append(row)

    df = pd.DataFrame(results, columns=["gyro_data", "accel_data", "label"])
    df.to_csv(output_csv, index=False)
    print(f"processed {len(df)} lines, output to {output_csv}")

def main():
    in_out_list = [
        ("data/origin/opensqa_10hz_shoaib.jsonl",    "data/extracted/shoaib_xyz.csv"),
        ("data/origin/opensqa_10hz_pamap2_v2.jsonl", "data/extracted/pamap2_xyz.csv"),
        ("data/origin/opensqa_10hz_motion.jsonl",    "data/extracted/motion_xyz.csv"),
        ("data/origin/opensqa_10hz_hhar.jsonl",      "data/extracted/hhar_xyz.csv"),
    ]

    for (infile, outfile) in in_out_list:
        if not os.path.exists(infile):
            print(f"Warning: file not found: {infile}")
            continue
        jsonl_to_csv_raw_xyz(infile, outfile)

if __name__ == "__main__":
    main()
