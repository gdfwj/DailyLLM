import json
from datetime import datetime

input_path = "summary_output.jsonl"
output_path = "summary_prompt.jsonl"

system_prompt = (
    "Please summarize the user's activities over the past few hours based on each activity log entry. Your summary should include:\n"
    "a). The user's movement trajectory and locations visited.\n"
    "b). Changes in activity types and their time distribution.\n"
    "c). Description of changes in environmental conditions (e.g., lighting, temperature, noise levels).\n"
    "d). Overall trends in physiological indicators.\n\n"
    "Then analyze the data, and if find any of the following anomalies, provide users with natural conversational feedback to raise awareness and suggest timely adjustments:\n"
    "a). Environmental anomalies: prolonged exposure to extreme darkness, heat, or high noise levels.\n"
    "b). Behavioral anomalies: extended periods of inactivity or frequent, non-purposeful movement.\n"
    "c). Health anomalies: elevated heart rate, low blood oxygen levels, or abnormal body temperature.\n"
    "Please summarize and analyze all activity logs step by step and produce a natural, logical piece of summarization text."
)

start_date = datetime.strptime("2013-03-27", "%Y-%m-%d")
end_date = datetime.strptime("2013-04-03", "%Y-%m-%d")

def in_date_range(date_str):
    try:
        dt = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
        return start_date <= dt <= end_date
    except:
        return False

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for line in infile:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)

        if not in_date_range(data["Date_time"]):
            continue  # skip dates outside range

        # Compose user message
        user_parts = [
            "Here are some user's activities logs over the past few hours. "
            f"Date-time: {data['Date_time']}",
            f"Location: {data['location information']['Specific address']} "
            f"({data['location information']['Location type']})",
            f"Detail: {data['location information']['Detail information']}",
            f"Activity type: {data['Activity type']}",
            f"Scenario: {data['Scenario information']}",
            f"Environmental parameters: {json.dumps(data['enviromental_parameter'])}",
            f"Physiological indicators: {json.dumps(data['physiological indicators'])}"
        ]
        user_message = "\n".join(user_parts)

        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "label": "default"
        }

        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
