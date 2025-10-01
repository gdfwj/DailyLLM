import pandas as pd
import json

def load_data(csv_file):
    return pd.read_csv(csv_file, header=None)

def create_jsonl_from_dataframe(df, output_jsonl_file):
    feature_columns = list(range(1, 121))  # 特征列索引从1到120
    label_column = 121  # 标签列索引为121
    
    
    label_mapping = {
        'beach': 1,
        'cafe/restaurant': 2,
        'city_center': 3,
        'forest_path': 4,
        'metro_station': 5,
        'tram': 6,
        'park': 7,
        'residential_area': 8,
        'home': 9,
        'bus': 10,
        'grocery_store': 11,
        'car': 12,
        'train': 13,
        'office': 14,
        'library': 15
    }
    '''
    label_mapping = {
        0: 'airport',
        1: 'shopping_mall',
        2: 'metro_station',
        3: 'street_pedestrian',
        4: 'public_square',
        5: 'street_traffic',
        6: 'tram',
        7: 'bus',
        8: 'metro',
        9: 'park'
    }
'''
    message_list = []
    total_samples = len(df)
    label_counters = {label: -1 for label in label_mapping.values()}

    for i in range(total_samples):
        features = df.iloc[i][feature_columns]
        features = [round(float(x), 4) for x in features]
        #mod_value=label_mapping[df.iloc[i][label_column]]
        scene_label = df.iloc[i][label_column]
        system_content = (
            "You are an expert in signal analysis. We collected audio from the user's phone to analyze the scene they were in. "
            "We extracted 120-dimensional features from the original sound signal. The first 60 features are means of 20 MFCC static coefficients (including 0th) + 20 delta MFCC coefficients + 20 acceleration MFCC coefficients, the last 60 features are standard deviations of the same coefficients. "
"Analyze these characteristics and combine your knowledge "
            "to predict the user's environment. Choose and output one of the following labels: ['beach', 'cafe/restaurant', 'city_center', 'forest_path', 'metro_station', 'tram', 'park', 'residential_area', 'home','bus','grocery_store','car','train','office','library']."
        )

        user_content = (
            f"Here are features extracted from the user's phone audio:\n\n"
            f"features: {features}\n"
            "Please analyze these features and combine your professional knowledge to determine the user's environment."
            "Choose and output one of the following labels:['beach', 'cafe/restaurant', 'city_center', 'forest_path', 'metro_station', 'tram', 'park', 'residential_area', 'home','bus','grocery_store','car','train','office','library']."
        )
#['beach', 'cafe/restaurant', 'city_center', 'forest_path', 'metro_station', 'tram', 'park', 'residential_area', 'home','bus','grocery_store','car','train','office','library']
        message = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": scene_label}
        ]

        message_list.append({"messages": message})

    with open(output_jsonl_file, 'w') as f:
        for message in message_list:
            f.write(json.dumps(message) + '\n')

csv_file = r'data\audio\test_features_tut2016.csv'
output_jsonl_file = 'test_data_tut2016.jsonl'

df = load_data(csv_file)

create_jsonl_from_dataframe(df, output_jsonl_file)

