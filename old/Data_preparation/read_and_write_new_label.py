import numpy as np
import gzip
from io import StringIO

def parse_header_of_csv(csv_str):
    # Ensure csv_str is a proper string before proceeding
    if isinstance(csv_str, bytes):
        csv_str = csv_str.decode('utf-8')

    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamp:
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    assert columns[-1] == 'label_source'
    
    # Search for the column of the first label:
    first_label_ind = -1
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break

    if first_label_ind == -1:
        raise ValueError("No label columns found in the CSV header.")

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:')
        label_names[li] = label.replace('label:', '')
    
    return (feature_names, label_names)

def parse_body_of_csv(csv_str, n_features):
    # Ensure csv_str is a proper string before proceeding
    if isinstance(csv_str, bytes):
        csv_str = csv_str.decode('utf-8')

    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return (X, Y, M, timestamps)

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(uuid):
    user_data_file = '%s.features_labels.csv.gz' % uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read().decode('utf-8')

    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return (X, Y, M, timestamps, feature_names, label_names)

uuid = '0A986513-7828-4D53-AA1F-E02D6DF9561B'
(X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid)

#print(X, Y, M, timestamps, feature_names, label_names)

#X is the feature matrix. Each row is an example and each column is a sensor-feature:
#Y is the binary label-matrix. Each row represents an example and each column represents a label. Value of 1 indicates the label is relevant for the example:
# M is the missing label matrix

print(label_names)






"""
def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'
    
    #if label.endswith('_'):
        #label = label[:-1] + ')';
        #pass;
    
    #label = label.replace('__',' (').replace('_',' ');
    #label = label[0] + label[1:].lower();
    #label = label.replace('i m','I\'m');
    return label;


n_examples_per_label = np.sum(Y, axis=0)
labels_and_counts = zip(label_names, n_examples_per_label)
sorted_labels_and_counts = sorted(labels_and_counts, reverse=True, key=lambda pair: pair[1])

print("How many examples does this user have for each context-label:")
print("-" * 20)

for (label, count) in sorted_labels_and_counts:
    # Correct way to format a print statement in Python 3
    print("label %s - %d minutes" % (label, count))
"""

def rewrite_label(label):
    label_map = {
        'SITTING': 'sitting',
        'FIX_walking': 'walking',
        'FIX_running': 'running',
        'SLEEPING': 'sleeping',
        'BICYCLING': 'bicycling',
        'LOC_main_workplace': 'At main workplace',
        'OR_indoors': 'Indoors',
        'OR_outside': 'Outside',
        'LOC_home': 'At home',
        'FIX_restaurant': 'At a restaurant',
        'OR_exercise': 'Exercise',
        'LOC_beach': 'At the beach',
        'OR_standing': 'Standing',
        'WATCHING_TV': 'Watching TV'
    }
    
    return label_map.get(label, label)  # Returns the original label if no match is found

# Example usage
#labels = ['FIX_walking', 'LOC_home', 'OR_outside', 'UNKNOWN_TAG']
#rewritten_labels = [rewrite_label(label) for label in labels]


#print(type(label_names))
rewritten_labels = [rewrite_label(label_names) for label_names in label_names]
#print(rewritten_labels)  


#print(X, Y, M, timestamps, feature_names, rewritten_labels)

print(Y)