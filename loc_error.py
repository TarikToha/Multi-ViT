import pandas as pd
from sklearn.metrics import recall_score

from loc_matrix import *


def eval_per_split(split_labels, split_name):
    det_error, loc_error = [], []
    for idx, row in split_labels.iterrows():
        pred, gt = row['Prediction'], row['Ground Truth']

        bin_pred = 'anomaly' if pred != 'no_weapon' else 'clear'
        bin_gt = 'anomaly' if gt != 'no_weapon' else 'clear'
        det_error.append((bin_pred, bin_gt))

        if pred == 'no_weapon' or gt == 'no_weapon':
            continue

        pred, gt = labels_mapping[pred], labels_mapping[gt]
        error = error_matrix[gt, pred]
        loc_error.append(error)

    det_error = np.array(det_error)
    y_pred, y_true = det_error[:, 0], det_error[:, 1]
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, labels=bin_labels)
    # for label_name, rec in zip(bin_labels, recall):
    #     print(split_name, label_name, 'recall =', rec)

    loc_error = np.array(loc_error)
    loc_error = loc_error.mean() * units
    # print(split_name, 'loc_error =', loc_error, '\n')

    return recall[0], loc_error


units = 25.4  # mm
exp_name = 'overall.rgbd_mse.v9.1'
labels_file = f'{exp_name}/{exp_name}_labels_out.csv'
# labels_file = f'{exp_name}/{exp_name}_frame_labels_out.csv'
print(exp_name)

labels_data = pd.read_csv(labels_file)
weapon_data = pd.read_csv('weapon_labels_9.csv')
for key, val in object_sym.items():
    weapon_data['Dataset'].replace(to_replace=key, value=val, inplace=True)

labels_data = labels_data.merge(weapon_data, on=['Dataset', 'Capture_ID'])

results = []
recall, loc_error = eval_per_split(labels_data, 'overall')
results.append(('overall', recall, loc_error))

for label in ['fleece_jacket', 'leather_jacket', 'normal', 'snow_jacket']:
    selection = labels_data.query(f"Label_clothing == '{label}'")
    selection = selection[['Dataset', 'Capture_ID', 'Prediction', 'Ground Truth']]
    recall, loc_error = eval_per_split(selection, label)
    results.append((label, recall, loc_error))

for label in ['corridor', 'ladder', 'ladder_whiteboard', 'normal']:
    selection = labels_data.query(f"Label_env == '{label}'")
    selection = selection[['Dataset', 'Capture_ID', 'Prediction', 'Ground Truth']]
    recall, loc_error = eval_per_split(selection, label)
    results.append((label, recall, loc_error))

results = pd.DataFrame(results, columns=['split_name', 'recall', 'loc_error'])
results.to_csv(f'{exp_name}/{exp_name}_loc_results.csv', index=False)
print(results)
