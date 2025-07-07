import pandas as pd
from sklearn.metrics import recall_score, precision_score

from loc_matrix import *

class_names_8 = np.array(
    ['left_ankle', 'left_chest', 'left_pocket', 'left_waist', 'no_weapon',
     'right_ankle', 'right_chest', 'right_pocket', 'right_waist']
)

class_names_6 = np.array(
    ['left_chest', 'left_pocket', 'left_waist', 'no_weapon',
     'right_chest', 'right_pocket', 'right_waist']
)

class_names_4 = np.array(
    ['left_chest', 'left_waist', 'no_weapon', 'right_chest', 'right_waist']
)


def eval_per_split(split_labels, split_name):
    if split_name == 'Label_clothing':
        class_names = class_names_4
    elif split_name == 'Label_env':
        class_names = class_names_6
    else:
        class_names = class_names_8

    label_pair = []
    for idx, row in split_labels.iterrows():
        pred, gt = row['Prediction'], row['Ground Truth']
        label_pair.append((pred, gt))

    label_pair = np.array(label_pair)
    y_pred, y_true = label_pair[:, 0], label_pair[:, 1]
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=class_names)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=class_names)
    f1_score = 2 / (1 / precision + 1 / recall)
    # print(split_name, 'precision =', precision, 'recall =', recall, 'f1_score =', f1_score)

    return precision, recall, f1_score


# exp_name = 'overall.cvae_color_azi_gen64.v9.1'
# exp_name = 'overall.pix2pix_color_azi_gen256.v9.1'
exp_name = 'overall.rgbd_mse.v9.1'
labels_file = f'{exp_name}/{exp_name}_labels_out.csv'
print(exp_name)

labels_data = pd.read_csv(labels_file)
weapon_data = pd.read_csv('weapon_labels_9.csv')
for key, val in object_sym.items():
    weapon_data['Dataset'].replace(to_replace=key, value=val, inplace=True)

labels_data = labels_data.merge(weapon_data, on=['Dataset', 'Capture_ID'])

results = []
precision, recall, f1_score = eval_per_split(labels_data, 'overall')
results.append(('overall', precision, recall, f1_score))

for label in ['fleece_jacket', 'leather_jacket', 'normal', 'snow_jacket']:
    selection = labels_data.query(f"Label_clothing == '{label}'")
    selection = selection[['Dataset', 'Capture_ID', 'Prediction', 'Ground Truth']]
    precision, recall, f1_score = eval_per_split(selection, 'Label_clothing')
    if label == 'normal':
        label = 'normal_clothing'
    results.append((label, precision, recall, f1_score))

for label in ['corridor', 'ladder', 'ladder_whiteboard', 'normal']:
    selection = labels_data.query(f"Label_env == '{label}'")
    selection = selection[['Dataset', 'Capture_ID', 'Prediction', 'Ground Truth']]
    precision, recall, f1_score = eval_per_split(selection, 'Label_env')
    if label == 'normal':
        label = 'normal_env'
    results.append((label, precision, recall, f1_score))

results = pd.DataFrame(results, columns=['split_name', 'precision', 'recall', 'f1_score'])
results.to_csv(f'{exp_name}/{exp_name}_split_results.csv', index=False)
print(results)
