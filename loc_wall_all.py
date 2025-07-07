import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from loc_matrix import fall_error_matrix, loc_class_names, loc2idx


def get_loc_error(selection):
    loc_error = []
    for _, row in selection.iterrows():
        pred_loc, gt_loc = row['Prediction'], row['Ground_Truth']
        pred_loc, gt_loc = loc2idx[pred_loc], loc2idx[gt_loc]
        error = fall_error_matrix[gt_loc, pred_loc]
        loc_error.append(error)

    loc_error = np.array(loc_error)
    loc_error = loc_error.mean() * units
    return loc_error


def get_results(selection):
    y_pred, y_true = selection['Prediction'], selection['Ground_Truth']
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=loc_class_names)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=loc_class_names)
    f1_score = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1_score


def aggregate_frames(dataset):
    cases = dataset['Capture_ID'].unique()
    cases.sort()

    out = []
    for capture_id in cases:
        selection = dataset[dataset['Capture_ID'] == capture_id]
        wall = selection['Dataset'].iloc[0]

        pred = selection['Prediction'].mode().iloc[0]
        gt = selection['Ground_Truth'].mode().iloc[0]
        # print(wall, capture_id, frame_id, frame_id + 5, pred, gt)
        out.append(
            (wall, capture_id, pred, gt)
        )
        # print(out[-1])

    out = pd.DataFrame(out, columns=[
        'Dataset', 'Capture_ID', 'Prediction', 'Ground_Truth'
    ])
    return out


units = 12 * 2.54 / 100  # meter
# exp_name = 'resnet_azi'
# exp_name = 'cvae_color_azi_gen64'
# exp_name = 'pix2pix_color_azi_gen256'
exp_name = 'rgbd_mse'
loc_exp, pose_loc2d_exp = f'intrusion.{exp_name}.v7.1', f'pose_loc2d.{exp_name}.v13.1'
loc_data = pd.read_csv(f'{loc_exp}/{loc_exp}_frame_labels_out.csv')
multi_data = pd.read_csv(f'{pose_loc2d_exp}/{pose_loc2d_exp}_multi_frame_labels_out.csv')
multi_data = multi_data[
    ['Dataset', 'Capture_ID', 'Frame_ID', 'Prediction_Loc', 'Ground_Truth_Loc']
]
multi_data.rename(columns={
    'Prediction_Loc': 'Prediction', 'Ground_Truth_Loc': 'Ground_Truth'
}, inplace=True)

data = pd.concat([loc_data, multi_data])
data.sort_values(by=['Capture_ID', 'Frame_ID'], inplace=True)
print(data.shape)

if exp_name == 'rgbd_mse':
    data = aggregate_frames(data)
    print(data.shape)

results = []
precision, recall, f1_score = get_results(data)
loc_error = get_loc_error(data)
results.append(('overall', precision, recall, f1_score, loc_error))

for wall in [
    'brick', 'brick_ladder', 'curtain', 'furniture', 'furniture_ladder', 'gator', 'gator_ladder'
]:
    split = data[data['Dataset'] == wall]
    precision, recall, f1_score = get_results(split)
    loc_error = get_loc_error(split)
    results.append(
        (wall, precision, recall, f1_score, loc_error)
    )

results = pd.DataFrame(results, columns=['Dataset', 'Precision', 'Recall', 'F1 Score', 'MLE'])
results.to_csv(f'{exp_name}_loc_wall_results.csv')
print(results)
