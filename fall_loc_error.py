import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from loc_matrix import fall_error_matrix, loc2idx, loc_class_names, fall_class_names


def eval_per_split(selection):
    loc_error = []
    for idx, row in selection.iterrows():
        pred_loc, gt_loc = row['pred_loc'], row['gt_loc']
        pred_loc, gt_loc = loc2idx[pred_loc], loc2idx[gt_loc]
        error = fall_error_matrix[gt_loc, pred_loc]
        loc_error.append(error)

    loc_error = np.array(loc_error)
    loc_error = loc_error.mean() * units
    return loc_error


def loc_per_split(selection, class_names):
    y_true, y_pred = selection['gt_loc'], selection['pred_loc']

    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=class_names)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=class_names)
    f1_score = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1_score


def detect_per_split(selection, class_names):
    y_true, y_pred = selection['gt_pose'], selection['pred_pose']

    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=class_names)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=class_names)
    f1_score = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1_score


units = 12 * 2.54 / 100  # meter
# exp_name = 'resnet_azi'
# exp_name = 'cvae_color_azi_gen64'
# exp_name = 'pix2pix_color_azi_gen256'
exp_name = 'rgbd_mse'
data = pd.read_csv(f'{exp_name}_fall_labels.csv')
print(data.shape)

results = []
prec_pose, rec_pose, f1_pose = detect_per_split(data, fall_class_names)
loc_error = eval_per_split(data)
prec_loc, rec_loc, f1_loc = loc_per_split(data, loc_class_names)
results.append(
    ('overall', prec_pose, rec_pose, f1_pose, loc_error, prec_loc, rec_loc, f1_loc)
)

for dataset in [
    'brick', 'brick_ladder', 'curtain', 'furniture', 'furniture_ladder', 'gator', 'gator_ladder'
]:
    selection = data[data['dataset'] == dataset]
    prec_pose, rec_pose, f1_pose = detect_per_split(selection, fall_class_names)
    loc_error = eval_per_split(selection)
    prec_loc, rec_loc, f1_loc = loc_per_split(selection, loc_class_names)
    # print(dataset, precision, recall, f1_score, loc_error)
    results.append(
        (dataset, prec_pose, rec_pose, f1_pose, loc_error, prec_loc, rec_loc, f1_loc)
    )

results = pd.DataFrame(results, columns=[
    'dataset', 'prec_pose', 'rec_pose', 'f1_pose', 'loc_error', 'prec_loc', 'rec_loc', 'f1_loc'
])
results.to_csv(f'{exp_name}_fall_results.csv', index=False)
print(exp_name, *results.iloc[0].values)
