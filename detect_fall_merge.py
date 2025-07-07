from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from loc_matrix import pose2idx, loc2idx, idx2loc, idx2pose, loc_class_names, fall_class_names


def categorical_entropy(labels):
    counts = Counter(labels)
    probs = np.array(list(counts.values())) / len(labels)
    return entropy(probs)


def get_frequent_label(labels, idx2label):
    labels = np.bincount(labels)
    labels = np.argmax(labels)
    pred_label = idx2label[labels]
    pred_label = merge_labels(pred_label)
    return pred_label


def merge_labels(label):
    if label in ['standing', 'sitting']:
        label = 'normal'

    return label


def split_label(label):
    label = label.split('_')
    pose, loc = label[0], label[1]
    if len(label) > 2:
        loc = f'{loc}_{label[2]}'
    return pose, loc


def separate_pose_loc(dataset):
    data = []
    for _, row in dataset.iterrows():
        capture_dir, capture_id, frame_id = row['Dataset'], row['Capture_ID'], row['Frame_ID']
        pred_pose, pred_loc = split_label(row['Prediction'])
        gt_pose, gt_loc = split_label(row['Ground_Truth'])

        data.append(
            (capture_dir, capture_id, frame_id, pred_loc, gt_loc, pred_pose, gt_pose)
        )

    data = pd.DataFrame(data, columns=[
        'Dataset', 'Capture_ID', 'Frame_ID', 'Prediction_Loc',
        'Ground_Truth_Loc', 'Prediction_Pose', 'Ground_Truth_Pose'
    ])
    return data


def aggregate_frames(labels):
    labels = np.reshape(labels, (-1, 6))
    out = []
    for row in labels:
        label = np.bincount(row)
        label = np.argmax(label)
        out.append(label)
    out = np.array(out)
    return out


def get_label_pair(data_split):
    cases = data_split['Capture_ID'].unique()
    cases.sort()

    labels = []
    for capture_id in cases:
        selection = data_split[data_split['Capture_ID'] == capture_id]
        dataset = selection['Dataset'].iloc[0]

        gt_pose = selection['Ground_Truth_Pose'].iloc[0]
        gt_pose = merge_labels(gt_pose)

        gt_loc = selection['Ground_Truth_Loc'].iloc[0]
        gt_loc = merge_labels(gt_loc)

        poses, loc = [], []
        for frame_id in range(60):
            row = selection[selection['Frame_ID'] == frame_id]
            if len(row) == 0:
                continue
            pred_pose, pred_loc = row['Prediction_Pose'].iloc[0], row['Prediction_Loc'].iloc[0]
            pred_pose, pred_loc = pose2idx[pred_pose], loc2idx[pred_loc]
            poses.append(pred_pose), loc.append(pred_loc)

        poses, loc = np.array(poses), np.array(loc)
        if exp_name != 'rgbd_mse':
            poses, loc = aggregate_frames(poses), aggregate_frames(loc)
            ent_poses, ent_loc = categorical_entropy(poses), 0

        else:
            ent_poses, ent_loc = categorical_entropy(poses), categorical_entropy(loc)

        if ent_poses > 0.5 or ent_loc > 0:
            pred_pose = 'fall'
        else:
            pred_pose = get_frequent_label(poses, idx2pose)

        pred_loc = get_frequent_label(loc, idx2loc)

        labels.append((dataset, capture_id, pred_pose, gt_pose, pred_loc, gt_loc))
        # print(labels[-1], ent_poses, ent_loc)

    labels = np.array(labels)
    return labels


def get_results(y_true, y_pred, class_names, exp_name):
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=None, labels=class_names)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, labels=class_names)

    results = []
    for idx, label in enumerate(class_names):
        results.append((label, precision[idx], recall[idx]))
    results.append(('Mean', precision.mean(), recall.mean()))
    results = pd.DataFrame(results, columns=['Label', 'Precision', 'Recall'])
    results.to_csv(f'{exp_name}_results.csv')

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cm.plot(xticks_rotation='vertical', colorbar=False)
    plt.tight_layout()
    plt.savefig(f'{exp_name}_cm.png', dpi=300, bbox_inches='tight')

    return results


# exp_name = 'resnet_azi'
# exp_name = 'cvae_color_azi_gen64'
# exp_name = 'pix2pix_color_azi_gen256'
exp_name = 'rgbd_mse'
pose_loc2d_exp = f'pose_loc2d.{exp_name}.v13.1'
pose2d_data = pd.read_csv(f'{pose_loc2d_exp}/{pose_loc2d_exp}_frame_labels_out.csv')
multi_data = pd.read_csv(f'{pose_loc2d_exp}/{pose_loc2d_exp}_multi_frame_labels_out.csv')

pose2d_data = separate_pose_loc(pose2d_data)
data = pd.concat([pose2d_data, multi_data])
data['Ground_Truth_Pose'] = data['Ground_Truth_Pose'].replace(to_replace='slow-fall', value='fall')
data['Ground_Truth_Pose'] = data['Ground_Truth_Pose'].replace(to_replace='fast-fall', value='fall')
data.sort_values(by=['Capture_ID', 'Frame_ID'], inplace=True)
# data.query(f"Ground_Truth_Pose in ['blank', 'standing', 'sitting']", inplace=True)
print(data.shape)

label_pair = get_label_pair(data)
out_labels = pd.DataFrame(
    label_pair, columns=['dataset', 'capture_id', 'pred_pose', 'gt_pose', 'pred_loc', 'gt_loc']
)
out_labels.to_csv(f'{exp_name}_fall_labels.csv', index=False)
print(label_pair.shape)

pred, gt = label_pair[:, 2], label_pair[:, 3]
fall_results = get_results(y_pred=pred, y_true=gt, class_names=fall_class_names, exp_name='fall')
# for _, row in fall_results.iterrows():
fall_results = fall_results.iloc[-1].values
print('fall_results', fall_results[1], '\t', fall_results[2])

pred, gt = label_pair[:, 4], label_pair[:, 5]
loc_results = get_results(y_pred=pred, y_true=gt, class_names=loc_class_names, exp_name='loc')
# for _, row in loc_results.iterrows():
loc_results = loc_results.iloc[-1].values
print('loc_results', loc_results[1], '\t', loc_results[2])

print(exp_name)
