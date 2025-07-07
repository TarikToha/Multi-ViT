import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from kornia.filters import gaussian_blur2d
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, precision_score, \
    recall_score
from torch.nn.functional import interpolate
from torchvision import transforms
from tqdm import tqdm

from models.configs import get_b16_config
from models.modeling import VisionTransformer


def hook_fn(module, input, output, layer_idx):
    extracted_features[layer_idx] = output[0].detach()  # Save feature map


def normalizer(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


start_time = time.time()
np.set_printoptions(suppress=True)
extracted_features = {}
object_sym = {
    'std_brick_object': 'br',
    'std_t_object': 'nt',
    'old_t_object': 'ot'
}

dataset_type = 'case'
exp_name = f'{dataset_type}.azi.v9.1'
data_dir = f'si_{dataset_type}_rgbd_azi_gen_256.v9.1-azi_fft.v9.1'
ckpt_path = f'{exp_name}/{exp_name}_checkpoint.bin'

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

model = VisionTransformer(get_b16_config(), num_classes=9)
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt)
# print(model)
model = model.transformer
model.cuda().eval()
print('loaded model', time.time() - start_time)

layers_to_extract = [3, 6, 9, 11]
hooks = []
for layer_idx in layers_to_extract:
    hook = model.encoder.layer[layer_idx].register_forward_hook(
        lambda module, input, output, idx=layer_idx: hook_fn(module, input, output, idx)
    )
    hooks.append(hook)

data = pd.read_csv(f'{dataset_type}_selection_9.csv')

scores, labels = [], []
y_pred, y_true = [], []
for data_idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    # if data_idx > 100:
    #     break
    capture_id, target_dir, label = row['Capture_ID'], row['Target_Dir'], row['Label']

    mode1_in_path = f'data/{data_dir}/{target_dir}/mode1/{label}/'
    mode2_in_path = f'data/{data_dir}/{target_dir}/mode2/{label}/'

    capture_dir = row['Dataset']
    prefix = f'{object_sym[capture_dir]}'
    start_frame, end_frame = row['Start_Frame'], row['End_Frame']

    mode1_in, mode2_in = [], []
    for frame_id in range(start_frame, end_frame):
        mode1_img = f'{mode1_in_path}{prefix}_{capture_id}_{frame_id}.jpg'
        mode1_img = Image.open(mode1_img)
        mode1_img = image_transform(mode1_img).unsqueeze(dim=0).cuda()
        mode1_in.append(mode1_img)

        mode2_img = f'{mode2_in_path}{prefix}_{capture_id}_{frame_id}.jpg'
        mode2_img = Image.open(mode2_img)
        mode2_img = image_transform(mode2_img).unsqueeze(dim=0).cuda()
        mode2_in.append(mode2_img)

    mode1_in, mode2_in = torch.cat(mode1_in), torch.cat(mode2_in)

    with torch.no_grad():
        model(mode1_in)
        patch_tokens_r = []
        for layer, feature in extracted_features.items():
            patch_tokens_r.append(feature)
        extracted_features.clear()

        model(mode2_in)
        patch_tokens_i = []
        for layer, feature in extracted_features.items():
            patch_tokens_i.append(feature)

    sigma = 6
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    b, n, c = patch_tokens_i[0][:, 1:, :].shape
    h = int(n ** 0.5)
    anomaly_maps1 = torch.zeros((b, 1, 512, 512)).cuda()
    for idx in range(len(patch_tokens_i)):
        pi = patch_tokens_i[idx][:, 1:, :]
        pr = patch_tokens_r[idx][:, 1:, :]

        pi = pi / torch.norm(pi, p=2, dim=-1, keepdim=True)
        pr = pr / torch.norm(pr, p=2, dim=-1, keepdim=True)

        cos0 = torch.bmm(pi, pr.permute(0, 2, 1))

        anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
        anomaly_map1 = interpolate(anomaly_map1.reshape(-1, 1, h, h), size=512,
                                   mode='bilinear', align_corners=True)
        anomaly_maps1 += anomaly_map1

    anomaly_maps1 = gaussian_blur2d(anomaly_maps1, kernel_size=(kernel_size, kernel_size),
                                    sigma=(sigma, sigma))[:, 0]
    score = torch.topk(torch.flatten(anomaly_maps1, start_dim=1), 250)[0].mean(dim=1)
    label_idx = 0 if label == 'no_weapon' else 1

    scores.extend([s for s in score.cpu().numpy()])
    labels.extend([label_idx] * b)

    pred = np.array(
        [s for s in score.cpu().numpy()]
    ).mean()

    y_pred.append(pred)
    y_true.append(label_idx)

scores = normalizer(np.array(scores))
labels = np.array(labels)

bin_labels = [0, 1]
y_pred = normalizer(np.array(y_pred))
y_pred = np.round(y_pred).astype(dtype=int)
y_true = np.array(y_true)

precision = precision_score(y_true=y_true, y_pred=y_pred, average=None, labels=bin_labels)
recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, labels=bin_labels)
for label, pre, rec in zip(bin_labels, precision, recall):
    if label == 0:
        continue
    print(label, 'precision =', pre, 'recall =', rec)

print('precision_recall_curve')
precisions_image, recalls_image, _ = precision_recall_curve(labels, scores)
f1_scores_image = (2 * precisions_image * recalls_image) / (precisions_image + recalls_image)
best_f1_scores_image = np.max(f1_scores_image[np.isfinite(f1_scores_image)])

print('roc_auc_score')
auroc_image = roc_auc_score(labels, scores)

print('average_precision_score')
AP_image = average_precision_score(labels, scores)

print(f'I-AUROC/I-AP : {round(auroc_image, 4)}/{round(AP_image, 4)}')
