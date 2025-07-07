import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score
from tqdm import tqdm

from models.modeling_resnet import BiResNet
from param import get_prefix, image_transform, get_class_names

start_time = time.time()
np.set_printoptions(suppress=True)
assert torch.cuda.is_available(), '!!!you are not using cuda!!!'

dataset_type = sys.argv[1]
exp_name = f'{dataset_type}.{sys.argv[2]}'
data_dir = f'bi_res_{dataset_type}_{sys.argv[3]}'
ckpt_path = f'{exp_name}/{exp_name}_checkpoint.bin'

class_names = get_class_names(dataset_type)
num_class = len(class_names)

model = BiResNet(num_classes=num_class)
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt)
model.cuda().eval()
print('loaded model', time.time() - start_time)

data = pd.read_csv(f'{dataset_type}_selection_{num_class}.csv')
data.query('Target_Dir == "test"', inplace=True)

label_pair = []
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    start_time = time.time()
    capture_id, target_dir, label = row['Capture_ID'], row['Target_Dir'], row['Label']

    mode1_in_path = f'data/{data_dir}/{target_dir}/mode1/{label}/'
    mode2_in_path = f'data/{data_dir}/{target_dir}/mode2/{label}/'

    capture_dir = row['Dataset']
    prefix = get_prefix(capture_dir)
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
        out_val = model(mode1_in, mode2_in)
        # print('predicted', time.time() - start_time)
        class_prob = torch.softmax(out_val, dim=1)
        class_prob = class_prob.detach().cpu().numpy().squeeze()

    class_idx = np.argmax(class_prob, axis=-1)
    class_idx = np.bincount(class_idx)
    class_idx = np.argmax(class_idx)

    dataset_split = capture_dir if prefix == '' else prefix
    duration = time.time() - start_time
    label_pair.append((dataset_split, capture_id, class_names[class_idx], label, duration))
    # print(capture_id, class_names[class_idx], label)

out_labels = pd.DataFrame(label_pair, columns=[
    'Dataset', 'Capture_ID', 'Prediction', 'Ground Truth', 'Duration'
])
out_labels.to_csv(f'{exp_name}/{exp_name}_labels_out.csv', index=False)

label_pair = np.array(label_pair)
print(label_pair.shape)

y_pred, y_true = label_pair[:, 1], label_pair[:, 2]
precision = precision_score(y_true=y_true, y_pred=y_pred, average=None, labels=class_names)
recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, labels=class_names)

results = []
for idx, label in enumerate(class_names):
    results.append((label, precision[idx], recall[idx]))
    print(results[-1])
results.append(('Mean', precision.mean(), recall.mean()))
print(results[-1])

results = pd.DataFrame(results, columns=['Label', 'Precision', 'Recall'])
results.to_csv(f'{exp_name}/{exp_name}_results.csv')

cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_names)
cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cm.plot(xticks_rotation='vertical', colorbar=False)
plt.tight_layout()
plt.savefig(f'{exp_name}/{exp_name}_cm.png', dpi=300, bbox_inches='tight')
print(exp_name)
