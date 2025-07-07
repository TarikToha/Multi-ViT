import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score
from torchvision.transforms import transforms
from tqdm import tqdm

from models.configs import get_b16_config
from models.modeling import SiVisionTransformer
from visual import save_visual

start_time = time.time()
np.set_printoptions(suppress=True)
assert torch.cuda.is_available(), '!!!you are not using cuda!!!'

object_sym = {
    'std_brick_object': 'br',
    'std_t_object': 'nt',
    'old_t_object': 'ot'
}

dataset_type = 'case'
exp_name = f'{dataset_type}.si_rgbd_azi_gen256.v9.1.3'
data_dir = f'si_{dataset_type}_rgbd_azi_gen_256.v9.1-azi_fft.v9.1'
res_type = 'mae'
ckpt_path = f'{exp_name}/{exp_name}_checkpoint.bin'

class_names = np.array(
    ['left_ankle', 'left_chest', 'left_pocket', 'left_waist', 'no_weapon',
     'right_ankle', 'right_chest', 'right_pocket', 'right_waist']
)
num_class = len(class_names)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

model = SiVisionTransformer(get_b16_config(), res_type=res_type, num_classes=num_class)
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt)
model.cuda().eval()
print('loaded model', time.time() - start_time)

data = pd.read_csv(f'{dataset_type}_selection_{num_class}.csv')
data.query('Target_Dir == "test"', inplace=True)

label_pair = []
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
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
        out_val = model(mode1_in, mode2_in)
        # print('predicted', time.time() - start_time)
        class_prob = torch.softmax(out_val, dim=1)
        class_prob = class_prob.detach().cpu().numpy().squeeze()

    max_prob = class_prob.mean(axis=0)
    out_dir = f'viz/{data_dir}/{target_dir}/{label}/'
    os.makedirs(out_dir, exist_ok=True)
    save_visual(max_prob, out_path=f'{out_dir}{prefix}_{capture_id}.jpg', save=True)

    class_idx = np.argmax(class_prob, axis=-1)
    class_idx = np.bincount(class_idx)
    class_idx = np.argmax(class_idx)
    label_pair.append((capture_id, class_names[class_idx], label, prefix))
    # print(capture_id, class_names[class_idx], label)

out_labels = pd.DataFrame(label_pair, columns=['Capture ID', 'Prediction',
                                               'Ground Truth', 'Dataset'])
out_labels.to_csv(f'{exp_name}/{exp_name}_labels_out.csv', index=False)

label_pair = np.array(label_pair)
print(label_pair.shape, time.time() - start_time)

y_pred, y_true = label_pair[:, 1], label_pair[:, 2]
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=class_names)
cm.plot(xticks_rotation='vertical', colorbar=False)
plt.tight_layout()
plt.savefig(f'{exp_name}/{exp_name}_cm.png', dpi=300, bbox_inches='tight')

precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)

results = []
for idx, label in enumerate(class_names):
    results.append((label, precision[idx], recall[idx]))
    print(results[-1])
results.append(('Mean', precision.mean(), recall.mean()))
print(results[-1])

results = pd.DataFrame(results, columns=['Label', 'Precision', 'Recall'])
results.to_csv(f'{exp_name}/{exp_name}_results.csv')
print(exp_name, time.time() - start_time)
