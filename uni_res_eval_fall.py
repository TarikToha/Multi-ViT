import sys
import time

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from models.modeling_resnet import UniResNet
from param import *

start_time = time.time()
np.set_printoptions(suppress=True)
assert torch.cuda.is_available(), '!!!you are not using cuda!!!'

data_dir = 'multi_fall_rgbd_azi_gen_256.v2.1-azi_fft.v2.1'

exp_name = f'pose_loc2d.{sys.argv[1]}.v13.1'
ckpt_path = f'{exp_name}/{exp_name}_checkpoint.bin'

class_names = get_class_names('pose_loc2d')
num_class = len(class_names)

model = UniResNet(num_classes=num_class)
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.cuda().eval()
print('loaded model', time.time() - start_time)

data = pd.read_csv(f'fall_selection_2.csv')
data.query('Target_Dir == "test"', inplace=True)
# print(data)

label_pair = []
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    capture_dir, capture_id = row['Dataset'], row['Capture_ID']
    label_loc, label_pose = row['Label_loc'], row['Label_pose']
    start_frame, end_frame = row['Start_Frame'], row['End_Frame']

    mode2_in_path = f'data/{data_dir}/test/mode2/'

    mode2_in = []
    for frame_id in range(start_frame, end_frame):
        mode2_img = f'{mode2_in_path}{capture_id}_{frame_id}.jpg'
        mode2_img = Image.open(mode2_img)
        mode2_img = image_transform(mode2_img).unsqueeze(dim=0).cuda()
        mode2_in.append(mode2_img)

    mode2_in = torch.cat(mode2_in)

    with torch.no_grad():
        out_val = model(mode2_in)
        # print('predicted', time.time() - start_time)
        out_prob = torch.softmax(out_val, dim=1)
        out_prob = out_prob.detach().cpu().numpy().squeeze()

    out_idx = np.argmax(out_prob, axis=-1)
    out_idx = class_names[out_idx]

    for frame_id, pred_label in zip(range(start_frame, end_frame), out_idx):
        pred_label = pred_label.split('_')
        pred_pose, pred_loc = pred_label[0], pred_label[1]
        if len(pred_label) > 2:
            pred_loc = f'{pred_loc}_{pred_label[2]}'

        label_pair.append(
            (capture_dir, capture_id, frame_id, pred_loc, label_loc, pred_pose, label_pose)
        )
        # print(label_pair[-1])

out_labels = pd.DataFrame(label_pair, columns=[
    'Dataset', 'Capture_ID', 'Frame_ID', 'Prediction_Loc', 'Ground_Truth_Loc',
    'Prediction_Pose', 'Ground_Truth_Pose'
])
out_labels.to_csv(
    f'{exp_name}/{exp_name}_multi_frame_labels_out.csv', index=False
)
print(exp_name)
