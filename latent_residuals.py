import os
import time
from functools import partial
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from kornia.filters import gaussian_blur2d
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
from tqdm import tqdm

from models.configs import get_b16_config
from models.modeling import VisionTransformer

extracted_features = {}


def hook_fn(module, input, output, layer_idx):
    extracted_features[layer_idx] = output[0].detach()  # Save feature map


def save_thread(image_data, out_file):
    image_data = cv2.normalize(
        image_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    image_data = cv2.applyColorMap(image_data, cv2.COLORMAP_JET)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    image_data = Image.fromarray(image_data)
    image_data = image_data.resize(size=(1386, 1386), resample=Image.Resampling.LANCZOS)
    image_data.save(out_file, dpi=(300, 300))


def concat(tensor1, tensor2):
    tensor1 = tensor1.mean(dim=-1).view(-1, 56, 14)
    tensor2 = tensor2.mean(dim=-1).view(-1, 56, 14)
    return torch.cat((tensor1, tensor2), dim=-1)


def cosine(tensor1, tensor2, sigma=6):
    cos_sim = torch.cosine_similarity(tensor1, tensor2, dim=-1)
    anomaly_map = 1 - cos_sim
    anomaly_map = anomaly_map.view(-1, 4, 14, 14)
    anomaly_map = torch.sum(anomaly_map, dim=1)
    anomaly_map = torch.repeat_interleave(anomaly_map, repeats=16, dim=1)
    anomaly_map = torch.repeat_interleave(anomaly_map, repeats=16, dim=2)

    anomaly_map = anomaly_map.unsqueeze(1)
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    anomaly_map = gaussian_blur2d(
        anomaly_map, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma)
    )
    anomaly_map = anomaly_map.squeeze()
    return anomaly_map


def mae_mse_residuals(loss, tensor1, tensor2):
    return loss(tensor1, tensor2)


start_time = time.time()
np.set_printoptions(suppress=True)
object_sym = {
    'std_brick_object': 'br',
    'std_t_object': 'nt',
    'old_t_object': 'ot'
}

save = True
agg_type = 'res'
res_type = 'cosine'
dataset_type = 'case'
exp_name = f'{dataset_type}.azi.v9.1'
data_dir = f'si_{dataset_type}_rgbd_azi_gen_256.v9.1-azi_fft.v9.1'
ckpt_path = f'{exp_name}/{exp_name}_checkpoint.bin'

if res_type == 'mae':
    residuals = partial(mae_mse_residuals, L1Loss(reduction='none'))
elif res_type == 'mse':
    residuals = partial(mae_mse_residuals, MSELoss(reduction='none'))
elif res_type == 'concat':
    residuals = concat
elif res_type == 'cosine':
    residuals = cosine
else:
    print(f'{res_type} is not defined')
    exit()

print(f'you are using {agg_type}_{res_type}')

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

model = VisionTransformer(get_b16_config(), num_classes=9)
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt)
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
for data_idx, row in tqdm(data.iterrows(), total=data.shape[0]):
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
        patch_gen = []
        for layer, feature in extracted_features.items():
            patch_gen.append(feature[:, 1:])
        extracted_features.clear()

        model(mode2_in)
        patch_real = []
        for layer, feature in extracted_features.items():
            patch_real.append(feature[:, 1:])
        extracted_features.clear()

    patch_real, patch_gen = torch.cat(patch_real, dim=1), torch.cat(patch_gen, dim=1)
    out_data = residuals(patch_real, patch_gen)
    out_data = out_data.cpu().numpy()

    if save:
        out_dir = f'data/uni_{dataset_type}_{agg_type}_{res_type}.v9.1/{target_dir}/{label}/'
        os.makedirs(out_dir, exist_ok=True)

        if agg_type == 'agg':
            out_file = f'{out_dir}{prefix}_{capture_id}.jpg'
            save_thread(image_data=out_data, out_file=out_file)

        else:
            thread_list = []
            for frame_id, out_img in zip(range(start_frame, end_frame), out_data):
                out_file = f'{out_dir}{prefix}_{capture_id}_{frame_id}.jpg'
                # save_thread(image_data=out_img, out_file=out_file)
                t = Thread(target=save_thread, args=(out_img, out_file))
                t.start()
                thread_list.append(t)

            for t in thread_list:
                t.join()

    else:
        print(out_data.shape)
        plt.imshow(out_data[0], cmap='jet')
        plt.colorbar()
        plt.show()
        break
