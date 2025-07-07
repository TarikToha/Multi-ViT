import os.path
import time
from threading import Thread

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from camera_utility import *
from models.configs import get_b16_config
from models.modeling import SiVisionTransformer
from models.modeling_resnet import UniResNet
from param import get_class_names, generate_weapon_prompt, cls_transform, gen_transform
from pix2mm.pix2pix_turbo import Pix2Pix_Turbo
from project import project


def read_camera_thread(cam_data_dir, start_frame, end_frame, color_case):
    timestamp['color_start'] = time.time()
    color_case[0] = read_camera_frame(cam_data_dir, start_frame, end_frame)
    timestamp['color_end'] = time.time()


def read_depth_thread(cam_data_dir, start_frame, end_frame, raw_depth_case):
    timestamp['depth_start'] = time.time()
    raw_depth_case[0] = read_depth_frame(cam_data_dir, start_frame, end_frame)
    timestamp['depth_end'] = time.time()


def read_azi_fft_thread(frames_dir, start_frame, end_frame, azi_case):
    timestamp['azi_start'] = time.time()
    azi_inputs = []
    for frame_id in range(start_frame, end_frame):
        azi_frame = f'{frames_dir}azi_fft_{capture_id}_{frame_id}.jpg'
        azi_frame = Image.open(azi_frame)
        azi_frame = cls_transform(azi_frame).unsqueeze(dim=0).cuda()
        azi_inputs.append(azi_frame)

    azi_inputs = torch.cat(azi_inputs)
    azi_case[0] = azi_inputs
    timestamp['azi_end'] = time.time()


def load_camera_tensor(color_case):
    color_inputs = []
    for color_frame in color_case:
        color_frame = Image.fromarray(color_frame)
        color_frame = cls_transform(color_frame).unsqueeze(dim=0).cuda()
        color_inputs.append(color_frame)

    color_inputs = torch.cat(color_inputs)
    return color_inputs


def run_cloth_model(color_case):
    with torch.no_grad():
        out_val = cloth_model(color_case)
        out_val = torch.softmax(out_val, dim=1)
        out_val = out_val.detach().cpu().numpy().squeeze()

    out_val = np.argmax(out_val, axis=-1)
    out_val = np.bincount(out_val)
    out_val = np.argmax(out_val)
    out_val = cloth_class_names[out_val]
    return out_val


def run_env_model(color_case):
    with torch.no_grad():
        out_val = env_model(color_case)
        out_val = torch.softmax(out_val, dim=1)
        out_val = out_val.detach().cpu().numpy().squeeze()

    out_val = np.argmax(out_val, axis=-1)
    out_val = np.bincount(out_val)
    out_val = np.argmax(out_val)
    out_val = env_class_names[out_val]
    return out_val


def run_context_extractor(color_case, context_val):
    timestamp['context_start'] = time.time()
    color_tensor = load_camera_tensor(color_case)
    num_frames = color_tensor.shape[0]

    cloth_label = run_cloth_model(color_tensor)
    env_label = run_env_model(color_tensor)

    prompt = generate_weapon_prompt(cloth_label, env_label)
    context_val[0] = [prompt] * num_frames
    timestamp['context_end'] = time.time()


def project_py(point_cloud):
    rgbd_frame = np.ones((80, 80, 3), dtype=np.float32)
    xs, zs = point_cloud[:, 0].astype(np.int16), point_cloud[:, 2].astype(np.int16)
    cs = point_cloud[:, 3:6] / 255
    valid_pts = (np.abs(xs) < 40) & (zs < 80)
    xs, zs, cs = xs[valid_pts] + 40, zs[valid_pts], cs[valid_pts]
    rgbd_frame[zs, xs] = cs
    return rgbd_frame


def project_thread(point_cloud):
    # return project_py(point_cloud)
    return project(point_cloud)


def projection(cam_data_dir, color_data, depth_data):
    rgbd_data = np.concatenate((color_data, depth_data[..., None]), axis=-1)
    rgbd_data = rgbd_data.astype(np.float32)
    rgbd_data[:, :, :, 3] /= 42.1

    pc_frames = make_point_cloud(cam_data_dir, rgbd_data)
    # print('pc_frames.shape =', pc_frames.shape, time.time() - start_time,
    #       pc_frames.dtype, pc_frames.nbytes / 2 ** 20)

    frame_count = pc_frames.shape[0]
    sort_idx = np.argsort(pc_frames[0][:, 1])[::-1]
    rgbd_data = np.ones((frame_count, 80, 80, 3), dtype=np.float32)
    for frame_idx in range(frame_count):
        rgbd_data[frame_idx] = project_thread(pc_frames[frame_idx][sort_idx])

    rgbd_data = fast_normalize(rgbd_data, dtype=np.uint8, alpha=0, beta=255)
    return rgbd_data


def run_aligner(color_case, raw_depth_case):
    depth_case = align_depth_camera(cam_data_dir, raw_depth_case, cpp=True)
    rgbd_case = projection(cam_data_dir, color_case, depth_case)

    rgbd_inputs = []
    for rgbd_frame in rgbd_case:
        rgbd_frame = Image.fromarray(rgbd_frame)
        rgbd_frame = gen_transform(rgbd_frame).unsqueeze(dim=0).cuda()
        rgbd_inputs.append(rgbd_frame)

    rgbd_inputs = torch.cat(rgbd_inputs)
    return rgbd_inputs


def run_pix2mm_model(vision, text):
    with torch.no_grad():
        output = gen_model(vision, text)
        output = output * 0.5 + 0.5
        output *= 255
        output = torch.permute(output, (0, 2, 3, 1))
        output = output.cpu().numpy().astype(dtype=np.uint8)
        return output


def load_gen_tensor(gen_case):
    gen_inputs = []
    for gen_frame in gen_case:
        gen_frame = Image.fromarray(gen_frame)
        gen_frame = cls_transform(gen_frame).unsqueeze(dim=0).cuda()
        gen_inputs.append(gen_frame)

    gen_inputs = torch.cat(gen_inputs)
    return gen_inputs


def run_loc_model(gen, real):
    with torch.no_grad():
        out_val = loc_model(gen, real)
        out_val = torch.softmax(out_val, dim=1)
        out_val = out_val.cpu().numpy().squeeze()

    out_val = np.argmax(out_val, axis=-1)
    out_val = np.bincount(out_val)
    out_val = np.argmax(out_val)
    out_val = weapon_class_names[out_val]
    return out_val


start_time = time.time()
np.set_printoptions(suppress=True)
assert torch.cuda.is_available(), '!!!you are not using cuda!!!'

object_mapping = {
    'std_brick_object': ['br', 'mini_pc_std_obj_cascaded_dataset'],
    'std_t_object': ['nt', 'mini_pc_new_cascaded_dataset'],
    'old_t_object': ['ot', 'cascaded_dataset']
}

dataset_type = 'overall'
res_type = 'mse'
cloth_exp_name = 'clothes.resnet_color.v9.1'
cloth_ckpt_path = f'{cloth_exp_name}/{cloth_exp_name}_checkpoint.bin'

env_exp_name = 'environments.resnet_color.v9.1'
env_ckpt_path = f'{env_exp_name}/{env_exp_name}_checkpoint.bin'

gen_ckpt_path = f'{dataset_type}_kld.rgbd_azi.v9.1/checkpoints/model_28801.pkl'

loc_exp_name = f'{dataset_type}.rgbd_{res_type}.v9.1'
loc_ckpt_path = f'{loc_exp_name}/{loc_exp_name}_checkpoint.bin'
data_root = '/work/users/t/t/ttoha12/dataset/weapon/'

cloth_class_names, env_class_names = get_class_names('clothes'), get_class_names('environments')
weapon_class_names = get_class_names(dataset_type)
num_weapon_class = len(weapon_class_names)

cloth_model = UniResNet(num_classes=len(cloth_class_names))
cloth_model.load_state_dict(torch.load(cloth_ckpt_path, weights_only=True))
cloth_model.cuda().eval()
print('loaded cloth_model', time.time() - start_time)

env_model = UniResNet(num_classes=len(env_class_names))
env_model.load_state_dict(torch.load(env_ckpt_path, weights_only=True))
env_model.cuda().eval()
print('loaded env_model', time.time() - start_time)

gen_model = Pix2Pix_Turbo(pretrained_name='', pretrained_path=gen_ckpt_path)
gen_model.set_eval()
print('loaded gen_model', time.time() - start_time)

loc_model = SiVisionTransformer(
    get_b16_config(), res_type=res_type, num_classes=num_weapon_class
)
loc_model.load_state_dict(torch.load(loc_ckpt_path, weights_only=True))
loc_model.cuda().eval()
print('loaded loc_model', time.time() - start_time)

data = pd.read_csv(f'{dataset_type}_selection_{num_weapon_class}.csv')
data.query('Target_Dir == "test"', inplace=True)
print(data.shape)

label_pair, timestamp = [], {}
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    timestamp['start_time'] = time.time()
    capture_id, capture_dir, label = row['Capture_ID'], row['Dataset'], row['Label']
    prefix, obj_dir = object_mapping[capture_dir][0], object_mapping[capture_dir][1]

    cam_data_dir = f'{data_root}{obj_dir}/capture_{capture_id:05d}/realsense/'
    rad_data_dir = f'{data_root}{capture_dir}/{capture_id:05d}/frames/'
    assert os.path.exists(cam_data_dir), f'{cam_data_dir} not found'
    assert os.path.exists(rad_data_dir), f'{rad_data_dir} not found'

    start_frame, end_frame = row['Start_Frame'], row['End_Frame']

    color_data, raw_depth_data, azi_data = ['none'], ['none'], ['none']
    color_thread = Thread(
        target=read_camera_thread, args=(cam_data_dir, start_frame, end_frame, color_data)
    )
    color_thread.start()

    depth_thread = Thread(
        target=read_depth_thread, args=(cam_data_dir, start_frame, end_frame, raw_depth_data)
    )
    depth_thread.start()

    azi_thread = Thread(
        target=read_azi_fft_thread, args=(rad_data_dir, start_frame, end_frame, azi_data)
    )
    azi_thread.start()

    color_thread.join()
    color_data = color_data[0]
    assert isinstance(color_data, np.ndarray), 'color_data is none'

    context_prompts = ['none']
    context_thread = Thread(target=run_context_extractor, args=(color_data, context_prompts))
    context_thread.start()

    depth_thread.join()
    raw_depth_data = raw_depth_data[0]
    assert isinstance(raw_depth_data, np.ndarray), 'raw_depth_data is none'

    timestamp['aligner_start'] = time.time()
    rgbd_data = run_aligner(color_data, raw_depth_data)
    timestamp['aligner_end'] = time.time()
    del color_data, raw_depth_data

    context_thread.join()
    context_prompts = context_prompts[0]
    assert isinstance(context_prompts, list), 'context_prompt is none'

    timestamp['gen_start'] = time.time()
    gen_data = run_pix2mm_model(rgbd_data, context_prompts)
    timestamp['gen_end'] = time.time()
    del rgbd_data, context_prompts

    gen_data = load_gen_tensor(gen_data)
    azi_thread.join()
    azi_data = azi_data[0]
    assert isinstance(azi_data, torch.Tensor), 'azi_data is none'

    timestamp['loc_start'] = time.time()
    loc_label = run_loc_model(gen_data, azi_data)
    timestamp['loc_end'] = time.time()
    del gen_data, azi_data

    label_pair.append((
        capture_id, loc_label, label, prefix, timestamp['start_time'],
        timestamp['color_start'], timestamp['color_end'],
        timestamp['depth_start'], timestamp['depth_end'],
        timestamp['azi_start'], timestamp['azi_end'],
        timestamp['context_start'], timestamp['context_end'],
        timestamp['aligner_start'], timestamp['aligner_end'],
        timestamp['gen_start'], timestamp['gen_end'],
        timestamp['loc_start'], timestamp['loc_end'],
    ))
    timestamp.clear()

response_time_out = pd.DataFrame(label_pair, columns=[
    'Capture_ID', 'Prediction', 'Ground_Truth', 'Dataset', 'Start_Time',
    'Color_Start', 'Color_End', 'Depth_Start', 'Depth_End', 'Azi_Start', 'Azi_End',
    'Context_Start', 'Context_End', 'Aligner_Start', 'Aligner_End',
    'Generation_Start', 'Generation_End', 'Localization_Start', 'Localization_End'
])
response_time_out.to_csv(f'{dataset_type}_debug_time_out.csv', index=False)

label_pair = np.array(label_pair)
print(label_pair.shape, time.time() - start_time)

y_pred, y_true = label_pair[:, 1], label_pair[:, 2]
precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)

results = []
for idx, label in enumerate(weapon_class_names):
    results.append((label, precision[idx], recall[idx]))
    print(results[-1])
results.append(('Mean', precision.mean(), recall.mean()))
print(results[-1])
