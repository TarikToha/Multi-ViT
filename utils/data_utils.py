import glob
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


class BiModalDataset(VisionDataset):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

        mode1_root, mode2_root = f'{root}/mode1/', f'{root}/mode2/'
        mode1_data = glob.glob(f'{mode1_root}*/*.jpg')

        self.data, self.labels = [], []
        for mode1_path in mode1_data:
            mode1_file = mode1_path.split('/')
            label, file_name = mode1_file[-2], mode1_file[-1]
            mode2_path = f'{mode2_root}{label}/{file_name}'

            self.data.append((mode1_path, mode2_path))
            self.labels.append(label)

        class_names = list(set(self.labels))
        class_names.sort()
        self.cls_to_idx = {}
        for idx, cls in enumerate(class_names):
            self.cls_to_idx[cls] = idx

    def __getitem__(self, index):
        (mode1_img, mode2_img), label = self.data[index], self.labels[index]

        mode1_img, mode2_img = Image.open(mode1_img), Image.open(mode2_img)
        mode1_img, mode2_img = self.transform(mode1_img), self.transform(mode2_img)

        label = self.cls_to_idx[label]
        label = torch.tensor(label)

        return mode1_img, mode2_img, label

    def __len__(self):
        return len(self.data)


class MultiDataset(VisionDataset):
    def __init__(self, root, split_name, transform):
        super().__init__(root, transform=transform)

        labels_data = pd.read_csv(f'{root}/{split_name}_files.csv')

        self.data, self.labels1, self.labels2 = [], [], []
        for _, row in labels_data.iterrows():
            file_name, label1, label2 = row['file_name'], row['label_loc'], row['label_pose']

            mode1_path = f'{root}/mode1/{file_name}'
            mode2_path = f'{root}/mode2/{file_name}'

            self.data.append((mode1_path, mode2_path))
            self.labels1.append(label1), self.labels2.append(label2)

        class_names1, class_names2 = list(set(self.labels1)), list(set(self.labels2))
        class_names1.sort(), class_names2.sort()

        self.cls_to_idx1 = {}
        for idx, cls in enumerate(class_names1):
            self.cls_to_idx1[cls] = idx

        self.cls_to_idx2 = {}
        for idx, cls in enumerate(class_names2):
            self.cls_to_idx2[cls] = idx

    def __getitem__(self, index):
        mode1_img, mode2_img = self.data[index]
        mode1_img, mode2_img = Image.open(mode1_img), Image.open(mode2_img)
        mode1_img, mode2_img = self.transform(mode1_img), self.transform(mode2_img)

        label1, label2 = self.labels1[index], self.labels2[index]
        label1, label2 = self.cls_to_idx1[label1], self.cls_to_idx2[label2]
        label1, label2 = torch.tensor(label1), torch.tensor(label2)

        return mode1_img, mode2_img, label1, label2

    def __len__(self):
        return len(self.data)


class UniModalDataset(VisionDataset):

    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

        mode1_root = f'{root}/'
        mode1_data = glob.glob(f'{mode1_root}*/*.jpg')

        self.data, self.labels = [], []
        for mode1_path in mode1_data:
            label = mode1_path.split('/')[-2]
            self.data.append(mode1_path)
            self.labels.append(label)

        class_names = list(set(self.labels))
        class_names.sort()
        self.cls_to_idx = {}
        for idx, cls in enumerate(class_names):
            self.cls_to_idx[cls] = idx

    def __getitem__(self, index):
        mode1_img, label = self.data[index], self.labels[index]

        mode1_img = Image.open(mode1_img)
        mode1_img = self.transform(mode1_img)

        label = self.cls_to_idx[label]
        label = torch.tensor(label)

        return mode1_img, label

    def __len__(self):
        return len(self.data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loader(args):
    set_seed(0)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            (args.img_size, args.img_size), scale=(0.08, 1.0),
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandAugment(0, 9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.0)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if args.dataset.startswith('bi_') or args.dataset.startswith('si_'):
        train_set = BiModalDataset(root=f'data/{args.dataset}/train', transform=train_transform)
        val_set = BiModalDataset(root=f'data/{args.dataset}/val', transform=val_transform)

    elif args.dataset.startswith('multi_'):
        train_set = MultiDataset(
            root=f'data/{args.dataset}/train', split_name='train', transform=train_transform
        )
        val_set = MultiDataset(
            root=f'data/{args.dataset}/val', split_name='val', transform=val_transform
        )

    else:
        train_set = UniModalDataset(root=f'data/{args.dataset}/train', transform=train_transform)
        val_set = UniModalDataset(root=f'data/{args.dataset}/val', transform=val_transform)

    assert len(train_set) > 0, 'training dataset cannot be empty'
    assert len(val_set) > 0, 'validation dataset cannot be empty'

    # train_sampler = RandomSampler(train_set)
    # val_sampler = SequentialSampler(val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
