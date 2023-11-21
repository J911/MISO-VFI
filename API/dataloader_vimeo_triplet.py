import imageio
import numpy as np
import os
from PIL import Image
import random

import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

class VIMEO(data.Dataset):
    def __init__(self, root, is_train=False, n_frames_input=10, n_frames_output=5):
        super(VIMEO, self).__init__()

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.root = root
        self.sequence_list = []
        self.target_list = []
        if self.is_train:
            with open(os.path.join(self.root, 'sep_trainlist.txt'), 'r') as txt:
                for line in txt:
                    self.sequence_list.append(os.path.join(self.root, 'sequences', line.strip()))
        else:
            with open(os.path.join(self.root, 'tri_testlist.txt'), 'r') as txt:
                for line in txt:
                    self.sequence_list.append(os.path.join(self.root, 'input', line.strip()))
                    self.target_list.append(os.path.join(self.root, 'target', line.strip()))


    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        frames = []
        for img_idx in range(1, 4):
            if img_idx == 2:
                _img = cv2.cvtColor(cv2.imread(os.path.join(self.target_list[idx], f'im{str(img_idx)}.png')), cv2.COLOR_BGR2RGB)
            else:
                _img = cv2.cvtColor(cv2.imread(os.path.join(self.sequence_list[idx], f'im{str(img_idx)}.png')), cv2.COLOR_BGR2RGB)
            _img = Image.fromarray(_img)
            _img = transforms.ToTensor()(_img).unsqueeze(0)
            frames.append(_img)
        frames = torch.cat([*frames], 0)
        inp = frames[:2]
        out = frames[2:]
        return inp, out

def load_data(
        batch_size, test_batch_size,
        data_root, num_workers):


    #train_set = VIMEO(root=data_root, is_train=True, n_frames_input=10, n_frames_output=5)
    test_set = VIMEO(root=data_root, is_train=False, n_frames_input=10, n_frames_output=5)

    #dataloader_train = torch.utils.data.DataLoader(
    #    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 255
    #return dataloader_train, dataloader_test, mean, std
    return True, dataloader_test, mean, std