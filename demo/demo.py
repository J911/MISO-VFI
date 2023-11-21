import sys
[sys.path.append(i) for i in ['.', '..']]

import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

from models.model import MISO 

def getVideoList(root='./sample-videos'):
    videoPaths = glob(os.path.join(root, '*.mp4'))
    videoPaths.sort()

    return videoPaths

def getFrameByVideoPath(path, size=(448, 256)):

    frames = []
    cap  = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        frame = cv2.resize(frame, size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.asarray(frames)

    return frames, fps


if __name__ == '__main__':
    size = (448, 256)
    videoPaths = getVideoList('./sample-videos')
    output = './out'

    model = MISO((2, 3, 256, 448), hid_S=64, hid_T=128, N_S=1, N_T=3, groups=4)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load('../weights/checkpoint_ema.pth'))

    for path in videoPaths:
        video_name = path.split('/')[-1]
        output_dir = os.path.join(output, video_name)
        os.makedirs(output_dir, exist_ok=True)

        frames, fps = getFrameByVideoPath(path, size)

        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        vout = cv2.VideoWriter(os.path.join(output_dir, 'out.mp4'), fcc, fps, size)

        if len(frames) % 2 != 0:
            frames = frames[:-1 * (len(frames) % 2)]
        frames = np.reshape(frames, (-1, size[1], size[0], 3))
        frames = [frames[i:i+2] for i in range(0, len(frames), 1) if len(frames[i:i+2]) == 2]
        frames = np.stack(frames, axis=0)
        for i, inout in enumerate(tqdm(frames)):
            inp = np.concatenate(([inout[0]], [inout[1]]), axis=0)
            inp = [transforms.ToTensor()(Image.fromarray(im)).unsqueeze(0) for im in inp]
            inp = torch.cat([*inp], 0).unsqueeze(0)
            gt = [inout[1]]
            gt = [transforms.ToTensor()(Image.fromarray(im)).unsqueeze(0) for im in gt]
            gt = torch.cat([*gt], 0).unsqueeze(0)

            for idx, giv in enumerate(inp[0]):
                if idx >= 1: break
                frame = torch.permute(giv, (1, 2, 0))
                frame = frame.numpy()
                frame = frame * 255
                im = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                vout.write(im.astype(np.uint8))

            for _time in range(1):
                time = torch.tensor(_time*100).repeat(inp.shape[0]).cuda()
                pred_y = model(inp.cuda(), time)

                pred_y = torch.permute(pred_y, (0, 2, 3, 1))
                pred_y = pred_y.detach().cpu().numpy()[0]
                pred_y = pred_y * 255
                pred_y[pred_y>255] = 255
                pred_y[pred_y<0] = 0

                im = cv2.cvtColor(pred_y, cv2.COLOR_RGB2BGR)
                im = im.astype(np.uint8)
                vout.write(im)
                
        vout.release()