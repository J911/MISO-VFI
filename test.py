import torch
import logging
import numpy as np
from models.model import MISO
from tqdm import tqdm
from API import *
from utils import *

from models.resnext import resnet101, ResNeXtBottleneck
from torchvision import models
from torch import nn
import argparse
import os


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--weight', default='./weights/vimeo-pretrained.pth', type=str)

    # dataset parameters
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='vimeo', choices=['vimeo', 'vimeo-triplet'])
    parser.add_argument('--num_workers', default=16, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[4, 3, 256, 448], type=int, nargs='*')
    parser.add_argument('--out_frame', default=3, type=int)
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=128, type=int)
    parser.add_argument('--N_S', default=1, type=int)
    parser.add_argument('--N_T', default=3, type=int)
    parser.add_argument('--groups', default=4, type=int)

    return parser

def test(model, device, test_loader, data_mean, data_std, args):
    model.eval()
    preds_lst, trues_lst, total_loss = [], [], []
    test_pbar = tqdm(test_loader)
    indicator = (args.in_shape[0]//2)
    for i, (batch_x, batch_y) in enumerate(test_pbar):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_in = torch.cat([batch_x[:,:indicator,:,:,:], batch_y], dim=1)
        batch_out = batch_x[:,indicator:,:,:,:]
        pred_list = []
        for time in range(args.out_frame):
            time = torch.tensor(time*100).repeat(batch_x.shape[0]).cuda()
            pred_y = model(batch_in, time)
            pred_list.append(pred_y.unsqueeze(1).detach().cpu())

        pred_y = torch.cat(pred_list, dim=1).detach().cpu()
        batch_out = batch_out.detach().cpu()

        list(map(lambda data, lst: lst.append(data.numpy()), [
             pred_y, batch_out], [preds_lst, trues_lst]))

    preds = np.concatenate(preds_lst, axis=0)
    trues = np.concatenate(trues_lst, axis=0)

    _, _, ssim, psnr = metric(preds, trues,  data_mean, data_std, True)
    print_log('test ssim:{:.4f}, psnr:{:.4f}'.format(ssim, psnr))

def build_model(args):
    model = MISO(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weight))
    model.to(args.device)
    return model

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_loader, test_loader, data_mean, data_std = load_data(**config)
    model = build_model(args)
    torch.backends.cudnn.benchmark = False
    test(model, args.device, test_loader, data_mean, data_std, args)
