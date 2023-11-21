import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mars_weight', default='./weights/kinetics-pretrained.pth', type=str)

    # dataset parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='vimeo', choices=['vimeo', 'vimeo-triplet'])
    parser.add_argument('--num_workers', default=16, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[4, 3, 256, 448], type=int,nargs='*')
    parser.add_argument('--out_frame', default=3, type=int)
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=128, type=int)
    parser.add_argument('--N_S', default=1, type=int)
    parser.add_argument('--N_T', default=3, type=int)
    parser.add_argument('--groups', default=4, type=int) 
    parser.add_argument('--perceptual', default=False, type=bool)
    parser.add_argument('--vgg_loss', default=False, type=bool)

    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lamb', default=10, type=float, help='perceptual loss lambda')
    parser.add_argument('--vgg_lamb', default=0.01, type=float, help='perceptual loss lambda')

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
