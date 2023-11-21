import copy
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from models.model import MISO
from tqdm import tqdm
from API import *
from utils import *

from models.resnext import resnet101, ResNeXtBottleneck
from torchvision import models
from torch import nn
torch.backends.cudnn.benchmark = False

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = 10
        self.step_start_ema = 2000
        self.step = 0
        self.p = [0.2, 0.2, 0.2, 0.2, 0.2]

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            device = 'cuda'
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = MISO(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        if self.args.vgg_loss:
            self.vgg = models.vgg16(pretrained=True).features[:9]
            self.vgg = nn.DataParallel(self.vgg)
            self.vgg.to(self.device)
            self.vgg.eval()
            self.vgg.requires_grad_(False)

        if self.args.perceptual:
            self.mars = resnet101(sample_size=112, sample_duration=16)
            self.mars.load_state_dict(torch.load(self.args.mars_weight))
            self.mars = nn.DataParallel(self.mars)
            self.mars.to(self.device)
            self.mars.eval()
            self.mars.requires_grad_(False)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), pct_start=0.0, epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.l2loss = torch.nn.MSELoss()
        self.l1loss = torch.nn.L1Loss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        torch.save(self.ema_model.state_dict(), os.path.join(
            self.checkpoints_path, name + '_ema.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        indicator = (self.args.in_shape[0]//2)
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_in = torch.cat([batch_x[:,:indicator,:,:,:], batch_y], dim=1)
                batch_out = batch_x[:,indicator:,:,:,:].clone()
                t = np.random.choice(self.args.out_frame, batch_x.shape[0])
                batch_yy = []

                for i in range(len(t)):
                    batch_yy.append(batch_out[i, t[i], :, :, :].unsqueeze(0))
                batch_yy  = torch.cat(batch_yy, dim=0)
                _t = torch.tensor(t*100).cuda()
                pred_y = self.model(batch_in, _t)
                loss = self.l1loss(pred_y, batch_yy)

                log_str = 'train loss: %.4f ' % (loss.item())
                if self.args.vgg_loss:
                    vgg_in = []
                    for i in range(len(t)):
                        vgg_in.append(batch_out[i, t[i], :, :, :].unsqueeze(0))

                    vgg_in = torch.cat(vgg_in, dim=0)

                    vgg_out = self.vgg(pred_y)
                    vgg_gt = self.vgg(vgg_in).detach()

                    perceptual_loss = self.l1loss(vgg_out, vgg_gt)
                    loss += (perceptual_loss * self.args.vgg_lamb)
                    log_str += 'p_loss: %.4f(%.4f) ' % (perceptual_loss.item(), (perceptual_loss * self.args.vgg_lamb))

                if self.args.perceptual:
                    mars_in = batch_out.clone()
                    for i in range(len(t)):
                        mars_in[i, t[i], :, :, :] = pred_y[i, :, :, :]

                    mars_in = torch.permute(mars_in, (0, 2, 1, 3, 4))
                    batch_out = torch.permute(batch_out, (0, 2, 1, 3, 4))
                    mars_out = self.mars(mars_in)
                    mars_gt = self.mars(batch_out).detach()
                    perceptual_loss = self.l1loss(mars_out, mars_gt)
                    loss += (perceptual_loss * self.args.lamb)
                    log_str += 'mp_loss: %.4f(%.4f) ' % (perceptual_loss.item(), (perceptual_loss * self.args.lamb))

                log_str += 'total_loss: %.4f' % (loss.item())
                train_pbar.set_description(log_str)

                train_loss.append(loss.item())
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                self.step += 1

            train_loss = np.average(train_loss)
            if epoch % args.log_step == 0:
                with torch.no_grad():
                    test_loss = self.test(self.test_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, test_loss))
                recorder(test_loss, self.model, self.path)
                recorder(test_loss, self.model, self.path, 'ema')

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, test_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        test_pbar = tqdm(test_loader)
        indicator = (self.args.in_shape[0]//2)
        for i, (batch_x, batch_y) in enumerate(test_pbar):

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_in = torch.cat([batch_x[:,:indicator,:,:,:], batch_y], dim=1)
            batch_out = batch_x[:,indicator:,:,:,:]
            pred_list = []
            for time in range(self.args.out_frame):
                time = torch.tensor(time*100).repeat(batch_x.shape[0]).cuda()
                pred_y = self.ema_model(batch_in, time)
                pred_list.append(pred_y.unsqueeze(1))
            pred_y = torch.cat(pred_list, dim=1)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_out], [preds_lst, trues_lst]))

            loss = self.l1loss(pred_y, batch_out)
            test_pbar.set_description(
                'test loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        mse, mae, ssim, psnr = metric(preds, trues, self.data_mean, self.data_std, True)
        print_log('test mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss

