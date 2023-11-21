import numpy as np
import torch

class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, mod='base'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, mod)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, mod)

    def save_checkpoint(self, val_loss, model, path, mod):
        if self.verbose:
            print(f'Test loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if mod == 'ema':
            torch.save(model.state_dict(), path+'/'+'checkpoint_ema.pth')
        else:
            torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
