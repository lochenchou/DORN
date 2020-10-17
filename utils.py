import torch
import glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image


def depth2label_sid(depth, K=80.0, alpha=1.0, beta=90.4414):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    if torch.cuda.is_available():
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()

    label = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    label = torch.max(label, torch.zeros(label.shape).cuda()) # prevent negative label.
    if torch.cuda.is_available():
        label = label.cuda()
        
    return label.int()


def label2depth_sid(label, K=80.0, alpha=1.0, beta=89.4648, gamma=-0.9766):
    if torch.cuda.is_available():
        alpha = torch.tensor(alpha).cuda()
        beta = torch.tensor(beta).cuda()
        K = torch.tensor(K).cuda()
    else:
        alpha = torch.tensor(alpha)
        beta = torch.tensor(beta)
        K = torch.tensor(K)

    label = label.float()
    ti_0 = torch.exp(torch.log(alpha) + torch.log(beta/alpha)*label/K) # t(i)
    ti_1 = torch.exp(torch.log(alpha) + torch.log(beta/alpha)*(label+1)/K) # t(i+1)
    depth = (ti_0 + ti_1) / 2 - gamma # avg of t(i) & t(i+1)
    return depth.float()


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:, :, :3]  # H, W, C


def batch_merge_into_row(input, depth_target, depth_pred):
    
    # _input, depth_target, and depth_pred should in shape N,C,H,W
    N,C,H,W = input.shape
    if N > 6:
        N = 6 # plot maximum 6 figures

    for i in range(N):
        _input = input[i,:,:,:]
        _depth_target = depth_target[i,:,:]
        _depth_pred = depth_pred[i,:,:]
        _input = 256 * np.transpose(_input.cpu().numpy(), (1, 2, 0))  # H, W, C
        _depth_target = 256* np.squeeze(_depth_target.cpu().numpy())
        _depth_pred = 256* np.squeeze(_depth_pred.data.cpu().numpy())
        d_min = min(np.min(_depth_target), np.min(_depth_pred))
        d_max = max(np.max(_depth_target), np.max(_depth_pred))
        _depth_target = colored_depthmap(_depth_target, d_min, d_max)
        _depth_pred = colored_depthmap(_depth_pred, d_min, d_max)
        if i==0:
            img_merge = np.hstack([_input, _depth_target, _depth_pred])
        else:
            row = np.hstack([_input, _depth_target, _depth_pred])
            img_merge = add_row(img_merge, row)
            
    return img_merge
            

def merge_into_row(_input, depth_target, depth_pred):
    
    if _input.dim() == 4:
        _input = _input[0,:,:,:]
    if depth_target.dim() == 4:
        depth_target = depth_target[0,:,:]
    if depth_pred.dim() == 4:
        depth_pred = depth_pred[0,:,:]
    
    _input = 255 * np.transpose(_input.cpu().numpy(), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([_input, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(_input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(_input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
    

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr