import torch
import math
import numpy as np

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    valid_mask = gt>0
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    thresh = torch.max((gt / pred), (pred / gt))
    d1 = float((thresh < 1.25).float().mean())
    d2 = float((thresh < 1.25 ** 2).float().mean())
    d3 = float((thresh < 1.25 ** 3).float().mean())
        
    rmse = (gt - pred) ** 2
    rmse = math.sqrt(rmse.mean())
    
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = math.sqrt(rmse_log.mean())
    
    abs_rel = ((gt - pred).abs() / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.sqrel = 0, 0
        self.lg10, self.rmse_log = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.sqrel = np.inf, np.inf
        self.lg10, self.rmse_log = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, sqrel, lg10, rmse_log, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.sqrel = absrel, sqrel
        self.lg10, self.rmse_log = lg10, rmse_log
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, pred, gt, cap=None):
        valid_mask = gt>0
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        
        if cap != None:
            cap_mask = gt <= cap
            pred = pred[cap_mask]
            gt = gt[cap_mask]
    
        abs_diff = (gt - pred).abs()
        abs_diff_log = torch.log(gt) - torch.log(pred)

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        rmse_log = float((torch.pow(abs_diff_log, 2)).mean())
        self.rmse_log =  math.sqrt(rmse_log)
        self.lg10 = float((log10(pred) - log10(gt)).abs().mean())
        self.absrel = float((abs_diff / gt).mean())
        self.sqrel = ((abs_diff**2) / gt).mean()
       
        maxRatio = torch.max(pred / gt, gt / pred)
        self.delta1 = float((maxRatio < 1.25).float().mean()) # diff_ratio < 1.25
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean()) # diff_ratio < 1.5625
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean()) # diff_ratio < 1.953125
        self.data_time = 0
        self.gpu_time = 0

        inv_pred= 1 / pred
        inv_gt = 1 / gt
        abs_inv_diff = (inv_pred - inv_gt).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_sqrel = 0, 0
        self.sum_lg10, self.sum_rmse_log = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_sqrel += n*result.sqrel
        self.sum_lg10 += n*result.lg10
        self.sum_rmse_log += n*result.rmse_log
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_sqrel / self.count, 
            self.sum_lg10 / self.count, self.sum_rmse_log / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg