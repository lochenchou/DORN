import os
import time
import torch
import numpy as np
import utils
from tqdm import tqdm
from metrics import AverageMeter, Result, compute_errors


# train
def train_one_epoch(device, train_loader, model, output_dir, ord_loss, optimizer, epoch, logger, PRINT_FREQ, BETA, GAMMA, ORD_NUM=80.0):
    
    avg80 = AverageMeter()
    
    model.train()  # switch to train mode

    iter_per_epoch = len(train_loader)
    trainbar = tqdm(total=iter_per_epoch)
    end = time.time()
    
    for i, (_input, _sparse_depth, _dense_depth) in enumerate(train_loader):
        _input, _sparse_depth, _dense_depth = _input.to(device), _sparse_depth.to(device), _dense_depth.to(device)
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        # compute output
        end = time.time()

        with torch.autograd.detect_anomaly():
            _pred_prob, _pred_label = model(_input) 
            loss = ord_loss(_pred_prob, _dense_depth) # calculate ord loss with dense_depth
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        pred_depth = utils.label2depth_sid(_pred_label, K=ORD_NUM, alpha=1.0, beta=BETA, gamma=GAMMA)
        # calculate metrices with ground truth sparse depth
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(_sparse_depth, pred_depth.to(device))
        
        result80 = Result()
        result80.evaluate(pred_depth, _sparse_depth.data, cap=80)
        avg80.update(result80, gpu_time, data_time, _input.size(0))
        end = time.time()

        # update progress bar and show loss
        trainbar.set_postfix(ORD_LOSS='{:.2f},RMSE/log= {:.2f}/{:.2f},delta={:.2f}/{:.2f}/{:.2f},AbsRel/SqRe;={:.2f}/{:.2f}'.format(loss,rmse,rmse_log,a1,a2,a3,abs_rel,sq_rel))
        trainbar.update(1)
        
        
        if (i + 1) % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'AbsRel={result.absrel:.2f}({average.absrel:.2f}) '
                  'SqRel={result.sqrel:.2f}({average.sqrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'RMSE_log={result.rmse_log:.3f}({average.rmse_log:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(train_loader), gpu_time=gpu_time, result=result80, average=avg80.average()))
            
            current_step = int(epoch*iter_per_epoch+i+1)
        
            img_merge = utils.batch_merge_into_row(_input, _dense_depth.data, pred_depth)
            filename = os.path.join(output_dir,'step_{}.png'.format(current_step))
            utils.save_image(img_merge, filename)
            
            logger.add_scalar('TRAIN/RMSE', avg80.average().rmse, current_step)
            logger.add_scalar('TRAIN/RMSE_log', avg80.average().rmse_log, current_step)
            logger.add_scalar('TRAIN/AbsRel', avg80.average().absrel, current_step)
            logger.add_scalar('TRAIN/SqRel', avg80.average().sqrel, current_step)
            logger.add_scalar('TRAIN/Delta1', avg80.average().delta1, current_step)
            logger.add_scalar('TRAIN/Delta2', avg80.average().delta2, current_step)
            logger.add_scalar('TRAIN/Delta3', avg80.average().delta3, current_step)
            
            # reset average meter
            result80 = Result()
            avg80 = AverageMeter()
        


def validation(device, data_loader, model, ord_loss, output_dir, epoch, logger, PRINT_FREQ, BETA, GAMMA, ORD_NUM=80.0):
    avg80 = AverageMeter()
    avg50 = AverageMeter()
    model.eval()
    
    end = time.time()
    skip = 1
    img_list = []
    
    evalbar = tqdm(total=len(data_loader))
    
    for i, (_input, _sparse_depth, _dense_depth) in enumerate(data_loader):
        _input, _sparse_depth, _dense_depth = _input.to(device), _sparse_depth.to(device), _dense_depth.to(device)
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        # compute output
        end = time.time()
        with torch.no_grad():
            _pred_prob, _pred_label = model(_input) 
            loss = ord_loss(_pred_prob, _dense_depth)
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        pred_depth = utils.label2depth_sid(_pred_label, K=ORD_NUM, alpha=1.0, beta=BETA, gamma=GAMMA)
        
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(_sparse_depth, pred_depth.to(device))
        
        # measure accuracy and record loss
        result80 = Result()
        result80.evaluate(pred_depth, _sparse_depth.data, cap=80)  
        result50 = Result()
        result50.evaluate(pred_depth, _sparse_depth.data, cap=50)       
        
        avg80.update(result80, gpu_time, data_time, _input.size(0))
        avg50.update(result50, gpu_time, data_time, _input.size(0))
        end = time.time()
        
        # save images for visualization 
        if i == 0:
            img_merge = utils.merge_into_row(_input, _dense_depth, pred_depth)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(_input, _dense_depth, pred_depth)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = os.path.join(output_dir,'eval_{}.png'.format(int(epoch)))
            print('save validation figures at {}'.format(filename))
            utils.save_image(img_merge, filename)

        if (i + 1) % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'AbsRel={result.absrel:.2f}({average.absrel:.2f}) '
                  'SqRel={result.sqrel:.2f}({average.sqrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'RMSE_log={result.rmse_log:.3f}({average.rmse_log:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result80, average=avg80.average()))
            
        # update progress bar and show loss
        evalbar.set_postfix(ORD_LOSS='{:.2f},RMSE/log= {:.2f}/{:.2f},delta={:.2f}/{:.2f}/{:.2f},AbsRel/SqRe;={:.2f}/{:.2f}'.format(loss,rmse,rmse_log,a1,a2,a3,abs_rel,sq_rel))
        evalbar.update(1)
        i = i+1
        

    print('\n**** CAP=80 ****\n'
          'RMSE={average.rmse:.3f}\n'
          'RMSE_log={average.rmse_log:.3f}\n'
          'AbsRel={average.absrel:.3f}\n'
          'SqRel={average.sqrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          'iRMSE={average.irmse:.3f}\n'
          'iMAE={average.imae:.3f}\n'
          't_GPU={average.gpu_time:.3f}\n'.format(
        average=avg80.average()))
    
    print('\n**** CAP=50 ****\n'
          'RMSE={average.rmse:.3f}\n'
          'RMSE_log={average.rmse_log:.3f}\n'
          'AbsRel={average.absrel:.3f}\n'
          'SqRel={average.sqrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          'iRMSE={average.irmse:.3f}\n'
          'iMAE={average.imae:.3f}\n'
          't_GPU={average.gpu_time:.3f}\n'.format(
        average=avg50.average()))
    
    logger.add_scalar('VAL_CAP80/RMSE', avg80.average().rmse, epoch)
    logger.add_scalar('VAL_CAP80/RMSE_log', avg80.average().rmse_log, epoch)
    logger.add_scalar('VAL_CAP80/AbsRel', avg80.average().absrel, epoch)
    logger.add_scalar('VAL_CAP80/SqRel', avg80.average().sqrel, epoch)
    logger.add_scalar('VAL_CAP80/Delta1', avg80.average().delta1, epoch)
    logger.add_scalar('VAL_CAP80/Delta2', avg80.average().delta2, epoch)
    logger.add_scalar('VAL_CAP80/Delta3', avg80.average().delta3, epoch)
    
    logger.add_scalar('VAL_CAP50/RMSE', avg50.average().rmse, epoch)
    logger.add_scalar('VAL_CAP50/RMSE_log', avg50.average().rmse_log, epoch)
    logger.add_scalar('VAL_CAP50/AbsRel', avg50.average().absrel, epoch)
    logger.add_scalar('VAL_CAP50/SqRel', avg50.average().sqrel, epoch)
    logger.add_scalar('VAL_CAP50/Delta1', avg50.average().delta1, epoch)
    logger.add_scalar('VAL_CAP50/Delta2', avg50.average().delta2, epoch)
    logger.add_scalar('VAL_CAP50/Delta3', avg50.average().delta3, epoch)
    