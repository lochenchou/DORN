import os
import torch
import os
import numpy as np
from os import path
from tqdm import tqdm
from PIL import Image
from scipy.interpolate import LinearNDInterpolator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_root',
                    help='root dir of KITTI dataset')
parser.add_argument('depth_dir',
                    help='depth dir under KITTI root, ex: depth_annotated')
parser.add_argument('raw_data_dir',
                    help='raw dir under KITTI root, ex: kitti_data')
args = parser.parse_args()


args = parser.parse_args()

def lin_interp(depth):
    # modified based on https://github.com/hunse/kitti
    points = np.zeros((len(depth[depth>0]), 2))
    x,y=np.where(depth>0)
    points[:,0] = x
    points[:,1] = y
    d = depth[x,y]
    m, n = depth.shape
    f = LinearNDInterpolator(points, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    interp_depth = f(IJ).reshape(depth.shape)
    return interp_depth


DATA_ROOT = args.data_root
DEPTH_DIR = args.depth_dir
COLOR_DIR = args.raw_data_dir
TRAIN = 'train'
VAL = 'val'

image_set = ['image_02', 'image_03']
train_list = []
val_list = []

TRAIN_NUM = 76842
VAL_NUM = 6694

print('DATA_ROOT:{}\nDEPTH_DIR:{}\nCOLOR_DIR:{}'.format(DATA_ROOT,DEPTH_DIR,COLOR_DIR))


pbar = tqdm(total=TRAIN_NUM)
print('preprocess training data')
f = open("./train_list.txt", "w")
for _date in os.listdir(path.join(DATA_ROOT,DEPTH_DIR,TRAIN)):
    _depth_date_dir = path.join(DEPTH_DIR,TRAIN,_date)
    _color_date_dir = path.join(COLOR_DIR,_date) 
    
    for _sync in os.listdir(path.join(DATA_ROOT, _depth_date_dir)):
        _depth_sync_dir = path.join(_depth_date_dir, _sync, 'proj_depth', 'groundtruth') 
        _color_sync_dir = path.join(_color_date_dir, _sync)
        
        for _set in image_set:
            _depth_set_dir = path.join(_depth_sync_dir, _set)
            _color_set_dir = path.join(_color_sync_dir, _set, 'data')
            _lin_interp_set_dir = path.join(_depth_sync_dir, _set+'_lin_interp')
            
            if not os.path.exists(path.join(DATA_ROOT,_lin_interp_set_dir)):
                os.makedirs(path.join(DATA_ROOT,_lin_interp_set_dir))
            
            for _image in os.listdir(path.join(DATA_ROOT, _depth_set_dir)):
                # depth_annotated/train/[date]/[sync]/proj_depth/groundtruth/[image_set]/xxxxxxxxxx.png
                _sparse_depth_path = path.join(_depth_set_dir, _image)
                # data_image/[date]/[sync]/[image_set]/data/xxxxxxxxxx.png
                _raw_image_path = path.join(_color_set_dir, _image)
                if not path.exists(path.join(DATA_ROOT,_raw_image_path) or not path.exists(path.join(DATA_ROOT,_sparse_depth_path))):
                                 continue
                # create lin_interp
                # depth_annotated/train/[date]/[sync]/proj_depth/groundtruth/[image_set]_lin_interp/xxxxxxxxxx.png
                _lin_interp_depth_path = path.join(_lin_interp_set_dir, _image)
                pbar.set_postfix(INTERP_PATH='{}'.format(_lin_interp_depth_path))
                
                # do linear interpolation based on groung truth sparse depth
                depth = Image.open(os.path.join(DATA_ROOT,_sparse_depth_path))
                depth = np.asarray(depth)
                interp_depth = lin_interp(depth)
                interp_depth = interp_depth.astype(np.int32)
                interp_depth = Image.fromarray(interp_depth)
                interp_depth.save(os.path.join(DATA_ROOT,_lin_interp_depth_path))
                
                # save as [raw_image] [sparse_depth] [lin_interp_depth]
                f.write('{} {} {}\n'.format(_raw_image_path, _sparse_depth_path, _lin_interp_depth_path))
                train_list.append(_sparse_depth_path)
                pbar.update(1)
            
f.close()
print('train_list len: {}'.format(len(train_list)))


pbar = tqdm(total=VAL_NUM)
print('preprocess validation data')
f = open("./val_list.txt", "w")
for _date in os.listdir(path.join(DATA_ROOT,DEPTH_DIR,VAL)):
    _depth_date_dir = path.join(DEPTH_DIR,VAL,_date) 
    _color_date_dir = path.join(COLOR_DIR,_date) 
    
    for _sync in os.listdir(path.join(DATA_ROOT, _depth_date_dir)):
        _depth_sync_dir = path.join(_depth_date_dir, _sync, 'proj_depth', 'groundtruth')
        _color_sync_dir = path.join(_color_date_dir, _sync)
        
        for _set in image_set:
            _depth_set_dir = path.join(_depth_sync_dir, _set)
            _color_set_dir = path.join(_color_sync_dir, _set, 'data')
            _lin_interp_set_dir = path.join(_depth_sync_dir, _set+'_lin_interp')
            
            if not os.path.exists(path.join(DATA_ROOT,_lin_interp_set_dir)):
                os.makedirs(path.join(DATA_ROOT,_lin_interp_set_dir))
            
            for _image in os.listdir(path.join(DATA_ROOT, _depth_set_dir)):
                # depth_annotated/val/[date]/[sync]/proj_depth/groundtruth/[image_set]/xxxxxxxxxx.png
                _sparse_depth_path = path.join(_depth_set_dir, _image)
                # data_image/[date]/[sync]/[image_set]/data/xxxxxxxxxx.png
                _raw_image_path = path.join(_color_set_dir, _image)
                if not path.exists(path.join(DATA_ROOT,_raw_image_path) or not path.exists(path.join(DATA_ROOT,_sparse_depth_path))):
                                 continue
                # create lin_interp
                # depth_annotated/val/[date]/[sync]/proj_depth/groundtruth/[image_set]_lin_interp/xxxxxxxxxx.png
                _lin_interp_depth_path = path.join(_lin_interp_set_dir, _image)
                pbar.set_postfix(INTERP_PATH='{}'.format(_lin_interp_depth_path))
                
                # do lin interp
                # do linear interpolation based on groung truth sparse depth
                depth = Image.open(os.path.join(DATA_ROOT,_sparse_depth_path))
                depth = np.asarray(depth)
                interp_depth = lin_interp(depth)
                interp_depth = interp_depth.astype(np.int32)
                interp_depth = Image.fromarray(interp_depth)
                interp_depth.save(os.path.join(DATA_ROOT,_lin_interp_depth_path))
                
                # save as [raw_image] [sparse_depth] [lin_interp_depth]
                f.write('{} {} {}\n'.format(_raw_image_path, _sparse_depth_path, _lin_interp_depth_path))
                val_list.append(_sparse_depth_path)
                pbar.update(1)
            
f.close()
print('val_list len: {}'.format(len(val_list)))

# pbar = tqdm(total=(len(train_depth)+len(val_depth)))
# for _date in date:
#     _date_dir = path.join('train',_date)
#     _raw_date_dir = path.join('kitti_raw_data',_date)
#     for _sync in os.listdir(path.join(DATA_ROOT, _date_dir)):
#         _sync_dir = path.join(_date_dir, _sync, 'proj_depth', 'groundtruth')
#         _raw_sync_dir = path.join(_raw_date_dir, _sync)

#         for _set in image_set:
#             _set_dir = path.join(_sync_dir, _set)
#             _raw_set_dir = path.join(_raw_sync_dir, _set, 'data')
#             _interp_depth_dir = path.join(_sync_dir, _set+'_interp')
#             if not os.path.exists(path.join(DATA_ROOT,_interp_depth_dir)):
#                 os.makedirs(path.join(DATA_ROOT,_interp_depth_dir))
            
#             for _image in os.listdir(path.join(DATA_ROOT, _set_dir)):
#                 _gt_depth = path.join(_set_dir, _image)
#                 _raw_image = path.join(_raw_set_dir, _image)    
#                 if not path.exists(path.join(DATA_ROOT,_raw_image) or not path.exists(path.join(DATA_ROOT,_gt_depth))):
#                                  continue    
#                 _interp_depth = path.join(DATA_ROOT, _interp_depth_dir, _image)

#                 imgRgb = Image.open(path.join(DATA_ROOT,_raw_image))
#                 imgRgb = imgRgb.convert('RGB')
#                 sparse_depth = Image.open(path.join(DATA_ROOT,_gt_depth))
#                 sparse_depth = np.asarray(sparse_depth)

#                 dense_depth = fill_depth_colorization(imgRgb=np.asarray(imgRgb), imgDepthInput=sparse_depth, alpha=1)
#                 dense_depth = dense_depth.astype(np.int32)
#                 dense_depth = Image.fromarray(dense_depth)
#                 dense_depth.save(_interp_depth)
                    
#                 pbar.update(1)
                

# for _date in date:
#     _date_dir = path.join('val',_date)
#     _raw_date_dir = path.join('kitti_raw_data',_date)
#     for _sync in os.listdir(path.join(DATA_ROOT, _date_dir)):
#         _sync_dir = path.join(_date_dir, _sync, 'proj_depth', 'groundtruth')
#         _raw_sync_dir = path.join(_raw_date_dir, _sync)
#         for _set in image_set:
#             _set_dir = path.join(_sync_dir, _set)
#             _raw_set_dir = path.join(_raw_sync_dir, _set, 'data')
#             _interp_depth_dir = path.join(_sync_dir, _set+'_interp')
#             if not os.path.exists(path.join(DATA_ROOT,_interp_depth_dir)):
#                 os.makedirs(path.join(DATA_ROOT,_interp_depth_dir))
            
#             for _image in os.listdir(path.join(DATA_ROOT, _set_dir)):
#                 _gt_depth = path.join(_set_dir, _image)
#                 _raw_image = path.join(_raw_set_dir, _image)  
#                 if not path.exists(path.join(DATA_ROOT,_raw_image) or not path.exists(path.join(DATA_ROOT,_gt_depth))):
#                                  continue      
#                 _interp_depth = path.join(DATA_ROOT, _interp_depth_dir, _image)
                    
#                 imgRgb = Image.open(path.join(DATA_ROOT,_raw_image))
#                 imgRgb = imgRgb.convert('RGB')
#                 sparse_depth = Image.open(path.join(DATA_ROOT,_gt_depth))
#                 sparse_depth = np.asarray(sparse_depth)

#                 dense_depth = fill_depth_colorization(imgRgb=np.asarray(imgRgb), imgDepthInput=sparse_depth, alpha=1)
#                 dense_depth = dense_depth.astype(np.int32)
#                 dense_depth = Image.fromarray(dense_depth)
#                 dense_depth.save(_interp_depth)
                    
#                 pbar.update(1)







