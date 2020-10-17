import os
import numpy as np
import torch
import PIL
from PIL import Image
from scipy import interpolate
from torchvision import transforms as T
from torch.utils.data import Dataset

def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')

def readPathFiles(root_dir, list_file):
    filepaths = []

    with open(list_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            color_path = os.path.join(root_dir, line.split()[0])
            depth_path = os.path.join(root_dir, line.split()[1])

            filepaths.append((color_path, depth_path))

    return filepaths

def lin_interp(sparse_depth):
    # modified from https://github.com/hunse/kitti
    m, n = sparse_depth.shape
    ij = np.zeros((len(sparse_depth[sparse_depth>0]), 2))
    x, y = np.where(sparse_depth>0)
    ij[:,0] = x
    ij[:,1] = y
    d = sparse_depth[x,y]
    f = interpolate.LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    interp_depth = f(IJ).reshape(sparse_depth.shape)
    return interp_depth

class KittiLoader(Dataset):
    """
        RGB image path:
        kitti_raw_data/2011_xx_xx/2011_xx_xx_drive_xxxx_sync/image_0x/data/xxxxxxxxxx.png
        
        Depth path:
        train: train/2011_xx_xx/2011_xx_xx_drive_xxxx_sync/proj_depth/groundtruth/image_0x/xxxxxxxxxx.png
        val: val/2011_xx_xx/2011_xx_xx_drive_xxxx_sync/proj_depth/groundtruth/image_0x/xxxxxxxxxx.png
        
        KITTI mean & std
        self.mean = torch.Tensor([0.3864, 0.4146, 0.3952])
        self.std = torch.Tensor([0.2945, 0.3085, 0.3134])
        
        ImageNet mean & std
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
        
    """
    
    def __init__(self, root_dir='/home/lochenchou/Datasets/KITTI/DORN',
                 mode='train', loader=pil_loader, size=(385, 513), 
                 train_list='./list/benchmark/train_list.txt', 
                 val_list='./list/benchmark/val_list.txt'):
        super(KittiLoader, self).__init__()
        self.root_dir = root_dir

        self.mode = mode
        self.filepaths = None
        self.loader = loader
        self.size = size
        
        # set ImageNet mean and std for image normalization
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
        self.uni_std = torch.Tensor([1, 1, 1])      
        
        # set color jitter parameter
        self.brightness =0.2
        self.contrast = 0.2 
        self.saturation = 0.2
        self.hue = 0.1
    
        if self.mode == 'train':
            self.filepaths = readPathFiles(root_dir, train_list)
        elif self.mode == 'val':
            self.filepaths = readPathFiles(root_dir, val_list)
    

    def __len__(self):
        return len(self.filepaths)
    

    def get_color(self, color_path):
        color = self.loader(color_path, rgb=True)
        
        return color

    def get_depth(self, depth_path):
        sparse_depth = self.loader(depth_path, rgb=False)
        sparse_depth = np.asarray(sparse_depth) / 256.
        interp_depth = lin_interp(sparse_depth)

        return sparse_depth, interp_depth
    
    def train_transform(self, color, sparse_depth, dense_depth):
        
        sparse_depth = Image.fromarray(sparse_depth)
        dense_depth = Image.fromarray(dense_depth)
        
        # augmentation parameters
        rotation_angle = 5.0 # random rotation degrees
        flip_p = 0.5  # random horizontal flip
        color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) # adjust color for input RGB image
        
        # garg/eigen crop, x=153:371, y=44:1197
        CROP_LEFT = 44
        CROP_TOP = 153
        CROP_RIGHT = 1197
        CROP_BOTTOM = 371
        _color = color.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
        _sparse_depth = sparse_depth.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
        _dense_depth = dense_depth.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))
    
        transform = T.Compose([
            T.Resize((385, CROP_RIGHT-CROP_LEFT), PIL.Image.BILINEAR), # resize x-axis, and remain y-axis
            T.CenterCrop(self.size),
        ])
        
        _color = transform(_color)
        _color = color_jitter(_color)
        _sparse_depth = transform(_sparse_depth)
        _dense_depth = transform(_dense_depth)
        
        _color = np.array(_color).astype(np.float32) / 256.0
        _sparse_depth = np.array(_sparse_depth).astype(np.float32)
        _dense_depth = np.array(_dense_depth).astype(np.float32)  

        _color = T.ToTensor()(_color)
        _sparse_depth = T.ToTensor()(_sparse_depth)
        _dense_depth = T.ToTensor()(_dense_depth)
        
#         if self.norm:
#             if self.uni:
#                 im_ = T.Normalize(mean=self.mean, std=self.std)(im_)
#             else:
#                 im_ = T.Normalize(mean=self.mean, std=self.uni_std)(im_)

        return _color, _sparse_depth, _dense_depth
    
    def val_transform(self, color, sparse_depth, dense_depth):
        
        sparse_depth = Image.fromarray(sparse_depth)
        dense_depth = Image.fromarray(dense_depth)
        
        # garg/eigen crop, x=153:371, y=44:1197
        CROP_LEFT = 44
        CROP_TOP = 153
        CROP_RIGHT = 1197
        CROP_BOTTOM = 371
        _color = color.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
        _sparse_depth = sparse_depth.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)) 
        _dense_depth = dense_depth.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))
    
        transform = T.Compose([
            T.Resize((385, CROP_RIGHT-CROP_LEFT), PIL.Image.BILINEAR), # resize x-axis, and remain y-axis
            T.CenterCrop(self.size),
        ])
        
        _color = transform(_color)
        _sparse_depth = transform(_sparse_depth)
        _dense_depth = transform(_dense_depth)
        
        _color = np.array(_color).astype(np.float32) / 256.0
        _sparse_depth = np.array(_sparse_depth).astype(np.float32)
        _dense_depth = np.array(_dense_depth).astype(np.float32)  

        _color = T.ToTensor()(_color)
        _sparse_depth = T.ToTensor()(_sparse_depth)
        _dense_depth = T.ToTensor()(_dense_depth)
        
#         if self.norm:
#             if self.uni:
#                 im_ = T.Normalize(mean=self.mean, std=self.std)(im_)
#             else:
#                 im_ = T.Normalize(mean=self.mean, std=self.uni_std)(im_)

        return _color, _sparse_depth, _dense_depth

    def __getitem__(self, idx):
        color_path, depth_path = self.filepaths[idx]
        
        color = self.get_color(color_path)
        sparse_depth, interp_depth = self.get_depth(depth_path)
        
        if self.mode == 'train':
            color, sparse_depth, interp_depth = self.train_transform(color, sparse_depth, interp_depth)
            return color, sparse_depth, interp_depth
        elif self.mode == 'val':
            color, sparse_depth, interp_depth = self.val_transform(color, sparse_depth, interp_depth)
            return color, sparse_depth, interp_depth

