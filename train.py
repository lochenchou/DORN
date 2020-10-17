from datetime import datetime
import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
import utils
import random
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils import PolynomialLRDecay
from dataloader.kitti_loader import KittiLoader
from loss import OrdinalRegressionLoss
from engine import train_one_epoch, validation
    

# set arguments
BATCH_SIZE = 3
EPOCHS = 14
LR = 0.0001
END_LR = 0.00001
POLY_POWER = 0.9
LR_PATIENCE = 10
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MAX_ITER = 300000
WORKERS = 3
SEED = 1984
PRINT_FREQ = 100

# min value (meter) for benchmark training set: 1.9766
# max value (meter) for benchmark training set: 90.4414
# min value (meter) for eigen training set: 0.704
# max value (meter) for eigen training set: 79.729

# we set a little bit larger range than 1.9766 - 90.4414, 1.5 - 91.0

ORD_NUM = 80
GAMMA = 1.0 - 1.5
ALPHA = 1.5 + GAMMA
BETA = 91 + GAMMA

# set random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# set dataloader
DATA_ROOT = '/datasets/KITTI/depth_prediction'
train_set = KittiLoader(DATA_ROOT, mode='train')
val_set = KittiLoader(DATA_ROOT, mode='val')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

# create model
from model import dorn
model = dorn.DORN(pretrained=True)

print('GPU number: {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if GPU number > 1, then use multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = PolynomialLRDecay(optimizer, max_decay_steps=MAX_ITER, end_learning_rate=END_LR, power=POLY_POWER)

# loss function
ord_loss = OrdinalRegressionLoss(ord_num=ORD_NUM, beta=BETA)
         
# create output dir,
_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
output_dir = os.path.join('./result',_now)
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'valid')
logdir = os.path.join(output_dir, 'log')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
logger = SummaryWriter(logdir)
epochbar = tqdm(total=EPOCHS)

for epoch in range(EPOCHS):
    
    train_one_epoch(device, train_loader, model, train_dir, ord_loss, optimizer, epoch, logger, PRINT_FREQ, BETA=BETA, GAMMA=GAMMA, ORD_NUM=80.0)  
    
    validation(device, val_loader, model, ord_loss, val_dir, epoch, logger, PRINT_FREQ, BETA=BETA, GAMMA=GAMME, ORD_NUM=80.0)
    
    # save model and checkpoint per epoch
    checkpoint_filename = os.path.join(output_dir, 'checkpoint-{}.pth.tar'.format(str(epoch)))
    torch.save(model, checkpoint_filename)
       
    epochbar.update(1)




