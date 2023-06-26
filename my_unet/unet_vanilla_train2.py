#training the nn with confounders

from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import scipy
import SimpleITK as sitk
import glob
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F 
import torch.nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import shutil
import random
from Models import modelB1, modelF1, modelP1
from losses import calc_loss, dice_loss, corr_loss
import time
from datetime import datetime
save_dir = '/data/users/uu85g9/'

print('torch version', torch.__version__)

#######################################################
#Setting the basic parameters of the model
#######################################################
sv =999  #  random seeds
torch.manual_seed(sv)
np.random.seed(sv)
random.seed(sv)
torch.cuda.manual_seed(sv)
torch.backends.cudnn.deterministic=True
print('seed', sv)

#load in data
mri_train = np.load(save_dir + 'mri_train2.npy')
mask_train = np.load(save_dir + 'mask_train2.npy')
#cf_train = np.float32(np.load(save_dir + 'cf_train_small10.npy'))

mri_val = np.load(save_dir + 'mri_val2.npy')
mask_val = np.load(save_dir + 'mask_val2.npy')
#cf_val = np.float32(np.load(save_dir + 'cf_val_small10.npy'))
print('(mri) Train size=', mri_train.shape )
print('(mri) Val size=', mri_val.shape)
print("Data Loaded")


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.label)

#######################################################
#Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda" if train_on_gpu else "cpu")

batch_size = 128    # number of batch size
print('batch_size = ' + str(batch_size))
valid_size = 0.1  # portion of validation data
epoch = 75         # number of epochs
print('epoch = ' + str(epoch))
#lambda = 1

trainset = MyDataset(mri_train,mask_train)
valset = MyDataset(mri_val,mask_val)

train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True,num_workers=1)
valid_loader = torch.utils.data.DataLoader(valset, batch_size, shuffle=False, pin_memory=True,num_workers=1)
n_batches = len(train_loader)

shuffle = True
valid_loss_min = np.Inf
num_workers = 1
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#######################################################
#Setting up the model
#######################################################


in_ch, out_ch = 4,1
modelF = modelF1(in_ch, out_ch)
modelP = modelP1(in_ch, out_ch)

modelF = torch.nn.DataParallel(modelF).to(device) # send tensor to device
modelP = torch.nn.DataParallel(modelP).to(device) # send tensor to device

#######################################################
#Using Adam as Optimizer
#######################################################
initial_lr = 0.0001
opt = torch.optim.Adam(list(modelF.parameters())+ list(modelP.parameters()), lr=initial_lr)
MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)

#######################################################
#Creating a Folder for every data of the program
#######################################################

New_folder = './model_vanilla2_' + datetime.now().strftime("%m%d%Y")

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)


#######################################################
#checking if the model exists and if true then delete
#######################################################
read_model_path = New_folder

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

#######################################################
#Training loop
#######################################################
for param in modelF.parameters():
    assert param.requires_grad==True, 'F Does not require grad'
    assert (param.is_leaf == True) or (param.grad_fn is None), 'F is not leaf'
for param in modelP.parameters():
    assert param.requires_grad==True, 'P Does not require grad'
    assert (param.is_leaf == True) or (param.grad_fn is None), 'P is not leaf'

t0 = time.time()
for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()

    #######################################################
    #Training Datavim
    #######################################################

    modelF.train()
    modelP.train()
    for x, y in train_loader: # b is bias
        x = x.permute(0,3,1,2)
        y = y.permute(0,3,1,2)
        x, y = x.to(device), y.to(device)

        y_pred = modelP(modelF(x))
        
        lossp = calc_loss(y_pred, y)     # Dice_loss Used predictor loss

        train_loss += lossp.item() #* x.size(0) what is this TODO

        opt.zero_grad(True)
        lossp.backward() #since lambda is 1?
        opt.step()
        scheduler.step()

        

    #######################################################
    #Validation Step
    #######################################################

    modelF.eval()
    modelP.eval()
    with torch.no_grad(): #to increase the validation process uses less memory
        for x1, y1 in valid_loader:
            x1= x1.permute(0,3,1,2)
            y1= y1.permute(0,3,1,2)
            x1,y1 = x1.to(device), y1.to(device)

            F_pred = modelF(x1)
            y_pred = modelP(F_pred)
            
            lossp = calc_loss(y_pred, y1)     # Dice_loss Used predictor loss

            valid_loss += lossp.item() #* x1.size(0) idk what this is TODO

    #######################################################
    #To write in Tensorboard
    #######################################################
    #train_idx = x.shape[0]
    #valid_idx = x1.shape[0]
    #train_loss = train_loss / train_idx
    #valid_loss = valid_loss / valid_idx TODO not sure why have all os this

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print('Epoch: {}/{}'.format(i+1, epoch))
    print('Train Loss: \t{:.6f} \tVal Loss: {:.6f}'.format( train_loss, valid_loss))
    print('-' * 10)
    if i == 0:
        print('Time per epoch: \t{:.1f} s'.format(time.time() - since))
        #estimated time to completion
        print('Estimated time to completion: \t{} s'.format((time.time() - since)*epoch))

    #######################################################
    #Early Stopping
    #######################################################

    if valid_loss < valid_loss_min and epoch_valid >= i: # and i_valid <= 2:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(modelF.state_dict(), read_model_path+'/modelF.pth')
        torch.save(modelP.state_dict(), read_model_path+'/modelP.pth')
      
        valid_loss_min = valid_loss
print('training time')
print('{} seconds'.format(time.time() - t0))
print('{} seconds per epoch'.format((time.time() - t0)/epoch))
