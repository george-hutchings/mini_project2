
from __future__ import print_function, division
#############################################################################################################
# section 1 - processing scans
import os
import time
import numpy as np
import warnings
import scipy.ndimage as ndi
from scipy import stats
import SimpleITK as sitk
from preprocess import over_preprocessing,less_preprocessing,maskless_preprocessing,maskover_preprocessing,xover_preprocessing,xless_preprocessing,xmaskless_preprocessing,xmaskover_preprocessing,extractbrain
import random
import pandas as pd

#training the nn with confounders
import scipy
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
import shutil
from Models import modelB1, modelF1, modelP1
import time
from datetime import datetime

from scipy.stats import wasserstein_distance
from scipy.stats.stats import pearsonr 

def remove_outliers(data):
    q1,q3 = np.percentile(data, [25,75])
    iqr = q3-q1
    minval = q1 - (1.5*iqr)
    maxval = q3 + (1.5*iqr)
    data_filtered = np.full_like(data, np.nan)
    cond = (data > minval) & (data < maxval)
    data_filtered[cond] = data[cond]
    return data_filtered



#cf model path and vanilla model path
cf_model_path = '/home/uu85g9/confounder_free_nn/my_unet/model_cf100_052023/'
vanilla_model_path = '/home/uu85g9/confounder_free_nn/my_unet/model_vanilla100_052023/' 
cf_of_interest =  ['UMAP10C_COMP1', 'UMAP10C_COMP2', 'UMAP10C_COMP3']

fig_dir = '/data/users/uu85g9/figs100/'


save_metrics = fig_dir + 'metrics.csv'
metrics_save_plot =  fig_dir + 'metrics_plot.png' #.png

# neural network preliminaries 
def metrics(prediction, truth, threshold=0.5):
    confusion_vector = (torch.sigmoid(prediction) > threshold).float() / truth.float()
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1, dim=(1, 2, 3), dtype=torch.float32)
    false_positives = torch.sum(confusion_vector == float('inf'), dim=(1, 2, 3), dtype=torch.float32)
    true_negatives = torch.sum(torch.isnan(confusion_vector),dim=(1, 2, 3), dtype=torch.float32)
    false_negatives = torch.sum(confusion_vector == 0, dim=(1, 2, 3), dtype=torch.float32)
    # calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # calculate dice
    dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    # # replace nans with 0
    # dice[torch.isnan(dice)] = 0
    # precision[torch.isnan(precision)] = 0
    # recall[torch.isnan(recall)] = 0

    #using yang's code
    smooth = 1.0
    intersection = torch.sum(prediction * truth, dim=(1, 2, 3))
    yang_dice = 1 - ((2. * intersection + smooth) / (torch.sum(prediction + truth ) + smooth))

    return (precision.tolist(), recall.tolist(), dice.tolist(), yang_dice.tolist())

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

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Running on CPU')
else:
    print('CUDA is available. Running on GPU')
device = torch.device("cuda" if train_on_gpu else "cpu")

#######################################################
#Setting up the model
#######################################################

in_ch, out_ch = 4,1
#load in confounder free modelF and modelP
modelF_cf = modelF1(in_ch, out_ch)
modelP_cf = modelP1(in_ch, out_ch)
modelP_cf = torch.nn.DataParallel(modelP_cf).to(device) # send tensor to device
modelF_cf = torch.nn.DataParallel(modelF_cf).to(device) # send tensor to device


#load in vanilla modelF and modelP
modelF_vanilla = modelF1(in_ch, out_ch)
modelP_vanilla = modelP1(in_ch, out_ch)
modelF_vanilla = torch.nn.DataParallel(modelF_vanilla).to(device) # send tensor to device
modelP_vanilla = torch.nn.DataParallel(modelP_vanilla).to(device) # send tensor to device


# load in data
class MyDataset(Dataset):
    def __init__(self, data, label, confounders):
        self.data = data
        self.label = label
        self.confounders = confounders
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.confounders[index]
    def __len__(self):
        return len(self.label)
    


if torch.cuda.device_count() > 1:
    batch_size = 1024
else:
    batch_size = 256
print('batch_size = ' + str(batch_size), flush=True)



modelF_cf.load_state_dict(torch.load(cf_model_path + 'modelF.pth'))
modelP_cf.load_state_dict(torch.load(cf_model_path + 'modelP.pth'))
modelF_vanilla.load_state_dict(torch.load(vanilla_model_path + 'modelF.pth'))
modelP_vanilla.load_state_dict(torch.load(vanilla_model_path + 'modelP.pth'))


csv_names = ['Utrain.csv', 'Useen.csv', 'Utest.csv']
#create df to store tp, fp, tn, fn
df_features = pd.DataFrame(columns=['precision_mean', 'precision_std', 'recall_mean', 'recall_std', 'dice_mean', 'dice_std', 'yang_dice_mean', 'yang_dice_std', 'csv_name', 'model_type', 'number_of_slices'])


def generate_data(patient, verbose=False):
    
    dirmask = y_dir + 'mask'
    dirmasks = os.listdir(dirmask)  # load subfolders names
    dirmasks.sort()  # sort folder names
    mask_image = sitk.ReadImage(dirmask+'/'+dirmasks[patient])
    mask_array = sitk.GetArrayFromImage(mask_image)

    mask_array[mask_array > 0] = 1.
    mask_array[mask_array <= 0] = 0.
    dir = y_dir + 'T1'
    dirs = os.listdir(dir)  # load subfolders names
    dirs.sort()  # sort folder names
    T1_image = sitk.ReadImage(dir+'/'+dirs[patient])
    T1_array = sitk.GetArrayFromImage(T1_image)  # load array of T1 image, 48*240*240
    ori_array = T1_array
    dir1 = y_dir + 'T2'
    dir1s = os.listdir(dir1)  # load subfolders names
    dir1s.sort()  # sort folder names
    T2_image = sitk.ReadImage(dir1+'/'+dir1s[patient])
    T2_array = sitk.GetArrayFromImage(T2_image)  # load array of T2 image, 48*240*240 
 
    dir2 = y_dir + 'PD'
    dir2s = os.listdir(dir2)  # load subfolders names
    dir2s.sort()  # sort folder names
    PD_image = sitk.ReadImage(dir2+'/'+dir2s[patient])
    PD_array = sitk.GetArrayFromImage(PD_image)  # load array of Gd image, 48*240*240
    
    dir3 = y_dir + 'FL'
    dir3s = os.listdir(dir3)  # load subfolders names
    dir3s.sort()  # sort folder names
    FL_image = sitk.ReadImage(dir3+'/'+dir3s[patient])
    FL_array = sitk.GetArrayFromImage(FL_image)  # load array of Gd image, 48*240*240

    mask_array = extractbrain(ori_array,mask_array)
    T1_array = extractbrain(ori_array,T1_array)
    T2_array = extractbrain(ori_array,T2_array)
    PD_array = extractbrain(ori_array,PD_array)
    FL_array = extractbrain(ori_array,FL_array)
    if verbose:
        return (dirmasks, dirs, dir1s, dir2s, dir3s)

    if mask_array.shape[2]>159:
        mask_train = maskover_preprocessing(mask_array)
    else:
        mask_train = maskless_preprocessing(mask_array)
    
    if mask_array.shape[1]>191:
        mask_train = xmaskover_preprocessing(mask_train)
    else:
        mask_train = xmaskless_preprocessing(mask_train)
   
    if T1_array.shape[2]>159:    
        T1_train = over_preprocessing(T1_array)
    else:
        T1_train = less_preprocessing(T1_array)

    if T2_array.shape[2]>159:
        T2_train = over_preprocessing(T2_array)
    else:
        T2_train = less_preprocessing(T2_array)

    if PD_array.shape[2]>159:
        PD_train = over_preprocessing(PD_array)
    else:
        PD_train=less_preprocessing(PD_array)
   
    if FL_array.shape[2]>159:
        FL_train = over_preprocessing(FL_array)
    else:
        FL_train=less_preprocessing(FL_array)
    
    if T1_array.shape[1]>191:
        T1_train = xover_preprocessing(T1_train)
    else:
        T1_train = xless_preprocessing(T1_train)
   
    if T2_array.shape[1]>191:
        T2_train = xover_preprocessing(T2_train)
    else:
        T2_train = xless_preprocessing(T2_train)
    
    if PD_array.shape[1]>191:
        PD_train = xover_preprocessing(PD_train)
    else:
        PD_train = xless_preprocessing(PD_train)
    
    if FL_array.shape[1]>191:
        FL_train = xover_preprocessing(FL_train)
    else:
        FL_train = xless_preprocessing(FL_train)
    
    full_train = np.concatenate((T1_train, T2_train, PD_train,FL_train), axis=3)

    return mask_train,full_train


def main():
    dir = y_dir + 'T1/'
    dirs = os.listdir(dir)  # load subfolders names
    dirs.sort()  # sort folder names
    x = len(dirs)
    print('number of training subjects')
    print(x)
    
    del2 = len('_T1_biascorr_brain.nii.gz')
    dirs_names = [name[:-del2] for name in dirs]
    masks = []
    fullimages = []
    img_names = []
    for patient in range(x):
        mask,fullimage= generate_data(patient)
        patient_name = dirs_names[patient]
        for i in range(mask.shape[0]):
            if np.max(mask[i,:,:,:])>0:
                lesionmask = np.expand_dims(mask[i,:,:,:],axis =0)
                fulllesionimage = np.expand_dims(fullimage[i,:,:,:],axis =0)
               
                masks.append(lesionmask)
                fullimages.append(fulllesionimage)
                img_names.append(patient_name)
    


    masks = np.concatenate(masks, axis=0)
    fullimages = np.concatenate(fullimages, axis=0)
    img_names = np.array(img_names)
   

    dirmasks, dirs, dir1s, dir2s, dir3s = generate_data(1, verbose=True)
    del1 = len('_T2Lesion_in_T1.nii.gz')
    dirmasks = [name[:-del1] for name in dirmasks]
    del2 = len('_T1_biascorr_brain.nii.gz')
    dirs = [name[:-del2] for name in dirs]
    del3 = len('_T2_biascorr_brain.nii.gz')
    dir1s = [name[:-del3] for name in dir1s]
    del4 = len('_PD_biascorr_brain.nii.gz')
    dir2s = [name[:-del4] for name in dir2s]
    del5 = len('_SEMI_FLAIR.nii.gz')
    dir3s = [name[:-del5] for name in dir3s]
    assert dirmasks==dirs==dir1s==dir2s==dir3s, 'not all names are equal'

    return masks, fullimages, img_names



#for section 2 
# columns in merged table to keep
col_to_keep = ['USUBJID', 'VISIT_MRI'] + cf_of_interest
df = pd.read_csv('/data/ms/processed/mri/MS_Share/merged_tables/mriqc_clusters_20230109.csv', sep=',', usecols=col_to_keep)
df.drop_duplicates(inplace=True)
#replace all x with underscore in table
df['USUBJID'].str.replace('x','_')

csv_name_dict = {'train':'Utrain.csv', 'seentest':'Useen.csv', 'test':'Utest.csv'}
for val in ['train', 'seentest','test']:
    print('Processing',val, 'val','.'*40, flush=True)
    y_dir = '/data/ms/processed/mri/Yang/ICML/Neurips/'+ val
    mask, train, img_names = main()

    print('Shape of image names', img_names.shape)
    print('# unique img_names', len(np.unique(img_names)))
    tmp = np.char.replace(img_names,'sub-','') #remove sub- prefix
    img_names_split = np.char.split(tmp, '_ses-') #split at _ses- and remove _ses-
    img_names_split = np.array(list(img_names_split))
    img_names_split[:,1] = ['ses-'+ r for r in img_names_split[:,1] ] # add ses prefix back
    print ('Image names \n', img_names_split[:5])

    
    #in examples
    img_names_split[:,0] = np.char.replace(img_names_split[:,0], 'x', '_')
    print('img names split shape', img_names_split.shape)
    print('img names split [:,0]', img_names_split[:,0])


    # create array of the confounders 
    cf = np.full((len(img_names), 3),  np.nan)
    counts = np.zeros(len(img_names), dtype=int)
    for i in range(len(img_names)):
        sub_ses = img_names_split[i, :]
        sub, ses = sub_ses[0], sub_ses[1]
        tmp = ((df['USUBJID']==sub)&(df['VISIT_MRI']==ses))
        counts[i] = sum(tmp)
        if counts[i] == 1:
            cf[i, :] = df.loc[tmp , cf_of_interest]
    print('counts', pd.Series(counts).value_counts()) # number of each occurances

    #standardise each column of cf
    med = np.nanmedian(cf, axis=0)
    cf = cf - med
    cf = cf / np.nanmedian(np.abs(cf), axis=0)
    print('new median', np.nanmedian(cf, axis=0))
    print('mad std', np.nanmedian(np.abs(cf-np.nanmedian(cf, axis=0)), axis=0))
    #np.save('/data/users/uu85g9/'+val+'confounders2.npy', cf)
    print('confounders', cf[:5, :])


    # code that combines the confounders and the  data
    cf_idx = np.logical_not(np.isnan(cf[:,0]))
    print('Total cf length', len(cf_idx))
    print('# nan', sum(cf_idx))
    print('cf_idx[:50]', cf_idx[:50])

    print('cf shape before', cf.shape)
    cf_cf = cf[cf_idx, :]
    print('cf shape after', cf_cf.shape)

    print('cf_full', cf[:5,:])
    print('cf_cf', cf_cf[:5,:])


    print('mask shape before', mask.shape)
    mask = mask[cf_idx,:,:,:]
    print('mask shape after', mask.shape)

    print('train shape before', train.shape)
    train = train[cf_idx,:,:,:]
    print('train shape after', train.shape)


    # taken from train code
    sv = 999
    np.random.seed(sv)
    # load numpy data, I have preprocessed them, you can change this to you own data
    mri = train
    cf = cf_cf                                                                          

    testdataset = MyDataset(mri, mask, cf)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size, shuffle=False, pin_memory=True, num_workers=1)
    n_batches = len(test_loader)
    print('n_batches for val',val,' = ' + str(n_batches), flush =True)

    #recording prediction metrics
    precision_cf, recall_cf, dice_cf, yang_dice_cf = [], [], [], []
    precision_vanilla, recall_vanilla, dice_vanilla, yang_dice_vanilla = [], [], [], [] 
    #######################################################
    #Testing the model
    #######################################################
    # test the model
    modelF_cf.eval()
    modelP_cf.eval()
    modelF_vanilla.eval()
    modelP_vanilla.eval()

    features_fg_cf_all = torch.empty(len(mri), device=device, dtype=torch.float32)
    features_bg_cf_all = torch.empty_like(features_fg_cf_all)
    features_fg_vanilla_all = torch.empty_like(features_fg_cf_all)
    features_bg_vanilla_all = torch.empty_like(features_fg_cf_all)
    # put model into no grad and pass through the test set
    with torch.no_grad():
        for j, (x, y, b) in enumerate(test_loader):
            current_time = time.time()
            x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            y = y.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

            # forward pass for cf model
            features = modelP_cf(modelF_cf(x))

            tmp = metrics(features, y)
            precision_cf += tmp[0]
            recall_cf += tmp[1]
            dice_cf += tmp[2]
            yang_dice_cf += tmp[3]

            features_fg = torch.mul(features, y).mean(dim=[1,2,3])
            features_bg = torch.mul(features, 1-y).mean(dim=[1,2,3])
            # save features being aware of last batch being different size
            features_fg_cf_all[j*batch_size:j*batch_size+len(features_fg)] = features_fg
            features_bg_cf_all[j*batch_size:j*batch_size+len(features_bg)] = features_bg

            #forward pass for vanilla model
            features = modelP_vanilla(modelF_vanilla(x))

            tmp = metrics(features, y)
            precision_vanilla += tmp[0]
            recall_vanilla += tmp[1]
            dice_vanilla += tmp[2]
            yang_dice_vanilla += tmp[3]

            features_fg = torch.mul(features, y).mean(dim=[1,2,3])
            features_bg = torch.mul(features, 1-y).mean(dim=[1,2,3])
            # save features being aware of last batch being different size
            features_fg_vanilla_all[j*batch_size:j*batch_size+len(features_fg)] = features_fg
            features_bg_vanilla_all[j*batch_size:j*batch_size+len(features_bg)] = features_bg

        if j ==0:
            #print estimated time to complete
            print('estimated time to complete ' + val  + str((time.time()-current_time)*n_batches/60) + ' minutes', flush=True)


    #add metrics to dataframe
    df_features = df_features.append({'precision_mean': np.nanmean(precision_cf), 
                    'precision_std': np.nanstd(precision_cf), 
                    'recall_mean': np.nanmean(recall_cf), 
                    'recall_std': np.nanstd(recall_cf), 
                    'dice_mean': np.nanmean(dice_cf), 
                    'dice_std': np.nanstd(dice_cf), 
                    'yang_dice_mean': np.nanmean(yang_dice_cf), 
                    'yang_dice_std': np.nanstd(yang_dice_cf), 
                    'csv_name': csv_name_dict[val], 
                    'model_type': 'cf', 
                    'number_of_slices': int(len(testdataset))}, 
                    ignore_index=True)
    df_features = df_features.append({'precision_mean': np.nanmean(precision_vanilla), 
                    'precision_std': np.nanstd(precision_vanilla), 
                    'recall_mean': np.nanmean(recall_vanilla), 
                    'recall_std': np.nanstd(recall_vanilla), 
                    'dice_mean': np.nanmean(dice_vanilla), 
                    'dice_std': np.nanstd(dice_vanilla), 
                    'yang_dice_mean': np.nanmean(yang_dice_vanilla), 
                    'yang_dice_std': np.nanstd(yang_dice_vanilla), 
                    'csv_name': csv_name_dict[val], 
                    'model_type': 'vanilla', 
                    'number_of_slices': int(len(testdataset))},
                    ignore_index=True)


    #numpy array of all features
    features_fg_cf_all = features_fg_cf_all.cpu().numpy()
    features_bg_cf_all = features_bg_cf_all.cpu().numpy()
    features_fg_vanilla_all = features_fg_vanilla_all.cpu().numpy()
    features_bg_vanilla_all = features_bg_vanilla_all.cpu().numpy()

    #combine numpy array of features and confounders as columns
    features_all = np.column_stack((cf, features_fg_cf_all, features_bg_cf_all, features_fg_vanilla_all, features_bg_vanilla_all))
    features_all = pd.DataFrame(features_all, columns=cf_of_interest+['FG_cf', 'BG_cf', 'FG_vanilla', 'BG_vanilla'])

    if val == 'train':
        df1 = features_all.copy()
    elif val == 'seentest':
        df2 = features_all.copy()
    elif val == 'test':
        df3 = features_all.copy()
    # add column with val name
    #save as csv
    # features_all.to_csv(save_dir + csv_names[i], index=False)
    # print('saved csv', csv_names[i], flush=True)
    # print (features_all.head(), flush=True)

    





df_features.to_csv(save_metrics, index=False)

d_fg_bg = {'FG': 'Foreground ', 'BG': 'Background '}
for bg_fg in ['FG','BG']:

    fg_bgname = d_fg_bg[bg_fg]

    cf_name = bg_fg + '_cf'
    vanilla_name =  bg_fg + '_vanilla'

    y1cf = remove_outliers(df1[cf_name])
    print('no. samples left, y1cf', sum(np.isnan(y1cf)), ' ', int(100*(len(y1cf) - sum(np.isnan(y1cf)))/len(y1cf)), '%', flush=True)
    y2cf = remove_outliers(df2[cf_name])
    print('no. samples left, y2cf', sum(np.isnan(y2cf)), ' ', int(100*(len(y2cf) - sum(np.isnan(y2cf)))/len(y2cf)), '%', flush=True)
    y3cf = remove_outliers(df3[cf_name])
    print('no. samples left, y3cf', sum(np.isnan(y3cf)), ' ', int(100*(len(y3cf) - sum(np.isnan(y3cf)))/len(y3cf)), '%', flush=True)
    y1vanilla = remove_outliers(df1[vanilla_name])
    print('no. samples left, y1vanilla', sum(np.isnan(y1vanilla)), ' ', int(100*(len(y1vanilla) - sum(np.isnan(y1vanilla)))/len(y1vanilla)), '%', flush=True)
    y2vanilla = remove_outliers(df2[vanilla_name])
    print('no. samples left, y2vanilla', sum(np.isnan(y2vanilla)), ' ', int(100*(len(y2vanilla) - sum(np.isnan(y2vanilla)))/len(y2vanilla)), '%', flush=True)
    y3vanilla = remove_outliers(df3[vanilla_name])
    print('no. samples left, y3vanilla', sum(np.isnan(y3vanilla)), ' ', int(100*(len(y3vanilla) - sum(np.isnan(y3vanilla)))/len(y3vanilla)), '%', flush=True)
    ylab = 'avg activation '


    #create histogram
    a = scipy.stats.wasserstein_distance(y1cf, y2cf)
    b = scipy.stats.wasserstein_distance(y1cf, y3cf)
    #plt.text(-3, 40, 'Wasserstein Distance \n Baseline and Moderate: 0.33 \n Baseline and Severe:1.23 ')
                    # '\n Baseline mean: -1.23 \n Moderate mean: -1.27\n Severe mean: -1.16', fontsize = 12)


    fig, (ax1, ax2) = plt.subplots(2,1,sharex=False, sharey=False)

    ax1.hist(y1cf, bins=int(np.sqrt(len(y1cf))),alpha=0.6,density=True,label='Baseline')
    ax1.hist(y2cf, bins=int(np.sqrt(len(y2cf))),alpha=0.6,density=True,label='Moderate')
    ax1.hist(y3cf, bins=int(np.sqrt(len(y3cf))),alpha=0.6,density=True,label='Severe')
    ax1.set_title('Confounder Free')

    ax2.hist(y1vanilla, bins=int(np.sqrt(len(y1vanilla))),alpha=0.6,density=True,label='Baseline')
    ax2.hist(y2vanilla, bins=int(np.sqrt(len(y2vanilla))),alpha=0.6,density=True,label='Moderate')
    ax2.hist(y3vanilla, bins=int(np.sqrt(len(y3vanilla))),alpha=0.6,density=True,label='Severe')
    ax2.set_title('Vanilla')
    ax2.set_xlabel(ylab)
    print('y1cf mean', np.nanmean(y1cf), 'y2cf mean', np.nanmean(y2cf), 'y3cf mean', np.nanmean(y3cf))


    fig.legend(['Baseline', 'Moderate', 'Severe'])
    fig.suptitle(fg_bgname + 'histogram')
    fig.tight_layout()
    plt.savefig(fig_dir + 'hist_' + bg_fg+'.png',bbox_inches='tight')
    plt.show()

    for component in cf_of_interest:
        COMP1=list(df1[component])
        COMP2=list(df2[component])
        COMP3=list(df3[component])


        #print min and max of COMP1, COMP2, COMP3
        # print('min, max COMP1', min(COMP1), max(COMP1))
        # print('min, max COMP2', min(COMP2), max(COMP2))
        # print('min, max COMP3', min(COMP3), max(COMP3))

        x1 = np.array(COMP1)
        x1 = x1/np.std(x1)
        x2 = np.array(COMP2)
        x2 = x2/np.std(x2)
        x3 = np.array(COMP3)
        x3 = x3/np.std(x3)
        print(component, bg_fg, flush=True)

        # print('x1 max, min; y1 max, min', min(x1), max(x1), min(y1cf), max(y1cf))
        # print('x2 max, min; y2 max, min', min(x2), max(x2), min(y2cf), max(y2cf))
        # print('x3 max, min; y3 max, min', min(x3), max(x3), min(y3cf), max(y3cf))
        xlimits = [min(np.concatenate((x1,x2,x3),axis=0))*1.1, max(np.concatenate((x1,x2,x3),axis=0))*1.1]

        #calculate number of discarded x1,x2,x3 values based off of xlimits
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=False)
        fig.suptitle(fg_bgname + 'scatter plot')

        ax1.set_xlabel(component)
        ax1.set_ylabel(ylab)
        ax1.scatter(x1, y1cf, marker='.', label='Baseline', alpha=0.8)
        ax1.scatter(x2, y2cf, marker='.', label='Moderate', alpha=0.4)
        ax1.scatter(x3, y3cf, marker='.', label='Severe', alpha=0.2)
        ax1.set_xlim(xlimits)
        ax1.set_title('Confounder Free')
        #ax1.set_yscale('symlog')

        ax2.set_xlabel(component)
        ax2.scatter(x1, y1vanilla, marker='.', label='Baseline', alpha=0.8)
        ax2.scatter(x2, y2vanilla, marker='.', label='Moderate', alpha=0.4)
        ax2.scatter(x3, y3vanilla, marker='.', label='Severe', alpha=0.2)
        ax2.set_title('Vanilla')
        #ax2.set_yscale('symlog')

        fig.legend(['Baseline', 'Moderate', 'Severe'])
        fig.tight_layout()
        plt.savefig(fig_dir + 'scatter_' + component + bg_fg+'.png',bbox_inches='tight')
        plt.show()


#create dataframe with columns: model, FG/BG, correlation
df = pd.DataFrame(columns=['model', 'component', 'correlation'])
#calculate correlation between vanilla_df and components
vanilla1 = np.array(list(df1['FG_vanilla']) + list(df2['FG_vanilla']) + list(df3['FG_vanilla'])) + np.array(list(df1['BG_vanilla']) + list(df2['BG_vanilla']) + list(df3['BG_vanilla']))
vanilla1*=0.5
cf1 = np.array(list(df1['FG_cf']) + list(df2['FG_cf']) + list(df3['FG_cf'])) + np.array(list(df1['BG_cf']) + list(df2['BG_cf']) + list(df3['BG_cf']))
cf1*=0.5
corr_umap_vanilla = np.empty(3)
corr_umap_cf = np.empty(3)
for i, component in enumerate(cf_of_interest):
    tmp=list(df1[component]) + list(df2[component]) + list(df3[component])
    tmp = np.array(tmp)
    #add correlation to df
    df = df.append({'model': 'vanilla',  'component':component,  'correlation':pearsonr(vanilla1, tmp)[0]}, ignore_index=True)
    df = df.append({'model': 'cf',  'component':component,  'correlation':pearsonr(cf1, tmp)[0]}, ignore_index=True)
print('Correlation df printing')
print(df.to_string())


d = { 'Utrain.csv': 'Baseline', 'Useen.csv': 'Severe', 'Utest.csv': 'Moderate'}
df = df_features.replace({"csv_name": d}) 
df['Name'] =  df['model_type'] + ' ' +  df['csv_name']
for val in ['precision', 'recall', 'dice', 'yang_dice']:
    x_pos = np.arange(len(df))
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, df[val+'_mean'], yerr=df[val+'_std'], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Avg' + val)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(df['Name']))
    #rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(metrics_save_plot)
    plt.show()