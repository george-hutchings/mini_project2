import os
import time
import numpy as np
import warnings
import scipy.ndimage as ndi
from scipy import stats
import SimpleITK as sitk
from preprocess import over_preprocessing,less_preprocessing,maskless_preprocessing,maskover_preprocessing,xover_preprocessing,xless_preprocessing,xmaskless_preprocessing,xmaskover_preprocessing,extractbrain
import random

rows_standard = 192
cols_standard = 160
smooth = 1.
print('=' * 60)
y_dir = '/data/ms/processed/mri/Yang/ICML/Neurips/'
save_dir = '/data/users/uu85g9/'


def generate_data(patient, verbose=False):
    
    dirmask = y_dir + 'trainmask'
    dirmasks = os.listdir(dirmask)  # load subfolders names
    dirmasks.sort()  # sort folder names
    mask_image = sitk.ReadImage(dirmask+'/'+dirmasks[patient])
    mask_array = sitk.GetArrayFromImage(mask_image)

    mask_array[mask_array > 0] = 1.
    mask_array[mask_array <= 0] = 0.
    dir = y_dir + 'trainT1'
    dirs = os.listdir(dir)  # load subfolders names
    dirs.sort()  # sort folder names
    T1_image = sitk.ReadImage(dir+'/'+dirs[patient])
    T1_array = sitk.GetArrayFromImage(T1_image)  # load array of T1 image, 48*240*240
    ori_array = T1_array
    dir1 = y_dir + 'trainT2'
    dir1s = os.listdir(dir1)  # load subfolders names
    dir1s.sort()  # sort folder names
    T2_image = sitk.ReadImage(dir1+'/'+dir1s[patient])
    T2_array = sitk.GetArrayFromImage(T2_image)  # load array of T2 image, 48*240*240 
 
    dir2 = y_dir + 'trainPD'
    dir2s = os.listdir(dir2)  # load subfolders names
    dir2s.sort()  # sort folder names
    PD_image = sitk.ReadImage(dir2+'/'+dir2s[patient])
    PD_array = sitk.GetArrayFromImage(PD_image)  # load array of Gd image, 48*240*240
    
    dir3 = y_dir + 'trainFL'
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
    
    mask_train = mask_train
    full_train = np.concatenate((T1_train, T2_train, PD_train,FL_train), axis=3)

    return mask_train,full_train


def main():
    dir = y_dir + 'trainT1/'
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
    
    np.save(save_dir + 'img_names.npy', np.array(img_names))
    print('finalgenerate AXI')

    masks = np.concatenate(masks, axis=0)
    fullimages = np.concatenate(fullimages, axis=0)
   
    np.save(save_dir + 'mask.npy', masks)
    np.save(save_dir + 'train.npy', fullimages)
   
    print('lesion mask')
    print(masks.shape)
    print('all mod')
    print(fullimages.shape)
   
if __name__ == '__main__':
    main()
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
    print('All names equal?')
    print(dirmasks==dirs==dir1s==dir2s==dir3s)