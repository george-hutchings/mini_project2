import os
import time
import numpy as np
import warnings
from scipy import stats
import scipy.ndimage as ndi
import scipy
import SimpleITK as sitk

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
rows_standard = 192
cols_standard = 160

def less_preprocessing(image):
    channel_num = 1
    start_cut = 0
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    image = np.float32(image)
    image_suitable = np.ndarray((num_selected_slice, image_rows_Dataset, cols_standard), dtype=np.float32)
    
    image_suitable[...] = np.min(image)

    brainmask = np.zeros(image.shape, dtype=np.float32)
    brainmask[image>0]=1
    for i in range(brainmask.shape[0]):
        brainmask[i,:,:] = scipy.ndimage.morphology.binary_fill_holes(brainmask[i,:,:])
    mad = stats.median_absolute_deviation(image[brainmask==1],axis=None)
    image -=np.median(image[brainmask==1])
    image /=(1.486*mad)

    image_suitable[:,:,(cols_standard - image_cols_Dataset) // 2:(cols_standard + image_cols_Dataset) // 2] = image[:,:, :]    
    image_suitable = image_suitable[:,:,:]
    image_suitable = image_suitable[..., np.newaxis]

    return image_suitable

def maskless_preprocessing(image):
    channel_num = 1
    start_cut = 0
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    image = np.float32(image)
    image_suitable = np.ndarray((num_selected_slice, image_rows_Dataset, cols_standard), dtype=np.float32)
 
    image_suitable[...] = np.min(image)
    image_suitable[:, :,(cols_standard - image_cols_Dataset) // 2:(cols_standard + image_cols_Dataset) // 2] = image[:,:, :]

    image_suitable = image_suitable[:,:,:]
    image_suitable = image_suitable[..., np.newaxis]

    return image_suitable

def maskover_preprocessing(image):
    channel_num = 1
    num_selected_slice = np.shape(image)[0]  # number of slices, 48
    image_rows_Dataset = np.shape(image)[1]  # row 240
    image_cols_Dataset = np.shape(image)[2]  # column 240
    image = np.float32(image)
    image = image[:,:,
                  (image_cols_Dataset // 2 - cols_standard // 2):(image_cols_Dataset // 2 + cols_standard // 2)]
  
    image = image[:, :, :]
    image = image[..., np.newaxis]  # 48*200*200*1， add one more dimension

    return image

def over_preprocessing(image):
    channel_num = 1
    num_selected_slice = np.shape(image)[0]  # number of slices, 48
    image_rows_Dataset = np.shape(image)[1]  # row 240
    image_cols_Dataset = np.shape(image)[2]  # column 240
    image = np.float32(image)
    image = image[:,:,
                  (image_cols_Dataset // 2 - cols_standard // 2):(image_cols_Dataset // 2 + cols_standard // 2)]

    brainmask = np.zeros(image.shape, dtype=np.float32)                                                      
    brainmask[image>0]=1                                                                                     
    for i in range(brainmask.shape[0]):
        brainmask[i,:,:] = scipy.ndimage.morphology.binary_fill_holes(brainmask[i,:,:])
    mad = stats.median_absolute_deviation(image[brainmask==1],axis=None)
    image -=np.median(image[brainmask==1])
    image /=(1.486*mad)
    
    # ---------------------------------------------------
    image = image[:, :, :]
    image = image[..., np.newaxis]  # 48*200*200*1， add one more dimension

    return image

def xless_preprocessing(image):
    channel_num = 1
    start_cut = 0
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    image = np.float32(image)
    image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard, 1), dtype=np.float32)
   
    image_suitable[...] = np.min(image)
    image_suitable[:, (rows_standard - image_rows_Dataset) // 2:(rows_standard + image_rows_Dataset) // 2,:, :] = image[:, :, :, :]
    image_suitable = image_suitable[:, :, :, :]

    return image_suitable

def xover_preprocessing(image):
    channel_num = 1
    num_selected_slice = np.shape(image)[0]  # number of slices, 48
    image_rows_Dataset = np.shape(image)[1]  # row 240
    image_cols_Dataset = np.shape(image)[2]  # column 240
    image = np.float32(image)

    image = image[:,
            (image_rows_Dataset // 2 - rows_standard // 2):(image_rows_Dataset // 2 + rows_standard // 2),
            (image_cols_Dataset // 2 - cols_standard // 2):(image_cols_Dataset // 2 + cols_standard // 2),
            :]
   
    # ---------------------------------------------------
    image = image[:, :, :, :]

    return image


def xmaskover_preprocessing(image):
    channel_num = 1
    num_selected_slice = np.shape(image)[0]  # number of slices, 48
    image_rows_Dataset = np.shape(image)[1]  # row 240
    image_cols_Dataset = np.shape(image)[2]  # column 240
    image = np.float32(image)

    image = image[:,
                  (image_rows_Dataset // 2 - rows_standard // 2):(image_rows_Dataset // 2 + rows_standard // 2),
                  (image_cols_Dataset // 2 - cols_standard // 2):(image_cols_Dataset // 2 + cols_standard // 2),:]

    image = image[:, :, :,:]

    return image

def xmaskless_preprocessing(image):
    channel_num = 1
    start_cut = 0
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    image = np.float32(image)

    image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)

    image_suitable[...] = np.min(image)
    image_suitable[:, (rows_standard - image_rows_Dataset) // 2:(rows_standard + image_rows_Dataset) // 2,:,:] = image[:,:,:,:]
    image_suitable = image_suitable[:,:,:,:]

    return image_suitable

def extractbrain(image,wholebrain):
    z_brain = []
    x_brain = []
    y_brain = []
    for i in range(image.shape[0]):
        if np.max(image[i,:,:])>0:
            z_brain.append(i)
    z_start = np.min(z_brain)
    z_end = np.max(z_brain)
    for j in range(image.shape[1]):
        if np.max(image[:,j,:])>0:
            x_brain.append(j)
    x_start = np.min(x_brain)
    x_end = np.max(x_brain)
    for k in range(image.shape[2]):
        if np.max(image[:,:,k])>0:
            y_brain.append(k)
    y_start = np.min(y_brain)
    y_end = np.max(y_brain)
    brain = wholebrain[z_start:z_end+1,x_start:x_end+1,y_start:y_end+1]
    
    return brain

def reconbrain(image,pred):
    z_brain = []
    x_brain = []
    y_brain = []
    for i in range(image.shape[0]):
        if np.max(image[i,:,:])>0:
            z_brain.append(i)
    z_start = np.min(z_brain)
    z_end = np.max(z_brain)
    for j in range(image.shape[1]):
       	if np.max(image[:,j,:])>0:
       	    x_brain.append(j)
    x_start = np.min(x_brain)
    x_end = np.max(x_brain)
    for k in range(image.shape[2]):
        if np.max(image[:,:,k])>0:
            y_brain.append(k)
    y_start = np.min(y_brain)
    y_end = np.max(y_brain)
    
    finalpred = np.zeros([image.shape[0], image.shape[1],image.shape[2]])
    finalpred[z_start:z_end+1,x_start:x_end+1,y_start:y_end+1] = pred[:,:,:]
    return finalpred
