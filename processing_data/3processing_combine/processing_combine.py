# code that consolodates the confounders and the  data
import pandas as pd
import numpy as np
save_dir = '/data/users/uu85g9/'
cf_full = np.load('/data/users/uu85g9/confounders.npy')
cf_idx = np.logical_not(np.isnan(cf_full[:,0]))
print('Total cf length', len(cf_idx))
print('# nan', sum(cf_idx))
print('cf_idx[:50]', cf_idx[:50])




print('cf shape before', cf_full.shape)
cf_cf = cf_full[cf_idx, :]
print('cf shape after', cf_cf.shape)
#np.save(save_dir + 'cf_cf.npy', cf_cf)

print('cf_full', cf_full[:5,:])
print('cf_cf', cf_cf[:5,:])


mask_cf = np.load('/data/users/uu85g9/mask.npy')
print('mask shape before', mask_cf.shape)
mask_cf = mask_cf[cf_idx,:,:,:]
print('mask shape after', mask_cf.shape)
#np.save(save_dir + 'mask_cf.npy', mask_cf)

train_cf = np.load('/data/users/uu85g9/train.npy')
print('train shape before', train_cf.shape)
train_cf = train_cf[cf_idx,:,:,:]
print('train shape after', train_cf.shape)
#np.save(save_dir + 'train_cf.npy', train_cf)

# taken from train code
sv = 999
np.random.seed(sv)
# load numpy data, I have preprocessed them, you can change this to you own data
mri_image = train_cf
lesion_mask = mask_cf
cf = cf_cf

# split whole data to train and validation
indices = list(range(mri_image.shape[0]))
val_indices = np.random.choice(indices,int(mri_image.shape[0]*0.1),replace = False)
train_indices = [x for x in range(mri_image.shape[0]) if x not in val_indices]

mri_train= []
mask_train= []
cf_train =[]

mri_val= []
mask_val= []
cf_val = []

for train_index in train_indices:
    mritrain = np.expand_dims(mri_image[train_index,:,:,:],axis =0)
    masktrain = np.expand_dims(lesion_mask[train_index,:,:,:],axis =0)
    cftrain = np.expand_dims(cf[train_index, :], axis=0)
    cf_train.append(cftrain)
    mri_train.append(mritrain)
    mask_train.append(masktrain)


for val_index in val_indices:
    mrival = np.expand_dims(mri_image[val_index,:,:,:],axis =0)                                            
    maskval = np.expand_dims(lesion_mask[val_index,:,:,:],axis =0)  
    cfval = np.expand_dims(cf[train_index, :], axis=0)
    mri_val.append(mrival)
    mask_val.append(maskval)
    cf_val.append(cfval)                                                                             


mri_train = np.concatenate(mri_train, axis=0)
np.save(save_dir + 'mri_train.npy', mri_train)
mask_train = np.concatenate(mask_train, axis=0)
np.save(save_dir + 'mask_train.npy', mask_train)
cf_train = np.concatenate(cf_train, axis=0)
np.save(save_dir + 'cf_train.npy', cf_train)


mri_val = np.concatenate(mri_val, axis=0)
np.save(save_dir + 'mri_val.npy', mri_val)
mask_val = np.concatenate(mask_val, axis=0)
np.save(save_dir + 'mask_val.npy', mask_val)
cf_val = np.concatenate(cf_val, axis=0)
np.save(save_dir + 'cf_val.npy', cf_val)



print('train shapes mri, mask, cf')
print(mri_train.shape)
print(mask_train.shape)
print(cf_train.shape)
print('val shapes mri, mask, cf')
print(mri_val.shape)
print(mask_val.shape)
print(cf_val.shape)

print('cf')
print(cf_train[:5,:])
print(cf_val[:5,:])



