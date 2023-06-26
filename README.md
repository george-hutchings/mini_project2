
# Confounder Free neural networks

Code for the first attempt at implimenting confounder free neural networks

## Directories

### my_unet
Contains 

`unet_cf_train100_2.py` is for training the confounder free nn
`unet_vanilla_train2.py` is for training the vanilla nn

`Models.py` contains each of the components that make up the neural networks: the network for the feature space, for the confounder prediction, and for the (segmentation) prediction.

`model_*` contains saved trained models with the name referencing to the date the model began training.

### processing_data

`1processing_scans`, `2processing_confounders`, `3processing_combine` preprocess the data as done by Yang et al.

`4processing_testdata` given the trained models and their folder names this compares the performance on a held out dataset 