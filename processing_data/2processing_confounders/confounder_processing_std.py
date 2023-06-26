# code that takes the merged table and the 
import pandas as pd
import numpy as np

val = ''

#columns in merged table to keep
col_to_keep = ['STUDY','USUBJID', 'VISIT_MRI', 'UMAP10C_COMP1', 'UMAP10C_COMP2', 'UMAP10C_COMP3']
df = pd.read_csv('/data/ms/processed/mri/MS_Share/merged_tables/mriqc_clusters_20230109.csv', sep=',', usecols=col_to_keep)
df.drop_duplicates(inplace=True)
def testing():
    #motivates dropping of STUDY column
    study = df['STUDY']
    subid = df['USUBJID']
    len_study = np.empty(df.shape[0], dtype=int)
    tmp = -1
    for i in range(df.shape[0]):
        len_study[i] = len(study[i])
        if (study[i] != subid[i][:len_study[i]]):
            tmp = i
            break
    if tmp == -1:
        print ("STUDY and USUBJID all have the same starting characters")
    else:
        print ("STUDY and USUBJID do NOT all have the same starting characters")

    if all(x == len_study[0] for x in len_study):
        print ("STUDY ID's are all of equal length")
    else:
        print ("STUDY ID's are NOT all of equal length")
df.drop(columns = 'STUDY', inplace=True)
print(df.head(10))

# Load the processed data 
print('='*40, 'img_names', '='*40) 
img_names = np.load('/data/users/uu85g9/img_names.npy')
print('Shape of image names', img_names.shape)
print('# unique img_names', len(np.unique(img_names)))
tmp = np.char.replace(img_names,'sub-','') #remove sub- prefix
img_names_split = np.char.split(tmp, '_ses-') #split at _ses-
img_names_split = np.array(list(img_names_split))
img_names_split[:,1] = ['ses-'+ r for r in img_names_split[:,1] ] # add ses prefix back
print ('Image names \n', img_names_split[:5])

#replace all x with underscore.
#in table
df['USUBJID'].str.replace('x','_')
#in examples
img_names_split[:,0] = np.char.replace(img_names_split[:,0], 'x', '_')
print('after replacing x with _')
print('Table', df['USUBJID'][:5])
print('img names split shape', img_names_split.shape)
print('img names split [:,0]', img_names_split[:,0])

# Checking that all values in img names are present in the table (they're not)
def testing2():
    tmp = ['USUBJID', 'VISIT_MRI']
    for j in range(2):
        a = np.sort(np.unique(img_names_split[:,j]))
        b = np.sort(np.unique(df[tmp[j]]))
        print('Sorted unique Img names[:,',j,'], Sorted table (',tmp[j],')')
        i = 10
        print(np.column_stack((a[:i], b[:i])))
        cond = any( a[0]==b )
        print('Is first value', a[0], 'in the sorted table?', cond)
        if cond:
            print('Are all img names[:,',j,'] in sorted table (',tmp[j],')?')
            tmp2 = [any(a_elt==b ) for a_elt in a]
            print(all(tmp2))
            if not all(tmp2):
                tmp3 = tmp2.index(False)
                print(a[tmp3], 'is in img names[:,',j,'], but not in in sorted table (',tmp[j],')?')


# create array of the confounders 
cf = np.full((len(img_names), 3),  np.nan)
counts = np.zeros(len(img_names), dtype=int)
for i in range(len(img_names)):
    sub_ses = img_names_split[i, :]
    sub, ses = sub_ses[0], sub_ses[1]
    tmp = ((df['USUBJID']==sub)&(df['VISIT_MRI']==ses))
    counts[i] = sum(tmp)
    #tmp2 = ((df['USUBJID']==str(sub))&(df['VISIT_MRI']==str(ses)))
    #assert tmp.equals(tmp2), 'using str DOES make a difference'
    if counts[i] == 1:
        cf[i, :] = df.loc[tmp , ['UMAP10C_COMP1', 'UMAP10C_COMP2', 'UMAP10C_COMP3'] ]
print('counts', pd.Series(counts).value_counts()) # number of each occurances


#standardise each column of cf
cf = cf - np.nanmean(cf, axis=0)
cf = cf / np.nanstd(cf, axis=0)
print('new mean', np.nanmean(cf, axis=0))
print('new std', np.nanstd(cf, axis=0))
np.save('/data/users/uu85g9/'+val+'confounders.npy', cf)
print('confounders', cf[:5, :])


