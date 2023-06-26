# code that takes the merged table and the 
import pandas as pd
import numpy as np

val = ''

#columns in merged table to keep
cf_of_interest = ['cjv', 'cnr', 'snrd_total']
col_to_keep = ['USUBJID', 'VISIT_MRI'] + cf_of_interest
df = pd.read_csv('/data/ms/processed/mri/MS_Share/merged_tables/mriqc_clusters_20230109.csv', sep=',', usecols=col_to_keep)
df.drop_duplicates(inplace=True)


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
        cf[i, :] = df.loc[tmp , cf_of_interest ]
print('counts', pd.Series(counts).value_counts()) # number of each occurances


#standardise each column of cf
med = np.nanmedian(cf, axis=0)
cf = cf - med
cf = cf / np.nanmedian(np.abs(cf), axis=0)
print('new median', np.nanmedian(cf, axis=0))
print('mad std', np.nanmedian(np.abs(cf-np.nanmedian(cf, axis=0)), axis=0))
np.save('/data/users/uu85g9/'+val+'confounders2.npy', cf)
print('confounders', cf[:5, :])


