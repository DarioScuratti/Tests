#%%
# Libraries

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from glob import glob

#%%
# Datasets
# Tweets and their classification
twts = pd.read_csv('Code/Datasets/Analyses/alltwts_majAtt.csv')
twts = twts.drop_duplicates('id_str')
res_two = pd.read_csv('Code/Datasets/Analyses/alltwts-two.csv')
res_two = res_two.drop_duplicates('id_str')
res_merged = pd.read_csv('Code/Datasets/Analyses/alltwts-merged.csv')
res_merged = res_merged.drop_duplicates('id_str')
join = res_two.set_index('id_str').join(res_merged.set_index('id_str'), how='inner', lsuffix = 'three_')
alltwts = join.join(twts[['id_str','country_or_territory','country_code', 'week', 'class_baseline', 'class_attnthree']].set_index('id_str'), how = 'inner')
alltwts = alltwts[['week', 'country_or_territory','class_baseline','country_code', 'class_attnthree', 'class_attnmerged', 'class_attntwo']]
alltwts = alltwts[alltwts['class_attnthree'] != 'inv URL']

#%%
# samples creation
weeks = twts['week'].unique()
path = 'Code/Datasets/samples_v2/'
#%%
for w in weeks:
    sam = twts[twts.week == w][['id_str', 'media_url', 'class', 'cls']].sample(250)
    sam.to_csv(path+'samp_'+w+'.csv', index = False)

#%%
# GROUND TRUTH
# Merge samples with results
samples = pd.DataFrame()
for w in weeks:
    sam = pd.read_csv(path+'samp_'+w+'.csv')
    sam = sam.drop_duplicates('id_str')
    sam['week'] = [w]*sam.shape[0]
    two = pd.read_csv(path+'res/ev_cl_'+w+'.csv')
    two = two.drop_duplicates('id')
    three = pd.read_csv(path+'res/ev_clt_'+w+'.csv')
    three = three.drop_duplicates('id')
    join = sam.set_index('id_str').join(two.set_index('id'), rsuffix = '_two')
    join = join.join(three.set_index('id'), rsuffix = '_three')
    samples = pd.concat([samples,join])

#%%
samples = samples[~samples.index.duplicated(keep='first')]

#%%
# Select only the required columns and substitute nans with T
samples.loc[samples['class_two'].isna(),['class_two']] = samples[samples['class_two'].isna()]['class']
samples.loc[samples['class_three'].isna(),['class_three']] = samples[samples['class_three'].isna()]['cls']

#%%
# Merge with classifications of other classifiers and compute overall scores

#PAY ATTENTION TO INVALID URLs
samples = samples.join(alltwts[['class_baseline','class_attnthree', 'class_attnmerged', 'class_attntwo']])
samples = samples[samples['class_three'] != 'inv URL']
samples = samples[samples['class_attnmerged'] != 'inv URL']

#%%
# Encode categoricals
cols = ['class_baseline','class_attnthree', 'class_attntwo', 'class_attnmerged']
ground = ['class_two', 'class_three']

for c in cols+['class_two', 'class_three']:
    samples[c] = samples[c].astype('category')
    samples[c+'_cat'] = samples[c].cat.codes

#%%
#Overall
for c in cols:
    col = c+'_cat'
    classes = samples[c].cat.categories
    print(c)
    if len(classes) == 3:
        print(classification_report(samples['class_three_cat'],samples[col],target_names = classes))
    else:
        print(classification_report(samples['class_two_cat'], samples[col], target_names = classes))

#%%
# By Week
for w in weeks:
    print('Week: {}'.format(w))
    red = samples[samples['week'] == w]
    for c in cols:
        col = c + '_cat'
        print(c)
        classes = red[c].cat.categories
        if len(classes) == 3:
            print(classification_report(red['class_three_cat'], red[col], target_names=classes))
        else:
            print(classification_report(red['class_two_cat'], red[col], target_names=classes))

#%%
alltwts.to_csv('Code/Datasets/Analyses/alltwts-final.csv')