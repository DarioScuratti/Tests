### PREPARATION ###
"""
    This script preprocess filtered tweets in order to be correctly loaded on crowdsourcing platforms
"""

#%%
# Libraries
import pandas as pd
import numpy as np
import glob

#%%
# Datasets Import
date = "w45"
dfs = glob.glob("Code/Datasets/FilteredCrawls/geoloc_*"+date+"*")
df = pd.DataFrame()
for d in dfs:
    df = pd.concat([df, pd.read_csv(d)])


#%%
print(df.isna().sum())

#%%
# We have to work on bounding_box, user_loc and country
columns = ['user_loc', 'bounding_box', 'country']
df[columns] = df[columns].fillna('-')
df = df.drop_duplicates(subset = 'id')

#%%
# Define correction functions
def f(s):
    return s[2:-1]

def r(s):
    if isinstance(s,str):
        s = s.replace("'", ' ')
        return s.replace(',',' -')
    else:
        return '-'

#%%
# Apply function to columns
df.full_text = df['full_text'].apply(f)
df.user_loc = df['user_loc'].apply(r)
df.country_or_territory = df['country_or_territory'].apply(r)

#%%
filter = df[['id','bounding_box','CIME_geolocation_string', 'CIME_coordinates']]
plat = df.drop(columns = ['CIME_geolocation_string', 'bounding_box', 'CIME_coordinates'])

#%%
filter.to_csv('Code/Datasets/FilteredCrawls/Boundings/bounding_boxes_'+date+'.csv', index = False,sep=',', quotechar='"', quoting=0)

#%%
# Save in chunks to load
plat.to_csv('Code/Datasets/FilteredCrawls/Filtered/filtered_crawl_'+date+'.csv', index = False,sep=',', quotechar='"', quoting=0)
#chunks = [plat[i*5000:min((i+1)*5000,len(plat))] for i in range(int(len(plat)/5000+1))]
#for c in range(len(chunks)):
    #chunks[c].to_csv('Code/Datasets/FilteredCrawls/Filtered/Chunks/filtered_crawl_'+date+'_'+str(c)+'.csv', index = False,sep=',', quotechar='"', quoting=0)