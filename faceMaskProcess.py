### Face Mask Process
# This script wants to build the dataset for the new classifier with attention
# Currently, tweets are 3602 + 5412 + 5592 =  images
# We could assume a train/ test split of 80/ 20 that means 7683 train images and 1921 test images
# Training test will be split again in training and validation to have more generalization potential

#%%
# Libraries
import pandas as pd
import numpy as np
import glob
from collections import Counter
from sklearn.model_selection import train_test_split
import requests

#%%
# Import useful Datasets
path = "Code/Datasets/Crowd/crowd_results/"
infos = glob.glob(path+"*_task.csv")
runs = glob.glob(path+"*_run.csv")
results = glob.glob(path+"*_result.csv")

#%%
def f(l):
    c = Counter(l)
    ans = c.most_common(1)[0][0]
    num = c.most_common(1)[0][1]
    if num > 1:
        return ans
    else:
        return np.nan

#%%
imgs = pd.DataFrame()
twts = pd.DataFrame()
sets = ['may', 'w32', 'w34']
for s in sets:
    for i in range(3):
        if s in infos[i]:
            tasks_info = pd.read_csv(infos[i])
        if s in runs[i]:
            tasks_run = pd.read_csv(runs[i])
        if s in results[i]:
            ress = pd.read_csv(results[i])

    res = tasks_run[tasks_run.task_id.isin(ress.task_id)].groupby('task_id')['info_3'].apply(list)

    res = pd.DataFrame(res)
    res['task_id'] = ress['task_id'].values
    res['answer'] = res.info_3.apply(f)
    #res = res[res.answer != 'Not answered']
    res = res[~res.answer.isna()]

    ids = tasks_info[tasks_info.id.isin(res.task_id)]
    ids = ids.join(res.set_index('task_id'), on= 'id')
    imgs = pd.concat([imgs,ids[['info_id','info_media_url','answer']]])

#%%
save_path = "Code/Datasets/Class_Attention/"
def download(v):
    id = v[0]
    url = v[1]
    c = v[2]
    try:
        resp = requests.get(url)
    except:
        print("Unable to download")
    sph = save_path + c + '/'
    if resp.status_code == 200:
        with open(sph+str(id)+'.jpg', "wb") as file:
            file.write(resp.content)

#%%
imgs.columns = ['id', 'url', 'class']
imgs.loc[(imgs['class'] == 'Not answered'), ['class']] = 'Cannot tell'
#%%
imgs[['id','url','class']].apply(download, axis = 1)

#%%
# Merge tasks info with classification from kaggle
classified = pd.read_csv("Code/Datasets/ClassifierResults/class_merged_w34.csv")

#%%
ground_labels = ids[['info_id', 'answer']]
ground_truth = classified[classified['id'].isin(ground_labels['info_id'])].join(ground_labels.set_index('info_id'), on= 'id')

#%%
ground_truth.to_csv('Code/Datasets/ClassifierResults/ground_truth.csv', index = False)

#%%
ground = pd.read_csv("Code/Datasets/ClassifierResults/Tests/ground_truth.csv")
#%%
ground[['id','media_url']].to_csv("Code/Datasets/evaluation_2.csv", index = False)

#%%
# Read from Datasets
periods = ['may', 'w34']
conc = pd.DataFrame()
for p in periods:
    cls = pd.read_csv("Code/Datasets/classes_"+p+'.csv')
    cls = cls.drop_duplicates(subset = 'tweet_id', keep = 'last')
    cro = pd.read_csv("Code/Datasets/Crowd/crowd_results/crowd_"+p+'_task.csv')
    mrg = cls.join(cro.set_index('info_id'), on = "tweet_id")
    mrg.to_csv("Code/Datasets/mrg_"+p+'.csv')
    conc = pd.concat([conc,mrg])

#%%
conc = conc.drop_duplicates(subset='tweet_id', keep = 'last')
#%%
save_path = "Code/Datasets/Attention2/"
def f(v):
    id = v[0]
    url = v[1]
    c = v[2]
    try:
        resp = requests.get(url)
    except:
        print("Unable to download")
    sph = save_path + c + '/'
    if resp.status_code == 200:
        with open(sph+str(id)+'.jpg', "wb") as file:
            file.write(resp.content)

#%%
conc[['tweet_id','info_media_url','class']].apply(f, axis = 1)
