##### FaceMaskProcess #####

"""
This script serves the purpose of aggregating tasks to build the dataset used to train the classifers based on:
    - Attention
"""

#%%
# LIBRARIES
import pandas as pd
import requests
import glob
from collections import Counter
import numpy as np

#%%
# Support Variables
path = "Code/Datasets/Crowd/crowd_results/"
save_path = "Code/Datasets/majVoting/"
times = ['may', 'w34', 'w32']
class_map = {
    'Yes': 'Yes',
    'No': 'No',
    'Some of them': 'No',
    'Cannot tell': "Don't know",
    'Not answered': "Don't know"
}

#%%
# CRITERIA
def majVoting(l):
    l = [x for x in l if str(x) != 'nan']
    if len(l) > 0:
        c = Counter(l)
        ans = c.most_common(1)[0][0]
        num = c.most_common(1)[0][1]
        if num > 1:
            return ans
        else:
            return np.nan
    return np.nan

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
merge = pd.DataFrame()
# Answer Assignment
for t in times:
    print(t)
    # Load datasets
    tasks = pd.read_csv(path+"crowd_"+t+"_task.csv")
    runs = pd.read_csv(path + "crowd_" + t + "_task_run.csv")
    results = pd.read_csv(path + "crowd_" + t + "_result.csv")

    # Apply Class Mappings
    for k, v in class_map.items():
        runs.loc[runs['info_3'] == k, ['info_3']] = v

    # merge answers for each task_id
    res = runs[runs.task_id.isin(results.task_id)].groupby('task_id')['info_3'].apply(list)

    res = pd.DataFrame(res)
    res.columns = ['answers']
    res = res.join(results.set_index('task_id'))
    res['answer'] = res.answers.apply(majVoting)
    res = res.join(tasks[['id', 'info_id', 'info_media_url', 'info_country_code']].set_index('id'))
    res['week'] = [t] * res.shape[0]
    print("Aggregated!")
    merge = pd.concat([merge, res])
    #res = res.loc[~res.answer.isna(),['info_id', 'info_media_url', 'answer']]
    #res.apply(download, axis=1)
    #print("Downloaded!")

#%%
merge.to_csv("Code/Datasets/Crowd/crowd_results/merged.csv")