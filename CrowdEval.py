#%%
# Libraries
import pandas as pd
import numpy as np
import glob
import json

#%%
path = 'Code/Datasets/Crowd/crowd_results/'
##### Depending on the Period #####
period = 'w32'
threshold = 250
results = pd.read_csv(path+'crowd_'+period+'_result.csv')
runs = pd.read_csv(path+'crowd_'+period+'_task_run.csv')
tasks = pd.read_csv(path+'crowd_'+period+'_task.csv')

#%%
# General Stats
print(tasks.shape)
print("Participants: ",len(runs['user_id'].unique()))
print(runs.pivot_table(index='task_id', values = 'id', aggfunc='count').sort_values('id', ascending= False))
print(runs.pivot_table(index='user_id', values = 'id', aggfunc='count').sort_values('id', ascending= False))

#%%
# We want now to take the minimum number of tasks to evaluate most of the workers with great confidence
# For each worker we will estimate a minimum of 5 tasks up until the 5% of their total tasks  or a maximum of 20 tasks
# Notice that those with na in the id have same ip -> assign a dummy id
runs.loc[runs['user_id'].isna(),'user_id'] = [150]* len(runs[runs.user_id.isna()]['user_id'] )
rank = runs.pivot_table(index='user_id', values = 'id', aggfunc='count').sort_values('id')
resps = rank[rank['id'] >= 10]

# Map who answered each task
tsks = {}
def cnt_resps(t):
    rs = json.loads(t[1])
    for r in rs:
        resp = runs[runs['id'] == r]['user_id'].values[0]
        if t[0] not in tsks.keys():
            tsks[t[0]] = []
        tsks[t[0]].append(resp)

results[['task_id','task_run_ids']].apply(cnt_resps, axis = 1)

def score(rec, part):
    s = 0
    for r in rec:
        if r in part:
            s += 1
    return s

resps['cnt'] = [0] * len(resps.index)
done = False
ts = []

while(done == False):
    lu = resps.sort_values(by = ['cnt','id']).index[:4]
    scores = {}
    for t, r in tsks.items():
        scores[t] = score(r,lu)
    t_best = sorted(scores, key=scores.get, reverse= True)[0]
    ts.append(t_best)
    resps.loc[resps.index.isin(tsks[t_best]),'cnt'] += 1
    tsks.pop(t_best)
    resps = resps[resps['cnt']<threshold]
    resps = resps[resps['cnt'] != resps['id']]
    if resps.shape[0] == 0:
        done = True

#%%
eval = tasks[tasks['id'].isin(ts)][['id', 'info_media_url']]
eval.columns = ['id','url']
eval.to_csv(path+'eval_'+period+'.csv', index= False)

#%%
# Use ground truth to evaluate each worker
resps = rank[rank['id'] >= 10]
resps['right'] = [0] * len(resps.index)
resps['wrong'] = [0] * len(resps.index)
gt = pd.read_csv(path+'eval_classes_'+period+'.csv')

j = runs.join(gt.set_index('id'), on = 'task_id')

#%%
def ev(r):
    ans = r[0]
    rs = r[1]
    c = r[2]
    if rs in resps.index:
        if ans == c:
            resps.loc[rs,'right'] += 1
        else:
            resps.loc[rs,'wrong'] += 1

j[~j['class'].isna()][['info_3','user_id','class']].apply(ev, axis = 1)

#%%
def assign_prec(r):
    return r[0]/(r[0]+r[1])

resps['prec'] = resps[['right','wrong']].apply(assign_prec, axis = 1)
resps.to_csv(path+'workers_'+period+'.csv', index = False)