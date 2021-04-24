#%%
# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import kruskal

#%%
sample = pd.read_csv('Code/Datasets/sample.csv')
sample = sample[['id','cls','media_url']]
#%%
sample.to_csv('Code/Datasets/sample.csv', index = False)

#%%
# Read datasets
twts = pd.read_csv('Code/Datasets/Analyses/alltwts.csv')
corr = pd.read_csv('Code/Datasets/Analyses/correlations.csv')
pol = pd.read_csv('Code/Datasets/Analyses/policies.csv')

##### Descriptive Statistics/ Explorative Analyses #####
#%%
# Overall Mean score across weeks
sns.lineplot(data = twts, x = 'week', y = 'mean', palette = 'viridis')
plt.show()

#%%
# Mean distribution for each class
sns.displot(data = twts, x='mean', hue = 'class', palette = 'crest', kind = 'kde')
plt.show()

#%%
# Overall twts by week
ax = sns.countplot(data = twts, x = 'week', palette  = 'crest')
for p in ax.patches:
    height = p.get_height()
    ax.text(x = p.get_x() + (p.get_width()/2), y = height+100, s = '{:.0f}'.format(height), ha = 'center')
plt.show()

#%%
# Visualizing the mean and the number of people, to check if there's any correlation
sns.scatterplot(data = twts, x= '#', y = 'mean', hue = 'class',palette = 'viridis')
plt.show()

#%%
# Visualizing the country and the number of people colored by the mean
n = 20
top_n = twts.groupby('country_code').count().sort_values(by = 'id',ascending= False).iloc[:n,:].index
sns.scatterplot(data = twts[twts['country_code'].isin(top_n)], x='country_code', y='#', hue= 'mean', palette = 'crest')
plt.show()

#%%
# Visualizing the incidence of a certain country per number of tweets
n = 20
top_n = twts.groupby('country_code').count().sort_values(by = 'id',ascending= False).iloc[:n,:].index
ax = sns.countplot(data = twts[twts['country_code'].isin(top_n)], x = 'country_code', palette = 'viridis')
for p in ax.patches:
    height = p.get_height()
    ax.text(x = p.get_x() + (p.get_width()/2), y = height+100, s = '{:.0f}%'.format((height/ twts.shape[0])*100), ha = 'center')
plt.show()

#%%
# Visualizing the incidence of a certain country per number of tweets in a given week
n = 5
subset = twts[twts.week == 'w44']
top_n = subset.groupby('country_code').count().sort_values(by = 'id',ascending= False).iloc[:n,:].index
ax = sns.countplot(data = subset[subset['country_code'].isin(top_n)], x = 'country_code', palette = 'viridis')
for p in ax.patches:
    height = p.get_height()
    ax.text(x = p.get_x() + (p.get_width()/2), y = height+25, s = '{:.0f}%'.format((height/ subset.shape[0])*100), ha = 'center')
plt.show()

#%%
# Country distribution by week (shows only top n countries by posts)
n = 10
top_n = twts.groupby('country_code').count().sort_values(by = 'id',ascending= False).iloc[:n,:].index
sns.countplot(data = twts[twts['country_code'].isin(top_n)], x='country_code',hue='week', palette = 'crest')
plt.show()
# We should identify also the significance in change across weeks -> chi-squared (# tweets by week~country)

#%%
# Heatmap between county and week
# First we show a heatmap to visualize proportions
cont_viz = pd.crosstab(twts['week'], twts[twts['country_code'].isin(top_n)]['country_code'])
sns.heatmap(cont_viz, cmap='crest')
plt.show()

#%%
# scatterplot of mean by sde
sns.scatterplot(data = twts, x= 'mean', y= 'std', hue = 'week', palette = 'crest')
plt.show()

#%%
# scatterplot of mean by sde -> Notice how the higher the #, the lower the sde because it is composed of similar
# predictions. As the # increases, the sde will shrink because will be divided by n squared.
sns.scatterplot(data = twts, x= 'mean', y= 'std', hue = '#', palette = 'crest')
plt.show()

#%%
# Bar plot of posts by # divided by class
sns.countplot(data = twts, x= '#', hue= 'class', palette = 'crest')
plt.show()

#%%
# Relationship between standard deviation and number of people for the medium score
sns.lineplot(data = twts, x= '#', y= 'std', palette = 'crest')
plt.show()

#%%
# Relationship between standard deviation and number of people for the medium score
sns.countplot(data = twts, x= '')

#%%
###### Statistical tests between tweets variables: bi-variate analysis #####

# Country vs Week: Chi- Squared
cont = pd.crosstab(twts['week'], twts['country_code'])
# Then we compute the Chi-Squared test
c, p, dof, expected = chi2_contingency(cont)
print(str(p))
if p < 0.05:
    print("Significant Relationship between the two Variables")
else:
    print("The two variables are independent")

#%%
# Country vs Class: Chi- Squared
cont = pd.crosstab(twts['class'], twts['country_code'])
# Then we compute the Chi-Squared test
c, p, dof, expected = chi2_contingency(cont)
print(str(p))
if p < 0.05:
    print("Significant Relationship between the two Variables")
else:
    print("The two variables are independent")

#%%
# Country vs #: Chi- Squared
cont = pd.crosstab(twts['#'], twts['country_code'])
# Then we compute the Chi-Squared test
c, p, dof, expected = chi2_contingency(cont)
print(str(p))
if p < 0.05:
    print("Significant Relationship between the two Variables")
else:
    print("The two variables are independent")

#%%
# Helper for Kurskal tests
def krusk(x, y, data):
    tst = [data.loc[ids, y].values for ids in data.groupby(x).groups.values()]
    return kruskal(*tst)
#%%
# Country vs Mean: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
krusk(x = 'country_code', y = 'mean', data = twts)

#%%
# Country vs std: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
krusk(x = 'country_code', y = 'std', data = twts)

#%%
# Class vs Week: Chi- Squared
cont = pd.crosstab(twts['class'], twts['week'])
# Then we compute the Chi-Squared test
c, p, dof, expected = chi2_contingency(cont)
print(str(p))
if p < 0.05:
    print("Significant Relationship between the two Variables")
else:
    print("The two variables are independent")

#%%
# Class vs Mean: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
krusk(x = 'class', y = 'mean', data = twts)

#%%
# Class vs std: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
krusk(x = 'class', y = 'std', data = twts)

#%%
# Class vs #: Chi- Squared
cont = pd.crosstab(twts['class'], twts['#'])
# Then we compute the Chi-Squared test
c, p, dof, expected = chi2_contingency(cont)
print(str(p))
if p < 0.05:
    print("Significant Relationship between the two Variables")
else:
    print("The two variables are independent")

#%%
# Week vs mean: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
krusk(x = 'week', y = 'mean', data = twts)

#%%
# Week vs std: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
krusk(x = 'week', y = 'std', data = twts)

#%%
# Week vs #: Chi- Squared
cont = pd.crosstab(twts['week'], twts['#'])
# Then we compute the Chi-Squared test
c, p, dof, expected = chi2_contingency(cont)
print(str(p))
if p < 0.05:
    print("Significant Relationship between the two Variables")
else:
    print("The two variables are independent")

#%%
# Mean vs std: We cannot use the simple Anova, we are not respecting the three hypotheses
# Kruskal test
kruskal(twts['mean'].values, twts['std'].values)

#%%
# # vs Mean: kruskal
krusk(x= '#', y = 'mean', data = twts)

#%%
# # vs std: kruskal
krusk(x= '#', y = 'std', data = twts)

#%%
# Summary of all the distributions
twts.describe()

#%%
##### Policies information #####
# Summary of the policies data
# Limit to the interesting columns
cols = pol.columns
cols = cols[[0,1,2,3,4,5,7,8,36,37,38,39]]
pol = pol[cols]

#%%
pol.describe()

#%%
# trend of new cases by day for the countries where we have the most twts
sns.lineplot(data = pol[pol.iso_code.isin(top_n)], x= 'date', y='new_cases', hue= 'iso_code', palette = 'crest', legend= False)
plt.show()

#%%
# trend of total cases by day for the countries
sns.lineplot(data = pol[pol.iso_code.isin(top_n)], x= 'date', y='total_cases', hue= 'iso_code', palette = 'crest', legend= False)
plt.show()

#%%
# Trend for the stringency index by the top countries depending on the day
sns.lineplot(data = pol[pol.iso_code.isin(top_n)], x='date', y='stringency_index', hue = 'iso_code', palette = 'crest', legend = False)
plt.show()

#%%
# Stringency index by country (average)
sns.barplot(data = pol[pol.iso_code.isin(top_n)], x= 'iso_code', y= 'stringency_index')
plt.show()

#%%
# Helper function: weeknumber
import datetime

def wn(d):
    spl = d.split('-')
    date = datetime.date(int(spl[0]), int(spl[1]), int(spl[2]))
    return 'w'+ str(date.isocalendar()[1])

pol['week'] = pol['date'].apply(wn)

#%%
def mn(d):
    return 'm' + d.split('-')[1]

pol['month'] = pol['date'].apply(mn)

#%%
# Scatter to identify the correlation between total cases and population density
sns.scatterplot(data = pol[pol['week'] == 'w34'], x = 'population_density', y = 'total_cases')
plt.show()

#%%
# Stringency Index by week (average)
sns.lineplot(data = pol[pol.iso_code.isin(top_n)], x= 'month', y= 'stringency_index', hue = 'iso_code', palette = 'crest')
sns.despine()
plt.show()

#%%
# Monthly average for the stringengy index for the top countries
piv = pol[pol['iso_code'].isin(top_n)].pivot_table(index = 'month',columns = 'iso_code',values = 'stringency_index', aggfunc= 'mean')
sns.heatmap(piv, cmap= 'crest')
sns.despine()
plt.show()

#%%
# Monthly average of new cases for the top countries
piv = pol[pol['iso_code'].isin(top_n)].pivot_table(index = 'month',columns = 'iso_code',values = 'new_cases', aggfunc= 'mean')
sns.heatmap(piv, cmap= 'crest')
plt.show()

#%%


