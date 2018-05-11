
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


traindf = pd.read_csv('train_2.csv')
traindf2 = traindf.fillna(0)
df3=traindf2.iloc[:, 1:]
df3.mean(axis=1).sort_values()

traindf['agent'] = traindf['Page'].str.split('_').str[-1]
traindf['access'] = traindf['Page'].str.split('_').str[-2]
traindf['project'] = traindf['Page'].str.split('_').str[-3]

traindf['label'] = traindf['Page'].str.split('_').str[:-3]

traindf['lanuage'] = traindf['project'].str.split('.').str[0]

lanuagetype = traindf.groupby( [ "lanuage"] ).count()

lanuagetype


agenttype = traindf.groupby( [ "agent"] ).count()


agenttype


accesstype = traindf.groupby( [ "access"] ).count()

accesstype

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'

train['lang'] = train.Page.map(get_language)

from collections import Counter

print(Counter(traindf.lanuage))


# In[29]:

lang_sets = {}
lang_sets['en'] = traindf[traindf.lanuage=='en'].iloc[:,0:-1]
lang_sets['ja'] = traindf[traindf.lanuage=='ja'].iloc[:,0:-1]
lang_sets['de'] = traindf[traindf.lanuage=='de'].iloc[:,0:-1]
lang_sets['fr'] = traindf[traindf.lanuage=='fr'].iloc[:,0:-1]
lang_sets['zh'] = traindf[traindf.lanuage=='zh'].iloc[:,0:-1]
lang_sets['ru'] = traindf[traindf.lanuage=='ru'].iloc[:,0:-1]
lang_sets['es'] = traindf[traindf.lanuage=='es'].iloc[:,0:-1]

lang_sets['commons'] = traindf[traindf.lanuage=='commons'].iloc[:,0:-1]
lang_sets['www'] = traindf[traindf.lanuage=='www'].iloc[:,0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0).int()/ lang_sets[key].shape[0]


# In[ ]:

days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[10,10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'
       }

for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()

#downsample

downsample = traindf.sample(frac=0.068935565926528).sort_index()

downsample.to_csv("Downsample",index_label=False)

down = pd.read_csv('Downsample')

downsample100 = down.sample(frac=0.01).sort_index()
downsample100.to_csv("downsample100.csv",index_label=False)


# ## others


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from fbprophet import Prophet
import matplotlib.pyplot as plt
import math as math

get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:

# Load the data
train = pd.read_csv("train_1.csv")
keys = pd.read_csv("key_1.csv")
ss = pd.read_csv("sample_submission_1.csv")


# In[37]:

train.head()


# In[38]:

# Check the data
print("Check the number of records")
print("Number of records: ", train.shape[0], "\n")

print("Null analysis")
empty_sample = train[train.isnull().any(axis=1)]
print("Number of records contain 1+ null: ", empty_sample.shape[0], "\n")


# In[39]:

empty_sample.iloc[np.r_[0:10, len(empty_sample)-10:len(empty_sample)]]


# In[40]:

import re

def breakdown_topic(str):
    m = re.search('(.*)\_(.*).wikipedia.org\_(.*)\_(.*)', str)
    if m is not None:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    else:
        return "", "", "", ""

print(breakdown_topic("Рудова,_Наталья_Александровна_ru.wikipedia.org_all-access_spider"))
print(breakdown_topic("台灣災難列表_zh.wikipedia.org_all-access_spider"))
print(breakdown_topic("File:Memphis_Blues_Tour_2010.jpg_commons.wikimedia.org_mobile-web_all-agents"))


# In[41]:

page_details = train.Page.str.extract(r'(?P<topic>.*)\_(?P<lang>.*).wikipedia.org\_(?P<access>.*)\_(?P<type>.*)')

page_details[0:10]


# In[42]:

unique_topic = page_details["topic"].unique()
print(unique_topic)
print("Number of distinct topics: ", unique_topic.shape[0])


# In[51]:

fig, axs  = plt.subplots(3,1,figsize=(6,6))
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

page_details["lang"].value_counts().sort_index().plot.bar(ax=axs[0])
axs[0].set_title('Language - distribution')

page_details["access"].value_counts().sort_index().plot.bar(ax=axs[1])
axs[1].set_title('Access - distribution')

page_details["type"].value_counts().sort_index().plot.bar(ax=axs[2])
axs[2].set_title('Agent - distribution')

plt.tight_layout()

