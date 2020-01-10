#!/usr/bin/env python
# coding: utf-8

# In[341]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import datasets
from sklearn import metrics
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[342]:


sumoData = 'C:/Users/Magnus/Downloads/SumoData2.csv'
data = pd.read_csv(sumoData, sep=';')
data_shuffled = data.sample(frac=1).reset_index(drop=True)


# In[343]:


data_shuffled_clean = data_shuffled.dropna(subset = ['deviceCategory', 'startTime', 'genreCode'])


# In[344]:


gnb = GaussianNB()
mnb = MultinomialNB()
ohe = OneHotEncoder()
le = LabelEncoder()
cnb = ComplementNB()


# In[345]:


df_userID = data_shuffled_clean['userId']


# In[346]:


df_specific_user = data_shuffled_clean.loc[df_userID == df_userID.value_counts().idxmax()]


# In[347]:


main_genres = ['DR10', 'UH70', 'UH60', 'IN31', 'DR30', 'UH20', 'BA00', 'IN23']
df_specific_user = df_specific_user[df_specific_user.genreCode.isin(main_genres)]


# In[348]:


df_playingTimeSec_user = df_specific_user['playingTime_sec']
df_device_user = df_specific_user['deviceCategory']
df_genreCode_user = df_specific_user['genreCode']
df_startTime_user = df_specific_user['startTime']


# In[349]:


df_specific_user


# In[350]:


df_dates = pd.to_datetime(df_startTime_user)
df_dayOfTheWeek = df_dates.dt.weekday
df_datetime = pd.to_datetime(df_startTime_user)
df_hours = df_datetime.dt.hour


# In[351]:


# Function that splits startTime values into hours, and appends values into a DataFrame according to
# four hour intervals. Returns the DataFrame
def hour_extractor(list_of_hours):
    hour_list = []
    for hour in list_of_hours:
        if 0 <= hour < 4:
            hour_list.append(0)
        elif 4 <= hour < 8:
            hour_list.append(1)
        elif 8 <= hour < 12:
            hour_list.append(2)
        elif 12 <= hour < 16:
            hour_list.append(3)
        elif 16 <= hour < 20:
            hour_list.append(4)
        elif 20 <= hour <= 24:
            hour_list.append(5)
    df_start = pd.DataFrame({'start_hour': hour_list})
    return df_start


# In[352]:


df_startTime_hours = hour_extractor(df_hours)


# In[353]:


df_startTime_hours = df_startTime_hours.start_hour


# In[354]:


df_device_encoded = le.fit_transform(df_device_user)
df_dayOfTheWeek_encoded = le.fit_transform(df_dayOfTheWeek)
df_startTime_hours_encoded = le.fit_transform(df_startTime_hours)


# In[355]:


df_genreCode_encoded = le.fit_transform(df_genreCode_user)


# In[356]:


features = list(zip(df_device_encoded, df_dayOfTheWeek_encoded, df_startTime_hours_encoded))


# In[357]:


label = df_genreCode_encoded


# In[358]:


x_train, x_test, y_train, y_test = train_test_split(features, label, shuffle=False)


# In[359]:


gnb = gnb.fit(x_train, y_train)


# In[360]:


y_pred = gnb.predict(x_test)


# In[361]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[362]:


size = len(x_test)


# In[363]:


print("Number of mislabeled points out of a total %d points : %d"
      % (size,(y_test != y_pred).sum()))


# In[364]:


mnb = mnb.fit(x_train, y_train)


# In[365]:


y_pred_mnb = mnb.predict(x_test)


# In[366]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_mnb))


# In[367]:


size = len(x_test)


# In[368]:


print("Number of mislabeled points out of a total %d points : %d"
      % (size,(y_test != y_pred_mnb).sum()))


# In[369]:


cnb = cnb.fit(x_train, y_train)


# In[370]:


y_pred_cnb = cnb.predict(x_test)


# In[371]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_cnb))
size = len(x_test)


# In[372]:


print("Number of mislabeled points out of a total %d points : %d"
      % (size,(y_test != y_pred_cnb).sum()))


# In[ ]:





# In[ ]:




