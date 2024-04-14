#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame = True)
print(housing)


# In[3]:


housing['data'].head()


# In[4]:


housing['target'].head()


# In[5]:


df = pd.DataFrame(housing['data'])
df


# In[6]:


df['Price'] = housing['target']
df


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# In[9]:


df.describe().T


# In[10]:


df.hist(figsize=(10,8))
plt.show()


# In[11]:


df.corr()


# In[12]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)


# In[13]:


X = df.drop('Price',axis=1)
y = df['Price']


# In[14]:


X


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[17]:


df.shape


# In[18]:


X_train


# In[19]:


X_test


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# In[22]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[23]:


y_pred = model.predict(X_test)
y_pred


# In[24]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(mean_squared_error(y_pred,y_test))


# In[25]:


print(mean_absolute_error(y_pred,y_test))


# In[26]:


r2_score(y_pred,y_test)


# In[27]:


df.iloc[1,:].values


# In[28]:


model.predict([[ 8.30140000e+00,  2.10000000e+01,  6.23813708e+00,  9.71880492e-01,
        2.40100000e+03,  2.10984183e+00,  3.78600000e+01, -1.22220000e+02]])


# In[29]:


df.iloc[0,:].values


# In[30]:


model.predict([[8.3252    ,   41.        ,    6.98412698,    1.02380952,
        322.        ,    2.55555556,   37.88      , -122.23 ]])

