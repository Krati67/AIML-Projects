#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


df = pd.read_csv("housepricepred.csv")
df.head(15)


# In[7]:


df.shape


# In[9]:


df.dtypes


# In[10]:


# Convert the SalePrice column into float64 data type

df['SalePrice'].astype('float64')


# In[11]:


# count of null values

df.isnull().sum()


# In[13]:


# Compute total number of cells in dataframe and cells with missing values
total_cells = np.product(df.shape)
total_missing = df.isnull().sum().sum()
total_missing


# In[14]:


# Compute percentage
percentage_missing = total_missing / total_cells * 100
print(percentage_missing)


# In[15]:


# Drop columns with missing values

col_with_na_dropped = df.dropna(axis = 1)
col_with_na_dropped.head()


# In[16]:


col_with_na = df.columns[df.isnull().any()]
list(col_with_na)


# In[17]:


# since we were dropping a significant amount of columns (features) as in almost a quarter of it. We will go with other method.
# Features are the characteristics that deescribes the house. If we remove features that are significant in explaining the sale price of the house, our model will not be able to make accurate predictions.

# Suppose we want to fill missing data in the LotFrontage column 
# First let's examine the data type

df['LotFrontage'].dtype


# In[18]:


df['LotFrontage'].head(10)


# In[19]:


# Compute median

df['LotFrontage'].median()


# In[20]:


# Impute missing data in LotFrontage with median

df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['LotFrontage'].head(10)


# In[21]:


# Let's see the value counts in the Garage Type including the nulll value

df['GarageType'].value_counts(dropna = False)


# In[22]:


df['GarageType'].mode()[0]


# In[23]:


df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageType'].tail(10)


# In[24]:


df['GarageQual'].value_counts(dropna = False)


# In[25]:


df['GarageQual'] = df['GarageQual'].fillna('Unknown')
df['GarageQual'].value_counts(dropna = False)


# In[26]:


df.info()


# In[ ]:




