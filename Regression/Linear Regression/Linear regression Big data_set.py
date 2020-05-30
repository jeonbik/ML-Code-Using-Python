#!/usr/bin/env python
# coding: utf-8

# In[18]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[19]:


file=pd.read_csv("data.csv")
print(file)


# In[20]:


file.describe()
print(file.size)


# In[21]:


file.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[22]:


file.isnull().any()
file=file.fillna(0)
print(file.isnull().any())


# In[26]:


x=file[['Year',"Engine HP","Engine Cylinders",'Number of Doors','highway MPG','city mpg','Popularity']].values
y=file['MSRP'].values


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[28]:


linear_model=LinearRegression()
linear_model.fit(x_train,y_train)
#y_predict=linear_model(x_train,y_train)


# In[30]:


y_predict=linear_model.predict(x_test)


# In[32]:


data_frame=pd.DataFrame({"Predicted ":y_predict, "Real ": y_test})
print(data_frame)


# In[34]:


coef_model=pd.DataFrame(linear_model.coef_,columns=["Coefficient"])
print("The fixed value is:", linear_model.intercept_)
print(coef_model)


# 
# 

# In[35]:


absolute_error=metrics.mean_absolute_error(y_predict,y_test)
mean_error=metrics.mean_squared_error(y_predict,y_test)
mean_square=np.sqrt(mean_error)
print(absolute_error)
print(mean_error)
print(mean_square)


# In[ ]:




