#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as nb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


# In[47]:


file=pd.read_csv("linear_reg.csv")


# In[48]:


file.describe()


# In[49]:


#checking if any values is null
file.isnull().any()


# In[50]:


#removing null values (if any)
file=file.fillna(method='ffill')


# In[ ]:





# In[51]:


X=file[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates','alcohol']].values
y=file['quality'].values


# In[52]:


#splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[53]:


#training the data
linear_model=LinearRegression()
linear_model.fit(X_train,y_train)


# In[54]:


coeff_df = pd.DataFrame(linear_model.coef_,columns=['Coefficient'])


# In[55]:


coeff_df


# In[74]:


print(linear_model.intercept_)
print(linear_model.coef_)


# In[56]:


y_predict=linear_model.predict(X_test)


# In[57]:


df=pd.DataFrame({"Predicted: ":y_predict, "Actual ":y_test})
print(df.head(20))


# In[71]:


mean_error=metrics.mean_absolute_error(y_test,y_predict)
mean_sq_error=metrics.mean_squared_error(y_test,y_predict)
root_mean=nb.sqrt(mean_sq_error)


# In[72]:


print(mean_error)
print(mean_sq_error)
print(root_mean)


# In[ ]:




