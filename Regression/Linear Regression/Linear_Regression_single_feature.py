#!/usr/bin/env python
# coding: utf-8

# In[158]:




import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[146]:


file=pd.read_csv("linear_regression.csv")


# In[162]:


file


# In[ ]:


("")


# In[148]:


file.describe()


# In[149]:


file.plot(x="x",y="y", style='*')
plt.title(" First ML Model.   X vs Y")
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.show()


# In[150]:


#Assigning x and y values and providing required dimension
x=file['x'].values.reshape(-1,1)
y-file['y'].values.reshape(-1,1)
print(y)


# In[151]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[152]:


linear_model=LinearRegression()
linear_model.fit(x_train,y_train)


# In[153]:


print(linear_model.intercept_)
print(linear_model.coef_)


# In[154]:


y_predict=linear_model.predict(x_test)


# In[164]:


data_set=pd.DataFrame({'Actual Values': y_test.flatten(),'Predicted ': y_predict.flatten()})
print(data_set)


# In[156]:


plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,y_predict,color="red",linewidth=2)
plt.show()


# In[159]:


mean_absolute=metrics.mean_absolute_error(y_test,y_predict)
mean_square=metrics.mean_squared_error(y_test,y_predict)
root_mean_error=np.sqrt(mean_square)


# In[161]:


print(mean_absolute)
print(mean_square)
print(root_mean_error)


# In[165]:


print("Linear regression model is: ",linear_model.intercept_,"+",linear_model.coef_,"x")


# In[ ]:




