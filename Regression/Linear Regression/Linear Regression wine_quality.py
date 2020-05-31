#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sn


# In[26]:


file=pd.read_csv("winequality-red.csv",sep=";")


# In[27]:


file.shape


# In[28]:


file.info()


# In[29]:


plt.scatter(file['alcohol'],file['quality'])


# In[30]:


file.head(5)


# In[31]:


file['quality'].value_counts()


# In[32]:


print("Total values of fixed acidity is: " + str(len(list(file["fixed acidity"].value_counts()))))
print("Total values of volatile acidity is: " + str(len(list(file["volatile acidity"].value_counts()))))
print("Total values of citric acid is: " + str(len(list(file["citric acid"].value_counts()))))
print("Total values of residual sugar is: " + str(len(list(file["residual sugar"].value_counts()))))
print("Total values of chlorides is: " + str(len(list(file["chlorides"].value_counts()))))
print("Total values of total sulfur dioxide is: " + str(len(list(file["total sulfur dioxide"].value_counts()))))
print("Total values of density is: " + str(len(list(file["density"].value_counts()))))
print("Total values of pH is: " + str(len(list(file["pH"].value_counts()))))
print("Total values of alcohol is: " + str(len(list(file["alcohol"].value_counts()))))


# In[33]:


file.isnull().any()


# In[34]:


sn.countplot(x="quality",data=file)
plt.show()


# In[35]:


file["alcohol"].plot.hist()
plt.xlabel("Alcohol")
plt.ylabel("counts")
plt.show()
print()
file["density"].plot.hist()
plt.xlabel("density")
plt.ylabel("counts")
plt.show()
print()
file["pH"].plot.hist()
plt.xlabel("pH Values")
plt.ylabel("counts")
plt.show()
print()
file["residual sugar"].plot.hist()
plt.xlabel("Residual Sugar Amounts")
plt.ylabel("counts")
plt.show()
file["total sulfur dioxide"].plot.hist()
plt.xlabel("total sulfur dioxide")
plt.ylabel("counts")
plt.show()
 


# In[36]:


sn.boxplot(x="quality",y="alcohol",data=file)
plt.show()
print()
sn.boxplot(x="quality",y="density",data=file)
plt.show()
print()
sn.boxplot(x="quality",y="pH",data=file)
plt.show()
print()
sn.boxplot(x="quality",y="total sulfur dioxide",data=file)
plt.show()


# # Cleaning Data
# Lets drop some unwanted columns

# In[37]:


sn.heatmap(file.isnull(),yticklabels=False,cbar=False)
print()


# In[38]:


file.drop(["free sulfur dioxide",'total sulfur dioxide'],axis=1,inplace=True)


# In[39]:


file.shape
file.head(5)


# # Training and Testing the Model

# In[40]:


X=file.iloc[:,[0,1,2,3,4,5,6,7,8]].values
#X=file.iloc[:,6].values.reshape(-1,1)
y=file.iloc[:,9].values


# In[41]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[42]:


lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_predict=lin_reg.predict(x_test)


# In[43]:


metrics.mean_absolute_error(y_predict,y_test)


# In[44]:


a=metrics.mean_squared_error(y_predict,y_test)
a


# In[45]:


np.sqrt(a)


# # Finding theta and intercept

# In[51]:


data=pd.DataFrame(lin_reg.coef_,columns=['Coefficient'])
data


# In[47]:


lin_reg.intercept_


# In[ ]:





# In[ ]:





# In[ ]:




