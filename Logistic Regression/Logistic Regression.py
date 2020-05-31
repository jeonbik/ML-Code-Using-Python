#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


file=pd.read_csv("bank.csv",header=0,sep=';')


# In[3]:



file.head(10)


# In[4]:


print(file.shape)
file['education'].index


# # Exploring Data

# In[5]:


print(file['y'].value_counts())
sns.countplot(x='y',data=file)
plt.show()


# In[6]:


sns.countplot(x='housing',hue="loan",data=file)
plt.show()


# In[7]:


sns.countplot(x='loan',data=file)
plt.show()


# In[8]:


print(file['marital'].value_counts())
print()
print(file['education'].value_counts())
print()
print(file['education'].unique())


# In[9]:


sns.countplot(x='y',hue='marital',data=file)
plt.show()


# In[10]:


sns.countplot(x='y',hue='education',data=file)
plt.show()


# In[11]:


file['age'].plot.hist()
plt.show()


# In[12]:


sns.countplot(x='y',hue='job',data=file)
plt.show()


# In[13]:


file.info()


# In[14]:


file['balance'].plot.hist()
plt.show()


# # Cleaning the Data

# Once we anayzise the relation between differnt features, we move forward to clean the data and make them ready for trining our model.

# print(file.isnull().any())
# print()
# print(file.isnull().sum())

# In[15]:


sns.boxplot(x='y',y="age",data=file)
plt.show()


# Lets drop some unwanted features and NaN values

# In[16]:


file.drop("pdays",axis=1,inplace=True)
sns.heatmap(file.isnull(),yticklabels=False,cbar=False)
plt.show()


# In[17]:


file.head(2)


# Converting variable into catgorical data

# In[18]:


job=pd.get_dummies(file['job'],drop_first=True)
print(job.head(2))
print()
marital=pd.get_dummies(file["marital"],drop_first=True)
print(marital.head(2))
print()
education=pd.get_dummies(file["education"],drop_first=True)
print(education.head(2))
# print()
# housing=pd.get_dummies(file["housing"],drop_first=True)
# print(housing.head(2))
# print()
# loan=pd.get_dummies(file['loan'],drop_first=True)
# print(loan.head(2))
y=pd.get_dummies(file['y'],drop_first=True)
print()
print(y.head(2))


# Lets Concatinate these data into dataframe
# 

# In[20]:


file=pd.concat([file,job,marital,education,y],axis=1)
file.head(2)
file.info()


# Lets drop those column whose dummy variables are just been made

# In[21]:


file.drop(['job','marital','education','default','housing','loan','contact','day','month','duration','campaign','previous','poutcome','y'],axis=1,inplace=True)


# In[22]:


file.head(5)


# Its time to train and and test the data

# In[23]:


x=file.drop('yes',axis=1)
y=file['yes']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[25]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
y_predict=log_reg.predict(x_test)


# In[33]:


print(classification_report(y_test,y_predict))


# # Calculating Errors

# In[27]:


abs_error=metrics.mean_absolute_error(y_predict,y_test)
abs_sqr=metrics.mean_squared_error(y_predict,y_test)
mean_sqrt=np.sqrt(abs_sqr)


# In[28]:


print("The mean absolute error is: "+ str(abs_error))
print("The mean square error is: "+ str(abs_sqr))
print("The mean square root error is: "+ str(mean_sqrt))


# In[29]:


confusion_matrix(y_predict,y_test)


# In[30]:


(accuracy_score(y_predict,y_test))*100


# In[32]:


coef=pd.DataFrame({'coeff': log_reg.coef_.flatten()})
print(coef)
print()
print("The intercept for beta_0 is: "+str(log_reg.intercept_))


# In[47]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, log_reg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

