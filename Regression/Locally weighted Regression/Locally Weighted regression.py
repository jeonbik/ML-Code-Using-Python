#!/usr/bin/env python
# coding: utf-8

# In[43]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


file=pd.read_csv('data.csv')
print(file.describe())
print(file.size)
file.fillna(0)


# In[18]:


first_input=file['Engine HP']
second_input=file['highway MPG']
mfirst=np.mat(first_input)
msecond=np.mat(second_input)
rows_of_mfirst=np.shape(mfirst)[1]
ones_matrix=np.ones((1,rows_of_mfirst),dtype=int)
input_x=np.hstack((ones_matrix.T,mfirst.T))


# In[49]:


def not_gaussian_density(point,inputx,thau):
    m=np.shape(inputx)[0]
    weights=np.mat(np.eye(m))
    
    for j in range (m):
        difference_in_x=point-inputx[j]
        weights[j,j]=np.exp(difference_in_x*difference_in_x.T/(-2.0*thau**2))
    return weights
        




def local_theta(point,inputx,outputy,thau):
    weight=not_gaussian_density(point,inputx,thau)
    wt=(inputx.T*(weight*inputx)).I*(inputx.T*weight*outputy.T)
    return wt



def Locally_weighted_theta(input_x,output_y,thau):
    m=np.shape(input_x)[0]
    new_theta=np.zeros(m)
    for i in range (m):
        new_theta[i]=input_x[i]*local_theta(input_x[i],input_x,output_y,thau)
    return new_theta


# In[ ]:

y_predict=Locally_weighted_theta(input_x,msecond,0.5)

