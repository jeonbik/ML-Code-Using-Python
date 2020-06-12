#!/usr/bin/env python
# coding: utf-8

# In[29]:


import csv
import math
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


file1=pd.read_csv("pima-indians-diabetes.data.csv")
file1.head(5)


# This file has a column or features name missing So we will assign the features to the data file.

# In[31]:


column = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
file=pd.read_csv("pima-indians-diabetes.data.csv",names=column)
file.head(10)


# In[32]:


file.describe()


# In[33]:


file.info()


# In[34]:


file.isnull().any()


# In[35]:


file['Outcome'].value_counts()


# In[36]:


sn.countplot(x='Outcome',data=file)
plt.show()


# In[37]:


sn.countplot(x="Outcome",hue="Pregnancies",data=file)
plt.show()
file["Age"].plot.hist()
plt.show()


# In[38]:


sn.boxplot(x='Outcome',y="Age",data=file)
plt.show()


# Once we have analyzed the data, lets code the math behind it

# converting the givern data's into float.
# our DataFrame has a data in the form of the string. Since it is a pandaas dataframe we will not be able
# to convert the stringh with . and columns name into the float value. So, we will read it in the CSV form 
# and convert it into the float values.
# 

# In[39]:


def to_float():
    datas=csv.reader(open('pima-indians-diabetes.data.csv'))
    data_set=list(datas)
    for i in range (len(data_set)):
        data_set[i]=[float(x) for x in data_set[i]]
    return data_set


# In[40]:


def print_rows():
    for row in to_float():
        print (row)
print_rows()


# We have analyzed the data sofar.
# Lets split the data to training and test dataset to measure the accuracy of the model we get.

# In[41]:


def train_test(data_set,splitting_ratio):
    train_size=int(len(data_set)*splitting_ratio)
    training_set=[]
    copy_data=list(data_set)
    while len(training_set)<train_size:
        index=random.randrange(len(copy_data))
        training_set.append(copy_data.pop(index))
    return [training_set,copy_data]


# Lets seperate each class and input vectors
# 

# In[42]:


def seperating_features(data_set):
    features={}
    for i in range (len(data_set)):
        input_vector=data_set[i]
        if (input_vector[-1] not in features):
            features[input_vector[-1]]=[]
        features[input_vector[-1]].append(input_vector)
    return features


# In[43]:


def mean(numbers):
    total_sum=sum(numbers)
    mean=total_sum/float(len(numbers))
    return mean

def stdev(numbers):
    means=mean(numbers)
    var_num=sum([pow(x-means,2) for x in numbers])
    variance=var_num/float(len(numbers)-1)
    return math.sqrt(variance)


# In[44]:


def summarize(data_set):
    summeries=[(mean(attribute),stdev(attribute)) for attribute in zip (*data_set)]
    del summeries[-1]
    return summeries
summarize(to_float())


# In[45]:


file.describe(include='all')[1:3]


# In[46]:


def summarize_by_class(data_set):
    seperated=seperating_features(data_set)
    summaries={}
    for classValue, instances in seperated.items():
        summaries[classValue]=summarize(instances)
    return summaries


# In[47]:


def calculate_probability(x,mean,stdev):
    exp=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exp


# In[48]:


def calculate_class_probabilities(summaries,input_vector):
    probabilities={}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue]=1
        for i in range (len(classSummaries)):
            mean,stdev=classSummaries[i]
            x=input_vector[i]
            probabilities[classValue] *=calculate_probability(x,mean,stdev)
        return probabilities
    


# In[49]:


def predict(summaries, inputVector):
    probabilites= calculate_class_probabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilites.items():
        if bestLabel is None  or probability > bestprob:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
            


# In[50]:


def get_prediction(summaries,test_set):
    predictions=[]
    for i in range (len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


# In[51]:


def get_accuracy(test_set,predictions):
    correct = 0
    for x in range (len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct +=1
    return (correct/float(len(test_set)))*100.0
    


# In[52]:


def main():
    split_ratio = 0.70
    data_set= to_float()
    train_set, test_set = train_test(data_set,split_ratio)
    summaries=summarize_by_class(train_set)
    predictions=get_prediction(summaries,test_set)
    accuracy=get_accuracy(test_set, predictions)
    print("Accuracy is : {0}%".format(accuracy))

    
main()


# In[53]:


file.shape


# In[71]:


X = file.iloc[:,[0,1,2,3,4,5,6,7]].values
y=file.iloc[:,8].values
train_x,test_x,train_y,test_y = train_test_split (X,y,test_size=0.3, random_state=0)
model=GaussianNB()
model.fit(train_x,train_y)
y_predict=model.predict(test_x)


# In[72]:


print(metrics.classification_report(test_y, y_predict))


# In[70]:


print(metrics.confusion_matrix(test_y,y_predict))


# In[86]:


df=pd.DataFrame({"predicted : ": y_predict, "Actual: ":test_y})
df.head(20)

