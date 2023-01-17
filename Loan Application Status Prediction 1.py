#!/usr/bin/env python
# coding: utf-8

# # Loan Application Status Prediction
# 
# ### Problem Statement:
# 
# This dataset includes details of applicants who have applied for loan. The dataset includes details like credit history, loan amount, their income, dependents etc. 
# 
# ### Independent Variables:
# 
# - Loan_ID
# 
# - Gender
# 
# - Married
# 
# - Dependents
# 
# - Education
# 
# - Self_Employed
# 
# - ApplicantIncome
# 
# - CoapplicantIncome
# 
# - Loan_Amount
# 
# - Loan_Amount_Term
# 
# - Credit History
# 
# - Property_Area
# 
# ## Dependent Variable (Target Variable):
# 
# - Loan_Status
# 
# You have to build a model that can predict whether the loan of the applicant will be approved or not on the basis of the details provided in the dataset. 
# 
# 
# Downlaod Files:
# 
# https://github.com/dsrscientist/DSData/blob/master/loan_prediction.csv
# 

# ### To set a raw data from github

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[3]:


#importing data
df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# * There are 614 rows and 13 columns in data set
# * There are 3 different datatypes in the dataset which include float64(4), int64(1), object(8)
# 

# In[6]:


# cheaking the null values
sns.heatmap(df.isnull())
plt.title("Null values")
plt.show()


# In[7]:


df.isnull().sum()


# In[8]:


#filling the null value
df = df.fillna(df.mean().iloc[0])


# In[9]:


df.isnull().sum()


# ### using the fillna function to fill the missing values
# 

# In[10]:


# statistical measures
df.describe()


# In[11]:


# label incoding
df.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[12]:


df.head()


# In[13]:


#Dependents
df['Dependents'].value_counts()


# In[14]:


#replacing the value of 3+ to 4
df = df.replace(to_replace='3+', value= 4)


# In[15]:


#Dependents
df['Dependents'].value_counts()


# In[16]:


df.columns


# ## Data visualization

# In[17]:


# Gender and loan_status
sns.countplot(x = 'Gender', hue='Loan_Status', data=df)
plt.show()


# * Orange bar is approved loan
# * Blue bar is not-approved loan
# * As you can see male applicants have higher percentage of loan approved than female 
# * As compared to female male apply more for loan

# In[18]:


# Married and loan_status
sns.countplot(x = 'Married', hue='Loan_Status', data=df)
plt.show()


# * Married people apply more for loan and married people approved more than single people
# 

# In[19]:


# Dependents and loan_status
sns.countplot(x = 'Dependents', hue='Loan_Status', data=df)
plt.show()


# * More non Dependents people apply for loan
# * There is high chance of getting loan who don't have any dependents member in family

# In[20]:


# Education and loan_status
sns.countplot(x = 'Education', hue='Loan_Status', data=df)
plt.show()


# * Educated people approved more than non educated one 
# 

# In[21]:


# Self_Employed and loan_status
sns.countplot(x = 'Self_Employed', hue='Loan_Status', data=df)
plt.show()


# * Job people apply more for loan and get approved too
# * Self_Employed people not get approved that essly

# In[22]:


# Credit_History and loan_status
sns.countplot(x = 'Credit_History', hue='Loan_Status', data=df)
plt.show()


# * 0 means bad Credit History
# * 1 means good Credit History
# * The people have good credit history get approved more than bad one

# In[23]:


# Property_Area and loan_status
sns.countplot(x = 'Property_Area', hue='Loan_Status', data=df)
plt.show()


# * Higher percentage of loan approval is for semi-urban area
# * 2nd Urban area
# * Then Rural area

# In[24]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='Blues');


# In[25]:


# print out column with unique values
for col in df.columns:
    if df[col].dtypes == 'object':
        num_of_unique_cat = len (df[col].unique())
        print("Features '{col_name}' has '{unique_cat}' unique categories". format(col_name=col, unique_cat=num_of_unique_cat))


# ##### convert categorical columns to numerical values
# 

# In[26]:


df['Self_Employed'].replace({'Yes':1,'No':0},inplace=True)
df['Married'].replace({'Yes':1,'No':0},inplace=True)
df['Gender'].replace({'Male':1,'Female':0},inplace=True)
df['Education'].replace({'Graduate':1,'Not Graduate':0},inplace=True)
df['Property_Area'].replace({'Urban':2,'Semiurban':1,'Rural':0},inplace=True)
df['Loan_Status'].replace({'Y':1,'N':0},inplace=True)


# In[27]:


df.head()


# In[28]:


# separating the data and label
X = df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = df['Loan_Status']


# In[29]:


print(X)
print(Y)


# ## Spliting the data

# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=5)


# In[31]:


print(X.shape, X_train.shape, X_test.shape)


# ## Logistic Regression
# 
# Logistic regression is applied to predict the categorical dependent variable. In other words, it's used when the prediction is categorical, for example, yes or no, true or false, 0 or 1. The predicted probability or output of logistic regression can be either one of them, and there's no middle ground.

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[33]:


model = LogisticRegression() #define the model


# In[34]:


model.fit(X_train,Y_train)#fit the model


# In[35]:


model.score(X_train,Y_train)


# In[36]:


model.score(X_test,Y_test)


# In[37]:


y_pred = model.predict(X_test)#predict on test sample


# In[38]:


from sklearn.metrics import r2_score


# In[39]:


r2_score(Y_test,y_pred)


# In[ ]:




