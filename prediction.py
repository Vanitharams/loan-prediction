#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[17]:


test = pd.read_csv(r'C:\Users\User\Downloads\test_dataset.csv')


# In[18]:


test


# In[19]:


train=pd.read_csv(r'C:\Users\User\Downloads\train_dataset.csv')


# In[20]:


train


# In[21]:


train.info()


# In[22]:


test.head(10)


# In[23]:


train.describe()


# In[24]:


train.isnull().sum().sort_values(ascending=False)


# In[25]:


test.isnull().sum().sort_values(ascending=False)


# In[26]:


train['Credit_History']=train['Credit_History'].fillna(train['Credit_History'].median())


# In[27]:


train['Self_Employed']=train['Self_Employed'].fillna(train['Self_Employed'].mode()[0])


# In[28]:


train['Dependents']=train['Dependents'].fillna(train['Dependents'].mode()[0])


# In[29]:


train['Gender']=train['Gender'].fillna(train['Gender'].mode()[0])


# In[30]:


train['Married']=train['Married'].fillna(train['Married'].mode()[0])


# In[31]:


train['Loan_Amount_Term']=train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean())
train['LoanAmount']=train['LoanAmount'].fillna(train['LoanAmount'].mean())


# In[32]:


train.groupby("Gender")["Loan_Status"].value_counts(normalize=True)


# In[33]:


train.groupby("Gender")["Loan_Status"].value_counts()


# In[34]:


train.drop(['Loan_ID'],axis=1,inplace=True)


# In[35]:


sns.countplot(data=train,x='Loan_Status')


# In[36]:


sns.countplot(data=train["Loan_Status"],x=train["Gender"],hue=train["Loan_Status"])


# In[37]:


sns.countplot(data=train["Loan_Status"],x=train["Married"],hue=train["Loan_Status"])


# In[38]:


sns.countplot(data=train["Loan_Status"],x=train["Dependents"],hue=train["Loan_Status"])


# In[39]:


sns.countplot(data=train["Loan_Status"],x=train["Education"],hue=train["Loan_Status"])


# In[40]:


sns.pairplot(train)


# In[41]:


Y=train.Loan_Status


# In[42]:


Y=np.where(Y=="Y",1,0)


# In[43]:


X=pd.get_dummies(train.drop("Loan_Status",axis=1),drop_first=True)


# In[44]:


col_names=X.columns


# In[45]:


scaler=StandardScaler()


# In[46]:


scaler.fit(X)


# In[47]:


X_scaled=scaler.transform(X)


# In[48]:


X=pd.DataFrame(X_scaled,columns=col_names)


# In[49]:


X.head(10)


# In[50]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[51]:


models=[]


# In[52]:


models.append(('Logistic Regression',LogisticRegression()))


# In[53]:


models.append(('K Nearest Neighbors',KNeighborsClassifier()))


# In[55]:


models.append(('Decision Tree',DecisionTreeClassifier()))


# In[56]:


models.append(('Gaussian Naive Bayes',GaussianNB()))


# In[57]:


models.append(('Random Forest',RandomForestClassifier()))


# In[61]:


for name,algorithm in models:
    model=algorithm
    model.fit(X_train,Y_train)
    prediction=model.predict(X_test)
    print('The Accuracy of the %s is %f:'%(name,accuracy_score(prediction,Y_test)))
print('\n')


# In[ ]:




