#!/usr/bin/env python
# coding: utf-8

# In[185]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[186]:


df = pd.read_csv('/Users/sadegh/Desktop/DataSet GitHub/KNN/mammogram_weka_dataset.csv')


# In[187]:


df.head()


# In[188]:


df.info()


# In[189]:


df.corr()


# In[190]:


#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[191]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('severity',axis=1))


# In[192]:


scaled_features = scaler.transform(df.drop('severity',axis=1))


# In[193]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[194]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['severity'],
                                                    test_size=0.3)


# In[195]:


from sklearn.neighbors import KNeighborsClassifier


# In[196]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[197]:


knn.fit(X_train,y_train)


# In[198]:


pred = knn.predict(X_test)


# In[199]:


from sklearn.metrics import classification_report,confusion_matrix


# In[200]:


cm = confusion_matrix(y_test,pred)


# In[201]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[202]:


print(classification_report(y_test,pred))


# In[203]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[204]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[209]:


# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:




