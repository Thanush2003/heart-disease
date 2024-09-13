


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import os
import joblib
import pickle
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


file_path = r'C:\Users\N THANUSH\Desktop\dataset\heart.csv'
df=pd.read_csv(file_path)


# In[3]:


df.drop(['education'],axis=1,inplace=True)
df.head()


# In[4]:


df.rename(columns={'male':'Sex_male'},inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


count = 0
for i in df.isnull().sum(axis=1):
  if(i>0):
    count += 1
print("Total number of rows with missing values is ",count)
print('since its only',round((count/len(df.index))*100),' percent of the entire dataset the rows with missing values are excluded')


# In[7]:


df.dropna(axis=0,inplace=True)


# In[8]:


def draw_histograms(dataframe,features,rows,cols):
  fig = plt.figure(figsize=(20,20))
  for i, feature in enumerate(features):
    ax = fig.add_subplot(rows,cols,i+1)
    dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
    ax.set_title(feature+" Distribution",color='DarkRed')
  fig.tight_layout()
  plt.show()
draw_histograms(df,df.columns,6,3)


# In[9]:


df.TenYearCHD.value_counts()


# In[10]:


sn.countplot(x='TenYearCHD',data=df)


# In[11]:


print(df.describe())


# In[12]:


from statsmodels.tools import add_constant as add_constant
df_constant = add_constant(df)
df_constant.head()


# In[15]:


import sklearn
new_features = df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x = new_features.iloc[:,:-1]
y = new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=5)


# In[16]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)


# In[17]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
conf_matrix = pd.DataFrame(data=cm,columns=['predicted:0','predicted:1'],index=['Actual:0',"Actual:1"])
plt.figure(figsize=(8,5))
sn.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu")


# In[19]:


TN = cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity = TP/float(TP+FN)
specificity = TN/float(TN+FP)


# In[20]:


print("The accuracy of the model = TP+TN/(TP+TN+FP+FN) = ",(TP+TN)/float(TP+TN+FP+FN),'\n',
      "The Missclassification = 1-Accuracy = ",1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
      "Sensitivity or True Positive Rate = TP/(TP+FN) = ",TP/float(TP+FN),'\n',
      "Specificity or True Negative Rate = TN/(TN+FP) = ",TN/float(TN+FP),'\n',
      "Positive Predicted value = TP/(TP+FP) = ",TP/float(TP+FP),'\n',
      "Negative Predictive value = TN/(TN+FN) = ",TN/float(TN+FN),'\n',
      "Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ",sensitivity/(1-specificity),'\n',
      "Negative Likelihood Ratio = (1-Sensitivity)/Specificity = ", (1-sensitivity)/specificity)


# In[22]:


y_pred_prob = logreg.predict_proba(x_test)[:,:]
y_pred_prob_df = pd.DataFrame(data = y_pred_prob,columns = ['Prob of no heart disease(0)','Prob of heart disease (1)'])
y_pred_prob_df.head()


# In[23]:


from sklearn.preprocessing import binarize
for i in range(1,5):
  cm2 = 0
  y_pred_prob_yes = logreg.predict_proba(x_test)
  y_pred2 = binarize(y_pred_prob_yes)[:,1]
  cm2 = confusion_matrix(y_test,y_pred2)
  print("with ",i/10," threshold the confusion matrix is ","\n","with ",cm2[0,0]+cm2[1,1],' correct predictions and ',cm2[1,0],' type 2 erros(False Negative)','\n','sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n')


# In[24]:


from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("ROC curve for heart disease classifier")
plt.xlabel("False Positive rate(1-specificity)")
plt.ylabel("True Positive rate(sensitivity)")
plt.grid(True)


# In[25]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])


# In[ ]:


#path = r'C:\Users\N THANUSH\Desktop\AB1\cl.pkl'
#joblib.dump(logreg,path,compress=True)
pickle.dump(logreg,open("cl.pkl","wb"))


