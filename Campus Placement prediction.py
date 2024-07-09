#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data= pd.read_csv('D:\\MAnik\\Final year projects\\Campus Placemenet  prediction\\Placement_Data_Full_Class.csv')


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


data.isnull().sum()


# In[5]:


data.columns


# In[6]:


data['status'].unique()


# In[7]:


data[(data['degree_t']=="Sci&Tech") & (data['status']=="Placed")].sort_values(by="salary",ascending=False).head()


# In[8]:


data = data.drop(['sl_no','salary'],axis=1)


# In[9]:


data.head()


# In[10]:


data['ssc_b'].unique()


# In[11]:


data['ssc_b'] = data['ssc_b'].map({'Central':1,'Others':0})


# In[12]:


data['hsc_b'].unique()


# In[13]:


data['hsc_b'] = data['hsc_b'].map({'Central':1,'Others':0})


# In[14]:


data['hsc_s'].unique()


# In[15]:


data['hsc_s'] = data['hsc_s'].map({'Science':2,'Commerce':1,'Arts':0})


# In[16]:


data['degree_t'].unique()


# In[17]:


data['degree_t'] = data['degree_t'].map({'Sci&Tech':2,'Comm&Mgmt':1,'Others':0})


# In[18]:


data['specialisation'].unique()


# In[19]:


data['specialisation'] =data['specialisation'].map({'Mkt&HR':1,'Mkt&Fin':0})


# In[20]:


data['workex'].unique()


# In[21]:


data['workex'] = data['workex'].map({'Yes':1,'No':0})


# In[22]:


data.head(2)


# In[23]:


data['status'].unique()


# In[24]:


data['status'] = data['status'].map({'Placed':1,'Not Placed':0})


# In[25]:


data.head(2)


# In[26]:


data.columns


# In[27]:


X = data.drop('status',axis=1)
y= data['status']


# In[28]:


y


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


# In[33]:


le = LabelEncoder()
X_train = X_train.apply(le.fit_transform)

# Train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[34]:


svm = svm.SVC()
svm.fit(X_train, Y_train)


# In[35]:


knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)


# In[36]:


dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)


# In[37]:


rf=RandomForestClassifier()
rf.fit(X_train,Y_train)


# In[38]:


gb=GradientBoostingClassifier()
gb.fit(X_train,Y_train)


# In[39]:


X_test['gender'] = X_test['gender'].replace({'M': 1, 'F': 0})


# In[40]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rf.predict(X_test)
y_pred6 = gb.predict(X_test)


# ### Evaluating The Algorithms

# In[41]:


from sklearn.metrics import accuracy_score


# In[42]:


score1=accuracy_score(Y_test,y_pred1)
score2=accuracy_score(Y_test,y_pred2)
score3=accuracy_score(Y_test,y_pred3)
score4=accuracy_score(Y_test,y_pred4)
score5=accuracy_score(Y_test,y_pred5)
score6=accuracy_score(Y_test,y_pred6)


# In[43]:


print(score1,score2,score3,score4,score5,score6)


# In[44]:


final_data=pd.DataFrame({'model':['LR','SVC','KNN','DT','RF','GB'],'ACC':[score1*100,score2*100,score3*100,score4*100,score5*100,score6*100]})


# In[45]:


final_data


# In[48]:


import seaborn as sns
sns.barplot(x='model', y='ACC', data=final_data)


# ### Prediction Model

# In[49]:


new_data = pd.DataFrame({
    'gender':0,
    'ssc_p':67.0,
    'ssc_b':0,
    'hsc_p':91.0,
    'hsc_b':0,
    'hsc_s':1,
    'degree_p':58.0,
    'degree_t':2,
    'workex':0,
    'etest_p':55.0,
     'specialisation':1,
    'mba_p':58.8,   
},index=[0])


# In[50]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[51]:


p=lr.predict(new_data)
prob=lr.predict_proba(new_data)
if p==1:
    print('Placed')
    print(f"You will be placed with probability of {prob[0][1]:.2f}")
else:
    print("Not-placed")


# In[52]:


prob


# ### Save Model

# In[53]:


import joblib


# In[54]:


joblib.dump(lr,'model_campus_placement')


# In[55]:


model = joblib.load('model_campus_placement')


# In[56]:


model.predict(new_data)


# ### GUI 

# In[62]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
import tkinter.font as font
import pandas as pd

def show_entry_fields():
    text = clicked.get()
    if text == "Male":
        p1=1
        print(p1)
    else:
        p1=0
        print(p1)
    p2=float(e2.get())
    text = clicked1.get()
    if text == "Central":
        p3=1
        print(p3)
    else:
        p3=0
        print(p3)
    p4=float(e4.get())
    text = clicked6.get()
    if text == "Central":
        p5=1
        print(p3)
    else:
        p5=0
        print(p3)
    text = clicked2.get()
    if text == "Science":
        p6=2
        print(p6)
    elif text == "Commerce":
        p6=1
        print(p6)
    else:
        p6=0
        print(p6)
    p7=float(e7.get())
    text = clicked3.get()
    if text == "Sci&Tech":
        p8=2
        print(p8)
    elif text=="Comm&Mgmt":
        p8=1
        print(p8)
    else:
        p8=0
        print(p8)
    text = clicked4.get()
    model = joblib.load('model_campus_placement')
    new_data = pd.DataFrame({
    'gender':p1,
    'ssc_p':p2,
    'ssc_b':p3,
    'hsc_p':p4,
    'hsc_b':p5,
    'hsc_s':p6,
    'degree_p':p7,
    'degree_t':p8,
    'workex':p9,
    'etest_p':p10,
     'specialisation':p11,
    'mba_p':p12,   
},index=[0])
    result=model.predict(new_data)
    result1=model.predict_proba(new_data)
    
    if result[0] == 0:
        Label(master, text="Can't Placed").grid(row=31)
    else:
        Label(master, text="Student Will be Placed With Probability of",font=("Arial", 15)).grid(row=31)
        Label(master, text=round(result1[0][1],2)*100,font=("Arial", 15)).grid(row=33)
        Label(master, text="Percent",font=("Arial", 15)).grid(row=34)

master = Tk()
master.title("Campus Placement Prediction System")


label = Label(master, text = "Campus Placement Prediction System"
                          , bg = "green", fg = "white",font=("Arial", 20)) \
                               .grid(row=0,columnspan=2)


Label(master, text="Gender",font=("Arial", 15)).grid(row=1)
Label(master, text="Secondary Education percentage- 10th Grade",font=("Arial", 15)).grid(row=2)
Label(master, text="Board of Education",font=("Arial", 15)).grid(row=3)
Label(master, text="Higher Secondary Education percentage- 12th Grade",font=("Arial", 15)).grid(row=4)
Label(master, text="Board of Education",font=("Arial", 15)).grid(row=5)
Label(master, text="Specialization in Higher Secondary Education",font=("Arial", 15)).grid(row=6)
Label(master, text="Degree Percentage",font=("Arial", 15)).grid(row=7)
Label(master, text="Under Graduation(Degree type)- Field of degree education",font=("Arial", 15)).grid(row=8)
Label(master, text="Work Experience",font=("Arial", 15)).grid(row=9)
clicked = StringVar()
options = ["Male","Female"]

clicked1 = StringVar()
options1 = ["Central","Others"]

clicked2 = StringVar()
options2 = ["Science","Commerce","Arts"]

clicked3 = StringVar()
options3 = ["Sci&Tech","Comm&Mgmt","Others"]

clicked4 = StringVar()
options4 = ["Yes","No"]

e1 = OptionMenu(master , clicked , *options )
e1.configure(width=13)
e2 = Entry(master)
e3 = OptionMenu(master , clicked1 , *options1 )
e3.configure(width=13)
e4 = Entry(master)
e5 = OptionMenu(master , clicked6 , *options6)
e5.configure(width=13)
e6 = OptionMenu(master , clicked2 , *options2)
e6.configure(width=13)
e7 = Entry(master)
e8 = OptionMenu(master , clicked3 , *options3)
e8.configure(width=13)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
buttonFont = font.Font(family='Helvetica', size=16, weight='bold')
Button(master, text='Predict',height= 1, width=8,activebackground='#00ff00',font=buttonFont,bg='black', fg='white',command=show_entry_fields).grid()

mainloop()


# In[ ]:




