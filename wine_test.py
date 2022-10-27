# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:09:30 2022

@author: Sagar Arora
"""

import pandas as pd
import numpy as np
wine = pd.read_csv("winequality-red.csv")
wine.head()
wine.isnull().sum()
wine.info()
bins = [2,6.5,8]
label = ['bad','good']
wine['quality'] = pd.cut(x=wine['quality'], bins=bins,labels=label)
wine['quality'].unique()

from sklearn.preprocessing import LabelEncoder

label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(wine['quality'])

X = wine.iloc[:,:-1]
y = wine.iloc[:,-1]

from sklearn.model_selection import train_test_split


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

#SVM Classifier
from sklearn.svm import SVC

clf = SVC()
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)

print(confusion_matrix(y_test, pred_clf))



