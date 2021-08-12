import numpy as np
from numpy.lib.function_base import copy
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


traindata = pd.read_csv('all.csv')
# print(traindata)
target='delta'    
x_columns = [x for x in traindata.columns if x not in [target]]
headers = list.copy(x_columns)
headers.remove('Date')
x_columns.append('delta') 

# 数据清洗
def harmonize_data(posedata):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    for title in x_columns[1:len(x_columns)-1]:
        posedata[title] = posedata[[title]].apply(max_min_scaler)
    posedata.loc[posedata['delta'] >= 0, 'delta'] = int(1)
    posedata.loc[posedata['delta'] < 0, 'delta'] = int(0)
    
    return posedata
 
precessed_train_data = harmonize_data(traindata)

X = (precessed_train_data[headers].values)
Y = (precessed_train_data['delta'].values)

X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.15)
print(X)
print("SVM method")
# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,Y_train)
score_rbf = clf_rbf.score(X_test,Y_test)
score_rbf1 = clf_rbf.score(X_train,Y_train)
print("The score of SVM rbf is : %f"%score_rbf)
print(score_rbf1)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train,Y_train)
score_linear = clf_linear.score(X_test,Y_test)
print("The score of SVM linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,Y_train)
score_poly = clf_poly.score(X_test,Y_test)
print("The score of SVM poly is : %f"%score_poly)

# random forest
Classifier = RandomForestClassifier()
Classifier.fit(X_train, Y_train)
Classifier_score = Classifier.score(X_test,Y_test)
print("The score of random forest is : %f"%Classifier_score)

# xgboost
xgboost =  XGBClassifier(use_label_encoder=False)
xgboost.fit(X_train, Y_train)
xgboost_score = xgboost.score(X_test,Y_test)
print("The score of xgboost is : %f"% xgboost_score)

