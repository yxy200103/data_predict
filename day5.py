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
def harmonize_data(posedata,x_columns):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    for title in x_columns[1:14]:
        posedata[title] = posedata[[title]].apply(max_min_scaler)
    posedata.loc[posedata['delta'] >= 0, 'delta'] = int(1)
    posedata.loc[posedata['delta'] < 0, 'delta'] = int(0)
    
    return posedata

def calcu_data(posedata):
    
    new_data = posedata.copy()
    for i in range(len(posedata) - 5):
        for title in range(len(posedata.columns)):
            list_t = []
            for j in range (5):
                # print(posedata[title][i+j])
                list_t.append(posedata.iloc[i+j,title])
            print(list_t)
            new_data.iloc[title,i]= list_t.copy()
    return new_data
 
# precessed_train_data = harmonize_data(traindata)
# print(type(precessed_train_data))
# # new_data = calcu_data(precessed_train_data)
# new_data = precessed_train_data[x_columns].values
# [rows, cols] = new_data.shape
# print(rows, cols)
# for i in range(rows- 5) :
#     for title in range(len(precessed_train_data.columns)):
#         list_t = []
#         for j in range (5):
#             # print(posedata[title][i+j])
#             list_t.append(precessed_train_data.iloc[i+j,title])
#         # print(list_t)
#         new_data[i,title]= list_t.copy()
#         new_data[i,len(precessed_train_data.columns) - 1] = list_t[-1]
# print(new_data)
# # X = (precessed_train_data[headers].values)
# # Y = (precessed_train_data['delta'].values)
# X = new_data[0:rows-5,1:cols-2]
# print("X\n",X)

# Y = new_data[0:rows-5,cols - 1]
# print("Y\n",Y)
# X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.15)
# print(X_train,X_test,Y_train, Y_test )


traindata = pd.read_csv('all.csv')
# print(traindata)
target='delta'    
x_columns = [x for x in traindata.columns if x not in [target]]
headers = list.copy(x_columns)
headers.remove('Date')
x_columns.append('delta')    
precessed_train_data = harmonize_data(traindata,x_columns)
print(type(precessed_train_data))
# new_data = calcu_data(precessed_train_data)
new_data = precessed_train_data[x_columns].values
[rows, cols] = new_data.shape
print(rows, cols)
for i in range(rows- 5) :
    for title in range(len(precessed_train_data.columns)):
        list_t = []
        for j in range (5):
            # print(posedata[title][i+j])
            list_t.append(precessed_train_data.iloc[i+j,title])
        # print(list_t)
        new_data[i,title]= np.array(list_t.copy())
        new_data[i,len(precessed_train_data.columns) - 1] = np.array(list_t[-1])

T = new_data[0:rows-5,1:cols-2]
X = []
for row in T :
    X_list = []
    for list in row:
        for item in list:
            X_list.append(item)
    X.append(X_list)
X = np.array(X)
print("X\n",X.shape)
Y=[]
Q= new_data[0:rows-5,cols - 1]
for item in Q :
    X_list = []
    X_list.append(item)
    Y.append(X_list)
Y = np.array(Y)
print("Y\n",Y,Y.shape)
# print("Y\n",Y)
X_train,X_test,Y_train, Y_test = train_test_split(np.array(X),np.array(Y), test_size=0.15)

print("SVM method")
kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,Y_train)
score_rbf = clf_rbf.score(X_test,Y_test)
print("The score of SVM rbf is : %f"%score_rbf)

# # kernel = 'linear'
# clf_linear = svm.SVC(kernel='linear')
# clf_linear.fit(X_train,Y_train)
# score_linear = clf_linear.score(X_test,Y_test)
# print("The score of SVM linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,Y_train)
score_poly = clf_poly.score(X_test,Y_test)
print("The score of SVM poly is : %f"%score_poly)

# random forest
regressor = RandomForestClassifier()
regressor.fit(X_train, Y_train)
regressor_score = regressor.score(X_test,Y_test)
print("The score of random forest is : %f"%regressor_score)

# xgboost
xgboost =  XGBClassifier(use_label_encoder=False)
xgboost.fit(X_train, Y_train)
xgboost_score = xgboost.score(X_test,Y_test)
print("The score of xgboost is : %f"% xgboost_score)
