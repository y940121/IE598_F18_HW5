#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:24:15 2018

@author: yingzhaocheng
"""

#import data
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/wine/wine.data',
                         header=None)
df_wine.head()
df_wine.describe()
df_wine.info()

#process the wine data into training and testing
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,
                                                   stratify=y,random_state=42)

#EDA
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df_wine, size=2.5)
plt.tight_layout()
plt.show()






# standardize the features: X_train_std, X_test_std
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)




#fit df dataset to logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train, y_train)
X_train_Logisticpredict = lr.predict(X_train)
X_test_Logisticpredict = lr.predict(X_test)




from sklearn.metrics import accuracy_score

print('accuracy score for logisticRegression(trainning set):'
      ,accuracy_score(y_train, X_train_Logisticpredict))
print('accuracy score for logisticRegression(testing set):'
      ,accuracy_score(y_test, X_test_Logisticpredict))




#fit df dataset to SVM regression kernal = linear
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train,y_train)
X_train_svmpred = svm.predict(X_train)
X_test_svmpred = svm.predict(X_test)

print('accuracy score for SVMRegression(trainning set):'
      ,accuracy_score(y_train, X_train_svmpred))
print('accuracy score for SVMRegression(testing set):'
      ,accuracy_score(y_test, X_test_svmpred))


#fit df dataset to SVM regression kernal = rbf
#accuracy score for SVMRegressionrbf(trainning set): 1.0
#accuracy score for SVMRegressionrbf(testing set): 0.3888888888888889
#
from sklearn.svm import SVC
svmrbf = SVC(kernel='rbf', C=1.0, random_state=1)
svmrbf.fit(X_train,y_train)
X_train_svmpredrbf = svmrbf.predict(X_train)
X_test_svmpredrbf = svmrbf.predict(X_test)

print('accuracy score for SVMRegressionrbf(trainning set):'
      ,accuracy_score(y_train, X_train_svmpredrbf))
print('accuracy score for SVMRegressionrbf(testing set):'
      ,accuracy_score(y_test, X_test_svmpredrbf))

##perform a PCA on both datasets： transformed and fit to get accuracy score
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
lrPCA = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lrPCA.fit(X_train_pca, y_train)

X_train_PCAlogistic = lrPCA.predict(X_train_pca)
X_test_PCAlogistic = lrPCA.predict(X_test_pca)

print('accuracy score for PCA logistic model(trainning set):'
      ,accuracy_score(y_train, X_train_PCAlogistic))
print('accuracy score for PCA logistic model(testing set):'
      ,accuracy_score(y_test, X_test_PCAlogistic))

#PCA on svm(linear)
svmPCA = SVC(kernel='linear', C=1.0, random_state=1)
svmPCA.fit(X_train_pca,y_train)

X_train_PCAsvm = svmPCA.predict(X_train_pca)
X_test_PCAsvm = svmPCA.predict(X_test_pca)

print('accuracy score for PCA svm model(trainning set):'
      ,accuracy_score(y_train, X_train_PCAsvm))
print('accuracy score for PCA svm model(testing set):'
      ,accuracy_score(y_test, X_test_PCAsvm))


#perform a LDA on both data set :

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lrLDA = LogisticRegression()

svmLDA = SVC(kernel='linear', C=1.0, random_state=1)

X_train_lda = lda.fit_transform(X_train_std,y_train)
lrLDA.fit(X_train_lda,y_train)
X_test_lda = lda.transform(X_test_std)
X_train_ldapred = lrLDA.predict(X_train_lda)
X_test_ldapred = lrLDA.predict(X_test_lda) 
print('accuracy score for LDA logistic model(trainning set):'
      ,accuracy_score(y_train, X_train_ldapred))
print('accuracy score for LDA logistic model(testing set):'
      ,accuracy_score(y_test, X_test_ldapred))

# LDA on svm

svmLDA.fit(X_train_lda,y_train)
X_train_svmlda = svmLDA.predict(X_train_lda)
X_test_svmlda = svmLDA.predict(X_test_lda)
print('accuracy score for LDA on svm model(trainning set):'
      ,accuracy_score(y_train, X_train_svmlda))
print('accuracy score for LDA on svm model(testing set):'
      ,accuracy_score(y_test, X_test_svmlda))



#kPCA transform and fit

from sklearn.decomposition import KernelPCA
gamatest = [0.0001,0.001,0.01,0.1,1,5,10,15]
acc_kPCA_logi_train = []
acc_kPCA_logi_test =[]
acc_kPCA_svm_train = []
acc_kPCA_svm_test =[]

for x in gamatest:
    scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma=x)
    X_train_kpca = scikit_kpca.fit_transform(X_train_std)
    X_test_kpca = scikit_kpca.transform(X_test_std)
    lrKPCA = LogisticRegression()
    svmKPCA = SVC(kernel='linear', C=1.0, random_state=42)
    lrKPCA.fit(X_train_kpca, y_train)

    X_train_KPCAlogistic = lrKPCA.predict(X_train_kpca)
    X_test_KPCAlogistic = lrKPCA.predict(X_test_kpca)
    print(x)
    print('accuracy score for kPCA logistic model(trainning set):'
      ,accuracy_score(y_train, X_train_KPCAlogistic))
    
    acc_kPCA_logi_train.append(accuracy_score(y_train, X_train_KPCAlogistic))
    print('accuracy score for kPCA logistic model(testing set):'
      ,accuracy_score(y_test, X_test_KPCAlogistic))
    acc_kPCA_logi_test.append(accuracy_score(y_test, X_test_KPCAlogistic))


    svmKPCA.fit(X_train_kpca,y_train)

    X_train_kPCAsvm = svmPCA.predict(X_train_kpca)
    X_test_kPCAsvm = svmPCA.predict(X_test_kpca)

    print('accuracy score for kPCA svm model(trainning set):'
      ,accuracy_score(y_train, X_train_kPCAsvm))
    acc_kPCA_svm_train.append(accuracy_score(y_train, X_train_kPCAsvm))
    print('accuracy score for kPCA svm model(testing set):'
      ,accuracy_score(y_test, X_test_kPCAsvm))
    acc_kPCA_svm_test.append(accuracy_score(y_test, X_test_kPCAsvm))

print(acc_kPCA_logi_train)    

import matplotlib.pyplot as plt
plt.plot(gamatest,acc_kPCA_logi_train,linewidth=2.0)
plt.plot(gamatest,acc_kPCA_logi_test,linewidth=2.0)
plt.show()


print("My name is Sa Yang")
print("My NetID is: say2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")










