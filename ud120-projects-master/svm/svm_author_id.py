#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import pandas as pd
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf=svm.SVC(kernel="rbf",C=10000.0)

t0=time()
clf.fit(features_train,labels_train)
print "Training time: ",round(time()-t0,3),"s"
t=0
z=0
pred=clf.predict(features_test)

for count in pred:
	if(count==1):
		t=t+1
	else:
		z=z+1
		
print "t",t,"z",z




#########################################################
### your code goes here ###

#########################################################


