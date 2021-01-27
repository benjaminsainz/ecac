"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import math
import numpy as np
def support_vector_classifier(X_train, X_test, y_train, y_test, n_classes):
    clf = OneVsRestClassifier(LinearSVC(max_iter=5000))
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    svc_auc=sum(roc_auc.values())/len(roc_auc)
    return svc_auc

def knn_classifier(X_train, X_test, y_train, y_test, n_classes):
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 2))
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    knn_auc=sum(roc_auc.values())/len(roc_auc)
    return knn_auc

def logistic_regression(X_train, X_test, y_train, y_test, n_classes):
    clf = OneVsRestClassifier(LogisticRegression())
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lr_auc=sum(roc_auc.values())/len(roc_auc)
    return lr_auc

def fitness_value(X, ind, n_classes):
    classes_lst=[]
    for i in range(n_classes): classes_lst.append(i)
    if n_classes==2:
        classes_lst=[0,1,2]
        y_bin= label_binarize(ind, classes=classes_lst)
        y_bin = np.delete(y_bin, 2, 1)
    else: y_bin= label_binarize(ind, classes=classes_lst)
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.75)
    svc=support_vector_classifier(X_train, X_test, y_train, y_test, n_classes)
    knn=knn_classifier(X_train, X_test, y_train, y_test, n_classes)
    lr=logistic_regression(X_train, X_test, y_train, y_test, n_classes)
    if math.isnan(svc): svc=0
    if math.isnan(knn): knn=0
    if math.isnan(lr): lr=0
    return (svc+knn+lr)/3
