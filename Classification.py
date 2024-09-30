# 
import networkx as nx
import time
import argparse
import csv
import numpy as np
import os
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn import svm
from itertools import islice

def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains feature_CSV.', required=True)
    parser.add_argument('-o', '--output', help='The path of output.', required=True)
    # parser.add_argument('-t', '--type', help='The type of centrality: degree, closeness, harmonic, katz', required=True)

    args = parser.parse_args()
    return args

def feature_extraction(file):
    vectors = []
    labels = []
    with open(file, 'r') as f:
        csv_data = csv.reader(f)
        for line in islice(csv_data, 1, None):
            if not line:
                continue
            label = int(float(line[1]))
            vector = [float(i) for i in line[2:]]
            vectors.append(vector)
            labels.append(label)

    return vectors, labels

def centrality_feature(feature_dir, type):
    feature_csv = feature_dir + type + '_features.csv'
    print(feature_csv)
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels


def random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]

from sklearn.neighbors import KNeighborsClassifier
def knn_1(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_X, train_Y)


        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('knn-1')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]

def knn_3(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('knn-3')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]

from sklearn.ensemble import RandomForestClassifier
def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=64, random_state=0)
        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
        
        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        # print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('RandomForest')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]


from sklearn.tree import DecisionTreeClassifier
def decisiontree(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = DecisionTreeClassifier(max_depth=64, random_state=0)
        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
       
        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        # print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('decision tree')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]

from xgboost.sklearn import XGBClassifier
def xgboost(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = XGBClassifier(max_depth=64, random_state=0)
        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
    
        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        # print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('xgboost')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]

from sklearn.ensemble import AdaBoostClassifier
def adaboost(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=64),random_state=0)
        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
       
        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        # print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('adaboost')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]

from sklearn.ensemble import GradientBoostingClassifier
def GBDT(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = GradientBoostingClassifier(max_depth=64,random_state=0)
        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        # print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        # print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    print('GBDT')
    print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]




def classification(vectors, labels):
    Vectors, Labels = random_features(vectors, labels)


    csv_data = [[] for i in range(10)]
    csv_data[0] = ['ML_Algorithm', 'F1', 'Precision', 'Recall', 'Accuracy', 'TPR', 'FPR', 'TNR', 'FNR']

    csv_data[1].append('KNN-1')
    csv_data[1].extend(knn_1(Vectors, Labels))
    csv_data[2].append('KNN-3')
    csv_data[2].extend(knn_3(Vectors, Labels))
    csv_data[3].append('Decision Tree')
    csv_data[3].extend(decisiontree(Vectors, Labels))
    csv_data[4].append('Random Forest')
    csv_data[4].extend(randomforest(Vectors, Labels))
    csv_data[5].append('Adaboost')
    csv_data[5].extend(adaboost(Vectors, Labels))
    csv_data[6].append('GBDT')
    csv_data[6].extend(GBDT(Vectors, Labels))
    csv_data[7].append('XGBoost')
    csv_data[7].extend(xgboost(Vectors, Labels))
    
    
    return csv_data



def main():
    args = parseargs()
    feature_dir = args.dir
    out_put = args.output
    folder = os.path.exists(args.output)
    if not folder:
        os.makedirs(args.output)
    # type = args.type

    if feature_dir[-1] == '/':
        feature_dir = feature_dir
    else:
        feature_dir += '/'

    if out_put[-1] == '/':
        out_put = out_put
    else:
        out_put += '/'

    types = ['degree', 'katz', 'closeness', 'harmonic', 'pagerank', 'eigenvector', 'authority']
    for type in types:
        vectors, labels = centrality_feature(feature_dir, type)
        csv_data = classification(vectors, labels)
        results = out_put + type + '_result.csv'
        with open(results, 'w', newline='') as f:
            csvfile = csv.writer(f)
            csvfile.writerows(csv_data)




if __name__ == '__main__':
    main()
