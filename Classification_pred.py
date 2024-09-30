# 分类，输出预测结果（prediction）和评估（result）
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from itertools import islice
import pandas as pd

def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains feature_CSV.', required=True)
    parser.add_argument('-o', '--output', help='The path of output.', required=True)
    parser.add_argument('-p', '--prediction', help='The path of prediction.', required=True)

    args = parser.parse_args()
    return args

def feature_extraction(file):
    Features = []
    with open(file, 'r') as f:
        csv_data = csv.reader(f)
        for line in islice(csv_data, 1, None):
            if not line:
                continue
            feature = []
            apk = line[0]
            label = int(float(line[1]))
            vector = [float(i) for i in line[2:]]
            feature.append(apk)
            feature.append(label)
            feature.append(vector)
            Features.append(feature)
    random.shuffle(Features)
    return Features

def centrality_feature(feature_dir, type):
    feature_csv = feature_dir + type + '_features.csv'
    print(feature_csv)
    Features = feature_extraction(feature_csv)
    return Features

def write_result(csv_data, output, type):
    results = output + type + '_result.csv'
    with open(results, 'w', newline='') as f:
        csvfile = csv.writer(f)
        csvfile.writerows(csv_data)


def get_stacking(clf, vectors, labels, n_folds=10):
    X = np.array(vectors)
    Y = np.array(labels)
    num = X.shape[0]
    second_level_set = np.zeros((num,))

    kf = KFold(n_splits=n_folds)
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

        clf.fit(train_X, train_Y)

        y_pred = clf.predict(test_X)
        second_level_set[test_index] = y_pred
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

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

    print(F1s, FPRs)
    result = [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
            np.mean(TNRs), np.mean(FNRs)]

    return second_level_set, result

def centrality_prediction(feature_dir, out_put, type):
    Features = centrality_feature(feature_dir, type)
    csv_data = [[] for i in range(10)]
    csv_data[0] = ['ML_Algorithm', 'F1', 'Precision', 'Recall', 'Accuracy', 'TPR', 'FPR', 'TNR', 'FNR']
    apks = [m[0] for m in Features]
    labels = [m[1] for m in Features]
    vectors = [m[2] for m in Features]
    apk_pd = pd.Series(apks)
    label_pd = pd.Series(labels)
    if type == 'degree':
        df = pd.DataFrame({'SHA256': apk_pd, 'Label': label_pd})
    else:
        df = pd.DataFrame({'SHA256': apk_pd})
    clfs = {
        '1NN': KNeighborsClassifier(n_neighbors=1),
        '3NN': KNeighborsClassifier(n_neighbors=3),
        'RandomForest': RandomForestClassifier(max_depth=64, random_state=0),
        'DecisionTree': DecisionTreeClassifier(max_depth=64, random_state=0),
        'Adaboost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=64), random_state=0),
        'GBDT': GradientBoostingClassifier(max_depth=64, random_state=0),
        'Xgboost': XGBClassifier(max_depth=64, random_state=0)
    }
    i = 0
    for clf_key in clfs.keys():
        i += 1
        title = type + '_' + clf_key
        print('\n the classifier is:', title)
        clf = clfs[clf_key]
        pred, result = get_stacking(clf, vectors, labels)
        csv_data[i].append(clf_key)
        csv_data[i].extend(result)
        pred_pd = pd.Series(pred)
        df[title] = pred_pd
    write_result(csv_data, out_put, type)
    return df

def PredictionMerge(feature_dir, out_put, pred_path):
    types = ['closeness', 'harmonic', 'katz', 'eigenvector', 'pagerank', 'authority']
    df = centrality_prediction(feature_dir, out_put, 'degree')
    for type in types:
        df_centrality = centrality_prediction(feature_dir, out_put, type)
        df = pd.merge(df, df_centrality, how='inner', on='SHA256')
    df.to_csv(pred_path, index=0)

def main():
    args = parseargs()
    feature_dir = args.dir
    out_put = args.output
    pred_path = args.prediction

    # feature_dir = '../feature/2017'
    # pred_path = '../prediction/2017'
    # out_put = '../result/2017'

    folder = os.path.exists(out_put)
    if not folder:
        os.makedirs(out_put)

    if feature_dir[-1] != '/':
        feature_dir += '/'
    if out_put[-1] != '/':
        out_put += '/'
    if pred_path[-1] == '/':
        pred_path = pred_path[:-1] + '_predictions.csv'
    else:
        pred_path = pred_path + '_predictions.csv'

    print(feature_dir)
    print(out_put)
    print(pred_path)

    #PredictionMerge(feature_dir, out_put, pred_path)
if __name__ == '__main__':
    main()











