import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import argparse
from itertools import islice

from torch.utils.tensorboard import SummaryWriter
BATCH_SIZE = 128
EPOCHS = 50
learning_rate = 0.0001

def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains feature_CSV.', required=True)
    parser.add_argument('-o', '--output', help='The path of output.', required=True)
    parser.add_argument('-t', '--time', help='The year of apk.', required=True)

    args = parser.parse_args()
    return args

class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.x_data = features
        self.y_data = labels
        self.len = len(labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            # pred
            nn.Linear(49, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

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

    X = torch.tensor(vectors, dtype=torch.float)
    Y = torch.tensor(labels, dtype=torch.long)
    return X, Y

def train(model, criterion, optimizer, train_loader, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))
    return ave_loss

def test(model, test_loader):
    csv_data = [[] for i in range(15)]
    csv_data[0] = ['ML_Algorithm', 'F1', 'Precision', 'Recall', 'Accuracy', 'TPR', 'FPR', 'TNR', 'FNR']
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        pred = torch.max(output, 1)[1].view(target.size())
        y_pred = pred.numpy()
        test_Y = target.numpy()
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
        result = [f1, precision, recall, accuracy, TPR, FPR, TNR, FNR]
        csv_data[batch_idx+1].append(batch_idx+1)
        csv_data[batch_idx+1].extend(result)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    result = [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs),
           np.mean(TNRs), np.mean(FNRs)]
    csv_data[len(test_loader) + 1].append('mean')
    csv_data[len(test_loader) + 1].extend(result)
    print('F1:', F1s)
    print('result:', result)
    return csv_data

def main():
    args = parseargs()
    feature_dir = args.dir
    out_put = args.output
    time = args.time
    feature_dir = args.dir
    out_put = args.output
    folder = os.path.exists(out_put)
    if not folder:
        os.makedirs(out_put)

    if feature_dir[-1] != '/':
        feature_dir += '/'
    if out_put[-1] != '/':
        out_put += '/'

    file = feature_dir + time + '_predictions.csv'
    out_put = out_put + time + '/ensemble_result.csv'

    # print(file)
    # print(out_put)

    Vectors, Labels = feature_extraction(file)
    dataset = MyDataSet(Vectors, Labels)
    num = dataset.len
    train_size = int(num * 0.7)
    test_size = num - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = Model()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # writer = SummaryWriter('../log')
    for epoch in range(1, EPOCHS + 1):
        loss_train = train(model, criterion, optimizer, train_loader, epoch)
        # writer.add_scalar("train_loss", loss_train, epoch)
        # loss_val = val(model, criterion, val_loader)
        # writer.add_scalar("val_loss", loss_val, epoch)
    # torch.save(model, 'model.pth')
    # writer.close()
    csv_data = test(model, test_loader)

    with open(out_put, 'w', newline='') as f:
        csvfile = csv.writer(f)
        csvfile.writerows(csv_data)

if __name__ == '__main__':
    main()

