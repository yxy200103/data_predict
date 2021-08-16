import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
# from time_utils import time_for_file, print_log
from visualization import draw_loss_and_accuracy
import random
import numpy as np
import os.path as osp
import time
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.lib.function_base import copy
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import os
import sys
import time
from torch.utils.data import DataLoader

log_save_root_path = "./"
model_save_root_path = "./model"


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
name = str(time.time()) + "log"
sys.stdout = Logger(name+'.txt')

print(path)
print(os.path.dirname(__file__))
print('------------------')

def print_log(print_string, log):
    print("{}".format(print_string))
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()
 
 
def time_for_file():
    ISOTIMEFORMAT = '%d-%h-at-%H-%M-%S'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))

def harmonize_data(posedata, x_columns):
    def max_min_scaler(x): return (x-np.min(x))/(np.max(x)-np.min(x))
    for title in x_columns[1:len(x_columns)-1]:
        posedata[title] = posedata[[title]].apply(max_min_scaler)
    posedata.loc[posedata['delta'] >= 0, 'delta'] = int(1)
    posedata.loc[posedata['delta'] < 0, 'delta'] = int(0)

    return posedata


def work():
    traindata = pd.read_csv('new_data.csv')
    # print(traindata)
    target = 'delta'   # pose的值就是分类
    x_columns = [x for x in traindata.columns if x not in [target]]
    headers = list.copy(x_columns)
    headers.remove('Date')
    x_columns.append('delta')    # 得到标题列表
    precessed_train_data = harmonize_data(traindata, x_columns)
    #    / print(type(precessed_train_data))
    # new_data = calcu_data(precessed_train_data)
    new_data = precessed_train_data[x_columns].values
    [rows, cols] = new_data.shape
    new_data = np.empty([rows, cols], dtype=list)
    print(rows, cols)
    list_a = [0, 0, 0, 0, 0]
    for i in range(rows):
        for j in range(cols):
            new_data[i, j] = list_a
    # print(new_data)
    # print(rows, cols)
    for i in range(rows - 5):
        for title in range(len(precessed_train_data.columns)):
            if title == 0:
                continue
            list_t = []
            for j in range(5):
                # print(posedata[title][i+j])
                list_t.append(float(precessed_train_data.iloc[i+j, title]))
            # print(list_t)
            new_data[i, title] = np.array(list_t.copy())
            new_data[i, len(precessed_train_data.columns) -
                     1] = np.array(list_t[-1])
    X = new_data[0:rows-5, 1:cols-2]
    Y = new_data[0:rows-5, cols - 1]
    X_train, X_test, Y_train, Y_test = train_test_split(
        np.array(X), np.array(Y), test_size=0.15)
    return X_train, X_test, Y_train, Y_test


class MyDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        # data = self.loader(data)
        data = data.tolist()
        new_data = []
        for list in data:
            for item in list:
                new_data.append(float(item))
        data = new_data

        data = np.array(data)
        row_n = int(len(data)/5)
        # print(row_n)
        data = data.reshape(row_n, 5, 1)
        # print("data",data)
        # print(data.shape,type(data),data.dtype,data.size,data.ndim)
        data = torch.from_numpy(data.transpose((2, 0, 1))).double()
        labels = torch.from_numpy(labels).double()

        # data = data.float().div(255).unsqueeze(0)  # 255也可以改为256
        # data = data.ToTensor()
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼

    def __len__(self):
        return len(self.data)


def adjust_learning_rate(optimizer, epoch, train_epoch, learning_rate):
    lr = learning_rate * (0.6 ** (epoch / train_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target):
    output = output.to("cpu")
    target = target.to("cpu")
    output = np.array(output.numpy())
    target = np.array(target.numpy())
    prec = 0
    for i in range(output.shape[0]):
        pos = np.unravel_index(np.argmax(output[i]), output.shape)
        pre_label = pos[1]
        if pre_label == target[i]:
            prec += 1
    prec /= target.size
    prec *= 100
    return prec
# def evalute(model, loader):
#     model.eval()   #必须要加入 model.eval() ，因为训练和测试BN不一致
#     correct = 0
#     total = len(loader.dataset)
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         with torch.no_grad():    #不需要计算梯度，所以加上不求导，验证集一定要加上这几句话
#             logits = model(x.float())
#             pred = logits.argmax(dim=1)
#             # y = y.to(device=device, dtype=torch.int64)
#         correct += torch.eq(pred, y).sum().float().item()
#     return correct / totalv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def resnet_finute(train_epoch, print_freq, learning_rate_start):
    log = open(osp.join(log_save_root_path, 'cluster_seed_{}_{}.txt'.format(random.randint(1, 10000), time_for_file())),
               'w')
    net = models.resnet50(pretrained=True)
    net = net.to(device)
    channel_in = net.fc.in_features
    class_num = 2
    net.fc = nn.Sequential(
        nn.Linear(channel_in, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, class_num),
        nn.LogSoftmax(dim=1)
    )
    net.conv1 = nn.Sequential(
					           nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
					            nn.BatchNorm2d(64)
					        )
    for param in net.parameters():
        param.requires_grad = False

    for param in net.fc.parameters():
        param.requires_grad = True

    # # 输出网络的结构
    # for child in net.children():
    #     print(child)

    Loss_list = []
    Accuracy_list = []
    X_train,X_test,Y_train, Y_test= work()
    torch_data = MyDataset(X_train,Y_train)
    torch_data_test = MyDataset(X_test,Y_test)
    # print(torch_data,"  ",type(torch_data))
    trainloader = DataLoader(torch_data, batch_size=256, shuffle=True, drop_last=False)
    # print(datas,"  ",type(datas))
    datas_test = DataLoader(torch_data_test, batch_size=256, shuffle=False, drop_last=False)
    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate_start, momentum=0.9)
    criterion = nn.NLLLoss()
    net = net.to(device)
    for epoch in range(train_epoch):
        epoch_accuracy = 0
        epoch_loss = 0
        learning_rate = adjust_learning_rate(
            optimizer, epoch, train_epoch, learning_rate_start)
        print_log('epoch : [{}/{}] lr={}'.format(epoch,
                  train_epoch, learning_rate), log)
        net.train()
        for i, (inputs, target) in enumerate(trainloader):
            inputs = inputs.to(device)
            output = net(inputs.float())
            output = output.to(device)
            target = target.to(device=device, dtype=torch.int64)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec = accuracy(output.data, target)
            epoch_accuracy += prec
            epoch_loss += loss.float()
        #     if i % print_freq == 0 or i + 1 == len(trainloader):
        #         print_log(
        #             'after {} epoch, {}th batchsize, prec:{}%,loss:{},input:{},output:{}'.format(epoch + 1, i + 1, prec,
        #                                                                                          loss, inputs.size(),
        #                                                                                          output.size()), log)
        epoch_loss /= len(trainloader)
        epoch_accuracy /= len(trainloader)
        print_log(
                    'after {} epoch, prec:{}%,loss:{}'.format(epoch + 1, epoch_accuracy, epoch_loss), log)
        Loss_list.append(epoch_loss)
        Accuracy_list.append(epoch_accuracy)
        torch.save(net.state_dict(), osp.join(model_save_root_path, 'resnet50_{}_{}.pth'.format(epoch + 1,
                                                                                                time.strftime(
                                                                                                    "%Y-%m-%d_%H-%M-%S"))))
        val_loader = DataLoader(torch_data_test, batch_size=32, shuffle=False, drop_last=False)
        # val_loader = loadTestData()
        criterion = torch.nn.CrossEntropyLoss()
        sum_accuracy = 0
        for i, (inputs, target) in enumerate(val_loader):
            with torch.no_grad():
                inputs = inputs.to(device)
                output = net(inputs.float())
                output = output.to(device)
                target = target.to(device=device, dtype=torch.int64)
                loss = criterion(output, target)
                prec = accuracy(output.data, target)
                sum_accuracy += prec
                print('for {}th batchsize, Eval:Accuracy:{}%,loss:{},input:{},output:{}'.format(i + 1, prec, loss,
                                                                                                inputs.size(),
                                                                                                output.size()))
        sum_accuracy /= len(val_loader)
        print('sum of accuracy = {}'.format(sum_accuracy))    
    # print(Loss_list)
    # draw_loss_and_accuracy(Loss_list, Accuracy_list, train_epoch)


def resnet_eval(single_image=False, img_path=None):
    resnet = models.resnet50(pretrained=True)
    resnet = resnet.to(device)
    channel_in = resnet.fc.in_features
    class_num = 2
    resnet.fc = nn.Sequential(
        nn.Linear(channel_in, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, class_num),
        nn.LogSoftmax(dim=1)
    )
    resnet.load_state_dict(
        torch.load(osp.join(model_save_root_path, 'resnet50_500_2021-08-13_17-07-27.pth')))  # 这里填最新的模型的名字
    resnet.eval()
    X_train,X_test,Y_train, Y_test= work()
    torch_data = MyDataset(X_train,Y_train)
    torch_data_test = MyDataset(X_test,Y_test)
    # print(torch_data,"  ",type(torch_data))
    trainloader = DataLoader(torch_data, batch_size= 32, shuffle=True, drop_last=False)
    # print(datas,"  ",type(datas))
    val_loader = DataLoader(torch_data_test, batch_size=32, shuffle=False, drop_last=False)
    # val_loader = loadTestData()
    criterion = torch.nn.CrossEntropyLoss()
    sum_accuracy = 0
    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            output = resnet(inputs.float())
            output = output.to(device)
            target = target.to(device=device, dtype=torch.int64)
            loss = criterion(output, target)
            prec = accuracy(output.data, target)
            sum_accuracy += prec
            print('for {}th batchsize, Eval:Accuracy:{}%,loss:{},input:{},output:{}'.format(i + 1, prec, loss,
                                                                                            inputs.size(),
                                                                                            output.size()))
    sum_accuracy /= len(val_loader)
    print('sum of accuracy = {}'.format(sum_accuracy))


if __name__ == '__main__':
    train_epoch = 300
    print_freq = 5
    learning_rate_start = 0.005
    resnet_finute(train_epoch, print_freq, learning_rate_start)
    # resnet_eval()
