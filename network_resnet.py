import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)  # reproducible
import numpy as np
from numpy.lib.function_base import copy
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import os,sys,time

import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
from resnet import ResNet18



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
name = str(time.time())+ "log"
sys.stdout = Logger(name+'.txt')
 
print(path)
print(os.path.dirname(__file__))
print('------------------')

# 数据清洗
def harmonize_data(posedata,x_columns):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    for title in x_columns[1:len(x_columns)-1]:
        posedata[title] = posedata[[title]].apply(max_min_scaler)
    posedata.loc[posedata['delta'] >= 0, 'delta'] = int(1)
    posedata.loc[posedata['delta'] < 0, 'delta'] = int(0)
    
    return posedata
    
def work():
    traindata = pd.read_csv('all.csv')
    # print(traindata)
    target='delta'   # pose的值就是分类
    x_columns = [x for x in traindata.columns if x not in [target]]
    headers = list.copy(x_columns)
    headers.remove('Date')
    x_columns.append('delta')    # 得到标题列表
    precessed_train_data = harmonize_data(traindata,x_columns)
#    / print(type(precessed_train_data))
    # new_data = calcu_data(precessed_train_data)
    new_data = precessed_train_data[x_columns].values
    [rows, cols] = new_data.shape
    new_data =  np.empty([rows, cols], dtype = list) 
    print(rows, cols)
    list_a = [0,0,0,0,0]
    for i in range(rows):
        for j in range(cols):
            new_data[i,j] = list_a
    # print(new_data)
    # print(rows, cols)
    for i in range(rows - 5):
        for title in range(len(precessed_train_data.columns)):
            if title == 0 :
                continue
            list_t = []
            for j in range(5):
                # print(posedata[title][i+j])
                list_t.append(float(precessed_train_data.iloc[i+j, title]))
            # print(list_t)
            new_data[i, title] = np.array(list_t.copy())
            new_data[i, len(precessed_train_data.columns) - 1] = np.array(list_t[-1])
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
        row_n =int( len(data)/5)
        # print(row_n)
        data = data.reshape(row_n,5,1)
        # print("data",data) 
        # print(data.shape,type(data),data.dtype,data.size,data.ndim)
        data = torch.from_numpy(data.transpose((2,0, 1))).double()
        labels = torch.from_numpy(labels).double()
        
        # data = data.float().div(255).unsqueeze(0)  # 255也可以改为256
        # data = data.ToTensor()
        return data, labels
        
        
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


def evalute(model, loader):
    model.eval()   #必须要加入 model.eval() ，因为训练和测试BN不一致
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():    #不需要计算梯度，所以加上不求导，验证集一定要加上这几句话
            logits = model(x.float())
            pred = logits.argmax(dim=1)
            # y = y.to(device=device, dtype=torch.int64)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total
batchsz = 32
lr = 1e-3
epochs = 500
device = torch.device('cuda:0')
torch.manual_seed(1234)   #为了方便以后能复现同样结果
# viz = visdom.Visdom()    
def main():
    X_train,X_test,Y_train, Y_test= work()
    torch_data = MyDataset(X_train,Y_train)
    torch_data_test = MyDataset(X_test,Y_test)
    # print(torch_data,"  ",type(torch_data))
    datas = DataLoader(torch_data, batch_size= 32, shuffle=True, drop_last=False)
    # print(datas,"  ",type(datas))
    datas_test = DataLoader(torch_data_test, batch_size=32, shuffle=False, drop_last=False)
 
    # for i, data in enumerate(datas):
    #     # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    #     print("第 {} 个Batch \n{}".format(i, data))
    
    # tensor = tensor.to(torch.float32)
    model = ResNet18(512).to(device)  #把模型放在GPU上面
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) #设置优化器
    criteon = nn.CrossEntropyLoss()   #设置损失函数
    best_acc, best_epoch = 0, 0
    global_step = 0
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))  #初始化 loss,方便可视化
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        dataset_sizes =len(torch_data)
        print(dataset_sizes)
        for step, (x, y) in enumerate(datas):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)
            model.train()              #必须加入model.train()，防止和验证时候BN一样

            logits = model(x.float())
            y = y.to(device=device, dtype=torch.int64)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # viz.line([loss.item()], [global_step], win='loss', update='append') #loss可视化
            global_step += 1
            _,pred = torch.max(logits,1)  
            running_loss += loss.item()*x.size(0)
            running_acc += torch.sum(pred==y) 
        if epoch % 2 == 0:
            val_acc = evalute(model, datas_test)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), './best.mdl') # 保存模型权重
                # viz.line([val_acc], [global_step], win='val_acc',update='append')
        
        epoch_loss = running_loss/dataset_sizes
        epoch_acc = running_acc.double()/dataset_sizes
        print('epoch={}, Phase={}, Loss={:.4f}, ACC:{:.4f}'.format(epoch, step,  epoch_loss, epoch_acc))        
        test_acc = evalute(model, datas_test)  #验证模型，evalute需要我们自己写
        print('test acc:', test_acc)

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))  #加载最好的模型权重
    print('loaded from ckpt!')
    
    test_acc = evalute(model, datas_test)  #验证模型，evalute需要我们自己写
    print('test acc:', test_acc)

    
if __name__ == '__main__':
	main()