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
from xgboost import XGBClassifier
from xgboost import plot_importance

import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from resnet import ResNet18
 
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
    # X = new_data[0:rows-5,1:cols-2]
    # print("X\n",X)
    # X = np.array(precessed_train_data[headers].values)
    # Y = np.array(precessed_train_data['delta'].values)
    X = new_data[0:rows-5,1:cols-2]
    # print("X\n",X)

    Y = new_data[0:rows-5,cols - 1]
    # print("Y\n",Y)
    # print("Y\n",Y)
    X_train,X_test,Y_train, Y_test = train_test_split(np.array(X),np.array(Y), test_size=0.15)
    # print(X_train,X_test,Y_train, Y_test )
    return X_train,X_test,Y_train, Y_test

class MyDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label 
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
       
        data = data.reshape(44,5,1)
 
        data = torch.from_numpy(data.transpose((2,0, 1))).double()
        labels = torch.from_numpy(labels).double()
 
        return data, labels
        
        
   
    def __len__(self):
        return len(self.data)


def evalute(model, loader):
    model.eval()   
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): 
            logits = model(x.float())
            pred = logits.argmax(dim=1)
            # y = y.to(device=device, dtype=torch.int64)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total
batchsz = 32
lr = 1e-3
epochs = 1000
device = torch.device('cpu')
torch.manual_seed(1234)   
# viz = visdom.Visdom()    
def main():
    X_train,X_test,Y_train, Y_test= work()
    torch_data = MyDataset(X_train,Y_train)
    torch_data_test = MyDataset(X_test,Y_test)
    # print(torch_data,"  ",type(torch_data))
    datas = DataLoader(torch_data, batch_size= 10, shuffle=True, drop_last=False)
    # print(datas,"  ",type(datas))
    datas_test = DataLoader(torch_data_test, batch_size=10, shuffle=False, drop_last=False)
 
    # tensor = tensor.to(torch.float32)
    model = ResNet18(512).to(torch.float32)  #把模型放在GPU上面
    model = model.to(torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr) #设置优化器
    criteon = nn.CrossEntropyLoss()   #设置损失函数
    best_acc, best_epoch = 0, 0
    global_step = 0 
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        dataset_sizes =len(datas) * 10
        print(dataset_sizes)
        for step, (x, y) in enumerate(datas):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)
            model.train()    
            logits = model(x.float())
            y = y.to(device=device, dtype=torch.int64)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            global_step += 1
            _,pred = torch.max(logits,1)  
            running_loss += loss.item()*x.size(0)
            running_acc += torch.sum(pred==y) 
        if epoch % 2 == 0:
            val_acc = evalute(model, datas_test)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), './best.mdl') 
        
        epoch_loss = running_loss/dataset_sizes
        epoch_acc = running_acc.double()/dataset_sizes
        print('epoch={}, Phase={}, Loss={:.4f}, ACC:{:.4f}'.format(epoch, step,  epoch_loss, epoch_acc))        
        test_acc = evalute(model, datas_test)   
        print('test acc:', test_acc)

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl')) 
    print('loaded from ckpt!')
    
    test_acc = evalute(model, datas_test) 
    print('test acc:', test_acc)

    
if __name__ == '__main__':
	main()