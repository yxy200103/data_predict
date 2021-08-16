import pandas as pd
import random
class DATA():
    def __init__(self, close ,open, high, low):
        self.open = open
        self.high = high
        self.low = low
        self.close = close

    def __getitem__(self, index):
        return self.open[index],self.close[index]

    def __len__(self):
        return len(self.data)
    

class State():
    def __init__(self,status,action,getin,out):
        self.status = status
        self.action = action
        self.getin = getin
        self.out = out
    def change_status(self,data:DATA):
        act = self.action
        reward = 0
        if (act == 1): # 卖出
            if self.status != 0:
                reward = (data.open - self.getin) / self.getin
            self.out = data.open
            self.status = -1 # 变为空仓 等待买入信号
        elif (act == -1): # 买入
            if self.status != 0:
                reward = (self.out - data.open) / self.out
            self.getin = data.open
            self.status = 1 # 变为空仓 等待卖出信号

        return reward

def main():
    data = pd.read_csv('20_test.csv')# close	open	high	low
    x_columns = [x for x in data.columns] #get the name list of titles 
    data = data[x_columns].values
    [rows, cols] = data.shape
    state_list = []
    state = State(0,0,0,0)
    state_list.append(state)
    sum = 0
    sum_reward = 0
    for i in range (rows - 1):
        if sum == -1:
            action =  random.randint(0, 1) 
        elif sum == 1:
            action = random.randint(-1,0)
        else:
            action = random.randint(-1,1)
        sum += action
        state.action = action
        print("action is ",action, "tomorrow data is",data[i+1])
        data_T = DATA(data[i+1,0],data[i+1,1],data[i+1,2],data[i+1,3])
        reward = state.change_status(data_T)
        sum_reward += reward
        print("Time ",i, "reward ",reward)

    print(sum_reward)

if __name__ == '__main__':
    main()