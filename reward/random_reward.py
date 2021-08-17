import copy
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

def change_status(state, state_stack,data:DATA,act_sum):
    reward = 0
    action = state.action
    if (action == 1): # 卖出
        status = state.status
        state.out = data.open
        state.status = -1 # 变为空仓 等待买入信号
        if action * act_sum < 0 and status != 0:
            for i in range(0-act_sum):
                old_state = state_stack.pop()
                rew= (data.open - old_state.getin) / old_state.getin
                reward += rew
                print("sub reward",rew)
            state_stack.append(copy.copy(state))
            act_sum = action
        elif action * act_sum >= 0:
            state_stack.append(copy.copy(state))
            act_sum += action

    elif (action == -1): # 买入
        status = state.status
        state.getin = data.open
        state.status = 1 # 变为空仓 等待卖出信号
        if action * act_sum < 0 and status != 0:
            for i in range(act_sum):
                old_state = state_stack.pop()
                rew = (old_state.out - data.open) / old_state.out
                reward += rew
                print("sub reward",rew)
            state_stack.append(copy.copy(state))
            act_sum = action
        elif action * act_sum >= 0 :
            state_stack.append(copy.copy(state))
            act_sum += action

    return reward,act_sum

def main():
    data = pd.read_csv('20_test.csv')# close	open	high	low
    x_columns = [x for x in data.columns] #get the name list of titles 
    data = data[x_columns].values
    [rows, cols] = data.shape
    state_stack = []
    state = State(0,0,0,0)
    # state_stack.append(state)
    sum = 0
    sum_reward = 0
    for i in range (rows - 1):
        action = random.randint(-1,1)
        data_T = DATA(data[i+1,0],data[i+1,1],data[i+1,2],data[i+1,3])
        print("Time ",i ,"-------------------")
        state.action = action
        reward,sum = change_status(state,state_stack,data_T,sum)
        sum_reward += reward
        print("action is ",action,"sum_act ",sum, "tomorrow data is",data[i+1])
        print("reward ",reward, "sum_reward ",sum_reward,"\n==================")
    num = abs(sum)
    for i in range(num):
        data_T = DATA(data[rows - 1,0],data[rows - 1,1],data[rows -1 ,2],data[rows- 1,3])
        # print(len(state_stack),data_T.open)
        if sum < 0 :
            old_state = state_stack.pop()
            # print(old_state.getin)
            reward = (data_T.open - old_state.getin) / old_state.getin
        else:
            old_state = state_stack.pop()
            # print(old_state.out)
            reward = (old_state.out - data_T.open) / old_state.out
        sum_reward += reward
        print("reward ",reward)
    print("total ",sum_reward)

if __name__ == '__main__':
    main()