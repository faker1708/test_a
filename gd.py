import gym
import time

import mlp
import torch


class gt():


    # def rule():
 #  杆子角度不能大于0.26 小于-0.26

    def __init__(self):


        depth = 2


        # kn = 1 # 宽度为 2**kn
        # n= 2**kn
            
        n = 4
        batch_size = 1

        super_param = list()
        for i in range(depth):
            super_param.append(n)

        super_param[-1]=1
        print(super_param)


        # sa = [1,1]

        lr = 0.02
        self.nn  = mlp.mlp(super_param,lr,batch_size,'gpu')



    def acf(self):

        state = self.state

        # print(type(state))
        # print(state)

        if(self.init):
            ss = state

        else:
            ss = state[0]
            pass

        x = torch.from_numpy(ss).cuda()#.half()
        # x.reshape((4,1))
        # x.reshape(-1)

        x = torch.unsqueeze(x, dim=1)

        # x = torch.ones(4,1)

        # print('x.shape',x.shape)
        # print(ss)


        # print(b)
        # print(type(b))
        self.init = 1



        ac = 0


        # out = self.env.action_space.sample()
        
        # ac = self.foresee()
        
        tac = self.nn.test(x)
        # tac.backward()

        

        tac = torch.where(tac>0.5,1,0)


        fac = float(tac)

        print('fac',fac)

        ac = fac>0.5
        # print('ac',ac)
        

        return ac
        
    def main(self):


        # 生成环境
        env = gym.make('CartPole-v1',render_mode='human')
        # env = gym.make('CartPole-v1')
        # 环境初始化
        state = env.reset()
        self.state = state

        self.init = 0
        self.env = env

        # 循环交互

        ll = 0
        
        while True:
            # print(ll)
            ll+=1
            # 渲染画面
            # env.render()
            # 从动作空间随机获取一个动作
            # action = env.action_space.sample()
            action = self.acf()
            
            # agent与环境进行一步交互
            # state, reward, terminated, truncated, info 
            ob = env.step(action)
            # print('state = {0}; reward = {1}'.format(state, reward))
            state = ob[0]
            self.state = state

            terminated= ob[2]

            # 判断当前episode 是否完成
            if terminated:
                print('terminated',ob[3])
                # print(state, reward, terminated, truncated, info)
                break
            time.sleep(0.1)
        print(ll)
        # 环境结束
        # a = input()
        env.close()

a = gt()
a.main()
# a.rule()