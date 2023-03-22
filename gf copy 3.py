import mlp

import torch
import gym
import random
# import numpy as np


class game():
    # 游戏

    def play(g):
        # env = gym.make('CartPole-v1',render_mode='human')
        env = gym.make('CartPole-v1')
        env.reset()
        action = 0
        ob = env.step(action)
        g.ob = ob
        # print(ob)
        
        score = 0
        sal = list()
        while 1:
            action = g.agent.action()
            # print(action)
            ob = env.step(action)
            g.ob = ob

            sa= dict()
            sa['ob']=ob
            sa['action']=action
            sal.append(sa)

            # print(ob)
            terminate = ob[2]
            if(terminate==1):
                # print('terminate')
                break
            else:
                score+=1
                
        report = dict()
        report['score'] = score
        report['sal'] = sal
        g.agent.report = report
        


class agent():


    def __init__(a):
        depth = 4
        width = 4
        super_param = list()
        for i in range(depth):
            super_param.append(depth)

        super_param[0]=4
        super_param[-1]=2

        lr = 0.03
        batch_size = 10

        a.nn = mlp.mlp(super_param,lr,batch_size,'gpu')


    # 玩家
    def action(a):
        g= a.game
        ob = g.ob
        # print(type(ob))


        # 神经网络军师给一张价值表
        x = torch.from_numpy(ob[0]).cuda()
        x = torch.unsqueeze(x, dim=1)       # 注意，必须保证是列向量输入


        vl = a.nn.test(x)
        # print('vl',vl)

        # print(vl[0]>vl[1])

        ac = 0
        nac = int(vl[0]>vl[1])

        p = a.p # 以多大的概率探索
        ppc = 2**10
        rp = random.randint(0,ppc-1)
        if(rp>p*ppc):
            ac = nac
        else:
            ac = random.randint(0,1)

        # print(ac)
        return ac
    

    def to_tensor(a,na):

        x = torch.from_numpy(na).cuda()
        x = torch.unsqueeze(x, dim=1)       # 注意，必须保证是列向量输入


        return x
    
    # 构造训练集
    def to_train_set(a,sal):
        # print(sal)
        ts = list()
        for i,ele in enumerate(sal):
            fe = ele['ob'][0]
            fe = a.to_tensor(fe)
            # print(fe)
            la = ele['action']
            # print(la)
            la = torch.tensor(la)
            la = torch.nn.functional.one_hot(la,2).cuda()
            data = [fe,la]
            # print(data)
            ts.append(fe)
        return ts

    def main(a):

        g = a.game

        a.p = 0.5

        bs = 0
        brp = dict()
        for i in range(2**5):
            g.play()
            # print(a.score)
            score = a.report["score"]
            # print(score)
            if(score>bs):
                bs = score
                brp = a.report
        # print(brp)
        # 找到最优秀的样本
        # 现在根据这个样本构造训练集
        expt = brp['score']
        print(expt)
        sal = brp['sal']
        # print()
        # 由收集到的数据变换成训练集
        ts = a.to_train_set(sal)
        # print(ts)
        
        # 开始训练

        

        a.nn.batch_size = len(ts)
        tc = 2**5
        for epoch in range(tc):
            # loss = a.nn.loss_f()
            # y = 
            pass
               

g = game()
a = agent()
g.agent = a
a.game = g

a.main()