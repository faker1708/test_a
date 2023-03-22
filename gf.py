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
        depth = 8
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

        x = torch.from_numpy(na)#.cuda()
        x = torch.unsqueeze(x, dim=1)       # 注意，必须保证是列向量输入


        return x
    
    # 构造训练集
    def to_train_set(a,sal):
        # print(sal)
        ts = list()
        
        of = sal[0]['ob'][0]
        of = a.to_tensor(of)

        ol= sal[0]['action']
        ol = torch.tensor(ol)
        ol = torch.nn.functional.one_hot(ol,2)
        ol= torch.unsqueeze(ol,1)
        # print(ol)
        

        for i,ele in enumerate(sal):
            if(i==0):continue
            fe = ele['ob'][0]
            fe = a.to_tensor(fe)
            # print(fe)
            of = torch.cat((of,fe),1)

            la = ele['action']
            # print(la)
            la = torch.tensor(la)

            la = torch.nn.functional.one_hot(la,2)#.cuda()
            la= torch.unsqueeze(la,1)

            # print(i,ol,la)
            ol = torch.cat((ol,la),1)

        of = of.cuda()
        ol= ol.cuda()
        data = [of,ol]
        # print(data)
            # data = [fe,la]
            # print(data)
            # ts.append(fe)
        return data

    def main(a):

        g = a.game

        a.p = 0.5

        bs = 0
        oe = 0

        for j in range(2**6):
            # bs = oe
            brp = dict()

            # 模拟，收集数据
            for i in range(2**5):
                a.p = 0.5
                g.play()
                # print(a.score)
                score = a.report["score"]
                # print(score)
                if(score>bs):
                    bs = score
                    brp = a.report
                    print(bs)
            # print(brp)
            # 找到最优秀的样本
            # 现在根据这个样本构造训练集
            # print(brp)
            try:
                expt = brp['score']
                
                sal = brp['sal']
            except:
                expt = 0
            
            # if(expt):print('expt',expt)
            if(expt==0):continue
            
            # expt = oe
            # print()
            # 由收集到的数据变换成训练集
            data = a.to_train_set(sal)
            # print(ts)
            
            # 
            # print('开始训练')

            x=data[0]
            y_ref= data[1]

            # a.nn.batch_size = len(data[0][0])
            a.nn.batch_size = expt
            # print('',a.nn.batch_size)
            tc = 2**5
            for epoch in range(tc):
                # loss = a.nn.loss_f()
                y = a.nn.forward(x)
                loss = a.nn.loss_f(y,y_ref)
                loss.backward()
                a.nn.update()

                fl = float(loss)
                print(fl)
                # pass

        # for i in range(10):
            a.p = 0.
            g.play()
            score = a.report['score']
            # print('test',score)
            oe = score




        # a.test()

    def tt(a):
        a = torch.tensor([[1],[2]])          
        b = torch.tensor([[3],[4]])
        # c = torch.stack([a,b],dim=1)
        c = torch.cat((a,b),1)
        c = torch.cat((c,b),1)

        print(c)

g = game()
a = agent()
g.agent = a
a.game = g

# a.main()
a.main()