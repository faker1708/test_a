import mlp

import torch
import gym


class game():
    # 游戏

    def play(g):
        env = gym.make('CartPole-v1',render_mode='human')
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
                print('terminate')
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
        batch_size = 1

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
        ac = int(vl[0]>vl[1])

        # print(ac)
        return ac

    def main(a):
        g = a.game


        bs = 0
        bsal = list()
        for i in range(3):
            g.play()
            # print(a.score)
            score = a.report["score"]
            print(score)
            if(score>bs):
                bs = score
                bsal = a.report['sal']
        print(bsal)
               

g = game()
a = agent()
g.agent = a
a.game = g

a.main()