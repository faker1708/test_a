import gym
import time

import mlp
import torch
import random


class gt():


    # def rule():
 #  杆子角度不能大于0.26 小于-0.26

    def __init__(self):


        depth = 2   # 神经网络的深度


        # kn = 1 # 宽度为 2**kn
        # n= 2**kn
            
        n = 4
        batch_size = 1

        super_param = list()
        for i in range(depth):
            super_param.append(n)

        
        # 保证最初是4个元，  最终是1个元
        super_param[0]=4
        super_param[-1]=1
        # print(super_param)


        # sa = [1,1]

        lr = 0.02
        self.nn  = mlp.mlp(super_param,lr,batch_size,'gpu')



    def acf(self):

        state = self.state





        ac = 0


        
        tac = self.nn.test(self.state)
        # tac.backward()

        
        yuzhi = 0.5 # 这个阈值与下面的01无关，只是relu 会让输出大于等于0，yuzhi不要取太小。更不要小于0
        tac = torch.where(tac>yuzhi,1,0)
        fac = float(tac)



        nac = fac>0.5


        p = 0.1
        xfc = 2**10
        xf = random.randint(0,xfc-1)
        if(xf<p*xfc):
            # print('explore')
            ac = random.randint(0,1)
        else:
            # print('greedy')
            ac = nac

        return ac
        

    def liquidate(self):

        print('清算')
        q_line = 0  # 比这条线高，就当作榜样来学习
        dcl = self.dcl


        # le = len(dcl)
        # sum = 0
        # for i,dc in enumerate(dcl):
        #     # print(dc['score'])

        #     sum+=dc['score']
        # avg = sum/le
        # print('平均得分',avg)

        cc = self.cc

        # 把高于平均分的加入其中
        upl = list()
        for i,dc in enumerate(dcl):
            if (dc['score'] > cc):
                upl.append(dc)
                # print(dcl[i]['score'])
        
        # 其实可以加入一个评价训练的机制，暂时不管了。
        # 如果训练后成绩变差了，就放弃这次的训练成果。

        print('训练集')
        for i ,ele in enumerate(upl):
            print(ele['score'])

        print(upl[0]['list'])


    def main(self):


        # 生成环境
        env = gym.make('CartPole-v1',render_mode='human')
        env = gym.make('CartPole-v1')
        # 环境初始化

        self.init = 0
        self.env = env

        # self.data_list= list()


        # 目标就是提升这个 cc
        self.cc = 10

        # 循环交互


        # 模拟次数
        sc = 2**6
        dcl = list()

        for simu_index in range(sc):

            
            self.simu_index = simu_index



            state = env.reset()[0]
            state = torch.from_numpy(state).cuda()
            self.state = state



            score = 0
            

            dc = dict()
            dd = list()
            while True:


                action = self.acf()
                
                ob = env.step(action)
                self.state = torch.from_numpy(ob[0]).cuda()

                terminated = ob[2]

                d = dict()
                d['state'] = self.state
                d['action'] = action
                dd.append(d)

                # 判断当前episode 是否完成
                if terminated:
                    # print('terminated',ob[3])
                    break
                else:
                    
                    score+=1
                # time.sleep(0.1)
            # print(score)
            dc['list']= dd
            dc['score']= score
            # print(dc['score'])

            dcl.append(dc)

        self.dcl= dcl    

        # 清算

        self.liquidate()










        env.close()

a = gt()
a.main()