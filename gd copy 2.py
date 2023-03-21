import gym
import time

import mlp


class gt():


    # def rule():
 #  杆子角度不能大于0.26 小于-0.26

    def __init__(self):


        depth = 4


        # kn = 1 # 宽度为 2**kn
        # n= 2**kn
            
        n = 4
        batch_size = 1

        super_param = list()
        for i in range(depth):
            super_param.append(n)

        super_param[-1]=1


        # sa = [1,1]

        lr = 0.1
        self.nn  = mlp.mlp(super_param,lr,batch_size,'gpu')



    def acf(self):

        state = self.state
        # print('state',state)

        # try:
        #     ss = state[0]
        #     print(ss)
        #     # print(type(ss))
        # except :
        #     pass


        ss= 0
        if(self.init):
            ss = state[2]
            # print(ss)
            # print(ss>0)
            # print(ss<0)

        self.init = 1



        # out = ss<0
        # print('out',out)


        out = 0 # 左移
        if(ss>0):
            out = 1
        if(ss<0): # 右倾
            out = 0


        print(ss,out)

        # out = 1
        # print(type(state))

        # print(state[0])

        # ss = list(state[0])
        # print(ss)

        # ack = state[0][2]

        # ack = ss[2]
        # print(ack)

        out = self.env.action_space.sample()
        return out
        
    def main(self):


        # 生成环境
        env = gym.make('CartPole-v1',render_mode='human')
        # 环境初始化
        state = env.reset()
        self.state = state

        self.init = 0
        self.env = env

        # 循环交互

        ll = 0
        
        while True:
            print(ll)
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
        # 环境结束
        # a = input()
        env.close()

a = gt()
a.main()
# a.rule()