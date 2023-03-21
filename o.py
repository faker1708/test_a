









































import torch 

import matplotlib.pyplot as plt
import math


class dnn():
    
    mode = 'gpu'    # 'gpu'

    width_mi = 4
    depth_mi = 1

    super_param = list()
    for i in range(2**depth_mi):
        super_param.append(width_mi)

    # super_param[0]=7
    # super_param = [7,5,2,2]


    #　数据量
    batch_size = 2**13

    batch_hight = 2**2

    # 训练量
    print_period = 2**1
    train_count = print_period * 2**9



    depth = len(super_param)
    lr = 0.03
    rl = torch.nn.ReLU(inplace=False)   # 定义relu

    def test_a(self,x,true_y,param):
        # 不止被 test调用注意。
        y = self.forward(x,param)
        loss = self.loss_f(y,true_y,self.batch_size)
        return loss




    def test(self,true_param,param):
        kn = self.super_param[0]
        n = 2**kn
        
        test_count = 2**10
        fls = 0
        fll = list() #　float_loss_list


        valid_count = test_count

        invaild = 0
        for i in range(test_count):
            if(self.mode == 'cpu'):
                x = torch.normal(0,1,(n,self.batch_size)).cpu()
            elif(self.mode == 'gpu'):
                x = torch.normal(0,1,(n,self.batch_size)).cuda()


            true_y = self.forward(x,true_param)
            loss = self.test_a(x,true_y,param)
            fl = float(loss)
            fll.append(fl)

            if(fl>2**10):
                invaild = 1
                fl = 0
                valid_count -=1
            
            fls += fl

        if(valid_count==0):
            # flv = float('inf')
            flv = 2**10
        else:
            flv = fls /valid_count

        valid_ratio = valid_count/test_count
        


        return flv,valid_ratio


    def build_nn(self):
        super_param=self.super_param
        depth = len(super_param)

        param = dict()

        w_list = list()
        b_list = list()
        for i,ele in enumerate(super_param):
            if(i<=depth -2):
                n = super_param[i]
                m = super_param[i+1]

                n = 2**n
                m = 2**m



                if(self.mode == 'cpu'):
                    
                    w = torch.normal(0,1,(m,n)).cpu()
                    b = torch.normal(0,1,(m,self.batch_size)).cpu()
                elif(self.mode == 'gpu'):
                    w = torch.normal(0,1,(m,n)).cuda()
                    b = torch.normal(0,1,(m,self.batch_size)).cuda()

                    
                
                # 默认记录计算图
                w.requires_grad=True
                b.requires_grad=True


                w_list.append(w)
                b_list.append(b)
                    


        param['w_list'] = w_list
        param['b_list'] = b_list
        param['depth'] = depth

        return param

    def forward(self,x,param):
        # y = 0


        w_list= param['w_list']
        b_list= param['b_list']

        # print('forward')

        depth = param['depth']
        for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.rl(w @ x + b)

        return x


    def update(self,param):
        
        w_list= param['w_list']
        b_list= param['b_list']

        batch_size = self.batch_size
        lr = self.lr
        # print('update')

        with torch.no_grad():
            
            depth = param['depth']
            for i in range(depth-1): 
                # print('update',i)
                w = w_list[i]
                b = b_list[i]

                w -= lr * w.grad / batch_size
                w.grad.zero_()


                b -= lr * b.grad / batch_size
                b.grad.zero_()

    def loss_f(self,y,true_y,batch):

        diff_y = y-true_y
        # print(diff_y)

        torch.clamp(diff_y,0,255)

        pp = diff_y**2
        # pp = diff_y



        ps = pp/2

        la = ps.sum()
        loss = la/batch
        return loss
    


    def dlf(self,param):
        # 人工生成数据集


        # 解码对象数据
        # param = self.build_nn()

        batch_hight = self.batch_hight
        batch_size = self.batch_size

        n = self.super_param[0]
        n = 2**n



        data_list = list()
        for i in range(batch_hight):
            
            if(self.mode == 'cpu'):
                x = torch.normal(0,1,(n,batch_size)).cpu()
            elif(self.mode == 'gpu'):
                x = torch.normal(0,1,(n,batch_size)).cuda()

            y = self.forward(x,param)


            data = dict()
            data['x']=x
            data['y']=y
            
            data_list.append(data)
        # print('dlf end')
        return data_list


    def fa(self):
        print(self.super_param)
        print('深度',len(self.super_param))
        print('宽度',2**self.super_param[1])

        # batch_size = self.batch_size
        

        # 重新随一个目标网络
        true_param = self.build_nn()
        data_list = self.dlf(true_param)  

        plt.ion()
        plt.figure(1)

        
        try_index = 0
        while(1):
            print("try_index",try_index)
            try_index+=1


            find_it = 0
            patience = 2**4

            


            # 重新随一个初始训练网络
            train_param = self.build_nn()


            # plt 绘图
            x_index = list()
            y_index = list()
            z_index = list()
            pp = 0
            for epoch in range(self.train_count):


                for i in range(self.batch_hight):
                    
                    # print('fa')
                    # print('batch_hight',i)
                    data = data_list[i]
                    x = data['x']
                    true_y = data['y']

                    loss = self.test_a(x,true_y,train_param)
                    
                

                    loss.backward(retain_graph=True)

                    self.update(train_param)
                


                if(epoch%(self.print_period * 2**4)== 0):

                    fl = float(loss)
                    print(fl)

                if(epoch%(self.print_period * 2**8)== 0):
                    pp +=1

                    test_loss,valid_ratio = self.test(true_param,train_param)
                    
                        
                        
                    if(pp>0):
                        x_index .append(epoch)

                        z_index .append(fl)
                        plt.plot(x_index, z_index,c='blue',ls='-')  ## 保存历史数据

                        y_index .append(test_loss)
                        plt.plot(x_index, y_index,c='deeppink',ls='-')  ## 保存历史数据
                        
                        # if(fl>2**10):
                        #     flz = 2**10
                        plt.pause(0.1) # 显示图像
                        
                        

                # if(epoch%(self.print_period)== 0):
                    # pp = epoch//(self.print_period)
                    # 动态调整学习率
                    if(loss>2**10):
                        self.lr = 2**2 # 3
                    elif(loss>2**8):
                        self.lr = 2**2
                    elif(loss>10):
                        self.lr=2
                    elif(loss>1):
                        self.lr = 1 #0.1
                    else:
                        self.lr = 0.03
                    print(fl,test_loss,valid_ratio ,'lr = ',self.lr)

                    if(test_loss<2**8):
                        print('通过',test_loss)
                        break
            # epoch end
            test_loss,valid_ratio = self.test(true_param,train_param)
            print(test_loss,valid_ratio )

        print('\a')
        plt.pause(0)

a = dnn()
a.fa()

