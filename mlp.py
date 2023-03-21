


import torch

class mlp():

    def __init__(self,super_param,lr,batch_size,mode = 'gpu'):
        # super_param = [2,2] # 超参是个列表，记录有几层神经元，每层有几个神经元，列表里的值 不是个数 ，而是个数的对数。具体构造规则见build函数。
        self.super_param = super_param
        self.mode = mode
        self.batch_size = batch_size
        


        self.depth = len(super_param)
        self.lr = lr
        self.rl = torch.nn.ReLU(inplace=False)   # 定义relu

        self.build_nn()
    

    def set_super_param(self,super_param):
        self.set_super_param = super_param


    def build_nn(self):
        super_param=self.super_param


        depth = len(super_param)
        w_list = list()
        b_list = list()
        for i,ele in enumerate(super_param):
            if(i<=depth -2):
                # kn = super_param[i]
                # km = super_param[i+1]

                # n = 2**kn
                # m = 2**km
                
                n = super_param[i]
                m = super_param[i+1]
                

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
                    

        param = dict()
        param['w_list'] = w_list
        param['b_list'] = b_list
        param['depth'] = depth


        self.param = param
        # return param


    def forward(self,x):
        # y = 0

        # if(gr==0):

        param = self.param

        w_list= param['w_list']
        b_list= param['b_list']

        # print('forward')

        depth = param['depth']
        for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
            
            # print('forward',i)
            w = w_list[i]
            b = b_list[i]

            
            x = self.rl(w @ x + b)

        y = x
        return y

    def loss_f(self,y,y_ref):

        # diff_y = y - y_ref
        # pp = diff_y**2
        # ps = pp/2

        dd = (y-y_ref)**2 /2
        
        la = dd.sum()



        batch = 1 # batch 和 lr 共同控制收敛速度，没必要加batch了。
        loss = la / batch
        return loss
    
    def update(self):
        
        param = self.param
        batch_size = self.batch_size
        lr = self.lr

        w_list= param['w_list']
        b_list= param['b_list']

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

    def test(self,x):

        # 测试与训练不同，这是确实要一个值出来 ，并且不要算梯度。


        with torch.no_grad():
            param = self.param

            w_list= param['w_list']
            b_list= param['b_list']

            # print('forward')

            depth = param['depth']
            for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
                
                # print('forward',i)
                w = w_list[i]
                b = b_list[i]

                # print(x.shape)
                
                x = self.rl(w @ x + b)
                # print(w.shape)
                # print(b.shape)
                # print('x',x.shape)

                # print('\nwewfe\n')

            y = x
        return y
