
import torch

import mlp
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)


plt_x = []
plt_y = []

# 使用简单的，宽度相同的神经网络

depth = 4


# kn = 1 # 宽度为 2**kn
# n= 2**kn
    
n = 361
batch_size = 1

sa = list()
for i in range(depth):
    sa.append(n)

# sa = [1,1]

lr = 0.1
a = mlp.mlp(sa,lr,batch_size,'gpu')
# x = torch.ones(n,1).cuda()
x = torch.normal(0,1,(n,batch_size)).cuda()


print(a.super_param)



# print(a)



xxx = a.param['w_list'][0]



# y_ref = torch.zeros(n,batch_size).cuda()
# y_ref[0]=2
# y_ref[1]=0

y_ref = torch.normal(10,1,(n,batch_size)).cuda()


print(y_ref)

tc = 2**6
for epoch in range(tc):

    y = a.forward(x)



    loss  = a.loss_f(y,y_ref)

    fl = float(loss)


    loss.backward()



    a.update()


    
    if(epoch>20):
        
        plt_x.append(epoch)
        plt_y.append(fl)
        plt.plot(plt_x, plt_y,c='deeppink')  ## 保存历史数据
        plt.pause(0.1)
        print(fl)

    if(fl<2**-2):
        break
ty = a.test(x)


loss  = a.loss_f(y,y_ref)
fl = float(loss)

print('测试',fl)
# print(ty)

print(ty)
print(y_ref)


plt.pause(1000)