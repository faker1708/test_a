
import torch

import mlp
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)


plt_x = []
plt_y = []

# 使用简单的，宽度相同的神经网络

depth = 2


kn = 1 # 宽度为 2**kn
n= 2**kn
    
n = 9
batch_size = 1

sa = list()
for i in range(depth):
    sa.append(n)

# sa = [1,1]

lr = 0.03
a = mlp.mlp(sa,lr,batch_size,'gpu')
x = torch.ones(n,1).cuda()


print(a.super_param)



# print(a)



xxx = a.param['w_list'][0]

# x = torch.normal(0,1,(4,1)).cuda()


y_ref = torch.zeros(n,batch_size).cuda()
y_ref[0]=3
y_ref[1]=1

print(y_ref)

tc = 2**7
for epoch in range(tc):

    y = a.forward(x)



    # print(x)
    # print(y)



    loss  = a.loss_f(y,y_ref)

    fl = float(loss)
    # print(fl)



    loss.backward()



    # print(xxx.grad)
    a.update()


    
    if(epoch>20):
        
        plt_x.append(epoch)
        plt_y.append(fl)
        plt.plot(plt_x, plt_y,c='deeppink')  ## 保存历史数据
        plt.pause(0.1)
        print(fl)

    if(fl<2**-2):
        break
    # print(a.param)


ty = a.test(x)

# print(xxx.grad)
# print(ty.clamp())

# tz = torch.where(ty>0.5,1,0)
print(fl)
print(ty)
# print(tz)

plt.pause(1000)