'''
动态折线图演示示例
'''
 


import matplotlib.pyplot as plt
 
plt.ion()
plt.figure(1)




t_list = []
result_list = []
# t = 0
 
for t in range(100):

    t_list.append(t)
    result_list.append(t*t)

    plt.plot(t_list, result_list,c='deeppink')  ## 保存历史数据
    #plt.plot(t, np.sin(t), 'o')
    plt.pause(0.1)