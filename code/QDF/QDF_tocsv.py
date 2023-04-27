from math import exp
from sre_constants import JUMP
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import csv
import re
from datetime import datetime
import time


def func1(matrix_y):  # y=x*x
    y = 0
    for i in matrix_y:
        y += i * i
    return y


def func(matrix_y):    #double well function,dwf 双阱函数
    y=0
    global l,k,h
    for i in matrix_y:
        y+=h*pow((pow((fre*i), 2)-pow(l, 2)), 2)/pow(l, 4)+k*(fre*i)
    return y

def func1(matrix_y):  # Rastrigin
    A = 10
    w = 2 * np.pi
    y = A * dim
    for i in matrix_y:
        y += i * i - A * np.cos(w * i)
    return y


def func1(matrix_y):  # Griewank
    a, b, y = 0, 1, 0
    index = 1
    for i in matrix_y:
        a = a + i * i
        b = b * np.cos(i / np.sqrt(index))
        index += 1
    y = 1 + 1 / 4000 * a - b
    return y


scale_acc = 0.000001
acc = 0.000001

walker_n = 30
begin, end = -5, 5
reptime = 30  # 重复次数
fre = 1
l,k,h=3,1,5 # 双阱DWF参数
time_scale = 0.01  # 时间尺度
# D=np.linspace(1,10,10)
# K=[0.1, 0.2, 0.5, 1]

k1=h/pow(l,4)
k2=pow(l,4)*k/h
st=math.acos(3*math.sqrt(3)*k2/(8*pow(l,3)))
D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 70, 100]
meantimes = []
meanevo = []
meanttimes = []
meanglobal_min_arr = []
findrate = []
SR = 1
for d in D:
    # if SR <= 0:
    #     print("算法执行到此，成功率为0，dim=", d - 1)
    #     break
    rep = 0
    resetnum = 0
    dim = int(d)
    max_item = 10000 * dim
    atimes = []
    etimes = []
    ttimes = []
    global_min_arr = []
    SR1 = 0
    while rep < reptime:
        t = 0
        ite_times = 0  # 迭代次数
        evolutions = 0  # 函数进化次数
        ttb=np.ones((dim, walker_n))*(-2)*l*math.cos(st/3)/math.sqrt(3)  # ttb用于计算双阱函数在每个维度下的理论全局最优值
        op_f = func(ttb[:, 1])
        a = np.random.uniform(begin, end, (dim, walker_n))
        b = np.zeros((dim, walker_n))
        c = np.zeros((dim, walker_n))
        scale = end - begin
        funcV = np.zeros(walker_n)  # 存放每次func(a)的计算结果
        sampleV = np.zeros(walker_n)  # 存放每次func(b)的计算结果
        for i in range(walker_n):
            funcV[i] = func(a[:, i])
            evolutions += 1
        factor = 1
        ite_flag = True  # scale constraction flag
        rep += 1
        once_start_time = time.time()
        while min(funcV)-op_f > acc and evolutions < max_item:
            while ite_flag == True and min(funcV)-op_f > acc and evolutions < max_item:
                ite_times += 1
                for j in range(walker_n):
                    for i in range(dim):
                        b[i, j] = np.random.normal(a[i, j], scale)  # sampling
                        while b[i, j] < begin or b[i, j] > end:
                            # if b[i,j]<begin or b[i, j]>end:
                            #     b[i,j]=random.uniform(begin, end)
                            b[i, j] = np.random.normal(a[i, j], scale)
                    sampleV[j] = func(b[:, j])
                    evolutions += 1
                    if sampleV[j] < funcV[j]:  # 若采到好解
                        t += time_scale
                        a[:, j] = b[:, j]
                        funcV[j] = sampleV[j]
                # -----------------均值替换-------------------- #
                index_max = np.argmax(funcV)
                a[:, index_max] = np.mean(a, 1)  # 均值替换 找到历史a函数值中最大的数下标 用a矩阵中每个维分量求均值替换掉
                funcV[index_max] = func(a[:, index_max])  # 将历史最优值中最差的值也替换掉
                evolutions += 1
                t += time_scale
                ite_flag = False
                # 判断高斯u正态分布是否收敛
                for i in range(dim):
                    if np.var(a[i, :]) > pow(scale, 2):
                        ite_flag = True
            scale = scale / 2  # 尺度减半
            t = 0
            # print('尺度下降', scale, '次数', rep, '成功次数', SR1)
            ite_flag = True
        global_min = min(funcV)
        global_min_local = a[:, np.argmin(funcV)]
        once_end_time = time.time()
        once_time = once_end_time - once_start_time
        ttimes.append(once_time)
        if global_min - op_f < acc:  # 如果此轮找到的全局最优-理论全局最优 < 精度acc, 则认为算法成功找到解
            SR1 = SR1 + 1
        # print("-----------第" + str(dim) +"维，重复第" + str(rep) + "次-------------")
        # print("Dim=", d)
        # print("Scale=", scale, "Itetimes", ite_times, "函数进化次数", evolutions)
        # print("Min_location=", ttb[:,1])
        # print("My_location=", global_min_local)
        # print("Min_value=", op_f)
        # print("My_value=", global_min)
        atimes.append(ite_times)
        etimes.append(evolutions)
        global_min_arr.append(global_min)
        # print("--------------------------------------------")
    SR = SR1 / reptime  # 成功率
    print("Dim=", dim, "SR=", SR)
    # ttimes单次找到解的物理时间 atimes单次找到解的迭代次数 etimes单次找到解的函数进化次数
    meanttimes.append(np.mean(ttimes))
    meantimes.append(np.mean(atimes))
    meanevo.append(np.mean(etimes))
    meanglobal_min_arr.append(np.mean(global_min_arr))
    # meanttimes 每个执行的维度 重复reptime次实验 平均物理时间
    # meantimes 每个执行的维度 重复reptime次实验 平均迭代次数
    # meanevo 每个执行的维度 重复reptime次实验 平均函数进化次数
    findrate.append(SR)  # 每个维度findrate找到率即成功率
    x_dim = pd.DataFrame([[dim, k, meantimes, meanevo, findrate, meanglobal_min_arr, meanttimes]],
                         columns=['维度', 'CR', '平均迭代次数', '平均进化次数', '成功率', '最优值', '时间'])
    x_dim.to_csv(
        'QDF_DWF_k_' + str(k) + '_dim_' + str(dim) + '_acc_' +
            str(acc) + '_time_scale_' + re.sub(r'[^0-9]', '', datetime.now().strftime("%Y%m%d")) + 'test.csv')
    # [^0-9] 表示匹配除数字以外的任何字符，即非数字字符，'' 表示将匹配到的字符替换为空字符串。
plt.figure()
plt.plot(D, meantimes)
plt.savefig('k_'+str(format(k,'.4f'))+'_h_'+str(h)+'_l_'+str(l)+'_dim_'+str(dim)+'_acc_'+str(acc)+'_time_scale_'+ re.sub(r'[^0-9]','',datetime.now().strftime("%Y%m%d"))+'_找到次数.png',dpi=600,bbox_inches='tight')

plt.figure()
plt.plot(D,findrate)
plt.savefig('k_'+str(format(k,'.4f'))+'_h_'+str(h)+'_l_'+str(l)+'_dim_'+str(dim)+'_acc_'+str(acc)+'_time_scale_'+ re.sub(r'[^0-9]','',datetime.now().strftime("%Y%m%d"))+'_找到率.png',dpi=600,bbox_inches='tight')

plt.show()