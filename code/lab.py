#********************************************
#QDA algorithm with mean value replacement
#Copyright Parallel Computing Lab.
#Author:Wang Peng Date:2022.10.22
#********************************************
import numpy as np ##numpy库：python科学计算扩展库，主要用来处理任意维度数组和矩阵
import matplotlib.pyplot as plt
import time

def func1(matrix_y):# y=x*x
    y=0   ## ？dim=0表示的是行（对列进行操作），dim=1表示的是列（对行进行操作）
    for i in matrix_y:
        y+=i*i
    return y
def func(matrix_y):    #double well function
    y=0
    l,k,h=3.0,0.5,5.0
    for i in matrix_y:
        y+=h*pow((pow(i, 2)-pow(l, 2)), 2)/pow(l, 4)+k*i  ##pow（x,y） 返回x的y次方的值
    return y

def func1(matrix_y):    #Rastrigin  
    y=10*dim   ##dim代表不同维度
    for i in matrix_y:
        y+=i*i-10*np.cos(2*np.pi*i)  ##y=x^2-10cos(2pix)
    return y

def func1(matrix_y):    #Griewank
    a,b,y=0,1,0
    index=1
    for i in matrix_y:
        a=a+i*i ##连加
        b=b*np.cos(i/np.sqrt(index)) ##连乘  numpy.sqrt 开方
        index+=1
    y=1+1/4000*a-b
    return y

def mid_exchange(matrix):  #均值替换
    mid_walker=np.mean(matrix,1)  ##对matrix压缩列，对各行求均值
    matrix[:,max_min_id(matrix,0)]=mid_walker  ##所有的行，xx列
    return matrix

def max_min_id(matrix,flag): #obtain max or min walker's index   ##flag用来做数据的切换，初始设置为0，当想要切换数据时让其等于1
                                                                 #获取最大或最小粒子下标，flag=0找最大，flag=1找最小.
    walker_id=0
    temp=func(matrix[:,0])  ##表示所有的行，第一列
    for j in range(walker_n-1):
        if temp < func(matrix[:, j + 1]) and flag == 0:#max condition
            walker_id = j + 1
            temp = func(matrix[:, j + 1])  
        if temp > func(matrix[:,j+1]) and flag==1:#min condition
            walker_id=j+1
            temp = func(matrix[:, j + 1])
    return walker_id




# time.clock()默认单位为s
# time.clock 在py3.8之后已经弃用，用time.perf_counter替代
# 获取开始时间
start = time.perf_counter()
'''
代码开始
'''
acc=0.000001 #精度 
ite_times=0 #迭代次数
dim, walker_n=10, 20   #维度，粒子数  ##walker_n种群数，20个高斯不断采    种群数*维度=大的数组
begin, end=-6, 6 #目标函数搜索定义域
scale=end-begin  #初始尺度   跳出第二个循环后scale/2 
ite_flag=True        #scale constraction flag
a=np.random.uniform(begin, end, (dim, walker_n)) ##从一个均匀分布中随机采样，左闭右开，第一个参数是下限，第二个参数是上限，第三个参数是产生什么样子的随机数，比如（3，2）表示3行二列
b=np.zeros((dim,walker_n))  ##返回一个给定形状和类型用0填充的数组np.zeros(shape, dtype=float ,order ='C')   shape数组形状， dtype数据类型 orderC用于C的行数组，F用于FORTRAN的列数组
print(a)
# 核心
##每迭代一次用当前所有点的平均值去替换那个坐标
while scale>acc:
    while ite_flag==True:
        ite_times+=1
        for j in range(walker_n): #range()为迭代器，如果步长不设置则默认为1
                                  ##range(start, stop[, step]) 
                                  # start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）
                                  # stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
                                  # step：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
            for i in range(dim): 
                b[i, j]=np.random.normal(a[i, j], scale) #sampling #新的采样，生成高斯分布的概率密度随机数
                                                         ##正态分布，用于生成随机数，第一个参数表示均值，第二个参数表示方差，第三个参数表示size
                while b[i,j]<begin or b[i, j]>end:  #越界处理
                    b[i,j] = np.random.normal(a[i, j], scale)
            if func(b[:,j])>func(a[:, j]):  #b解第j个粒子有点差
                b[:, j]=a[:, j]
        mid_exchange(b)  #mean value replacement function.
        a[:, :] = b[:, :]
        ite_flag=False
        for i in range(dim):
            if np.var(a[i, :]) > pow(scale, 2): #np.var方差 np.pow次方 scale的二次方  如果a的方差>尺度，说明还不是全部都是最优解，里面还有搅屎棍，还得采样替换
                ite_flag=True  
    scale=scale/2   
    print(scale)
    ite_flag=True
    print("1---------------")
print("Scale=", scale,"Itetimes", ite_times)
print("Min_location=", a[:,max_min_id(a,1)])
print("Min_value=", func(a[:,max_min_id(a,1)]))


# 获取结束时间
end = time.perf_counter()
# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
