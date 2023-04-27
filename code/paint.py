import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"QDF/QDF_DWF_k_1_dim_100_acc_1e-06_time_scale_20230425test.csv")
evolve=df.loc[0,'平均进化次数']
interation=df.loc[0,'平均迭代次数']
best_value=df.loc[0,'最优值']
to_arr_evolve=np.array(evolve)
to_arr_interation=np.array(interation)
to_best_value=np.array(best_value)
D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 70, 100]
def arr_to_str_num(my_arr):
    #将每个元素转换成字符串并替换括号为空
    my_string=str(my_arr)
    my_string=my_string.replace("[",'').replace(']','')
    #将字符串转换回列表
    new_list=my_string.split(",")
    new_list=[float(x) for x in new_list]
    My_arr=new_list
    return My_arr

to_arr_evolve= arr_to_str_num(evolve)
to_arr_interation= arr_to_str_num(interation)
to_best_value=arr_to_str_num(to_best_value)

plt.plot(D,to_arr_interation)
plt.title('Evolve')
plt.show()
plt.plot(D,to_arr_interation)
plt.title('Interation')
plt.show()
plt.plot(D,to_best_value)
plt.title('Best_Velue')
plt.show()

#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# df=pd.read_csv(r"./QDF_DWF_k_1_dim_100_acc_1e-06_time_scale_20230425test.csv")
# evolve=df.loc[0,'平均进化次数']
# interation=df.loc[0,'平均迭代次数']
# best_value=df.loc[0,'最优值']
# to_arr_evolve=np.array(evolve)
# to_arr_interation=np.array(interation)
# to_best_value=np.array(best_value)
# D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 70, 100]
# #去掉数组外面的括号
# def arr_to_str_num(my_arr):
#     #将每个元素转换成字符串并替换括号为空
#     my_string=str(my_arr)
#     my_string=my_string.replace("[",'').replace(']','')
#     #将字符串转换回列表
#     new_list=my_string.split(",")
#     new_list=[float(x) for x in new_list]
#     My_arr=new_list
#     return My_arr
# to_arr_evolve= arr_to_str_num(evolve)
# to_arr_interation= arr_to_str_num(interation)
# to_best_value=arr_to_str_num(to_best_value)
#
# plt.plot(D,to_arr_interation)
# plt.title('Evolve')
# plt.show()
# plt.plot(D,to_arr_interation)
# plt.title('Interation')
# plt.show()
# plt.plot(D,to_best_value)
# plt.title('Best_Velue')
# plt.show()
