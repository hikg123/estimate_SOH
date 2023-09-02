import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")

# In[interp & feature extraction]
def interp1(V,Q,V_seq):
    fun = interpolate.interp1d(V, Q,kind='linear',fill_value='extrapolate')
    Q_int = fun(V_seq)
    return Q_int

def feature(Cyc,Stride,Cha_interval,seg_length,Cn):
    # Cyc = Cyc_Bat_arr
    Qmax = Cn
    Stride = Stride
    Cha_interval = Cha_interval
    seg_length = seg_length
    Vmax=4.19
    Vmin=3.75
    cnt=1
    feature_list = []
    
    # OG program
    for i in range(len(Cyc)):
        # i = 0
        cyc_tem = Cyc[i]
        cyc_tem.reset_index(drop=True, inplace=True)
        plt.figure(figsize=(5,3),dpi=400)
        plt.plot(cyc_tem.Vmean,label='mean')
        plt.plot(cyc_tem.Vmax,label='max')
        plt.legend(loc=0)
        plt.title('charge cut voltage');plt.xlabel('Sample');plt.ylabel('Voltage')
        plt.show()
        
        V_start = max( np.min(cyc_tem.Vmax) , np.min(cyc_tem.Vmean) )
        V_end = min( np.max(cyc_tem.Vmax) , np.max(cyc_tem.Vmean) )
        
        try:
            Seq_min = max(cyc_tem[cyc_tem.Vmean <= V_start].index)+1# 寻找插值起点
            Seq_max = min(cyc_tem[cyc_tem.Vmax >= V_end].index)+1;# 寻找插值终点
        except:
            print('data out')
            continue
        
        cyc_cut = cyc_tem[Seq_min:Seq_max]
        cyc_cut.reset_index(drop=True, inplace=True)
        
        for j in range(1,len(cyc_cut)):
            if cyc_cut.Vmean[j-1] >= cyc_cut.Vmean[j]:
                cyc_cut.Vmean[j] = cyc_cut.Vmean[j-1]*(1+1e-6)
            if cyc_cut.Vmax[j-1] >= cyc_cut.Vmax[j]:
                cyc_cut.Vmax[j] = cyc_cut.Vmax[j-1]*(1+1e-6)
        
        V_seq = np.arange(V_start,V_end,Cha_interval)# 生成目标电压序列，采样间隔1mV
        Q1 = interp1(cyc_cut.Vmax, cyc_cut.Q, V_seq)
        Q2 = interp1(cyc_cut.Vmean, cyc_cut.Q, V_seq)
        
        plt.figure(figsize=(5,3),dpi=400)
        plt.plot(cyc_cut.Vmean, cyc_cut.Q, label='Og')
        plt.plot(V_seq, Q2, label='inter')
        plt.legend(loc=0)
        plt.title('Q interpolate');plt.xlabel('Vmean');plt.ylabel('Q')
        plt.show()
        
        V_seq = (V_seq-Vmin)/(Vmax-Vmin)# 电压最大最小归一化
        Q1=Q1/Qmax# 电量增量归一化（以额定容量为量度）
        Q2=Q2/Qmax# 电量增量归一化（以额定容量为量度）
        if (len(Q1)<seg_length or len(Q2)<seg_length ): # 保证增量片段长度
            continue 
        sample_all = pd.DataFrame(np.vstack([V_seq,Q1,Q2]).T,columns=['V','Q1','Q2'])
        
        num = int((len(V_seq)-seg_length)/Stride)
        cnt = 1
        cut_start = 0
        feature = pd.DataFrame()
        for j in range(num):
            # j = 1
            sample_cut = sample_all[cut_start:cut_start+seg_length]
            sample_cut.reset_index(drop=True, inplace=True)
            Q1temp = sample_cut.Q1[:]-sample_cut.Q1[0]# 部分充电片段的Q1增量序列
            Q2temp = sample_cut.Q2[:]-sample_cut.Q2[0]# 部分充电片段的Q2增量序列
            Vtemp = sample_cut.V
            feature_tmp = pd.DataFrame(np.vstack([Vtemp,Q1temp,Q2temp]).T)
            feature = pd.concat([feature,feature_tmp])
            cut_start = cut_start+Stride
        feature_list.append(feature)
    return feature_list
    
# In[Feature extraction]
Stride = 10# 步长
Cha_interval = 0.001# 电压间隔
seg_length = 60# 片段长度
Cn = 152# 标称容量，单位为Ah
Ca = 140.8# 实测容量
N_cell = 108
Qmax = Cn# 标称容量，单位为Ah
Sample_val = []
Label_val = []

data = pd.read_csv('欧拉好猫-152.1ah-HPPC工况2-38-70.csv')
col = data.columns

# 创建结构体，存储车辆的测试工况数据并进行可视化检查
time = data.TIME.values
I_load = data.BMS_SUM_CHARGE_CUTTENT.values
Vmean_cell = data.BMS_SUM_CHARGE_VOLTAGE.values/N_cell
Vmax = data.MAX_CELL_VOLT.values
Vmean = (Vmean_cell+Vmax)/2
plt.figure(figsize=(5,3),dpi=400)
plt.plot(Vmean,label='mean')
plt.plot(Vmax,label='max')
plt.legend(loc=0)
plt.show()

# 计算电量增量,Ah
Q = np.zeros((len(I_load),1))
for k in range(len(I_load)):
    Q[k,0]=-sum(I_load[:k])/3600
Cyc_Bat = pd.DataFrame(np.vstack([time,I_load,Vmax,Vmean,Q[:,0]]).T,
                        columns=['time','I_load','Vmax','Vmean','Q'])

index = Cyc_Bat[Cyc_Bat.I_load<-10].index.values# 寻找所有充电数据的行索引
index_start=0
Cyc_Bat_arr=[]# 存储恒流充电片段
for i in range(len(index)-1):# 原始数据中存在非充电数据，需要进行切分，单独提取每个恒流充电片段
    if (index[i+1]-index[i])>100: #原表格中有连续超过100个不满足充电要求的数据（Cyc_Bat.I_load<-10），就将其作为临界分段
        # print(i, index[i+1])
        index_end=i
        Cyc_Bat_arr.append(Cyc_Bat.iloc[index[index_start]:index[index_end]])
        index_start=i+1
Cyc_Bat_arr.append(Cyc_Bat.iloc[index[index_start]:index[-1]])

feature_list = feature(Cyc_Bat_arr,Stride,Cha_interval,seg_length,Cn)

if len(feature_list)>1:
    x_pd = feature_list[0]
    for i in range(len(feature_list)):
        x_pd = pd.concat([x_pd,feature_list[i]])
else:
    x_pd = feature_list[0]

x = (x_pd.values).reshape(-1,60,3)

# In[estimation]
from keras.models import load_model
import tensorflow_probability as tfp

# model_est = load_model('model_base_1.h5')
model_est = load_model('model_tl_1.h5')
y_est = model_est.predict(x)
y_ave = np.mean(y_est[:])
soh = Ca/Cn
print(soh,y_ave)







