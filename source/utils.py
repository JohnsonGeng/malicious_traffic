#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@File    :   utils.py    
@Author  :   你牙上有辣子
@Contact :   johnsogunn23@gmail.com
@Create Time : 2021-10-14 16:08     
@Discription :

工具类，包含数据预处理和画图操作

'''


# import lib
import csv
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


data_path = '../data/detection/'
result_path = '../result/'

# 数据格式转换用
protocol_type = ['tcp', 'udp', 'icmp']
service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo',
           'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
           'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
           'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
           'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
           'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
           'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']


# 数据读取类，用于读取数据和标签，将其传递给检测器类
class DataLoader():
    def __init__(self, feature=None):
        self.data = []
        self.label = []
        self.feature = feature

    def load_data(self, filename):
        with open(data_path+filename) as f:
            for row in f:
                row = row.split(',')
                self.data.append(row[:-1])
                # 去掉换行
                temp = row[-1].replace('\n', '')
                # 获取标签
                self.label.append(temp)

                # 存放提取后的特征
                x = []

            if self.feature == None:
                x = self.data
            else:
                # 把对应的特征取出来
                feature = set(self.feature)
                for data_row in self.data:
                    new_data_row = []
                    for i in feature:
                        new_data_row.append(data_row[i])
                    x.append(new_data_row)

            return self.data, self.label


# 读取csv,针对两种不同的csv文件
def load_csv(fiename, original=False):
    data = []
    label = []
    if original == True:
        index = 2
    else:
        index = 1
    with open(fiename) as csvfile:
        for row in csvfile:
            row = row.split(',')
            # 处理protocol_type
            row[1] = str(protocol_type.index(row[1]))
            # 处理service
            row[2] = str(service.index(row[2]))
            # 处理flag
            row[3] = str(flag.index(row[3]))
            # 将str类型转化为float类型
            for i in range(len(row)-index):
                row[i] = float(row[i])
            # 处理label
            row[-index] = row[-index].replace('\n', '')
            if row[-index] == 'normal':
                label.append(0)
            else:
                label.append(1)
            data.append(row[:-index])

    return data, label


# 保存csv
def save_csv(path, data, label):

    if os.path.exists(path):
        os.remove(path)

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            data[i].append(label[i])
            writer.writerow(data[i])

    print("Save data successfully to: {}".format(path))


# 读出CSV数据
def load_data(path):

    data = []
    label = []

    with open(path) as f:
        for row in f:
            row = row.split(',')
            data.append(row[:-1])
            # 去掉换行
            temp = row[-1].replace('\n', '')
            label.append(temp)

    return data, label


# 数据预处理
def data_preprocessing(data):

    # Min-Max数据
    # min_max_scaler = preprocessing.MinMaxScaler()
    # temp_data = min_max_scaler.fit_transform(data)
    # 正则化数据,这个效果回比上面Min-Max好一些
    preprocessed_data = preprocessing.normalize(data, norm='l2')

    return preprocessed_data


# 绘制折线图
def draw_line(x, y, metric):
    plt.xticks(x)
    plt.plot(x, y['DT'], color='deepskyblue', marker='o', ls='-', label='DT')
    plt.plot(x, y['RF'], color='lavender', marker='o', ls='-', label='RF')
    plt.plot(x, y['KNN'], color='wheat', marker='o', ls='-', label='KNN')
    plt.plot(x, y['NB'], color='plum', marker='o', ls='-', label='NB')
    plt.plot(x, y['MLP'], color='teal', marker='o', ls='-', label='MLP')
    plt.plot(x, y['Ada'], color='red', marker='o', ls='-', label='Ada')
    plt.plot(x, y['Bagging'], color='grey', marker='o', ls='-', label='Bagging')
    plt.plot(x, y['GBDT'], color='orangered', marker='o', ls='-', label='GBDT')
    plt.xlabel("Feature used during model training")
    plt.ylabel("Accuracy(%)")
    plt.legend(loc='best', ncol=2)
    plt.savefig(metric+".png")


# 绘制柱状图
def draw_bar(data_list1):
    bar_label_list = ['ARC', 'CS', 'CP', 'CG', 'MR', 'DU']
    bar_locs = np.arange(6)
    bar_width = 0.40
    # 柱状图 X 轴标识
    xtick_labels = [bar_label_list[i] for i in range(6)]

    # 用 matplotlib 库中的 pyplot 来画出 图形
    plt.figure()
    # 柱状图
    rect1 = plt.bar(bar_locs, data_list1, width=bar_width, color='orange', alpha=0.7, xtick_labels=bar_label_list)
    # rect2 = plt.bar(bar_locs + bar_width, data_list2, width=bar_width, color='orange', alpha=0.7, label='攻击后')
    # 显示数值
    for rect in rect1:
        plt.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='right', va='bottom', fontsize=12)

    plt.ylabel = '次数'
    plt.ylim(0, 2000)
    plt.title('智能体所作出的修改动作统计')
    plt.tight_layout()
    plt.savefig('statistics_bar.png',dpi=1000)
    plt.show()



def my_drawline(x, y, metric):
    plt.plot(x, y, c='deepskyblue')
    # plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.xlim(0, 1000)
    plt.ylim(-1, 2)
    plt.savefig(metric+".png")
    plt.show()



if __name__ == '__main__':
    # 折线图绘制
    # x = [6, 7, 8, 9, 10]
    # y = {'DT':[87.43, 87.65, 87.93, 88.11, 87.98],
    #      'RF':[86.93, 86.80, 87.19, 87.76, 88.06],
    #      'KNN':[85.24, 84.89, 85.26, 86.02, 86.13],
    #      'NB':[72.30, 73.28, 73.84, 73.94, 74.21],
    #      'MLP':[79.20, 79.13, 79.94, 80.13, 80.67],
    #      'Ada':[83.11, 83.78, 84.31, 84.52, 84.89],
    #      'Bagging':[81.20, 81.33, 81.84, 82.40, 83.06],
    #      'GBDT':[84.56, 84.77, 85.03, 85.43, 85.50]}
    # draw_line(x, y, 'test')
    pass