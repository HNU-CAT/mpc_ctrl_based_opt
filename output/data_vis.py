'''
Author: XXLiu-HNU 3031316709@qq.com
Date: 2025-08-17 19:48:29
LastEditors: XXLiu-HNU 3031316709@qq.com
LastEditTime: 2025-08-17 19:49:49
FilePath: /mpc_ctrl_based_opt/output/data_vis.py
Description: 

Copyright (c) 2025 by xingxun, All Rights Reserved. 
'''
# TODO

import os

import matplotlib.pyplot as plt

def read_txt_files_and_plot():
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    for filename in txt_files:
        data = []
        with open(filename, 'r') as file:
            for line in file:
                try:
                    # 假设每行是一个数字或用空格/逗号分隔的数字
                    nums = [float(x) for x in line.replace(',', ' ').split()]
                    data.append(nums if len(nums) > 1 else nums[0])
                except ValueError:
                    continue  # 跳过无法解析的行
        if data:
            plt.figure()
            plt.title(filename)
            if isinstance(data[0], list):
                for i in range(len(data[0])):
                    plt.plot([row[i] for row in data], label=f'col {i}')
                plt.legend()
            else:
                plt.plot(data)
            plt.xlabel('Index')
            plt.ylabel('Value')
    plt.show()

if __name__ == '__main__':
    read_txt_files_and_plot()