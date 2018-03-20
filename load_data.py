"""
Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)
Authors: XingYu Fu; YiFeng Guo; YuYan Shi; MingWen Liu;
Institution: Sun Yat_sen University
Contact: fuxy28@mail2.sysu.edu.cn
All Rights Reserved
"""

"""Import Necessary packages"""
import os
import numpy as np


"""Load Stock Data from Disk"""
def Load_Stock_Data():
    path = r'./database'
    Database = []
    # Loading price information
    for data_name in os.listdir(path):
        Database.append(np.loadtxt( (path+ '/' +data_name),dtype=np.float,delimiter=",",usecols=[1,2,3,4,5]))
    # Loading date information
    Database.append(list(np.loadtxt( (path+ '/' +data_name),dtype=np.str,delimiter=",",usecols=[0])))
    return Database