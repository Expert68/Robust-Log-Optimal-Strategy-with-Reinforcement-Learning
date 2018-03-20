"""
Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)
Authors: XingYu Fu; YiFeng Guo; YuYan Shi; MingWen Liu;
Institution: Sun Yat_sen University
Contact: fuxy28@mail2.sysu.edu.cn
All Rights Reserved
"""


"""Import modules"""
import numpy as np


class Agent:
    """
    Note: This is the implementation of Follow the Winner Portfolio Management Strategy
    i.e. we invest the money to the asset with the highest return
    """
    def __init__(self):
        pass
    
    """Decision Making"""
    def predict(self, y_yesterday):
        weight = np.zeros( len(y_yesterday) )
        index_max = np.argmax(y_yesterday)
        weight[index_max] = 1
        return weight