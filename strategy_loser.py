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
    Note: This is the implementation of Follow the Loser Portfolio Management Strategy
    i.e. we invest the money to the asset with the lowest return
    """
    def __init__(self):
        pass
    
    """Decision Making"""
    def predict(self, y_yesterday):
        weight = np.zeros( len(y_yesterday) )
        index_min = np.argmin(y_yesterday)
        weight[index_min] = 1
        return weight