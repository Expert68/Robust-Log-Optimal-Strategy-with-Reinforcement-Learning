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
    Note: This is the implementation of Naive average Portfolio Management Strategy
    i.e. we equally invest the money to all the assets
    """
    def __init__(self):
        pass
    
    """Decision Making"""
    def predict(self, y_yesterday):
        return np.ones( len(y_yesterday) ) / len(y_yesterday)