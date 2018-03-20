"""
Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)
Authors: XingYu Fu; YiFeng Guo; YuYan Shi; MingWen Liu;
Institution: Sun Yat_sen University
Contact: fuxy28@mail2.sysu.edu.cn
All Rights Reserved
"""


"""Import Modules"""
# Deep Learning
from NN import Network


class Agent:
    """Note: this is the implementation of Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)"""
    
    """Initialization Function"""
    def __init__(self, M, N, Feature):       
        # define the hyperparameters for RLOSRL
        self.assets_num = M
        self.rl_history_len = N 
        self.features = Feature
        # define Deep Neural Network
        self.network = Network(1e-3, 'Adam')
        self.network.inputs([1, M, N, Feature], [1, 1], [1, M], [M], [1, M, N, Feature])
        conv1_depth = 2
        conv2_depth = 10
        conv3_depth = 1
        self.network.conv_layer(1, 3, Feature, conv1_depth, activation=True, insert_w = False, name='conv1')
        self.network.conv_layer(1, N-2, conv1_depth, conv2_depth, activation=True, insert_w = True, name='conv2')
        self.network.conv_layer(1, 1, conv2_depth + 1, conv3_depth, activation=False, insert_w = False, name='conv3')
        self.network.fc_layer(M , 1, name='fc')
        self.network.graph()
    
    """Decision Making"""
    def predict(self, x_t, v_t):
        w_rlosrl , r_predicted = self._RL(x_t, v_t)
        return w_rlosrl[0], r_predicted[0][0]
    
    """Update the CNN for latest market information"""
    def train(self, material, assets_num, rl_history_len, features):
        # (x_t, v_t, w_target, r_target)
        self.network.train(material, self.assets_num, self.rl_history_len, self.features)
    
    """Reinforcement Learning"""
    def _RL(self, x_t, v_t):
        return self.network.test(x_t, v_t)