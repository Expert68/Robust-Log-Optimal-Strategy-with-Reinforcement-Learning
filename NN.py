"""
Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)
Authors: XingYu Fu; YiFeng Guo; YuYan Shi; MingWen Liu;
Institution: Sun Yat_sen University
Contact: fuxy28@mail2.sysu.edu.cn
All Rights Reserved
"""


"""Import Modules"""
import tensorflow as tf
import numpy as np
import os


"""Loss function related"""
alpha = 0.1
beta = 0.1
sigma = 0.1
c = 0.1

class Network():
    def __init__(self, learning_rate, optimizer, path = 'model/default.ckpt'):
        # Hyper parameters       
        self.learning_rate = learning_rate
        self.optimizer = optimizer
      
        # Network structure
        self.conv_config = [] 
        self.conv_w = []
        self.conv_b = []
        self.fully_connected_config = [] 
        self.fully_connected_w = []
        self.fully_connected_b = []
        
        # Graph
        self.w = None
        self.tf_train_data = None
        self.tf_train_r = None
        self.tf_train_weights = None
        self.tf_test_data = None

        # Tensorboard
        self.writer = None
        self.train_summaries = []

        # Save
        self.saver = None
        self.path = path
     
    def inputs(self, train_data_shape, train_r_shape, train_weights_shape, w_shape, test_data_shape):
        with tf.name_scope('inputs'):
            self.tf_train_data = tf.placeholder(tf.float32, shape = train_data_shape, name='train_data')
            self.tf_train_r = tf.placeholder(tf.float32, shape = train_r_shape, name='train_r')
            self.tf_train_weights = tf.placeholder(tf.float32, shape = train_weights_shape, name='train_weights')
            self.w = tf.placeholder(tf.float32, shape = w_shape, name='insert_w')
            self.tf_test_data = tf.placeholder(tf.float32, shape = test_data_shape, name='test_data')

    def conv_layer(self, filter_height, filter_width, in_depth, out_depth, activation, insert_w, name):
        self.conv_config.append({
            'filter_height': filter_height, 
            'filter_width': filter_width,
            'in_depth': in_depth,
            'out_depth': out_depth,
            'activation': activation,
            'insert_w' : insert_w,
            'name': name
        })
        with tf.name_scope(name+ '_'):
            w = tf.Variable(tf.truncated_normal([filter_height, filter_width, in_depth, out_depth], stddev=0.1), name=name + '_w')
            b = tf.Variable(tf.constant(0.1, shape=[out_depth]), name=name + '_b')
            self.conv_w.append(w)
            self.conv_b.append(b)

    def fc_layer(self, num_in, num_out, name):
        self.fully_connected_config.append({'num_in': num_in,'num_out': num_out,'name': name})
        with tf.name_scope(name + '_'):
            w = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1), name=name + '_w')
            b = tf.Variable(tf.constant(0.1, shape=[num_out]), name=name + '_b')
            self.fully_connected_w.append(w)
            self.fully_connected_b.append(b)

    def graph(self):        
        def model(data, insertion):
            for i, (w, b, config) in enumerate(zip(self.conv_w, self.conv_b, self.conv_config)):
                with tf.name_scope(config['name']):
                    with tf.name_scope('conv2d'):
                        data = tf.nn.conv2d(data, filter = w, strides=[1, 1, 1, 1], padding = 'VALID')
                    if config['activation']:
                        data = data + b
                        data = tf.nn.relu(data)
                if config['insert_w']:
                    with tf.name_scope('insert_w'):
                        width = data.get_shape()[2]
                        height = data.get_shape()[1]
                        features = data.get_shape()[3]
                        data = tf.reshape(data, [-1, int(height), 1, int(width*features)])
                        reshape_w = tf.reshape(insertion, [-1, int(height), 1, 1])
                        data = tf.concat([data, reshape_w], axis=3)

            data = data[:, :, 0, 0]
            
            for i, (w, b, config) in enumerate(zip(self.fully_connected_w, self.fully_connected_b, self.fully_connected_config)):
                with tf.name_scope(config['name']):
                    predicted_r = tf.matmul(data, w) + b
                    
            with tf.name_scope('softmax'):
                predicted_weights = tf.nn.softmax(data)
            
            return predicted_weights,predicted_r
        
        predicted_weights, predicted_r = model(self.tf_train_data, self.w)
        
        with tf.name_scope('loss'):
            # Squared error between the predicted and the true log return
            loss1 = (predicted_r[0][0]- self.tf_train_r[0][0])**2
            # Cross entropy
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.tf_train_weights, logits= predicted_weights, name="xent"))
            loss3 = self.tf_train_r[0][0]
            # L2 regularization
            regularization = 0
            for w, b in zip(self.fully_connected_w, self.fully_connected_b):
                regularization = regularization + tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            loss4 = regularization* 1e-5
            # Loss function
            self.loss = alpha * loss1 - beta * loss2 - sigma * loss3 + c * loss4
            self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

        # Optimizer
        with tf.name_scope('optimizer'):
            if (self.optimizer == 'Gradient'):
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif (self.optimizer == 'Adam'):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Predictions
        with tf.name_scope('train'):
            self.train_predicted_weights = predicted_weights
            self.train_predicted_r = predicted_r
        with tf.name_scope('test'):
            self.tf_test_predicted_weights,self.tf_test_predicted_r  = model(self.tf_test_data, self.w)

        self.merged_train_summary = tf.summary.merge(self.train_summaries)     
        self.saver = tf.train.Saver()
            
    def train(self, material, m, n, num_features):
        self.writer = tf.summary.FileWriter('board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            for  i in range(0, len(material) - 1):
                # Train
                train_weights = np.reshape(material[i][2],(-1, m))
                train_data = np.reshape(material[i][0],(-1,m,n,num_features))
                train_r = np.reshape(material[i][3],(-1,1))
                w = material[i][1]
                weights, r, summary = session.run(
                    [self.train_predicted_weights, self.train_predicted_r, self.merged_train_summary],
                    feed_dict={self.tf_train_data: train_data, self.tf_train_r: train_r, self.tf_train_weights: train_weights, self.w: w}
                )
                self.writer.add_summary(summary, i)
            
            # Save
            if os.path.isdir(self.path.split('/')[0]):
                self.saver.save(session, self.path)
            else:
                os.makedirs(self.path.split('/')[0])
                self.saver.save(session, self.path)
                
            return weights, r
        
    def test(self, test_data, w):
        if self.writer is None:
            self.writer = tf.summary.FileWriter('board', tf.get_default_graph())
        if self.saver is None:
            self.define_graph()
            
        with tf.Session(graph=tf.get_default_graph()) as session:
            # Read model
            if os.path.exists('model'):
                self.saver.restore(session, self.path)
            else:
                tf.initialize_all_variables().run()
            weights, r = session.run([self.tf_test_predicted_weights, self.tf_test_predicted_r],
                feed_dict={self.tf_test_data: test_data, self.w: w}
            )
            
            return weights, r