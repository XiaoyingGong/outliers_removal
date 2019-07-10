import tensorflow as tf
import numpy as np


# 输入为32维的向量， 输出为一个标记
# 输出为两个维度的向量，[1， 0] 第一个维度为是inliers的可能性，第二个为是outliers的可能性
class ORNet:
    def __init__(self, restore_from):
        self.inputs = tf.placeholder(tf.float32, [None, 16])
        self.labels = tf.placeholder(tf.float32, [None, 2])
        # 定义网络
        self.layer_1 = self.add_layer(self.inputs, 16, 16, activation_func=tf.nn.relu)
        self.layer_2 = self.add_layer(self.layer_1, 16, 8, activation_func=tf.nn.relu)
        self.outputs = self.add_layer(self.layer_2, 8, 2, activation_func=None)
        # Session
        self.sess = tf.Session()
        # 是否加载
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:
            self.loss = tf.losses.softmax_cross_entropy(self.labels, self.outputs)
            #self.loss = -tf.reduce_mean(tf.reduce_sum(self.labels * tf.log(self.outputs), reduction_indices=[1]))
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())


    # 定义训练
    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.inputs: x, self.labels: y})
        return loss

    # 定义预测
    def predict(self, x):
        outputs = self.sess.run(self.outputs, {self.inputs: x})
        return outputs

    # 保存
    def save(self, path="./model/model"):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

    # 16-> 16 -> 8 -> 2
    # 定义某一层
    def add_layer(self, inputs, insize, outsize, activation_func=None):
        Weights = tf.Variable(tf.random_normal([insize, outsize]))
        bias = tf.Variable(tf.zeros([1, outsize]) + 0.1)
        wx_plus_b = tf.matmul(inputs, Weights) + bias
        if activation_func:
            return activation_func(wx_plus_b)
        else:
            return wx_plus_b
