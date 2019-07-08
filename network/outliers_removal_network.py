import tensorflow as tf
import numpy as np


# 输入为32维的向量， 输出为一个标记
# 输出为两个维度的向量，[1， 0] 第一个维度为是inliers的可能性，第二个为是outliers的可能性
class ORNet:
    def __init__(self, sess, restore_from):
        self.data = tf.placeholder(tf.float32, [None, 32])
        self.label = tf.placeholder(tf.float32, [None, 2])
        # 定义网络
        self.predict = self.network(self.data)

        if restore_from:
            saver = tf.train.Saver()
            saver.restore(sess, restore_from)
        else:
            self.loss = -tf.reduce_mean(tf.reduce_sum(self.label * tf.log(self.predict), reduction_indices=[1]))
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    # 32 -> 32 -> 16 -> 8 -> 4 -> 2 -> softmax
    def network(self, data):
        layer1 = tf.layers.dense(data, 32, tf.nn.relu, name='layer1')
        layer2 = tf.layers.dense(layer1, 16, tf.nn.relu, name='layer2')
        layer3 = tf.layers.dense(layer2, 8, tf.nn.relu, name='layer3')
        layer4 = tf.layers.dense(layer3, 4, tf.nn.relu, name='layer4')
        layer5 = tf.layers.dense(layer4, 2, tf.nn.relu, name='layer5')
        layer_output = tf.layers.dense(layer5, 2, tf.nn.softmax, name='layer_output')
        return layer_output

