import tensorflow as tf
import numpy as np
from network.outliers_removal_network import ORNet

with tf.Session() as sess:
    # or_net = ORNet(sess, None)
    # labels = [[1, 0], [0, 1], [0, 1], [0, 1]]
    # rd = np.random.rand(4, 32)
    # sess.run(tf.global_variables_initializer())
    #
    # # training
    # for i in range(10):
    #     sess.run(tf.global_variables_initializer())
    #     result, _, loss = sess.run([or_net.predict, or_net.train_op, or_net.loss],
    #                        feed_dict={or_net.data: rd, or_net.label: labels})
    #     print('第', i, '轮训练: ', result)
    #     print('第', i, '轮训练: ', loss)
    #
    # saver = tf.train.Saver()
    # saver.save(sess, './model/model', write_meta_graph=False)

    # 预测
    # 前向传播
    or_net = ORNet(sess, "./model/model")
    rd = np.random.rand(4, 32)
    sess.run(tf.global_variables_initializer())
    predict = sess.run(or_net.predict, feed_dict={or_net.data: rd})
    print(predict)
