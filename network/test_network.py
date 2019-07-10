import tensorflow as tf
import numpy as np

# y = 2 * x^2 + 1
class TestNetwork:
    def __init__(self, sess, restore_from=None):
        self.data = tf.placeholder(tf.float32, [None, 1])
        self.label = tf.placeholder(tf.float32, [None, 1])
        # 定义网络
        self.predict = self.network(self.data)

        if restore_from:
            saver = tf.train.Saver()
            saver.restore(sess, restore_from)
        else:
            self.loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.data, labels=self.label))

            self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

    # 32 -> 32 -> 16 -> 8 -> 4 -> 2 -> softmax
    def network(self, data):
        layer1 = tf.layers.dense(data, 10, tf.nn.relu, name='layer1')
        output = tf.layers.dense(layer1, 1, tf.nn.relu, name='layer_output')
        return output


if __name__ == "__main__":
    x = np.random.randint(1000, size=[200])
    y = 2 * x ** 2 + 1
    x = x.reshape([len(x), 1])
    y = y.reshape([len(x), 1])
    # print(x)
    # print(y)

    with tf.Session() as sess:
        testNetwork = TestNetwork(sess, None)
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            index = np.random.randint(0, 200, size=[16])
            input = x[index]
            label = y[index]
            predict, _, loss = sess.run([testNetwork.predict, testNetwork.train_op, testNetwork.loss],
                                        feed_dict={testNetwork.data: input, testNetwork.label: label})
        saver = tf.train.Saver()
        saver.save(sess, './model/model', write_meta_graph=False)
