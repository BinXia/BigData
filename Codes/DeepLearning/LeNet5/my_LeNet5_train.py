#coding=UTF-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

# 加载my_LeNet5_inference中定义的常亮和前向传播的函数
import my_LeNet5_inference


# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 600
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH="../Models/MNIST_1/"
MODEL_NAME="LeNet5_model_1"


def train(mnist):
    # 定义输入输出placeholder
    '''
    x = tf.placeholder(tf.float32, [None, my_LeNet5_inference.INPUT_NODE], name='x-input')
    '''
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,                           # 第一维表示一个batch中样例的个数
            my_LeNet5_inference.IMAGE_SIZE,    # 第二维和第三维表示图片的尺寸
            my_LeNet5_inference.IMAGE_SIZE,
            my_LeNet5_inference.NUM_CHANNELS   # 第四维表示图片的深度，对于RBG格式的图片，深度为5
        ], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, my_LeNet5_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用my_LeNet5_inference.py中定义的前向传播过程
    y = my_LeNet5_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)


    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)#移动平均，反向传播
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                -1,
                my_LeNet5_inference.IMAGE_SIZE,    # 第二维和第三维表示图片的尺寸
                my_LeNet5_inference.IMAGE_SIZE,
                my_LeNet5_inference.NUM_CHANNELS   # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                  ))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                merged_summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs',sess.graph)
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str,i)
        '''
        merged_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
        total_step = 0
        while train:
              total_step += 1
              sess.run(train_op)
              if total_step % 100 == 0:
                  summary_str = session.run(merged_summary_op)
                  summary_writer.add_summary(summary_str,total_step)
        '''






def main(argv=None):
    mnist = input_data.read_data_sets("../Dataset/MNIST_data_1", one_hot=True)
    train(mnist)

if __name__ == '__main__':tf.app.run()


