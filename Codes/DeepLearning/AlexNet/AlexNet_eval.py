#coding=UTF-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import my_AlexNet_inference
import my_AlexNet_train


# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            5000,
            28,    # 第二维和第三维表示图片的尺寸
            28,
            1   # 第四维表示图片的深度，对于RBG格式的图片，深度为5
        ], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        validate_feed = {x: np.reshape(mnist.validation.images,(-1,28,28,1)), y_: mnist.validation.labels}

        y = my_AlexNet_inference.inference(x)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(my_AlexNet_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(my_AlexNet_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)



def main(argv=None):
    mnist = input_data.read_data_sets("../Dataset/MNIST_data_2", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()
