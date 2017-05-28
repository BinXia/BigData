#coding=UTF-8
import tensorflow as tf



'''
定义一个LSTM结构。在TensorFlow中通过一句简单的命令就可以实现一个完整的LSTM结构。
LSTM中使用的变量也会在该函数中自动被声明。
'''
lstm = tf.contrib.rnn.BasicLSTMCell(10)


'''
将LSTM中的
'''
state = lstm.zero_state(32,tf.float32)





loss = 0.0

for i in range(100):
	if i > 0: tf.get_varable_scope().reuse_variables()

	