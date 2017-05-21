import tensorflow as tf


# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点格式
FC_SIZE = 512



'''
定义一个用来显示网络每一层结构的函数print_actications，展示每一个卷积层或池化层输出tensor的尺寸。
这个函数接受一个tensor作为输入，并显示其名称(t.op.name)和tensor的尺寸(t.get_shape.as_list())。
'''
def print_activations(t):
	print(t.op.name, ' ', t.get_shape().as_list())

def inference(images):
	parameters = []
	

	# conv1
	with tf.name_scope('layer1-conv1') as scope:
		kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope)
		print_activations(conv1)
		parameters += [kernel, biases]

	# lrn1
	with tf.name_scope('layer2-lrn1') as scope:
		lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
		print_activations(lrn1)

	# pool1
	with tf.name_scope('layer3-pool1') as scope:
		pool1 = tf.nn.max_pool(lrn1,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool1')
		print_activations(pool1)

	# conv2
	with tf.name_scope('layer4-conv2') as scope:
		kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv2)

	# lrn2
	with tf.name_scope('layer5-lrn2') as scope:
		lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
		print_activations(lrn2)

	# pool2
	with tf.name_scope('layer6-pool2') as scope:
		pool2 = tf.nn.max_pool(lrn2,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool2')
		print_activations(pool2)

	# conv3
	with tf.name_scope('layer7-conv3') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],dtype=tf.float32,stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv3)

	# conv4
	with tf.name_scope('layer8-conv4') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],dtype=tf.float32,stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv4)

	# conv5
	with tf.name_scope('layer9-conv5') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],dtype=tf.float32,stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv5)

	# pool5
	with tf.name_scope('layer10-pool5') as scope:
		pool5 = tf.nn.max_pool(conv5,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool5')
		print_activations(pool5)


	# fc1-2
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
									  initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1, 0.5)

	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
									  initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1, fc2_weights) + fc2_biases

	return logit












	
