import tensorflow as tf




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
		print_activations(conv1)

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

	return pool5, parameters












	