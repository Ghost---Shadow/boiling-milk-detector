import tensorflow as tf

videoWidth = 176
videoHeight = 144

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a.get_shape())
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    print(l1.get_shape())

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l2a.get_shape())
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    print(l2.get_shape())

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,              
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l3a.get_shape())
    
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],           
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])   
    l3 = tf.nn.dropout(l3, p_keep_conv)

    print(l3.get_shape())

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    print(l4.get_shape())

    pyx = tf.matmul(l4, w_o)

    print(pyx.get_shape()) # (?, 2)
    return pyx

X = tf.placeholder("float", [None, videoHeight, videoWidth, 3])
Y = tf.placeholder("float", [None, 2]) # True of False

w = init_weights([3, 3, 3, 16])       # 3x3x3 conv, 32 outputs
w2 = init_weights([3, 3, 16, 32])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 32, 64])    # 3x3x32 conv, 128 outputs
w4 = init_weights([16 * 36 * 44, 512]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([512, 2])         # FC 625 inputs, 2 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
