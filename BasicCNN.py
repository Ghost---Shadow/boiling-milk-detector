#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pickle

print('Starting')

batch_size = 32
#test_size = 256
EPOCHS = 10
directory = './dataset/processed/'
pickleName = 'dataset.pickle'

videoWidth = 176
videoHeight = 144

with open(directory+pickleName,'rb') as fp:
    dataset = pickle.load(fp)

trX = []
trY = []
teX = []
teY = []

for fileName in dataset:
    for frame in dataset[fileName][0]:
        trX.append(frame)
    for vec in dataset[fileName][1]:
        trY.append(vec)

tr = np.array(list(zip(trX,trY)))
np.random.shuffle(tr)
test_size = int(len(tr)/10) # 10% for testing

trX = np.array(list(tr[:-test_size,0]),dtype=np.float32) / 255.0
trX = trX.reshape((trX.shape[0],trX.shape[1],trX.shape[2],1))
trY = np.array(list(tr[:-test_size,1]),dtype=np.uint8)

teX = np.array(list(tr[-test_size:,0]),dtype=np.float32) / 255.0
teX = teX.reshape((teX.shape[0],teX.shape[1],teX.shape[2],1))
teY = np.array(list(tr[-test_size:,1]),dtype=np.uint8)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 144, 176, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a.get_shape())
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 72, 88, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    print(l1.get_shape())

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 72, 88, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l2a.get_shape())
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 36, 44, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    print(l2.get_shape())

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 36, 44, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l3a.get_shape())
    
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 2048)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 625)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    print(l3.get_shape())

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    print(l4.get_shape())

    pyx = tf.matmul(l4, w_o)

    print(pyx.get_shape()) # (?, 2)
    return pyx

X = tf.placeholder("float", [None, videoHeight, videoWidth, 1])
Y = tf.placeholder("float", [None, 2]) # True of False

w = init_weights([3, 3, 1, 16])       # 3x3x1 conv, 32 outputs
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

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(EPOCHS):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
