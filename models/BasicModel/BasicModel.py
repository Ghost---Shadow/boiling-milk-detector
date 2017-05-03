import tensorflow as tf

videoWidth = 176
videoHeight = 144

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

W = []
h = []
layers = [1,4,6,6,4,4]

def model(X, p_keep_conv,p_keep_hidden,LAYERS = 5):
    h.append(X)

    for layer in range(LAYERS):
      W.append(weight_variable([5,5,layers[layer],layers[layer+1]]))
      hTemp = tf.nn.relu(conv2d(h[-1], W[-1]))
      h.append(tf.nn.dropout(hTemp,p_keep_conv))
      print(h[-1].get_shape().as_list())

    lastH = h[-1].get_shape().as_list()
    flatDimensions = lastH[1] * lastH[2] * lastH[3]

    lastHFlat = tf.reshape(h[-1],[-1, flatDimensions])

    Wfc = weight_variable([flatDimensions,2])

    y_conv=tf.nn.dropout(tf.matmul(lastHFlat, Wfc),p_keep_hidden)
    h.append(y_conv)
    
    y_conv=tf.nn.softmax(y_conv)    
    return h[-1]

X = tf.placeholder("float", [None, videoHeight, videoWidth, 1])
Y = tf.placeholder("float", [None, 2]) # True of False

p_keep_hidden = tf.placeholder("float")
p_keep_conv = tf.placeholder("float")
py_x = model(X,p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
