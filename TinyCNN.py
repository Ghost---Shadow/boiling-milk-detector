#!/usr/bin/env python
import numpy as np
import pickle
from models.TinyModel.TinyModel import *

print('Starting')

batch_size = 32
#test_size = 256
EPOCHS = 10
directory = './dataset/processed/'
pickleName = 'dataset.pickle'
modelDirectory = './models/TinyModel/'

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

del dataset

saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
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
        saver.save(sess,modelDirectory)

