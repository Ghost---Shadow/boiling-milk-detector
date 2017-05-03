import os
import sys
import numpy as np
import cv2

modelDirectory = "../../models/TemporalModel/"
scriptpath = "../../models/TemporalModel/"
sys.path.append(os.path.abspath(scriptpath))

from TemporalModel import *

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess,modelDirectory)

CHANNELS = 3

def dream(layer = -1,ITERATIONS = 50):
  img_noise = np.random.uniform(size=(videoHeight,videoWidth,CHANNELS))
  #img_noise = np.ones((videoHeight,videoWidth,CHANNELS)) * .5
  total_image = None

  for channel in range(h[layer].get_shape().as_list()[-1]):
    try:
      t_obj = h[layer][:,:,:,channel]
    except:
      t_obj = h[layer][:,channel]
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,X)[0]
    img = img_noise.copy()
    img = img.reshape((1,videoHeight,videoWidth,CHANNELS))

    for i in range(ITERATIONS):
      g,score = sess.run([t_grad,t_score],{X:img,p_keep_conv: 1,p_keep_hidden: 1})
      g /= g.std()+1e-8
      step = 1
      img += g*step
    print(channel,score)

    img = (img-img.mean())/max(img.std(), 1e-4)*.1 + 0.5     
    if total_image is None:
      total_image = img.reshape((videoHeight,videoWidth,CHANNELS))
    else:
      total_image = np.hstack((total_image,img.reshape((videoHeight,videoWidth,CHANNELS))))
  cv2.imwrite('Total_%s.png'%layer,total_image * 255)

def dreamAll(ITERATIONS = 50):
  for i in range(len(h)):
    dream(i,ITERATIONS)
