from BasicModel import *
import cv2
import numpy as np

modelDirectory = './'
videoDirectory = '../../dataset/processed/'
fileName = '004.avi'

cap = cv2.VideoCapture(videoDirectory+fileName)

out = ""

#sess = tf.Session()
with tf.Session() as sess:
#if True:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,modelDirectory)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = np.array(list(frame),dtype=np.float32) / 255.0
        frame = frame.reshape((frame.shape[0],frame.shape[1],1))
        #print(frame.shape)
        
        confidenceVec,prediction = sess.run((py_x, predict_op),
                                            feed_dict={X: [frame],
                                                       p_keep_conv: 1.0,
                                                       p_keep_hidden: 1.0})
        #print(prediction)
        out += str(confidenceVec[0])+'\n'
        if prediction[0] == 0: # True
            cv2.imshow('win',frame)
        else: # Show the image darker
            cv2.imshow('win',(frame*.25))

        if cv2.waitKey(int(1000/20)) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

with open('./debug.log','w') as f:
    f.write(out)
