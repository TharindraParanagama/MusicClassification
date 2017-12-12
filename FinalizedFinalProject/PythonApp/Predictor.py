import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import matplotlib.pyplot as plt

def setSourcePath(Path):
    #dataset=np.loadtxt(Path,delimiter=',')
    dataset=pd.read_csv(Path,header=None)
    if (dataset.ndim == 1):
        dataset = dataset.reshape(1, -1)
    else:
        dataset = dataset

    features=dataset.iloc[:,1:4]

    names=dataset.iloc[:,0]

    scale=StandardScaler()
    norm_features=scale.fit_transform(features)

    with tf.Session() as sess:

      new_saver = tf.train.import_meta_graph('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/saveGeetha.ckpt.meta')
      new_saver.restore(sess,tf.train.latest_checkpoint('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification'))

      #sess.run(tf.global_variables_initializer())

      graph = tf.get_default_graph()

      X = graph.get_tensor_by_name("features:0")
      Y = graph.get_tensor_by_name("labels:0")
      W1 = graph.get_tensor_by_name("weights1:0")
      b1 = graph.get_tensor_by_name("biases1:0")
      W2 = graph.get_tensor_by_name("weights2:0")
      b2 = graph.get_tensor_by_name("biases2:0")
      Wo = graph.get_tensor_by_name("weightsOut:0")
      bo = graph.get_tensor_by_name("biasesOut:0")
      y1 = graph.get_tensor_by_name("activationLayer1:0")
      y2 = graph.get_tensor_by_name("activationLayer2:0")
      a = graph.get_tensor_by_name("activationOutputLayer:0")
      keep_prob = graph.get_tensor_by_name("drop_prob:0")


      feed_dict1 = {X: norm_features,keep_prob:1.0}
      pred = (tf.argmax(a, 1))
      final = sess.run(pred,feed_dict1)
      correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      final=pd.DataFrame(final)
      final=pd.concat([names,final],axis=1)
      final=str(pd.DataFrame(final).as_matrix())
    return final




