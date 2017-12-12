#declaring inputs
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

#loading data from text file
dataset=np.loadtxt('/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/DatasetAfterClustering.csv',delimiter=',',skiprows=1)

#specifying feature and label columns
features=dataset[:,0:3]
labels=dataset[:,3]

#binarizing labels/one-hot encoding
le=LabelBinarizer()
labels=le.fit_transform(labels)

#standardizing data
scale=StandardScaler()
norm_features=scale.fit_transform(features)

#splitting data to train and test split
tr_features,ts_features,tr_labels,ts_labels=train_test_split(norm_features,labels,test_size=0.5,random_state=42)

#printing results for testing purposes
print ("tr_features", tr_features)
print ("tr_encode", tr_labels)
print ("ts_features", ts_features)
print ("ts_encode", ts_labels)

# setting hyper parameters & other variables
training_epochs = 200
n_features = tr_features.shape[1]
n_classes = tr_labels.shape[1]
n_neurons_in_h1 = 120
n_neurons_in_h2 = 120
learning_rate = 0.001

#printing results for testing purposes
print (n_classes)

# placeholdr tensors built to store features(in X) , labels(in Y) and dropout probability(in keep_prob)
X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')
keep_prob=tf.placeholder(tf.float32,name='drop_prob')

#The point for using truncated normal is to overcome saturation of functions like sigmoid (where if the value is too big/small, the neuron stops learning).
#Weights allow you to change the steepness of the activation function in such a way that you will yield better results.while the biases allow you to shift your activation function left or right.
# network parameters(weights and biases) are set and initialized(Layer1)
W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1],mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
# activation function(tanh)
y1 = tf.nn.tanh((tf.matmul(X, W1)+b1), name='activationLayer1')
#dropout layer 1
drop_out_layer1 = tf.nn.dropout(y1, keep_prob)

# network parameters(weights and biases) are set and initialized(Layer2)
W2 = tf.Variable(tf.truncated_normal([n_neurons_in_h1, n_neurons_in_h2],mean=0, stddev=1 / np.sqrt(n_features)), name='weights2')
b2 = tf.Variable(tf.truncated_normal([n_neurons_in_h2],mean=0, stddev=1 / np.sqrt(n_features)), name='biases2')
# activation function(tanh)
y2 = tf.nn.tanh((tf.matmul(drop_out_layer1, W2)+b2), name='activationLayer2')
#dropout layer 2
drop_out_layer2 = tf.nn.dropout(y2, keep_prob)

# network parameters(weights and biases) are set and initialized(output layer)
Wo = tf.Variable(tf.truncated_normal([n_neurons_in_h2, n_classes],mean=0, stddev=1 / np.sqrt(n_features)), name='weightsOut')
bo = tf.Variable(tf.truncated_normal([n_classes],mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
# activation function(softmax)
a = tf.nn.softmax((tf.matmul(drop_out_layer2, Wo) + bo), name='activationOutputLayer')

# tensorboard histograms on summary operations
tf.summary.histogram("weights1", W1)
tf.summary.histogram("biases1", b1)
tf.summary.histogram("weights2", W2)
tf.summary.histogram("biases2", b2)
tf.summary.histogram("weightsOut", Wo)
tf.summary.histogram("biasesOut", bo)

# name scope for the cost function for more clarity on tensorboard
with tf.name_scope('Cost'):
    # cost function(cross entropy)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a),reduction_indices=[1]))#reduction indices=1 means row wise mean
    #optimization function
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # scalar summary for plotting cost variation againt epoches
    tf.summary.scalar('Cost', cross_entropy)

# name scope for the accuracy for more clarity on tensorboard
with tf.name_scope('Accuracy'):
    # compare predicted value from network with the expected value/target
    correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
    # accuracy determination
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    # scalar summary for plotting accuracy variation against epoches
    tf.summary.scalar('Accuracy', accuracy)

# initialization of all variables
initial = tf.global_variables_initializer()

#creating an instance of a session object to execute the computational graph
with tf.Session() as sess:
    sess.run(initial)
    #writing summery values to file
    writer = tf.summary.FileWriter("/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/Geetha/TrainRuns")
    #saving the computational graph
    writer.add_graph(sess.graph)
    #merging all summary operations
    merged_summary = tf.summary.merge_all()

    # training in batches of samples
    batchsize=50
    for epoch in range(training_epochs):

        for i in range(len(tr_features)):

            start=i
            end=i+batchsize
            x_batch=tr_features[start:end]
            y_batch=tr_labels[start:end]


            # feeding training data/examples
            sess.run(train_step, feed_dict={X:x_batch , Y:y_batch,keep_prob:0.5})
            i+=batchsize
        # feeding testing data to determine model accuracy
        y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: ts_features,keep_prob:1.0})
        y_true = sess.run(tf.argmax(ts_labels, 1))
	    #accuracy for each epoch
        summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: ts_features, Y: ts_labels,keep_prob:1.0})
        # write results to summary file
        writer.add_summary(summary, epoch)
        # print accuracy for each epoch
        print('epoch',epoch, acc)
        print ('---------------')
        print(y_pred, y_true)

    #saving model
    saver = tf.train.Saver()
    saver.save(sess,"/home/tharindra/PycharmProjects/WorkBench/FinalizedFinalProject/MusicClassification/saveGeetha.ckpt")
    print("Model saved")








