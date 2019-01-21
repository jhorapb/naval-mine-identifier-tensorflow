
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_dataset():
    """
    Read the dataset
    """
    data_frame = pd.read_csv('../sonar_data/sonar.all-data.csv')
    X_read = data_frame[data_frame.columns[0:60]].values
    y_base = data_frame[data_frame.columns[60]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y_base)
    y = encoder.transform(y_base)
    Y_read = one_hot_encoder(y)
    # print(X_read.shape)
    return X_read, Y_read, y_base


def one_hot_encoder(labels):
    """
    Define the encoder function
    """
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Read the dataset
X, Y, y_original = read_dataset()

# Shuffle the dataset to mix up the rows (the current dataset is presented in order)
X, Y = shuffle(X, Y, random_state=1)

# Split the dataset into training and testing parts
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# Inspect the shape of the training and testing data
print(train_x.shape)
print(train_y.shape)
print('testx', test_x)
print('testy', test_y)

# Define the important parameters and variable to work with the tensors
learning_rate = 0.03
training_epochs = 400
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print('n_dim', n_dim)
num_classes = 2
model_path = '../models/nmi'

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y_ = tf.placeholder(tf.float32, [None, num_classes])


def multilayer_perceptron(x, weights, biases):
    """
    Define the model
    """
    # Hidden layer with RELU activations
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden layer with sigmoid activations
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with sigmoid activations
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden layer with RELU activations
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out'] + biases['out'])

    return out_layer


# Define the weights and biases for each layer
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, num_classes])),
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([num_classes])),
}

# Initialize all the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Call the model defined
y = multilayer_perceptron(x, weights, biases)

# Define cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# Restore model
saver.restore(sess, model_path)

prediction = tf.argmax(y, 1)
# Print the final accuracy
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('+' * 50)
print('Mine (M) -> 0, Rock (R) -> 1')
print('+' * 50)
# print(X)
print('------------')
# print(Y)
print('............')
for i in range(93, 101):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 60)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 60), y_: Y[i].reshape(1, 2)})
    print('Original label/class:', y_original[i], ' Predicted value:', prediction_run[0], ' Accuracy:', accuracy_run)