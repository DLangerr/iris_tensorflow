import tensorflow as tf
import numpy as np

from preprocess_data import get_data



X, T, Y, N, D, K, M, means, stds = get_data()


def print_type(num, i):
    if num[i] == 0:
        return "Setosa"
    elif num[i] == 1:
        return "Versicolor"
    elif num[i] == 2:
        return "Virginica"
    

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


def forward(X, W1, b1, W2, b2, W3, b3):
    Z = tf.nn.tanh(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.tanh(tf.matmul(Z, W2) + b2)
    return tf.matmul(Z2, W3) + b3

M2 = M


W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, M2])
b2 = init_weights([M2])
W3 = init_weights([M2, K])
b3 = init_weights([K])

phX = tf.placeholder(tf.float32, [None, D])
phY = tf.placeholder(tf.float32, [None, K])

Y_ = forward(phX, W1, b1, W2 ,b2, W3, b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=phY, logits=Y_))
lr = 0.01

train = tf.train.AdamOptimizer(lr).minimize(cost)
predictions = tf.argmax(Y_, 1)

init = tf.global_variables_initializer()

steps = 1000


sess = tf.Session()
sess.run(init)

for i in range(steps):
    sess.run(train, feed_dict={phX: X, phY: T})
    pred = sess.run(predictions, feed_dict={phX: X})
    if i % 100 == 0:
        acc = np.mean(pred == Y)
        print(f"Accuracy: {acc}")
        
# Test predictions        
test_input = np.array([[5.3, 4.8, 1.1, 0.3], [5.0, 3.0, 4.1, 1.5], [7.7, 2.8, 6.7, 2]])
print()
print("Trying to predict:")
print("==============================")
print()
print(test_input)
test_input = (test_input - means) / stds
pred2 = sess.run(predictions, feed_dict={phX: test_input})
print()
print("Prediction:")
print("==============================")
print()
for i in range(len(pred2)):
    flower = print_type(pred2, i)
    print(flower)
                