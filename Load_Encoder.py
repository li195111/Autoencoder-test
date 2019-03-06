import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
# dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

model_dir = os.path.join(os.getcwd(), "model")
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
	pass

n_pixels = 28*28
latent_dim = 20
h_dim = 500

X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

def weight_variables(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def FC_layer(x, w, b):
	return tf.matmul(x, w) + b

# Encoder -----------------------------------------------------------------------------------------
# layer 1
W_enc = weight_variables([n_pixels, h_dim], 'W_enc')
b_enc = bias_variable([h_dim], 'b_enc')
# tanh activation
h_enc = tf.nn.tanh(FC_layer(X, W_enc, b_enc))

# layer 2
W_mu = weight_variables([h_dim, latent_dim], 'W_mu')
b_mu = bias_variable([latent_dim], 'b_mu')
mu = FC_layer(h_enc, W_mu, b_mu) # mean

# standard deviation
W_logstd = weight_variables([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = FC_layer(h_enc, W_logstd, b_logstd) # std

# RANDOMNESSSSSSSSSSSSSssss
noise = tf.random_normal([1, latent_dim])

# z is the ultimate output of our encoder
z = mu + tf.multiply(noise, tf.exp(.5 * logstd))
# Encoder -----------------------------------------------------------------------------------------

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
saver.restore(sess, os.path.join(model_dir, "autoencoder_model.ckpt"))

num_pair = 10
image_indices = np.random.randint(0, 200, num_pair)
for pair in range(num_pair):
	x = np.reshape(mnist.test.images[image_indices[pair]], (1, n_pixels))
	x_image = np.reshape(x, (28, 28))
	plt.figure()
	plt.subplot(111)
	plt.imshow(x_image)
	x_logstd_vector = logstd.eval(feed_dict= {X:x})

	print (x_logstd_vector)
#plt.show()