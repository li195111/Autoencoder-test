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

n_pixels = 28*28
# input the images (gateway)
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))


# Encoder -----------------------------------------------------------------------------------------
def weight_variables(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def FC_layer(x, w, b):
	return tf.matmul(x, w) + b

latent_dim = 20
h_dim = 500

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

# Decoder -----------------------------------------------------------------------------------------

# layer 1
W_dec = weight_variables([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')
h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))


# layer 2
W_reconstruct = weight_variables([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')

reconstruction = tf.nn.sigmoid(FC_layer(h_dec, W_reconstruct, b_reconstruct))

# Decoder -----------------------------------------------------------------------------------------

# Loss Function

log_likelihood = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9), reduction_indices= 1)

# KL Divergence
KL_tern = -.5 * tf.reduce_sum(1 + 2 * logstd - tf.pow(mu, 2) - tf.exp(2 * logstd), reduction_indices= 1)

variational_lower_bound = tf.reduce_mean(log_likelihood - KL_tern)
optimizer = tf.train.AdadeltaOptimizer().minimize( - variational_lower_bound)

# Training
saver = tf.train.Saver()
init = tf.global_variables_initializer()
config = tf.ConfigProto(log_device_placement=True )
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)
sess.run(init)

model_dir = os.path.join(os.getcwd(), "model")
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
	pass

update = True

if update:
	saver.restore(sess, os.path.join(model_dir, "autoencoder_model.ckpt"))

import time
num_iterations = 100000
recording_interval = 1000
variational_lower_bound_array = []
log_likelihood_array = []
KL_tern_array = []
interation_array = [i * recording_interval for i in range(int(num_iterations / recording_interval))]
for i in range(num_iterations):
	x_batch = np.round(mnist.train.next_batch(200)[0])
	sess.run(optimizer, feed_dict = {X: x_batch})
	if (i % recording_interval == 0):
		vlb_eval = variational_lower_bound.eval(feed_dict = {X: x_batch})
		print ("Iteration : {}, Loss : {}".format(i, vlb_eval))
		variational_lower_bound_array.append(vlb_eval)
		log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict = {X: x_batch})))
		KL_tern_array.append(np.mean(KL_tern.eval(feed_dict = {X: x_batch})))
plt.figure()
plt.plot(interation_array, variational_lower_bound_array)
plt.plot(interation_array, KL_tern_array)
plt.plot(interation_array, log_likelihood_array)
plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
plt.title('Loss per interation')


saver.save(sess, os.path.join(model_dir, "autoencoder_model.ckpt"))

if update:
	files = []
	for file in os.listdir(model_dir):
		files.append(file.split('.')[0])
	times_str = [file.replace('autoencoder_model', '') for file in files if 'autoencoder_model' in file]
	times_list = []
	for time_str in times_str:
		if time_str != '':
			times = int(time_str)
			times_list.append(times)
		else:
			times = 0
			times_list.append(times)
	max_times = max(times_list)
	max_times += num_iterations
	saver.save(sess, os.path.join(model_dir, "autoencoder_model" + str(max_times) + ".ckpt"))

num_pair = 10
image_indices = np.random.randint(0, 200, num_pair)
for pair in range(num_pair):
	x = np.reshape(mnist.test.images[image_indices[pair]], (1, n_pixels))
	plt.figure()
	x_image = np.reshape(x, (28, 28))
	plt.subplot(121)
	plt.imshow(x_image)

	x_reconstruction = reconstruction.eval(feed_dict= {X:x})

	x_reconstruction_image = (np.reshape(x_reconstruction, (28, 28)))

	plt.subplot(122)
	plt.imshow(x_reconstruction_image)
plt.show()