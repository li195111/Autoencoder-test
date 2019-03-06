import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

model_dir = os.path.join(os.getcwd(), "model")
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

n_pixels = 28*28
latent_dim = 20
h_dim = 500

Z = tf.placeholder(tf.float32, shape=([None, latent_dim]))

def weight_variables(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def FC_layer(x, w, b):
	return tf.matmul(x, w) + b

# Decoder -----------------------------------------------------------------------------------------

# layer 1
W_dec = weight_variables([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')
h_dec = tf.nn.tanh(FC_layer(Z, W_dec, b_dec))


# layer 2
W_reconstruct = weight_variables([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')

reconstruction = tf.nn.sigmoid(FC_layer(h_dec, W_reconstruct, b_reconstruct))

# Decoder -----------------------------------------------------------------------------------------

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
saver.restore(sess, os.path.join(model_dir, "autoencoder_model.ckpt"))

num_pair = 25
plt.figure(figsize = (10, 10))
for pair in range(num_pair):
	plt.subplot(int(np.sqrt(num_pair)), int(np.sqrt(num_pair)), pair + 1)
	#z = np.random.standard_normal(size=20)
	#z = np.reshape(z, (1, 20))
	
	#i = (pair - int(num_pair/2)) / 10
	i = np.random.standard_normal()
	z = np.array([[i,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

	#z[0,pair // num_pair] = i

	print (z)
	
	x_reconstruction = reconstruction.eval(feed_dict= {Z:z})
	x_reconstruction_image = (np.reshape(x_reconstruction, (28, 28)))
	plt.imshow(x_reconstruction_image, cmap=plt.cm.binary)
plt.show()