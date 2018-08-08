# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
#from ops import * #what does this do anyway?
import numpy as np

class ConvoAutoencoder(object):
#Convolutional variational autoencoder

  def __init__(self, latent_dim = 10, batch_size = 50):
    # apparently something wrong with the indentation here . . . 
    # should I load the pretrained variables here? Or outside of this document/module?
    #VARIABLES
    self.latent_dim = latent_dim
    self.batch_size = batch_size

    #PLACEHOLDERS
    # input images placeholder
    self.input_images = tf.placeholder(tf.float32, [None, 64, 64, 3]) 

    #ENCODER
    # first convolutional layer 64x64x3 -> 32x32x16
    self.w = tf.get_default_graph().get_tensor_by_name('conv1/w:0')
    self.b = tf.get_default_graph().get_tensor_by_name('conv1/b:0')

    self.conv_1 = tf.nn.conv2d(self.input_images, self.w, strides = [1,2,2,1], padding = 'SAME') + self.b

    # ReLu
    self.conv_1_r = tf.nn.relu(self.conv_1)

    # second convolutional layer 32x32x16 -> 16x16x32
    self.w2 = tf.get_default_graph().get_tensor_by_name('conv2/w:0')
    self.b2 = tf.get_default_graph().get_tensor_by_name('conv2/b:0')

    self.conv_2 = tf.nn.conv2d(self.conv_1_r, self.w2, strides = [1,2,2,1], padding = 'SAME') + self.b2

    # ReLu
    self.conv_2_r = tf.nn.relu(self.conv_2)

    # Reshape (flatten)
    self.conv_2_r_flat = tf.reshape(self.conv_2_r, [-1, 16*16*32])

    # fully connected layer
    # w for z_mean
    self.w_mean = tf.get_default_graph().get_tensor_by_name('z_mean_dense/w:0')
    # b for z_mean
    self.b_mean = tf.get_default_graph().get_tensor_by_name('z_mean_dense/b:0')
    # w for z_logstd
    self.w_logstd = tf.get_default_graph().get_tensor_by_name('z_stddev_dense/w:0')
    # b for z_logstd
    self.b_logstd = tf.get_default_graph().get_tensor_by_name('z_stddev_dense/b:0')

    self.z_mean = tf.matmul(self.conv_2_r_flat, self.w_mean) + self.b_mean
    self.z_logstd = tf.matmul(self.conv_2_r_flat, self.w_logstd) + self.b_logstd


    #DECODER
    # calculate z, the decoder input
    self.samples = tf.random_normal([batch_size, latent_dim], 0, 1, dtype = tf.float32)
    self.z = self.z_mean + (tf.exp(.5*self.z_logstd) * self.samples) 
    # I don't quite understand why we need to generate samples

    # fully connected layer
    # w_fc
    self.w_fc = tf.get_default_graph().get_tensor_by_name('z_fc_dense/w:0')
    # b_fc
    self.b_fc = tf.get_default_graph().get_tensor_by_name('z_fc_dense/b:0')

    self.z_fc = tf.matmul(self.z, self.w_fc) + self.b_fc

    # Reshape (unflatten) and relu
    self.z_matrix = tf.nn.relu(tf.reshape(self.z_fc, [-1, 16, 16, 32]))

    # first deconvolutional layer 16x16x32 -> 32x32x16
    # w_deconv
    self.w_deconv = tf.get_default_graph().get_tensor_by_name('deconv1/w:0')
    # b_deconv
    self.b_deconv = tf.get_default_graph().get_tensor_by_name('deconv1/b:0')

    self.deconv_1 = tf.nn.conv2d_transpose(self.z_matrix, self.w_deconv, output_shape = [batch_size, 32, 32, 16], strides = [1,2,2,1]) + self.b_deconv

    # ReLu
    self.deconv_1_r = tf.nn.relu(self.deconv_1)

    # second deconvolutional layer 32x32x16 -> 64x64x3
    # w_deconv2
    self.w_deconv2 = tf.get_default_graph().get_tensor_by_name('deconv2/w:0')
    # b_deconv2
    self.b_deconv2 = tf.get_default_graph().get_tensor_by_name('deconv2/b:0')

    self.deconv_2 = tf.nn.conv2d_transpose(self.deconv_1_r, self.w_deconv2, output_shape = [batch_size, 64, 64, 3], strides = [1,2,2,1]) + self.b_deconv2

    # apply activation to outputs
    self.output_images = tf.nn.sigmoid(self.deconv_2)

    # clean up outputs for plotting
    #outputs_clean = self.output_images * 255
    #outputs_clean = self.outputs_clean.astype(np.uint8)


  def train_step(self, input_image_batch, sess, learning_rate = 1e-3):
    # training info
    # loss function
    # generated_images variable --> basically the outputs before sigmoid
    generated_images = self.deconv_2
    # the flattened generated images
    generated_images_flat = tf.reshape(generated_images, [-1, 64*64*3])
    # the flattened input images
    input_images_flat = tf.reshape(self.input_images, [-1, 64*64*3])

    # the actual loss function calculation
    generation_loss = tf.reduce_sum(tf.maximum(generated_images_flat, 0) - generated_images_flat * input_images_flat\
                             + tf.log(1 + tf.exp(-tf.abs(generated_images_flat))), 1)

    latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.exp(2*self.z_logstd) - 2*self.z_logstd - 1, 1)

    loss = tf.reduce_mean(generation_loss + latent_loss)

    # setting up the optimizer, which is an AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # sess.run on the optimizer
    return sess.run(optimizer, feed_dict={self.input_images: input_image_batch})

  def return_encoding(self, sess, input_frame):
    # sess.run on self.z_mean?
    return sess.run(self.z_mean, feed_dict={self.input_images: input_frame})

  def reconstruct_image(self, sess, input_image):
    # sess.run on self.output_images
    out = sess.run(self.output_images, feed_dict={self.input_images: input_image})
    # clean up the output images for plotting
    out = out * 255
    out = out.astype(np.uint8)
    return out

