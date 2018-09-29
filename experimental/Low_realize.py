import os
import gc
import sys
import math
import tempfile
import threading
import numpy as np
from scipy import stats
import tensorflow as tf
sys.path.append("../..")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Send_Server_Message import send_message

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def random_mini_batches(X, Y, Z, mini_batch_size=64, seed=None):
	# number of training examples
	m = Y.shape[0]
	mini_batches = []
	np.random.seed(seed)

	# Step 1 : Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation]
	shuffled_Y = Y[permutation, :]
	shuffled_Z = Z[permutation, :]

	# Step 2 : Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m / mini_batch_size)
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
		mini_batch_Z = shuffled_Z[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
		mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Z)
		mini_batches.append(mini_batch)

	# Step 3: Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
		mini_batch_Z = shuffled_Z[num_complete_minibatches * mini_batch_size : m, :]
		for index in range(mini_batch_size - m % mini_batch_size):
			mini_batch_X = np.append(mini_batch_X, shuffled_X[-1])
			mini_batch_Y = np.append(mini_batch_Y, shuffled_Y[-1, :])
			mini_batch_Z = np.append(mini_batch_Z, shuffled_Z[-1, :])
		mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[0], 1)
		mini_batch_Z = mini_batch_Z.reshape(mini_batch_Y.shape[0], -1)
		mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Z)
		mini_batches.append(mini_batch)

	return mini_batches

def data_loader(root_path, metafile_path, top=True, context=None):
	if top:
		lines = open(metafile_path, 'r').readlines()
	else:
		lines = context

	image_path = [] 
	label = []

	for line in lines:
		picture_path, only_label = line.split()
		image_path.append(os.path.abspath(os.path.join(root_path, picture_path)))
		label.append(float(only_label))

	return np.array(image_path), np.array(label, dtype=np.float32)

def mini_batch_read_picture(picture_path):
	pictures = []

	with tf.variable_scope("pre_process"):
		for index in range(picture_path.shape[0]):
			image_raw_data = tf.gfile.FastGFile(picture_path[index],'rb').read()
			image_data = tf.image.decode_jpeg(image_raw_data, channels=3)
			image_data = tf.random_crop(image_data, [224, 224, 3])
			image_data = tf.image.random_flip_left_right(image_data)
			image_data = tf.image.per_image_standardization(image_data)
			image_data = tf.expand_dims(image_data, 0)
			pictures.append(image_data)

		pictures = tf.concat(pictures, 0).eval()

	return pictures
		
def label_to_discribrate(label_x, StandardDeviation):
	label_discribrate = {}
	for label_mean in [ x for x in range(64) ]:
		p = stats.norm.pdf(label_x, loc=label_mean, scale=StandardDeviation)
		label_discribrate[label_mean] = p / p.sum()
	return label_discribrate

class ResNet(object):
	def __init__(self, model_save_path='./model_saving_DLDL/ResNet_Low'):
		self.model_save_path = model_save_path

	def identuty_block(self, x_input, kernel_size, in_filter, out_filters, stage, block, training):
		block_name = 'res' + str(stage) + block
		f1, f2, f3 = out_filters
		with tf.variable_scope(block_name):
			x_shortcut = x_input

			# first
			W_conv1 = self.weight_variable([1, 1, in_filter, f1])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_conv1))
			x = tf.nn.conv2d(x_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
			x = tf.layers.batch_normalization(x, axis=3, training=training)
			x = tf.nn.relu(x)

			# second
			W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_conv2))
			x = tf.nn.conv2d(x, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
			x = tf.layers.batch_normalization(x, axis=3, training=training)
			x = tf.nn.relu(x)

			#thrid
			W_conv3 = self.weight_variable([1, 1, f2, f3])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_conv3))
			x = tf.nn.conv2d(x, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
			x = tf.layers.batch_normalization(x, axis=3, training=training)

			add = tf.add(x, x_shortcut)
			add_result = tf.nn.relu(add)
		return add_result

	def convolutional_block(self, x_input, kernel_size, in_filter, out_filters, stage, block, training, stride=2):
		block_name = 'res' + str(stage) + block
		with tf.variable_scope(block_name):
			f1, f2, f3 = out_filters

			x_shortcut = x_input
			# frist
			W_conv1 = self.weight_variable([1, 1, in_filter, f1])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_conv1))
			x = tf.nn.conv2d(x_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
			x = tf.layers.batch_normalization(x, axis=3, training=training)

			# second
			W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_conv2))
			x = tf.nn.conv2d(x, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
			x = tf.layers.batch_normalization(x, axis=3, training=training)
			x = tf.nn.relu(x)

			# third
			W_conv3 = self.weight_variable([1, 1, f2, f3])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_conv3))
			x = tf.nn.conv2d(x, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
			x = tf.layers.batch_normalization(x, axis=3, training=training)

			# shortcut path
			W_shortcut = self.weight_variable([1, 1, in_filter, f3])
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(W_shortcut))
			x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

			#final
			add = tf.add(x_shortcut, x)
			add_result = tf.nn.relu(add)
		return add_result

	def deepnn(self, x_input, classes=1):

		with tf.variable_scope('reference'):
			training = tf.placeholder(tf.bool, name='training')

			# stage 1
			x = tf.layers.conv2d(x_input, filters=64, kernel_size=(7, 7),  strides=(2, 2), padding='VALID')
			x = tf.layers.batch_normalization(x, axis=3, training=training)
			x = tf.nn.relu(x)
			x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

			# stage 2
			x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
			x = self.identuty_block(x, 3, 256, [64, 64, 256], 2, 'b', training=training)
			x = self.identuty_block(x, 3, 256, [64, 64, 256], 2, 'c', training=training)

			# strage 3
			x = self.convolutional_block(x, 3, 256, [128, 128, 512], 3, 'a', training)
			x = self.identuty_block(x, 3, 512, [128, 128, 512], 3, 'b', training)
			x = self.identuty_block(x, 3, 512, [128, 128, 512], 3, 'c', training)
			x = self.identuty_block(x, 3, 512, [128, 128, 512], 3, 'd', training)

			# strage 4
			x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
			x = self.identuty_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training)
			x = self.identuty_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training)
			x = self.identuty_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training)
			x = self.identuty_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training)
			x = self.identuty_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training)

			# strage 5
			x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
			x = self.identuty_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training)
			x = self.identuty_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training)

			x = tf.nn.avg_pool(x, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

			flatten = tf.layers.flatten(x)
			with tf.name_scope('dropout'):
				keep_prob = tf.placeholder(tf.float32)
				flatten = tf.nn.dropout(flatten, keep_prob)

			logits = tf.layers.dense(flatten, units=64, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
			sorce = tf.layers.dense(flatten, units=1)

			sigmod = tf.nn.softmax(logits)

		return sigmod, sorce, keep_prob, training, flatten

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def train(self, X_train, Y_train, label_discribrate):
		mode = tf.placeholder(tf.bool, name='mode')
		pictures_name = tf.placeholder(tf.string, [None])
		labels = tf.placeholder(tf.float32, [None, 1])
		labels_p = tf.placeholder(tf.float32, [None, 64])
		labels_k = tf.constant([ [1.0 * x] for x in range(64) ]) 

		pictures = []
		for index in range(32):
			image_data = tf.image.decode_jpeg(tf.read_file(pictures_name[index]), channels=3)
			#-----------
			image_data = tf.image.random_brightness(image_data, max_delta=32./255)
			# image_data = tf.image.random_contrast(image_data, lower=0.5, upper=1.5)
			# image_data = tf.image.random_hue(image_data, max_delta=0.2)
			# image_data = tf.image.random_saturation(image_data, lower=0.5, upper=1.5)
			#-----------
			image_data = tf.random_crop(image_data, [224, 224, 3])
			image_data = tf.image.random_flip_left_right(image_data)
			image_data = tf.image.per_image_standardization(image_data)
			image_data = tf.expand_dims(image_data, 0)
			pictures.append(image_data)
		features = tf.concat(pictures, 0)

		
		sigmod, sorce, keep_prob, train_mode = self.deepnn(features)
		#----------------------------------------------------------------------
		# ld = tf.reduce_sum(tf.multiply(labels_p, tf.log(sigmod + 1e-10)), axis=1)
		ld = -tf.reduce_mean(tf.reduce_sum(tf.multiply(labels_p, tf.log(sigmod + 1e-10)), axis=1))
		tf.add_to_collection("losses", ld)
		#----------------------------------------------------------------------
		ld_show = tf.reduce_mean(ld)
		er = tf.abs(tf.matmul(sigmod, labels_k) - labels)
		er_show = tf.reduce_mean(er)

		#----------------------------------------------------------------------
		# seloss = tf.reduce_mean(0.000 * er - ld)
		seloss = tf.reduce_mean(tf.square(sorce - labels))
		#----------------------------------------------------------------------
		tf.summary.scalar('seloss', seloss)
		tf.add_to_collection("losses", seloss)

		l2_loss = tf.losses.get_regularization_loss()
		tf.summary.scalar('l2_loss', l2_loss)
		tf.add_to_collection("losses", l2_loss)

		loss = tf.add_n(tf.get_collection("losses"))

		mae = tf.reduce_mean( tf.abs(tf.matmul(sigmod, labels_k) - labels))
		mse = tf.sqrt(tf.reduce_mean(tf.square(tf.matmul(sigmod, labels_k) - labels)))

		global_step = tf.placeholder(tf.int32)
		tf.summary.scalar("global step", global_step)
		learning_rate = tf.train.exponential_decay(0.0001, global_step, decay_steps= 23564 / 32, decay_rate=0.999,staircase=True) #23564 / 32
		tf.summary.scalar("Learning Rate", learning_rate)
		with tf.name_scope('adam_optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

		graph_location = './model_saving_DLDL'
		print('Saving graph to : %s' % graph_location)

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(graph_location)
		train_writer.add_graph(tf.get_default_graph())

		saver = tf.train.Saver(max_to_keep=1000)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# saver.restore(sess, 'model_saving_DLDL/ResNet_Low-18000')

			mini_batches = random_mini_batches(X_train, Y_train, label_discribrate, mini_batch_size=32, seed=0)
			for index, (X_mini_batch, Y_mini_batch, Z_mini_batch) in enumerate(mini_batches):
				val_x, val_y, val_z = X_mini_batch, Y_mini_batch, Z_mini_batch
				break
			step = 0
			for i in range(1, 2000):
				mini_batches = random_mini_batches(X_train, Y_train, label_discribrate, mini_batch_size=32, seed=None)

				for index, (X_mini_batch, Y_mini_batch, Z_mini_batch) in enumerate(mini_batches):

					step = step + 1
					
					S, L, E, summary, _= sess.run([sigmod, ld, er, merged, train_step], feed_dict={pictures_name: X_mini_batch, labels: Y_mini_batch, labels_p: Z_mini_batch, keep_prob: 0.6, train_mode: True, mode:True, global_step:step})
					train_writer.add_summary(summary, step)

					if step % 200 == 0:
						train_ld, train_er, tarin_mae, tarin_mse, train_seloss, train_loss = sess.run([ld_show, er_show, mae, mse, seloss, loss], feed_dict={pictures_name: val_x, labels: val_y, labels_p: val_z, keep_prob: 1.0, train_mode: False, mode:True})
						print('Step: %d | ld: %g, er: %g, seloss: %g, loss: %g, train_mae: %g, tarin_mse: %g ' % (step, train_ld, train_er, train_seloss, train_loss, tarin_mae, tarin_mse))
					
					if not (step > 40000):
						if step % 10000 == 0:
							saver.save(sess, self.model_save_path, global_step=step)

					else:
						if step % 1000 == 0:
							saver.save(sess, self.model_save_path, global_step=step)

					if step > 150000:
						return
					gc.collect()

	def test(self, X_test, Y_test, label_discribrate):
		tf.reset_default_graph()
		mode = tf.placeholder(tf.bool, name='mode')
		pictures_name = tf.placeholder(tf.string, [None])
		labels = tf.placeholder(tf.float32, [None, 1])
		labels_p = tf.placeholder(tf.float32, [None, 64])
		labels_k = tf.constant([ [x * 1.0] for x in range(64) ]) 

		pictures = []
		for index in range(32):
			image_data = tf.image.decode_jpeg(tf.read_file(pictures_name[index]), channels=3)
			image_data = tf.image.resize_image_with_crop_or_pad(image_data, 224, 224)
			image_data = tf.image.per_image_standardization(image_data)
			image_data = tf.expand_dims(image_data, 0)
			pictures.append(image_data)
		features = tf.concat(pictures, 0)

		logits, keep_prob, train_mode = self.deepnn(features)
		mae = tf.reduce_mean(tf.abs(logits - labels))
		mse = tf.sqrt(tf.reduce_mean(tf.square(logits - labels)))

		sigmod, keep_prob, train_mode = self.deepnn(features)
		mae = tf.reduce_mean( tf.abs(tf.matmul(sigmod, labels_k) - labels))
		mse = tf.sqrt(tf.reduce_mean(tf.square(tf.matmul(sigmod, labels_k) - labels)))

		saver = tf.train.Saver()
		all_mae = 0
		all_mse = 0
		# with tf.Session(config=tf.ConfigProto(device_count={'gpu':-1})) as sess:
		with tf.Session() as sess:
			saver.restore(sess, 'model_saving_DLDL/ResNet_Low-10000')
			mini_batches = random_mini_batches(X_test, Y_test, label_discribrate, mini_batch_size=32, seed=0)
			for index, (X_mini_batch, Y_mini_batch, Z_mini_batch) in enumerate(mini_batches):
				tarin_mae, tarin_mse = sess.run([mae, mse], feed_dict={pictures_name: X_mini_batch, labels: Y_mini_batch, labels_p: Z_mini_batch, keep_prob: 1.0, train_mode: False, mode:True})
				all_mae = all_mae + tarin_mae
				all_mse = all_mse + tarin_mse
				if (index + 1) % 100 == 0:
					print('...........', (index+1), '/', len(mini_batches))
		all_mae = all_mae / (index+1)
		all_mse = all_mse / (index+1)
		print(all_mae, all_mse)

	def val(self, X_test, Y_test, Z_test, num):
		tf.reset_default_graph()
		mode = tf.placeholder(tf.bool, name='mode')
		pictures_name = tf.placeholder(tf.string, [None])
		labels = tf.placeholder(tf.float32, [None, 1])
		labels_p = tf.placeholder(tf.float32, [None, 64])
		labels_k = tf.constant([ [1.0 * x] for x in range(64) ])

		pictures = []
		for index in range(1):
			image_data = tf.image.decode_jpeg(tf.read_file(pictures_name[index]), channels=3)
			image_data = tf.image.resize_image_with_crop_or_pad(image_data, 224, 224)
			image_data = tf.image.per_image_standardization(image_data)
			image_data = tf.expand_dims(image_data, 0)
			pictures.append(image_data)
		features = tf.concat(pictures, 0)

		sigmod, sorce, keep_prob, train_mode, feature  = self.deepnn(features)
		KL = tf.reduce_mean(tf.reduce_sum(tf.multiply(labels_p, tf.log(sigmod + 1e-10)), axis=1))

		train_people_pre_label = (tf.matmul(sigmod, labels_k) + sorce) / 2

		# train_label = tf.reduce_mean( tf.matmul(sigmod, labels_k))
		train_label_s = tf.reduce_mean(sorce)

		# train_label = (train_label + train_label_s) / 2
		train_label = tf.reduce_mean(train_people_pre_label)

		pro = tf.reduce_mean(sigmod, axis=0)

		saver = tf.train.Saver()

		pople_pre_picture = []
		pople_pre_label = []
		pople_pre_feature = []
		pople_pre_distribution = []

		avg_label = []
		avg_label_ = []
		pro_disc = []
		with tf.Session() as sess:
			saver.restore(sess, 'model_saving_DLDL/ResNet_Low-51000')
			step = 0
			LABEL = 0
			for index in range(len(num) - 1):
				X = X_test[num[index] : num[index + 1]]
				Y = Y_test[num[index] : num[index + 1], :]
				Z = Z_test[num[index] : num[index + 1], :]

				avla = 0
				avlas = 0
				avpro = [0 for n in range(64)]
				poprlabel = []
				poprfea = []
				poprdis = []

				for index_index in range(len(X)):
					Y_ = Y[index_index, :].reshape(-1, 1)
					Z_ = Z[index_index, :].reshape(-1, 64)
					train_feature, train_kl, train_pro, train_label_, train_label_ss= sess.run([feature, KL, pro, train_label, train_label_s], feed_dict={pictures_name: [X[index_index]], labels: Y_, labels_p: Z_, keep_prob: 1.0, train_mode: False, mode:True})
					avla = avla + train_label_
					avlas = avlas + train_label_ss
					avpro = avpro + train_pro
					poprlabel.append(train_label_)
					poprfea.append(train_feature)
					poprdis.append(train_pro)

				if (index + 1) % 10 == 0:
					print(index+1, '|', len(num)-1)
				avla = avla / len(X)
				avlas = avlas / len(X)
				avpro = avpro / len(X)

				pople_pre_picture.append(X)
				pople_pre_label.append(poprlabel)
				pople_pre_feature.append(poprfea)
				pople_pre_distribution.append(poprdis)

				avg_label.append(avla)
				avg_label_.append(avlas)
				pro_disc.append(train_pro)

		return avg_label, avg_label_, pro_disc, pople_pre_picture, pople_pre_label, pople_pre_feature, pople_pre_distribution

def anaysis(people_path, picture_path):

	people_lines = open(people_path, 'r').readlines()

	people = []
	people_label = []
	for people_line in people_lines:
		only_people = people_line.split()[0]
		only_people_label = float(people_line.split()[-1])
		people.append(only_people)
		people_label.append(only_people_label)

	people_num = [0] * len(people)
	picture_lines = open(picture_path, 'r').readlines()
	for picture_line in picture_lines:
		only_people = picture_line.split('\\')[1]
		people_num[people.index(only_people)] = people_num[people.index(only_people)] + 1

	return people, people_label, people_num

def main():
	Root_path = "D://deeplearning//Dataset//Depression//AVEC2014//AVEC2014"
	Train_file = "D://deeplearning//Dataset//Depression//AVEC2014//AVEC2014//pp_trn_0_img.txt"
	Test_file = "D://deeplearning//Dataset//Depression//AVEC2014//AVEC2014//pp_tst_img.txt"
	Val_file = "D://deeplearning//Dataset//Depression//AVEC2014//AVEC2014//pp_tst.txt"

	Image_path, label = data_loader(Root_path, Train_file)
	TsT_Image_path, TsT_label = data_loader(Root_path, Test_file)

	label_x = np.arange(0, 64, 1)
	label_discribrate = label_to_discribrate(label_x, 2.0)
	label = label.reshape(label.shape[0], 1)
	discribrate = np.array([label_discribrate[x[0]] for x in label])

	TsT_label = TsT_label.reshape(TsT_label.shape[0], 1)
	TsT_discribrate = np.array([label_discribrate[x[0]] for x in TsT_label])

	model = ResNet()
	# model.train(Image_path, label, discribrate)
	# model.test(Image_path, label, discribrate)
	# model.test(TsT_Image_path, TsT_label, TsT_discribrate)
	# model.val(Val_Image_path, Val_label, Val_discribrate)

	#-----------------VAL--------------------#
	people, people_label, people_num = anaysis("D://deeplearning//Dataset//Depression//AVEC2014//AVEC2014//pp_tst.txt",
											   "D://deeplearning//Dataset//Depression//AVEC2014//AVEC2014//pp_tst_img.txt")

	people_discribrate = np.array([label_discribrate[x] for x in people_label])

	for index in range(1, len(people_num)):
		people_num[index] = people_num[index] + people_num[index - 1]
	people_num.insert(0, 0)
	pre_label, pre_label_, pro_disc, pople_pre_picture, pople_pre_label, pople_pre_feature, pople_pre_distribution = model.val(TsT_Image_path, TsT_label, TsT_discribrate, people_num)
	pre_mae = 0
	pre_mse = 0
	pre_mae_ = 0
	pre_mse_ = 0

	for index in range(len(pre_label)):
		pre_mae = pre_mae + abs(pre_label[index] - people_label[index])
		pre_mse = pre_mse + pow((pre_label[index] - people_label[index]), 2)
		pre_mae_ = pre_mae_ + abs(pre_label_[index] - people_label[index])
		pre_mse_ = pre_mse_ + pow((pre_label_[index] - people_label[index]), 2)
	pre_mae = pre_mae / len(pre_label)
	pre_mse = math.sqrt(pre_mse / len(pre_label))

	pre_mae_ = pre_mae_ / len(pre_label)
	pre_mse_ = math.sqrt(pre_mse_ / len(pre_label))

	print('mae:', pre_mae, '|', 'mse:', pre_mse)
	print('mae_:', pre_mae_, '|', 'mse_:', pre_mse_)

	#--#---------------Creat CSV picture truth forward 2014_feature -------------------#--#
	# import pandas as pd
	# csv_list = []
	# for index in range(len(people)):
	# 	for n in range(len(pople_pre_feature[index])):
	# 		info = [pople_pre_picture[index][n], people_label[index], pople_pre_label[index][n], pople_pre_distribution[index][n].tolist(), pople_pre_feature[index][n].tolist()]
	# 		csv_list.append(info)
	# name = ['picture_path', 'truth_label', 'forward_label', 'distribution', 'feature']
	# csv_file = pd.DataFrame(columns=name,data=csv_list)
	# csv_file.to_csv('./51000_test_info.csv',encoding='gbk')
	#--#---------------------------------end-------------------------------------------#--#

	# #--#-----Pyecharts--Label--Bar---------------#
	# from pyecharts import Bar, Grid
	# np_people_label = np.array(people_label)
	# np_people = np.array(people)
	# label_sort = np.argsort(np_people_label)
	# np_people_label = np_people_label[label_sort].tolist()

	# attr = np_people_label
	# grid = Grid(height=500, width=2000)
	# bar = Bar('label--different')
	# v = [ (pre_label[index] - people_label[index]) for index in label_sort ]
	# bar.add("label different", attr, v, mark_line=["average"], mark_point=["max", "min"], legend_top='5%')
	# grid.add(bar, grid_top= "10%", grid_bottom= "10%")
	# grid.render("model_saving_DLDL/train_label_30000.html")
	# #--#----------------END----------------------#

	# #--#----Pyecharts--People--Bar--------------#
	# from pyecharts import Bar, Grid
	# attr = [people[index] for index in range(len(people))]
	# grid = Grid(height=1000, width=500)
	# bar = Bar('people--label--different')
	# v = [ abs(pre_label[index] - people_label[index]) for index in range(len(people))]
	# bar.add("label differebt", attr, v, mark_line=["average"], is_convert=True, legend_top='5%')
	# bar.add("label forward", attr, [pre_label[index] for index in range(len(people))], is_convert=True, legend_top= "5%")
	# grid.add(bar, grid_top= "10%", grid_bottom= "10%")
	# grid.render("model_saving_DLDL/train_30000.html")
	# #--#--------------END--------------------#

	# # #--#----Pyecharts--People--Pre--Bar---------#
	# will_look_people = ['325_2_Freeform_video', '325_2_Northwind_video', '246_2_Freeform_video', '246_2_Northwind_video', '315_2_Freeform_video', '315_2_Northwind_video']
	# will_look_index = []
	# for look_people in will_look_people:
	# 	for n, only_people in enumerate(people): 
	# 		if look_people == only_people:
	# 			will_look_index.append(n)
	
	# from pyecharts import Bar, Grid
	# for num, num_index in enumerate(will_look_index):
	# 	attr = pople_pre_picture[num_index]
	# 	grid = Grid(height=500, width=1500)
	# 	bar = Bar(will_look_people[num] + '--pre--forward--label')
	# 	v = pople_pre_label[num_index]
	# 	print(sum(v) / len(v), pre_label[num_index])
	# 	bar.add("forward label", attr, v, mark_line=["average"], mark_point=["max", "min"], legend_top='5%')
	# 	grid.add(bar, grid_top= "10%", grid_bottom= "10%")
	# 	grid.render("model_saving_DLDL/" + will_look_people[num] + "_test_30000.html")


	# #--#--------------END--------------------#

	# #--#-----------Pyecharts-----------------#
	# from pyecharts import Line, Grid

	# attr = [index for index in range(64)]
	# grid = Grid(height=20000, width=1500)

	# for index in range(len(people)):
	# 	line = Line(people[index], title_top=str(index + 0.1) + "%")
	# 	v1 = pro_disc[index]
	# 	v2 = people_discribrate[index]
	# 	line.add("TRAIN_PRO_" + str(index) , attr, v1, is_smooth=True, legend_top=str(index + 0.1) + "%")
	# 	line.add("TRUTH_PRO_" + str(index) , attr, v2, is_smooth=True, legend_top=str(index + 0.1) + "%", mark_point=["max"])
	# 	grid.add(line, grid_top= str(index + 0.3) + "%", grid_bottom= str(100.3 - (index +1) )+ "%")

	# grid.render("model_saving_DLDL/para_2/test_58000.html")

	# #--#--------------END--------------------#
	#-----------------END--------------------#


if __name__ == '__main__':
	main()