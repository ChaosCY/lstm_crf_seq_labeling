import tensorflow as tf
import numpy as np

class Model():
	def __init__(self, args):
		self.args = args
		
		self.input_data = tf.placeholder(tf.int32, [None, None])
		self.targets = tf.placeholder(tf.int32, [None, None])
		self.seq_len = tf.placeholder(tf.int32, [None])
		onehot_targets = tf.one_hot(self.targets, args.label_size)
		
		lstm_fw = tf.contrib.rnn.BasicLSTMCell(args.rnn_size)
		lstm_bw = tf.contrib.rnn.BasicLSTMCell(args.rnn_size)
		
		self.keep_prob = tf.Variable(0.0, trainable=False)
		cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
		cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
		
		with tf.name_scope('weights'):
			softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.label_size], initializer=tf.random_normal_initializer())
			tf.summary.histogram('softmax_w', softmax_w)
		with tf.name_scope('bias'):
			softmax_b = tf.get_variable("softmax_b", [args.label_size], initializer=tf.constant_initializer(0.0))
			tf.summary.histogram('softmax_b', softmax_w)
		
		embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)
		
		output, _ = tf.nn.bidirectional_dynamic_rnn(
			cell_fw,
			cell_bw,
			inputs,
			dtype=tf.float32
			#sequence_length=args.seq_length
		)
		#output, state=tf.nn.dynamic_rnn(lstm_fw, inputs, sequence_length=self.seq_len, dtype=tf.float32)
		output = tf.reduce_mean(output, 0)
		
		output = tf.reshape(output, [-1, args.rnn_size])
		y_ = tf.reshape(onehot_targets, [-1, args.label_size])
		
		logits = tf.matmul(output, softmax_w) + softmax_b
		y = tf.nn.softmax(logits)
		
			
		self.mask = tf.sequence_mask(self.seq_len)
		total_labels = tf.reduce_sum(self.seq_len)
		
		#crf
		if args.is_crf:
			y_scores = tf.reshape(logits, [-1, tf.reduce_max(self.seq_len), args.label_size])
			log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_scores, self.targets, self.seq_len)
			self.loss = tf.reduce_mean(-log_likelihood)
			
			self.predict_sequence, _ = tf.contrib.crf.crf_decode(y_scores, transition_params, self.seq_len)
		else:
			#这里输入应该是logits
			cost = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
			cost = tf.reshape(cost, [-1, tf.reduce_max(self.seq_len)])
			self.loss = tf.reduce_mean(tf.reduce_sum(cost, 1))
			
			y_scores = tf.reshape(y, [-1, tf.reduce_max(self.seq_len), args.label_size])
			self.predict_sequence = tf.argmax(y_scores, axis=2)
		
		with tf.name_scope('loss'):
			self.loss = tf.reduce_sum(self.loss)
			tf.summary.scalar('loss', self.loss)
			
		self.lr = tf.Variable(0.0, trainable=False)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		
	
	def init_embeddings(vocabulary_size, embedding_size):
		return tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		
