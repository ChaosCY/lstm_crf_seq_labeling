import tensorflow as tf

import argparse
import os
import codecs
import numpy as np

from my_utils import MyTextReader
from my_model import Model


def main():
	#设置参数
	parser = argparse.ArgumentParser(
						formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data_path', type=str, default='data',
						help='data directory containing input.txt')
	parser.add_argument('--batch_size', type=int, default=64,
						help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=200,
						help='RNN sequence length')
	parser.add_argument('--rnn_size', type=int, default=128,
						help='size of RNN hidden state')
	parser.add_argument('--learning_rate', type=float, default=0.01,
						help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=1,
						help='decay rate for Adam')
	parser.add_argument('--num_epochs', type=int, default=100,
						help='number of epochs')
	parser.add_argument('--is_crf', type=int, default=True,
						help='if add crf_layer')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='directory to store tensorboard logs')
	args = parser.parse_args()
	train(args)

def train(args):
	data_loader = MyTextReader(args.data_path, args.batch_size, args.seq_length)
	args.vocab_size = data_loader.vocab_size
	args.label_size = data_loader.label_size
	print("vocab_size: %4d" % data_loader.vocab_size)
	print("label_size: %4d" % data_loader.label_size)
	
	model = Model(args)
	
	with tf.Session() as sess:
		summaries = tf.summary.merge_all()
		writer = tf.summary.FileWriter(os.path.join(args.log_dir))
		writer.add_graph(sess.graph)
		
		sess.run(tf.global_variables_initializer())
		x_test, y_test, seq_len_test = data_loader.test_data()
		best_test_accuracy = 0.0
		for epoch in range(args.num_epochs):
			sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** epoch)))
			data_loader.reset_batch_pointer()
			loss_epoch = []
			accuracy_epoch = []
			for iteration in range(data_loader.num_batches):
				x, y, seq_len = data_loader.next_batch()
				feed = {model.input_data: x, model.targets: y, model.seq_len: seq_len}
				sess.run(tf.assign(model.keep_prob, 0.5))
				summ, _, loss, tf_predict_sequence, mask = sess.run([summaries, model.optimizer, model.loss, model.predict_sequence, model.mask], feed)
				
				if epoch%99 == 0:
					with codecs.open("epoch" + str(epoch), 'a') as f:
						for i in range(len(tf_predict_sequence)):
							f.write(str(y[i]))
							f.write("\n")
							f.write(str(tf_predict_sequence[i]))
							f.write("\n\n")
				
				total_labels = np.sum(seq_len)
				correct_labels = np.sum((y == tf_predict_sequence) * mask)
				accuracy = 100.0 * correct_labels / float(total_labels)
				print("epoches: %3d, iteration: %2d, train loss: %2.6f, train precision: %.6f" % (epoch, iteration, loss, accuracy))
				
				loss_epoch.append(loss)
				accuracy_epoch.append(accuracy)
				
			with codecs.open("model\\loss", 'a') as f:
				f.write(str(np.mean(loss_epoch)) + "\n")
			with codecs.open("model\\train_accuracy", 'a') as f:
				f.write(str(np.mean(accuracy_epoch)) + "\n")
			
			writer.add_summary(summ)
			feed = {model.input_data: x_test, model.targets: y_test, model.seq_len: seq_len_test}
			sess.run(tf.assign(model.keep_prob, 1.0))
			tf_predict_sequence_test, mask = sess.run([model.predict_sequence, model.mask], feed)
			total_labels = np.sum(seq_len_test)
			correct_labels = np.sum((y_test == tf_predict_sequence_test) * mask)
			test_accuracy = 100.0 * correct_labels / float(total_labels)
			if test_accuracy > best_test_accuracy:
				best_test_accuracy = test_accuracy
			print("test precision: %.6f, best test precision: %.6f" % (test_accuracy, best_test_accuracy))
			
			with codecs.open("model\\test_accuracy", 'a') as f:
				f.write(str(test_accuracy) + "\n")

if __name__ == '__main__':
	main()