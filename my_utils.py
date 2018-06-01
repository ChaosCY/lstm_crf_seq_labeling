import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import math

class MyTextReader():
	def __init__(self, data_path, batch_size, seq_length, encoding='utf-8'):
		self.data_path = data_path
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding
		
		# 获取输入及输出文件路径
		input_X = os.path.join(data_path, "trainX.in")
		input_Y = os.path.join(data_path, "trainY.in")
		test_X = os.path.join(data_path, "testX.in")
		test_Y = os.path.join(data_path, "testY.in")
		vocab_file = os.path.join(data_path, "vocab.pkl")
		label_file = os.path.join(data_path, "label.pkl")
		tensor_X_file = os.path.join(data_path, "tensorX.npy")
		tensor_Y_file = os.path.join(data_path, "tensorY.npy")
		
		# 读取数据及预处理
		if not (os.path.exists(vocab_file) and os.path.exists(tensor_X_file) and os.path.exists(label_file) and os.path.exists(tensor_Y_file)):
			self.preprocess(input_X, input_Y, test_X, test_Y, vocab_file, label_file, tensor_X_file, tensor_Y_file)
		else:
			self.load_proprecess(vocab_file, label_file, tensor_X_file, tensor_Y_file)
		
		self.create_batches()
		self.create_testdata(test_X, test_Y)
		self.reset_batch_pointer()
		
	def preprocess(self, input_X_file, input_Y_file, test_X_file, test_Y_file, vocab_file, label_file, tensor_X_file, tensor_Y_file):
		with codecs.open(input_X_file, "r", encoding=self.encoding) as f:
			data = f.read()
		with codecs.open(test_X_file, "r", encoding=self.encoding) as f:
			data1 = f.read()
		# 对训练文件及测试文件统一处理，待修正
		data = data + data1
		# 删除\r\n\t字符
		data = re.sub('[\r\n\t]', '', data)
		counter = collections.Counter(data)
		count_pairs = sorted(counter.items(), key=lambda x: x[1])
		self.chars, a = zip(*count_pairs)
		self.vocab_size = len(self.chars)
		# 字典
		self.vocab = dict(zip(self.chars, range(len(self.chars))))
		with open(vocab_file, "wb") as f:
			cPickle.dump(self.chars, f)
		self.tensor_X = []
		with codecs.open(input_X_file, "r", encoding=self.encoding) as f:
			for line in f.readlines():
				line = re.sub('[\r\n\t]', '', line)
				self.tensor_X.append(list(map(self.vocab.get, line)))
		self.tensor_X = np.array(self.tensor_X)
		np.save(tensor_X_file, self.tensor_X)
		
		with codecs.open(input_Y_file, "r", encoding=self.encoding) as f:
			data = f.read()
		data = re.sub('[\r\n\t]', '', data)
		data = data.split(' ')
		counter = collections.Counter(data)
		count_pairs = sorted(counter.items(), key=lambda x: x[1])
		self.chars, _ = zip(*count_pairs)
		print(self.chars)
		self.label_size = len(self.chars)
		self.label = dict(zip(self.chars, range(len(self.chars))))
		with open(label_file, "wb") as f:
			cPickle.dump(self.chars, f)
		self.tensor_Y = []
		with codecs.open(input_Y_file, "r", encoding=self.encoding) as f:
			for line in f.readlines():
				line = re.sub('[\r\n\t]', '', line)
				line = line.split(' ')[:-1]
				self.tensor_Y.append(list(map(self.label.get, line)))
		print(np.array(self.tensor_Y).shape)
		self.tensor_Y = np.array(self.tensor_Y)
		#print(self.tensor_Y)
		np.save(tensor_Y_file, self.tensor_Y)
	
	def load_proprecess(self, vocab_file, label_file, tensor_X_file, tensor_Y_file):
		# 直接从文件读取
		with open(vocab_file, 'rb') as f:
			self.chars = cPickle.load(f);
		self.vocab_size = len(self.chars)
		self.vocab = dict(zip(self.chars, range(len(self.chars))))
		self.tensor_X = np.load(tensor_X_file)
		print(self.tensor_X.shape)
		
		with open(label_file, 'rb') as f:
			self.chars = cPickle.load(f);
		print(self.chars)
		self.label_size = len(self.chars)
		self.label = dict(zip(self.chars, range(len(self.chars))))
		self.tensor_Y = np.load(tensor_Y_file)
		
	def create_batches(self):
		self.num_batches = math.ceil(self.tensor_X.size / self.batch_size)
		
		if self.num_batches == 0:
			assert False, "Not enough data."
		
		num_samples = len(self.tensor_X)
		indexs = np.arange(num_samples)
		np.random.shuffle(indexs)
		
		xdata = self.tensor_X[indexs]
		ydata = self.tensor_Y[indexs]
		self.x_batches = np.array_split(xdata, self.num_batches)
		self.y_batches = np.array_split(ydata, self.num_batches)
		
		self.batch_maxlen = []
		for i in range(len(self.x_batches)):
			batch_x_array = self.x_batches[i]
			batch_y_array = self.y_batches[i]
			lengths = [len(s) for s in batch_x_array]
			self.batch_maxlen.append(lengths)
			max_length = max(lengths)
			padding_X = np.zeros([len(batch_x_array), max_length])
			padding_Y = np.zeros([len(batch_y_array), max_length])
			for idx,seq in enumerate(batch_x_array):
				padding_X[idx, :len(seq)] = seq
			for idx,seq in enumerate(batch_y_array):
				padding_Y[idx, :len(seq)] = seq
			self.x_batches[i] = padding_X
			self.y_batches[i] = padding_Y
		print("x_batches: " + str(len(self.x_batches)))

	def next_batch(self):
		x, y, seq_len = self.x_batches[self.pointer], self.y_batches[self.pointer], self.batch_maxlen[self.pointer]
		self.pointer += 1
		return x, y, seq_len
	
	def reset_batch_pointer(self):
		self.pointer = 0
	
	def create_testdata(self, test_x_file, test_y_file):
		self.tensor_X_test = []
		with codecs.open(test_x_file, "r", encoding=self.encoding) as f:
			for line in f.readlines():
				line = re.sub('[\r\n\t]', '', line)
				self.tensor_X_test.append(list(map(self.vocab.get, line)))
		self.tensor_Y_test = []
		with codecs.open(test_y_file, "r", encoding=self.encoding) as f:
			for line in f.readlines():
				line = re.sub('[\r\n\t]', '', line)
				line = line.split(' ')[:-1]
				self.tensor_Y_test.append(list(map(self.label.get, line)))
	
	def test_data(self):
		lengths = [len(s) for s in self.tensor_X_test]
		max_length = max(lengths)
		padding_X = np.zeros([len(self.tensor_X_test), max_length])
		padding_Y = np.zeros([len(self.tensor_X_test), max_length])
		for idx,seq in enumerate(self.tensor_X_test):
			padding_X[idx, :len(seq)] = seq
		for idx,seq in enumerate(self.tensor_Y_test):
			padding_Y[idx, :len(seq)] = seq
		x, y, seq_len = padding_X, padding_Y, lengths
		return x, y, seq_len