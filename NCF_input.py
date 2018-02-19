from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset


DATA_URI = '/home/yang/Datasets/ml-1m/ratings.dat'
COLUMN_NAMES = ['user', 'item']

def load_data(num_neg):
	full_data = pd.read_csv(
		DATA_URI, sep='::', header=None, names=COLUMN_NAMES, 
		usecols=[0,1], dtype={0: np.int32, 1: np.int32}, engine='python')

	def index_set(s):
		"""Mainly useful for reindex items"""
		i = 0
		s_map = {}
		for key in s:
			s_map[key] = i
			i += 1
		return s_map
	
	#Forcing the user index begining from 0.
	full_data.user = full_data.user - 1
	user_set = set(full_data.user.unique())
	user_size = len(user_set)

	item_list = full_data.item.unique()
	item_size = len(item_list)
	item_map = index_set(item_list)
	item_set = set(item_map.values())

	item_index_new = []
	for i in range(len(full_data)):
		item_index_new.append(
						item_map.get(full_data['item'][i]))

	full_data['item'] = item_index_new

	#Group each user's interactions(purchased items) into dictionary.
	user_bought = {}
	for i in range(len(full_data)):
		u = full_data['user'][i]
		t = full_data['item'][i]
		if u not in user_bought:
			user_bought[u] = []
		user_bought[u].append(t)

	#Group each user's negative interactions.
	user_negative = {}
	for key in user_bought:
		user_negative[key] = list(item_set - set(user_bought[key]))

	#Splitting the full set into train and test sets.
	user_length = full_data.groupby('user').size().tolist() #User transaction count.
	split_train_test = [] 

	for i in range(len(user_set)):
		for _ in range(user_length[i] - 1):
			split_train_test.append('train')
		split_train_test.append('test') #Last one for test.

	full_data['split'] = split_train_test
	train_data = full_data[full_data['split'] == 'train'].reset_index(drop=True)
	test_data = full_data[full_data['split'] == 'test'].reset_index(drop=True)
	del train_data['split']
	del test_data['split']

	#For implicit feedback, first one is positive, other num_neg sanmples are negative.
	labels = np.zeros(num_neg+1, dtype=int)
	labels[0] = 1
	labels = np.tile(labels, (len(train_data), 1))

	train_features = train_data
	train_labels = labels
	test_features = test_data
	test_labels = test_data['item']
	#Take the groundtruth item as test labels.

	return (train_features, train_labels), (test_features, test_labels), (
			user_size, item_size), (user_set, item_set), (
			user_bought, user_negative)


class NCFDataset(Dataset):
	def __init__(self, features, labels, num_neg, user_neg):
		"""
		After load_data processing, read train or test data. Num_neg is different for
		train and test. User_neg is the items that users have no explicit interaction.
		In order to sample different negative items, add_neg() is a must when train.
		"""
		self.features = features
		self.labels = labels
		self.num_neg = num_neg
		self.user_neg = user_neg

	def add_neg(self, numbers):
		"""
		Add negative items to the train or test feature. When at train mode, every 
		iteration sample different negative samples.
		"""
		items_final = []
		for i in range(len(self.features)):
			item_tmp = []
			item_tmp.append(self.features['item'][i]) #first add positive one.
			u = self.features['user'][i]

			#uniformly sample negative ones from candidate negative items
			neg_samples = np.random.choice(
							self.user_neg.get(u), size=numbers, replace=False)
			item_tmp.extend(neg_samples)

			items_final.append(np.array(item_tmp))

		#add the negative items to item column.
		self.features['item_final'] = items_final

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		user = self.features['user'][idx]
		item = self.features['item_final'][idx]
		label = self.labels[idx]

		sample = {'user': user, 'item': item, 'label': label}

		return sample