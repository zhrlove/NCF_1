from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 
from torch.autograd import Variable

class NCF(nn.Module):
	def __init__(self, user_size, item_size, embed_size):
		super(NCF, self).__init__()
		self.user_size = user_size
		self.item_size = item_size

		initStd = 0.01 #Standard deviation

		self.embed_user_GMF_layer = nn.Linear(self.user_size, embed_size)
		self.embed_user_MLP_layer = nn.Linear(self.user_size, embed_size)
		self.embed_item_GMF_layer = nn.Linear(self.item_size, embed_size)
		self.embed_item_MLP_layer = nn.Linear(self.item_size, embed_size)

		self.embed_user_GMF_layer.weight.data.normal_(std=initStd)
		self.embed_user_MLP_layer.weight.data.normal_(std=initStd)
		self.embed_item_GMF_layer.weight.data.normal_(std=initStd)
		self.embed_item_MLP_layer.weight.data.normal_(std=initStd)
		self.embed_user_GMF_layer.bias.data.fill_(0)
		self.embed_user_MLP_layer.bias.data.fill_(0)
		self.embed_item_GMF_layer.bias.data.fill_(0)
		self.embed_item_MLP_layer.bias.data.fill_(0)

		self.MLP_layer1 = nn.Linear(embed_size*2, embed_size*2)
		self.MLP_layer2 = nn.Linear(embed_size*2, embed_size)
		self.MLP_layer3 = nn.Linear(embed_size, embed_size//2)

		self.MLP_layer1.weight.data.normal_(std=initStd)
		self.MLP_layer2.weight.data.normal_(std=initStd)
		self.MLP_layer3.weight.data.normal_(std=initStd)
		self.MLP_layer1.bias.data.fill_(0)
		self.MLP_layer2.bias.data.fill_(0)
		self.MLP_layer3.bias.data.fill_(0)

		self.predict_layer = nn.Linear(embed_size*3//2, 1)

		self.predict_layer.weight.data.normal_(std=initStd)
		self.predict_layer.bias.data.fill_(0)


	def convert_onehot(self, feature, feature_size, num_total):
		"""Convert user and item to one-hot format"""
		feature = feature.view(-1, num_total, 1)

		#Num_neg is different when test and train for items.
		batch_size = feature.shape[0] 
		f_onehot = torch.cuda.FloatTensor(
				 				batch_size, num_total, feature_size)
		f_onehot.zero_()
		f_onehot.scatter_(2, feature.data, 1)

		return Variable(f_onehot)


	def forward(self, user, item, num_total):
		user = self.convert_onehot(user, self.user_size, 1)
		item = self.convert_onehot(item, self.item_size, num_total)

		embed_user_GMF = self.embed_user_GMF_layer(user)
		embed_user_MLP = self.embed_user_MLP_layer(user)
		embed_item_GMF = self.embed_item_GMF_layer(item)
		embed_item_MLP = self.embed_item_MLP_layer(item)

		#GMF part begins. 
		output_GMF = embed_user_GMF * embed_item_GMF

		#MLP part begins.
		#Must expand embed_user_MLP to the same shape as item.
		embed_user_MLP = embed_user_MLP.expand(-1, num_total, -1)
		interaction = torch.cat((embed_user_MLP, embed_item_MLP), 2)

		inter_layer1 = F.relu(self.MLP_layer1(interaction))
		inter_layer2 = F.relu(self.MLP_layer2(inter_layer1))
		inter_layer3 = F.relu(self.MLP_layer3(inter_layer2))

		#Concatenation part begins.
		concat = torch.cat((output_GMF, inter_layer3), 2)
		prediction = F.sigmoid(self.predict_layer(concat))

		return prediction.view(-1, num_total)


