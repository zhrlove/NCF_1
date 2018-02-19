from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time, argparse
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader 
from tensorboardX import SummaryWriter

import NCF_input
from NCF_input import NCFDataset
from NCF import NCF 
import evaluate

parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=0.001, type=float, help="learning rate.")
parser.add_argument("--batch_size", default=128, type=int, help="batch size when training.")
parser.add_argument("--gpu", default="0", type=str, help="gpu card ID.")
parser.add_argument("--epochs", default=20, type=str, help="training epoches.")
parser.add_argument("--top_k", default=10, type=int, help="compute metrics@top_k.")
parser.add_argument("--clip_norm", default=5.0, type=float, 
					help="clip norm for preventing gradient exploding.")
parser.add_argument("--embed_size", default=16, type=int, 
					help="embedding size for users and items.")
parser.add_argument("--num_neg", default=4, type=int, 
					help="sample negative items for training.")
parser.add_argument("--test_num_neg", default=99, type=int, 
					help="sample part of negative items for testing.")

FLAGS = parser.parse_args()

opt_gpu = FLAGS.gpu 
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu

(train_features, train_labels), (test_features, 
test_labels), (user_size, item_size), (user_set, 
item_set), (user_bought, user_negative) = NCF_input.load_data(FLAGS.num_neg)

print("User Number: %d\tItem Number: %d\t" %(user_size, item_size))

#Model construction begins.
model = NCF(user_size, item_size, FLAGS.embed_size)
model.cuda()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

writer = SummaryWriter() #For visualization

#Construct the train and test datasets
train_dataset = NCFDataset(
				train_features, train_labels, FLAGS.num_neg, user_negative)
test_dataset = NCFDataset(
				test_features, test_labels, FLAGS.test_num_neg, user_negative)

#Test dataset should be the same, sample negative items one time.
test_dataset.add_neg(FLAGS.test_num_neg)
test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size,
							shuffle=False, num_workers=4)

for epoch in range(FLAGS.epochs):
	model.train() #Enable dropout (if have).
	strat_time = time.time()

	train_dataset.add_neg(FLAGS.num_neg)
	train_dataloader = DataLoader(train_dataset,
		batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)

	for idx, batch_data in enumerate(train_dataloader):
		#Assign the user and item on GPU later.
		user = Variable(batch_data['user'].long()).cuda()
		item = Variable(batch_data['item']).cuda()
		label = Variable(batch_data['label'].float()).cuda()

		model.zero_grad()
		prediction = model(user, item, FLAGS.num_neg+1)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()

		writer.add_scalar('data/loss', loss.data[0], idx)

	model.eval() #Disable dropout (if have).
	HRR, NDCG = evaluate.metrics(
					model, test_dataloader, FLAGS.test_num_neg, FLAGS.top_k)

	elapsed_time = time.time() - strat_time
	print("Epoch: %d" %epoch + "Epoch time: " + time.strftime(
					"%H: %M: %S", time.gmtime(elapsed_time)))
	print("Hit ratio is %.3f\tNdcg is %.3f" %(
					HRR/user_size, NDCG/user_size))
