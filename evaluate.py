from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 

import torch
from torch.autograd import Variable

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def metrics(model, test_dataloader, test_num_neg, top_k):
	(HR, NDCG) = 0.0, 0.0

	for batch_data in test_dataloader:
		user = Variable(batch_data['user'].long()).cuda()
		item = Variable(batch_data['item']).cuda()
		#label = batch_data['label'].numpy()

		prediction = model(user, item, test_num_neg+1).cpu().data

		gt = 0 #Index 0 is the groundtruth one.
		for p in prediction:
			(_, p_topk) = torch.topk(p, top_k)

			HR += hit(gt, p_topk)
			NDCG += ndcg(gt, p_topk)

	return HR, NDCG

		

