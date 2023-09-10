from utils import creat_dataset, create_dataloader, train_task, DNN
import torch
from torch import nn
from torch.nn import functional as F
import itertools


"""
AFM细节 https://zhongqiang.blog.csdn.net/article/details/109532346
"""
class AttentionLayer(nn.Module):
	def __init__(self, embed_dim, att_dim):
		super(AttentionLayer, self).__init__()
		self.att_weight = nn.Sequential(
			nn.Linear(embed_dim, att_dim),
			nn.ReLU(),
			nn.Linear(att_dim, 1),
			nn.Softmax(dim=1)
		)

	def forward(self, bi_interaction):
		"""
		:param bi_interaction: (batch_size, field_num*(field_num-1)/2, embed_dim)
		:return:
		"""
		# 不同的特征交互有不同的weight
		# (batch_size, field_num*(field_num-1)/2, 1)
		att_weight = self.att_weight(bi_interaction)
		# (batch_size, embed_dim)
		att_out = torch.sum(att_weight * bi_interaction, dim=1)
		return att_out


class AFM(nn.Module):
	def __init__(self, feat_columns, embed_dim, mode, att_dim, use_dnn, hidden_units, dropout):
		super(AFM, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.mode = mode
		self.use_dnn = use_dnn
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		if self.mode == 'att':
			self.attention = AttentionLayer(embed_dim, att_dim)
		if self.use_dnn:
			hidden_units.insert(
				0,
				len(dense_feat_columns) + embed_dim
			)
			self.dnn = nn.Sequential(
				nn.BatchNorm1d(hidden_units[0]),
				DNN(hidden_units, dropout),
				nn.Linear(hidden_units[-1], 1)
			)
		else:
			self.dnn = nn.Linear(len(dense_feat_columns) + embed_dim, 1)

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		# (batch_size, field_num, embed_dim)
		sparse_embeds = torch.stack(sparse_embeds, dim=1)
		first_indices = []
		second_indices = []
		# combinations(list, x)， 在list里面去取元素按x的长度进行不重复组合
		for f, s in itertools.combinations(range(sparse_embeds.shape[1]), 2):
			first_indices.append(f)
			second_indices.append(s)
		p = sparse_embeds[:, first_indices, :]
		q = sparse_embeds[:, second_indices, :]
		# (batch_size, field_num * (field_num - 1) / 2, embed_dim)
		bi_interaction = p * q

		if self.mode == 'max':
			att_out = torch.max(bi_interaction, dim=1)
		elif self.mode == 'avg':
			att_out = torch.mean(bi_interaction, dim=1)
		else:
			att_out = self.attention(bi_interaction)

		all_input = torch.cat([dense_input, att_out], dim=1)
		outputs = F.sigmoid(self.dnn(all_input))
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0
	mode, att_dim, use_dnn = 'att', 10, True
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = AFM(feat_columns, embed_dim, mode, att_dim, use_dnn, hidden_units, dropout)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)



