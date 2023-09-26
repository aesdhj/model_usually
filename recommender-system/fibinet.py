import torch
from torch import nn
from torch.nn import functional as F
from itertools import combinations
from utils import DNN, creat_dataset, create_dataloader, train_task


class SENetAttention(nn.Module):
	def __init__(self, num_fields, reduction_ratio=3):
		super(SENetAttention, self).__init__()
		reduced_size = max(1, int(num_fields / reduction_ratio))
		self.excitation = nn.Sequential(
			nn.Linear(num_fields, reduced_size, bias=False),
			nn.ReLU(),
			nn.Linear(reduced_size, num_fields, bias=False),
			nn.ReLU()
		)

	def forward(self, x):
		"""
		:param x: (batch_size, num_features, embed_dim)
		:return:
		"""
		# (batch_size, num_features)
		z = torch.mean(x, dim=-1)
		# 属于attention机制的一种，抓取num_features之间的权重关系
		# (batch_size, num_features)
		a = self.excitation(z)
		# (batch_size, num_features, embed_dim)
		v = x * a.unsqueeze(dim=-1)
		return v


class BilinearInteraction(nn.Module):
	def __init__(self, num_fields, embed_dim, bilinear_type):
		super(BilinearInteraction, self).__init__()
		self.bilinear_type = bilinear_type
		if self.bilinear_type == 'field_all':
			self.bilinear_layer = nn.Linear(embed_dim, embed_dim, bias=False)
		elif self.bilinear_type =='field_each':
			self.bilinear_layer = nn.ModuleList([
				nn.Linear(embed_dim, embed_dim, bias=False) for i in range(num_fields)])
		elif self.bilinear_type == 'field_interaction':
			self.bilinear_layer = nn.ModuleList([
				nn.Linear(embed_dim, embed_dim, bias=False) for i, j in combinations(range(num_fields), 2)])
		else:
			raise NotImplementedError

	def forward(self, x):
		"""
		:param x: (batch_size, num_features, embed_dim)
		:return:
		"""
		embed_list = torch.split(x, 1, dim=1)
		if self.bilinear_type == 'field_all':
			bilinear_list = [
				self.bilinear_layer(v_i) * v_j for v_i, v_j in combinations(embed_list, 2)]
		elif self.bilinear_type == 'field_each':
			bilinear_list = [
				self.bilinear_layer[i](embed_list[i]) * embed_list[j] for i, j in combinations(range(len(embed_list)))]
		elif self.bilinear_type == 'field_interaction':
			bilinear_list = [
				self.bilinear_layer[i](v[0]) * v[1] for i, v in enumerate(combinations(embed_list, 2))]
		return torch.cat(bilinear_list, dim=1)


"""
FiBiNET 细节 https://zhongqiang.blog.csdn.net/article/details/118439590
重点：
	1，SENet的attention机制类似特征筛选，在特征交叉之前判断特征的重要性，AFM是交叉以后在判断
	2，Bilinear内部计算是哈达玛积各个元素加一个权重，权重矩阵利用了FFM的思想
"""
class FiBiNET(nn.Module):
	def __init__(self, feat_columns, embed_dim, hidden_units, dropout, bilinear_type):
		super(FiBiNET, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		hidden_units.insert(
			0,
			len(sparse_feat_columns) * (len(sparse_feat_columns)-1) * embed_dim
		)
		# wide
		self.linear = nn.Linear(len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim, 1)
		# deep
		self.se_attention = SENetAttention(len(sparse_feat_columns))
		self.bilinear = BilinearInteraction(len(sparse_feat_columns), embed_dim, bilinear_type)
		self.dnn = DNN(hidden_units, dropout)
		self.dnn_linear = nn.Linear(hidden_units[-1], 1)

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_embeds = torch.cat(sparse_embeds, dim=1)
		all_input = torch.cat([sparse_embeds, dense_input], dim=1)
		# wide
		wide_output = self.linear(all_input)
		# dnn
		# batch_size, num_sparse_feats, embed_dim
		sparse_embeds = sparse_embeds.reshape(sparse_embeds.shape[0], sparse_input.shape[1], -1)
		se_embeds = self.se_attention(sparse_embeds)
		# (batch_size, len(sparse_feat_columns) * (len(sparse_feat_columns)-1) / 2, embed_dim)
		ffm_out = self.bilinear(sparse_embeds)
		# (batch_size, len(sparse_feat_columns) * (len(sparse_feat_columns)-1) / 2, embed_dim)
		se_ffm_out = self.bilinear(se_embeds)
		interaction_out = torch.flatten(torch.cat([ffm_out, se_ffm_out], dim=1), start_dim=1)
		deep_out = self.dnn_linear(self.dnn(interaction_out))
		outputs = F.sigmoid(wide_output+deep_out)
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout, bilinear_type = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0, 'field_interaction'
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = FiBiNET(feat_columns, embed_dim, hidden_units, dropout, bilinear_type)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)

