import torch
from torch import nn
from torch.nn import functional as F
from utils import DNN, creat_dataset, create_dataloader, train_task


class CIN(nn.Module):
	def __init__(self, field_num, hidden_units, activation='relu'):
		super(CIN, self).__init__()
		self.hidden_units = hidden_units
		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'prelu':
			self.activation = nn.PReLU()
		elif activation == 'sigmoid':
			self.activation = nn.Sigmoid()
		self.conv1ds = nn.ModuleList()
		self.field_nums = [field_num]
		for idx, size in enumerate(self.hidden_units):
			# conv1d input_size = (batch_size, embed_dim, seq_len)
			self.conv1ds.append(
				nn.Conv1d(
					in_channels=self.field_nums[-1]*self.field_nums[0],
					out_channels=size,
					kernel_size=1
				)
			)
			self.field_nums.append(size)

	def forward(self, x):
		"""
		:param x: batch_size, num_sparse_feats, embed_dim
		:return:
		"""
		batch_size = x.shape[0]
		dim = x.shape[2]
		hidden_layers = [x]
		res_layers = []

		for idx, size in enumerate(self.hidden_units):
			# (batch_size, hi, m, dim)
			x = torch.einsum('bhd, bmd -> bhmd', hidden_layers[-1], hidden_layers[0])
			# (batch_size, hi*m, dim)
			x = x.reshape(batch_size, hidden_layers[-1].shape[1]*hidden_layers[0].shape[1], dim)
			# 相当于在每一个dim上对hi*m 元素加权求和
			# (batch_size, size, dim)
			x = self.conv1ds[idx](x)
			x = self.activation(x)
			hidden_layer = x
			res_layer = x
			hidden_layers.append(hidden_layer)
			res_layers.append(res_layer)

		# (batch_size, sum(sizes), dim)
		result = torch.cat(res_layers, dim=1)
		# (batch_size, sum(sizes))
		result = torch.sum(result, dim=-1)
		return result


"""
xDeepFm 细节 https://zhongqiang.blog.csdn.net/article/details/116379857
重点：
	1， DCN中的CrossNetwork隐形的计算，忽略了embedding后特征整体的概念
	2，显性特征组合vector-wise的DCI + 隐形特征组合bit-wise的DNN + linear
	3，DCI的特征提取方式可以用nn.conv1d代替
"""
class xDeepFM(nn.Module):
	def __init__(self, feat_columns, embed_dim, dnn_hidden_units, dnn_dropout, cin_hidden_units):
		super(xDeepFM, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		dnn_hidden_units.insert(
			0,
			len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim
		)
		self.embed_dim = embed_dim
		# linear
		self.linear = nn.Linear(dnn_hidden_units[0], 1)
		# dnn
		self.dnn = DNN(dnn_hidden_units, dnn_dropout)
		self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1)
		# cin
		self.cin = CIN(len(sparse_feat_columns), cin_hidden_units)
		self.cin_layer = nn.Linear(sum(cin_hidden_units), 1)

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_embeds = torch.cat(sparse_embeds, dim=1)
		all_input = torch.cat([sparse_embeds, dense_input], dim=1)
		# linear
		linear_output = self.linear(all_input)
		# dnn
		dnn_output = self.dnn_linear(self.dnn(all_input))
		# cin
		# batch_size, num_sparse_feats, embed_dim
		sparse_embeds = sparse_embeds.reshape(sparse_embeds.shape[0], sparse_input.shape[1], -1)
		cin_out = self.cin_layer(self.cin(sparse_embeds))
		outputs = F.sigmoid(linear_output+dnn_output+cin_out)
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, dnn_hidden_units, cin_hidden_units, dnn_dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64], [128, 128], 0.0
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = xDeepFM(feat_columns, embed_dim, dnn_hidden_units, dnn_dropout, cin_hidden_units)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)



