import torch
from torch import nn
from torch.nn import functional as F
from utils import DNN, creat_dataset, create_dataloader, train_task


class InteractLayer(nn.Module):
	def __init__(self, embed_dim, num_heads, use_residual=True):
		super(InteractLayer, self).__init__()
		self.att_embed_dim = int(embed_dim//num_heads)
		self.num_heads = num_heads
		self.use_residual = use_residual
		self.queries_linear = nn.Linear(embed_dim, embed_dim, bias=False)
		self.keys_linear = nn.Linear(embed_dim, embed_dim, bias=False)
		self.values_linear = nn.Linear(embed_dim, embed_dim, bias=False)
		if use_residual:
			self.res_linear = nn.Linear(embed_dim, embed_dim, bias=False)

	def forward(self, x):
		"""
		:param x: batch_size, num_fields, embed_dim
		:return:
		"""
		# batch_size, num_fields, embed_dim
		queries = self.queries_linear(x)
		keys = self.queries_linear(x)
		values = self.values_linear(x)
		# num_heads, batch_size, num_fields, embed_dim//num_heads
		queries = torch.stack(torch.split(queries, self.att_embed_dim, dim=2), dim=0)
		keys = torch.stack(torch.split(keys, self.att_embed_dim, dim=2), dim=0)
		values = torch.stack(torch.split(values, self.att_embed_dim, dim=2), dim=0)
		# num_heads, batch_size, num_fields, num_fields
		att_scores = torch.einsum('nbik, nbjk -> nbij', queries, keys)
		att_scores /= self.att_embed_dim ** 0.5
		att_scores = F.softmax(att_scores, dim=-1)
		# num_heads, batch_size, num_fields, embed_dim//num_heads
		output = torch.matmul(att_scores, values)
		# batch_size, num_fields, embed_dim
		output = torch.cat(torch.split(output, 1, dim=0), dim=3).squeeze()
		if self.use_residual:
			output += self.res_linear(x)
		return F.relu(output)

"""
AutoInt 细节详解 https://zhongqiang.blog.csdn.net/article/details/118682806
重点：利用多层attention叠加实现特征多阶交互
"""
class AutoInt(nn.Module):
	def __init__(self, feat_columns, embed_dim, hidden_units, dnn_dropout, att_layers, num_heads):
		super(AutoInt, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers_sparse = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		self.embed_layers_dense = nn.ModuleList([
			nn.Embedding(1, embed_dim) for _ in range(len(dense_feat_columns))
		])
		# linear
		self.linear = nn.Linear((len(dense_feat_columns)+len(sparse_feat_columns))*embed_dim, 1)
		# dnn
		hidden_units.insert(0, (len(dense_feat_columns)+len(sparse_feat_columns))*embed_dim)
		self.dnn = DNN(hidden_units, dnn_dropout)
		self.dnn_linear = nn.Linear(hidden_units[-1], 1)
		# attention
		self.embed_dim = embed_dim
		self.att_layers = att_layers
		self.att_layers = nn.ModuleList([
			InteractLayer(embed_dim, num_heads) for _ in range(att_layers)
		])
		self.att_dnn = DNN(hidden_units, dnn_dropout)
		self.att_dnn_linear = nn.Linear(hidden_units[-1], 1)

	def forward(self, dense_input, sparse_input):
		dense_embeds = [
			self.embed_layers_dense[i](torch.zeros(size=(dense_input.shape[0], ), dtype=torch.long).to(dense_input.device)) * \
			dense_input[:, i].unsqueeze(dim=-1) for i in range(dense_input.shape[1])
		]
		sparse_embeds = [
			self.embed_layers_sparse[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		all_input = torch.cat(dense_embeds+sparse_embeds, dim=1)
		# linear
		linear_output = self.linear(all_input)
		# dnn
		dnn_output = self.dnn_linear(self.dnn(all_input))
		# attention
		att_input = all_input.reshape(all_input.shape[0], -1, self.embed_dim)
		for att_layer in self.att_layers:
			att_input = att_layer(att_input)
		att_output = torch.flatten(att_input, start_dim=1)
		att_output = self.att_dnn_linear(self.att_dnn(att_output))
		outputs = F.sigmoid(linear_output+dnn_output+att_output)
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0
	att_layers, num_heads = 2, 2
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = AutoInt(feat_columns, embed_dim, hidden_units, dropout, att_layers, num_heads)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)
