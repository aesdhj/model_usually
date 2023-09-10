from utils import creat_dataset, create_dataloader, train_task, DNN
import torch
from torch import nn


class CrossNetwork(nn.Module):
	def __init__(self, layer_num, input_dim):
		super(CrossNetwork, self).__init__()
		self.layer_num = layer_num
		self.cross_weights = nn.ParameterList([
			nn.Parameter(torch.randn(input_dim, 1)) for _ in range(layer_num)
		])
		self.cross_bias = nn.ParameterList([
			nn.Parameter(torch.randn(input_dim, 1)) for _ in range(layer_num)
		])

	def forward(self, x):
		# cross部门作用是原始特征高阶交互，看成fm的泛化形式,fm只存在二阶的交互
		# (batch_size, dim, 1)
		x_0 = torch.unsqueeze(x, dim=2)
		x = x_0.clone()
		# (batch_size, 1, dim)
		x_t = x_0.clone().permute(0, 2, 1)
		for i in range(self.layer_num):
			# torch.bmm(x_0, x_t)类似于pnn的外积形式， (batch_size, dim, dim)
			# (batch_size, dim, 1)
			x = torch.matmul(torch.bmm(x_0, x_t), self.cross_weights[i]) + self.cross_bias[i] + x
			x_t = x.permute(0, 2, 1)
		return x.squeeze()


"""
DCN细节详解， https://zhongqiang.blog.csdn.net/article/details/109254498
重点:
	1, 相对于widedeep，不用人工选择特征进入deep还是wide部分
	2，cross部门属于fm的泛化形式， 有高阶交互，不需要人工干预
"""
class DCN(nn.Module):
	def __init__(self, feat_columns, embed_dim, layer_num, hidden_units, dropout):
		super(DCN, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		hidden_units.insert(
			0,
			len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim
		)
		self.cross = CrossNetwork(layer_num, hidden_units[0])
		self.deep_dnn = DNN(hidden_units, dropout)
		self.linear = nn.Linear(hidden_units[0] + hidden_units[-1], 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_embeds = torch.cat(sparse_embeds, dim=1)
		all_input = torch.cat([sparse_embeds, dense_input], dim=1)
		# cross
		cross_out = self.cross(all_input)
		# deep
		deep_out = self.deep_dnn(all_input)

		outputs = torch.cat([cross_out, deep_out], dim=1)
		outputs = self.sigmoid(self.linear(outputs))
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = DCN(feat_columns, embed_dim, len(hidden_units), hidden_units, dropout)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)


