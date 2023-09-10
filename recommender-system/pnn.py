from utils import creat_dataset, create_dataloader, train_task, DNN
import torch
from torch import nn
from torch.nn import functional as F


class ProductLayer(nn.Module):
	def __init__(self, mode, hidden_units, sparse_feat_columns, embed_dim):
		super(ProductLayer, self).__init__()
		self.mode = mode
		self.linear_z = nn.Linear(len(sparse_feat_columns) * embed_dim, hidden_units[0])
		if mode == 'inner':
			self.linear_p = nn.Linear(len(sparse_feat_columns) * len(sparse_feat_columns), hidden_units[0])
		else:
			self.linear_p = nn.Linear(embed_dim * embed_dim, hidden_units[0])
		self.l_b = nn.Parameter(torch.rand(hidden_units[0],))

	def forward(self, sparse_embeds):
		"""
		:param sparse_embeds: (batch_size, embed_dim) * sparse_feat_num
		:return:
		"""
		# batch_size, embed_dim * sparse_feat_num
		sparse_embeds_z = torch.cat(sparse_embeds, dim=1)
		l_z = self.linear_z(sparse_embeds_z)

		# batch_size, sparse_feat_num, embed_dim
		sparse_embeds_p = torch.stack(sparse_embeds, dim=1)
		# 'inner' 不同特征向量之间的内积运算，结果是一个数
		# 'outer' 不同特征向量之间的外积运算，结果是一个矩阵
		if self.mode == 'inner':
			# batch_size, sparse_feat_num, sparse_feat_num
			p = torch.matmul(sparse_embeds_p, sparse_embeds_p.permute(0, 2, 1))
		else:
			# sparse_feat_num*sparse_feat_num个(embed_dim, embed_dim)矩阵，对应位置求和 1个(embed_dim, embed_dim)矩阵
			# 等同于 sparse_feat_num方向上求和embed_dim向量，外积1个(embed_dim, embed_dim)矩阵
			# batch_size, 1, embed_dim
			p_sum = torch.unsqueeze(torch.sum(sparse_embeds_p, dim=1), dim=1)
			# batch_size, embed_dim, embed_dim
			p = torch.matmul(p_sum.permute(0, 2, 1), p_sum)
		l_p = self.linear_p(p.reshape(p.shape[0], -1))

		return l_z + l_p + self.l_b


"""
PNN 细节 https://zhongqiang.blog.csdn.net/article/details/108985457
重点：不管是内积还是外积，都是想求得特征之间非线性的交叉特征
"""
class PNN(nn.Module):
	def __init__(self, feat_columns, embed_dim, hidden_units, mode, dropout):
		super(PNN, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		self.product = ProductLayer(mode, hidden_units, sparse_feat_columns, embed_dim)
		hidden_units[0] += len(dense_feat_columns)
		self.dnn = DNN(hidden_units, dropout)
		self.linear_final = nn.Linear(hidden_units[-1], 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_product = self.product(sparse_embeds)
		outputs = F.relu(torch.cat([dense_input, sparse_product], dim=1))
		outputs = self.dnn(outputs)
		outputs = self.sigmoid(self.linear_final(outputs))
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout, mode = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0, 'inner'
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = PNN(feat_columns, embed_dim, hidden_units, mode, dropout)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)

