from utils import creat_dataset, create_dataloader, train_task, DNN
import torch
from torch import nn
from torch.nn import functional as F


"""
WideDeep细节， https://zhongqiang.blog.csdn.net/article/details/109254498
重点 这个模型的wide和deep端接收的特征是不一样的， wide端一般会接收一些重要的交互特征，高维的稀疏离散特征， 而deep端接收的是一些连续特征
"""
class WideDeep(nn.Module):
	def __init__(self, feat_columns, embed_dim, hidden_units, dropout):
		super(WideDeep, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		self.wide_linear = nn.Linear(len(dense_feat_columns), 1)
		# self.wide_linear = nn.Linear(len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim, 1)
		hidden_units.insert(
			0,
			len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim
		)
		self.deep_dnn = DNN(hidden_units, dropout)
		self.deep_linear = nn.Linear(hidden_units[-1], 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_embeds = torch.cat(sparse_embeds, dim=1)
		dnn_input = torch.cat([sparse_embeds, dense_input], dim=1)
		# wide
		wide_out = F.relu(self.wide_linear(dense_input))
		# wide_out = self.wide_linear(dnn_input)
		# deep
		deep_out = self.deep_dnn(dnn_input)
		deep_out = self.deep_linear(deep_out)

		out = self.sigmoid(0.5 * (wide_out + deep_out))
		return out.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = WideDeep(feat_columns, embed_dim, hidden_units, dropout)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)

