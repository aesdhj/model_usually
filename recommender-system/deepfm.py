from utils import creat_dataset, create_dataloader, train_task, DNN
import torch
from torch import nn
from torch.nn import functional as F


"""
DeepFM细节 https://zhongqiang.blog.csdn.net/article/details/109532267
重点:
	1, wide_deep=lr+dnn,dcn=dnn+cross,deepfm=dnn+fm，都是对交叉特征的改进
	2，deepfm fm部门都是1-weights,所以不用fm那样预训练一个（n,k)的参数
"""
class DeepFM(nn.Module):
	def __init__(self, feat_columns, embed_dim, hidden_units, dropout):
		super(DeepFM, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		hidden_units.insert(
			0,
			len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim
		)
		self.fm_linear = nn.Linear(hidden_units[0], 1)
		self.dnn = DNN(hidden_units, dropout)
		self.dnn_linear = nn.Linear(hidden_units[-1], 1)

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_embeds = torch.cat(sparse_embeds, dim=1)
		all_input = torch.cat([sparse_embeds, dense_input], dim=1)
		# fm
		fm_first = self.fm_linear(all_input)
		# batch_size, num_sparse_feats, embed_dim
		sparse_embeds = sparse_embeds.reshape(sparse_input.shape[0], sparse_input.shape[1], -1)
		fm_second_square_of_sum = torch.pow(torch.sum(sparse_embeds, dim=1), 2)
		fm_second_sum_of_square = torch.sum(torch.pow(sparse_embeds, 2), dim=1)
		fm_second = 0.5 * (torch.sum(fm_second_square_of_sum - fm_second_sum_of_square, dim=1, keepdim=True))
		fm = fm_first + fm_second
		# dnn
		dnn = self.dnn_linear(self.dnn(all_input))
		outputs = F.sigmoid(fm + dnn)
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64], 0.0
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = DeepFM(feat_columns, embed_dim, hidden_units, dropout)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)


