from utils import creat_dataset, create_dataloader, train_task
import torch
from torch import nn


"""
FM,FFM细节参考 https://zhongqiang.blog.csdn.net/article/details/108719417
	FM对于交叉系数矩阵(对称)进行分解，参数规模(feature_num,k)
	FFM在FM的基础上引入filed概念，参数规模(feature_num, field_num, k)
FM代码实现 https://zhuanlan.zhihu.com/p/364613247
FFM代码实现 https://zhuanlan.zhihu.com/p/545963418
"""
class FM(nn.Module):
	def __init__(self, feat_columns, embed_dim, k):
		super(FM, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		n = len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim
		self.w = nn.Linear(n, 1, bias=True)
		self.v = nn.Parameter(torch.rand(n, k))
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [
			self.embed_layers[i](sparse_input[:, i])for i in range(sparse_input.shape[1])
		]
		sparse_embeds = torch.cat(sparse_embeds, dim=1)
		all_input = torch.cat([sparse_embeds, dense_input], dim=1)
		# linear
		first_fm = self.w(all_input).squeeze()
		# fm
		square_of_sum = torch.pow(torch.mm(all_input, self.v), 2)
		sum_of_square = torch.mm(torch.pow(all_input, 2), torch.pow(self.v, 2))
		second_fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
		output = self.sigmoid(first_fm + second_fm)
		return output


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, k, lr, epochs, embed_dim = 0.2, 256, 10, 0.03, 20, 10
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = FM(feat_columns, embed_dim, k)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)


