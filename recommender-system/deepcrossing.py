from utils import creat_dataset, create_dataloader, train_task
import torch
from torch import nn


class Residual(nn.Module):
	def __init__(self, hidden_unit, dim_stack):
		super(Residual, self).__init__()
		self.linear1 = nn.Linear(dim_stack, hidden_unit)
		self.linear2 = nn.Linear(hidden_unit, dim_stack)
		self.relu = nn.ReLU()

	def forward(self, x):
		y = self.linear2(self.linear1(x))
		return self.relu(y + x)


class DeepCrossing(nn.Module):
	def __init__(self, feat_columns, embed_dim, hidden_units, dropout):
		super(DeepCrossing, self).__init__()
		dense_feat_columns, sparse_feat_columns = feat_columns
		self.embed_layers = nn.ModuleList([
			nn.Embedding(feat['feat_num'], embed_dim) for feat in sparse_feat_columns
		])
		dim_stack = len(dense_feat_columns) + len(sparse_feat_columns) * embed_dim
		self.res_layers = nn.ModuleList([
			Residual(hidden_unit, dim_stack) for hidden_unit in hidden_units
		])
		self.res_dropout = nn.Dropout(dropout)
		self.linear = nn.Linear(dim_stack, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, dense_input, sparse_input):
		sparse_embeds = [self.embed_layers[i](sparse_input[:, i]) for i in range(sparse_input.shape[1])]
		sparse_embed = torch.cat(sparse_embeds, dim=1)
		stack = torch.cat([dense_input, sparse_embed], dim=1)
		for res in self.res_layers:
			stack = res(stack)
		outputs = self.res_dropout(stack)
		outputs = self.sigmoid(self.linear(outputs))
		return outputs.squeeze()


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_size, batch_size, embed_dim, lr, epochs, hidden_units, dropout = 0.2, 256, 10, 0.03, 20, [256, 128, 64, 32], 0.0
	data, feat_columns, dense_feats, sparse_feats = creat_dataset(5000)
	train_loader, test_loader = create_dataloader(data, dense_feats, sparse_feats, batch_size)

	model = DeepCrossing(feat_columns, embed_dim, hidden_units, dropout)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler)

































