import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
import itertools
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')


def process_dense_feats(data, dense_feats):
	data[dense_feats] = data[dense_feats].fillna(0)
	for f in tqdm(dense_feats, desc='process dense feats'):
		mean = data[f].mean()
		std = data[f].std()
		data[f] = (data[f] - mean) / (std + 1e-12)
	return data


def process_sparse_feats(data, sparse_feats):
	data[sparse_feats] = data[sparse_feats].fillna('-1')
	for f in tqdm(sparse_feats, desc='process sparse feats'):
		label_encoder = LabelEncoder()
		data[f] = label_encoder.fit_transform(data[f])
	return data


def dense_feat(feat):
	return {'feat': feat}


def sparse_feat(feat, feat_num):
	return {'feat': feat, 'feat_num': feat_num}


def creat_dataset(sample_count, path='data/criteo_sampled_50.csv'):
	data = pd.read_csv(path)
	data = data.sample(n=sample_count, random_state=0, ignore_index=True)
	dense_feats = [col for col in data.columns if col[0] == 'I']
	sparse_feats = [col for col in data.columns if col[0] == 'C']
	data = process_dense_feats(data, dense_feats)
	data = process_sparse_feats(data, sparse_feats)
	feat_columns = [[dense_feat(feat) for feat in dense_feats]] + [[sparse_feat(feat, len(data[feat].unique())) for feat in sparse_feats]]
	return data, feat_columns, dense_feats, sparse_feats


def create_dataloader(data, dense_feats, sparse_feats, batch_size):
	train, test = train_test_split(data, test_size=0.2, random_state=2022)
	train_dataset = torch.utils.data.TensorDataset(
		torch.tensor(train[dense_feats].values, dtype=torch.float32),
		torch.tensor(train[sparse_feats].values, dtype=torch.long),
		torch.tensor(train['label'].values, dtype=torch.float32)
	)
	test_dataset = torch.utils.data.TensorDataset(
		torch.tensor(test[dense_feats].values, dtype=torch.float32),
		torch.tensor(test[sparse_feats].values, dtype=torch.long),
		torch.tensor(test['label'].values, dtype=torch.float32)
	)
	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=True
	)
	test_loader = torch.utils.data.DataLoader(
		dataset=test_dataset,
		batch_size=batch_size,
		shuffle=False
	)
	return train_loader, test_loader


def train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler):
	for epoch in range(epochs):
		model.train()
		l_sum, batch_count = 0.0, 0
		for (dense_data, sparse_data, y) in train_loader:
			dense_data = dense_data.to(device)
			sparse_data = sparse_data.to(device)
			y = y.to(device)
			y_hat = model(dense_data, sparse_data)

			l = loss(y_hat, y)
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			l_sum += l.item()
			batch_count += 1
		scheduler.step()

		l_sum_val, batch_count_val = 0.0, 0
		model.eval()
		with torch.no_grad():
			for (dense_data, sparse_data, y) in test_loader:
				dense_data = dense_data.to(device)
				sparse_data = sparse_data.to(device)
				y = y.to(device)
				y_hat = model(dense_data, sparse_data)

				l = loss(y_hat, y)
				l_sum_val += l.item()
				batch_count_val += 1
		print(f'epoch={epoch + 1}, train_loss={l_sum / batch_count:.4f}, val_loss={l_sum_val / batch_count_val:.4f}')


class DNN(nn.Module):
	def __init__(self, hidden_units, dropout):
		super(DNN, self).__init__()
		self.dnn_layers = nn.ModuleList([
			nn.Linear(layer[0], layer[1]) for layer in zip(hidden_units[:-1], hidden_units[1:])
		])
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		for dnn_layer in self.dnn_layers:
			x = F.relu(dnn_layer(x))
		return self.dropout(x)


