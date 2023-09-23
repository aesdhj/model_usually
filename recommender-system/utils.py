import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader


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


def train_task(epochs, model, train_loader, test_loader, device, loss, optimizer, scheduler=None):
	for epoch in range(epochs):
		model.train()
		l_sum, batch_count = 0.0, 0
		for batch in train_loader:
			if model.__class__.__name__ == 'DIN':
				for k, v in batch.items():
					batch[k] = v.to(device)
				y = batch['label']
				y_hat = model(batch)
				l = loss(y_hat, y)
			elif model.__class__.__name__ == 'DIEN':
				for k, v in batch.items():
					batch[k] = v.to(device)
				y = batch['label']
				y_hat, auxiliary_loss = model(batch)
				l = loss(y_hat, y)
				if auxiliary_loss is not None:
					l += auxiliary_loss
			else:
				dense_data, sparse_data, y = batch
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
		if scheduler is not None:
			scheduler.step()

		l_sum_val, batch_count_val = 0.0, 0
		model.eval()
		with torch.no_grad():
			for batch in test_loader:
				if model.__class__.__name__ == 'DIN':
					for k, v in batch.items():
						batch[k] = v.to(device)
					y = batch['label']
					y_hat = model(batch)
					l = loss(y_hat, y)
				elif model.__class__.__name__ == 'DIEN':
					for k, v in batch.items():
						batch[k] = v.to(device)
					y = batch['label']
					y_hat, auxiliary_loss = model(batch)
					l = loss(y_hat, y)
					if auxiliary_loss is not None:
						l += auxiliary_loss
					l += auxiliary_loss
				else:
					dense_data, sparse_data, y = batch
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


def mark_last_timestamp(df):
	last = df[['userid', 'movieid']].groupby('userid', as_index=False).tail(1).copy()
	last['last'] = 1
	df = df.merge(last, on=['userid', 'movieid'], how='left')
	df.loc[~df['last'].isnull(), 'last'] = 1
	df.loc[df['last'].isnull(), 'last'] = 0
	return df


def neg_sampling(candidates, filters, length):
	max_length = len(candidates)
	res = []
	for _ in range(length):
		while True:
			c = candidates[np.random.randint(0, max_length)]
			if c not in filters:
				res.append(str(c))
				filters.add(c)
				break
	return res


def get_hist_movie_ids(df, candidate_movie_ids, max_len=30):
	hist_movie_ids = list()
	neg_hist_movie_ids = list()
	for _, group in df.groupby('userid'):
		tmp_hist_movie_ids = list()
		for _, row in group.iterrows():
			# rating >= 4才能算正样本
			if row['rating'] >= 4 and row['last'] == 0:
				tmp_hist_movie_ids.append(str(int(row['movieid'])))
		# 取最近的正样本
		tmp_hist_movie_ids = tmp_hist_movie_ids[-max_len:]
		# 负样本采样
		tmp_neg_hist_movie_ids = neg_sampling(candidate_movie_ids, set(tmp_hist_movie_ids), len(tmp_hist_movie_ids))
		hist_movie_ids.append('|'.join(tmp_hist_movie_ids))
		neg_hist_movie_ids.append('|'.join(tmp_neg_hist_movie_ids))
	return hist_movie_ids, neg_hist_movie_ids


def data_preprocess(data_path='data/ml-1m/'):
	# 读取数据
	users_df = pd.read_csv(data_path + 'users.dat', sep='::', names=['userid', 'gender', 'age', 'occupation', 'zipcode'])
	movies_df = pd.read_csv(data_path + 'movies.dat', sep='::', names=['movieid', 'title', 'genres'], encoding='latin-1')
	ratings_df = pd.read_csv(data_path + 'ratings.dat', sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
	# 训练集，验证集
	train_userids, test_userids = train_test_split(list(set(users_df['userid'])), test_size=0.2, random_state=2023)
	train_ratings_df = ratings_df[ratings_df['userid'].isin(train_userids)]
	test_ratings_df = ratings_df[ratings_df['userid'].isin(test_userids)]
	# 标记每个用户最后一部电影
	candidate_movie_ids = list(set(movies_df['movieid']))
	train_ratings_df = train_ratings_df.sort_values(['userid', 'timestamp'])
	test_ratings_df = test_ratings_df.sort_values(['userid', 'timestamp'])
	train_ratings_df = mark_last_timestamp(train_ratings_df)
	test_ratings_df = mark_last_timestamp(test_ratings_df)
	# 对每个用户生成历史列表和负采样列表
	train_hist_movie_ids, train_neg_hist_movie_ids = get_hist_movie_ids(train_ratings_df, candidate_movie_ids)
	test_hist_movie_ids, test_neg_hist_movie_ids = get_hist_movie_ids(test_ratings_df, candidate_movie_ids)
	train_ratings_df = train_ratings_df[train_ratings_df['last'] == 1]
	test_ratings_df = test_ratings_df[test_ratings_df['last'] == 1]
	train_ratings_df['hist_movieids'] = train_hist_movie_ids
	train_ratings_df['neg_hist_movieids'] = train_neg_hist_movie_ids
	test_ratings_df['hist_movieids'] = test_hist_movie_ids
	test_ratings_df['neg_hist_movieids'] = test_neg_hist_movie_ids
	# 合并users_df, movies_df的数据
	train_ratings_df = train_ratings_df.merge(users_df, on='userid', how='inner')
	test_ratings_df = test_ratings_df.merge(users_df, on='userid', how='inner')
	train_ratings_df = train_ratings_df.merge(movies_df, on='movieid', how='inner')
	test_ratings_df = test_ratings_df.merge(movies_df, on='movieid', how='inner')
	# 最后一部电影rating >=4为正样本，label=1
	train_ratings_df['label'] = 0
	train_ratings_df.loc[train_ratings_df['rating'] >= 4, 'label'] = 1
	test_ratings_df['label'] = 0
	test_ratings_df.loc[test_ratings_df['rating'] >= 4, 'label'] = 1
	train_ratings_df.to_csv(data_path + 'train.csv', index=False)
	test_ratings_df.to_csv(data_path + 'test.csv', index=False)
	return train_ratings_df, test_ratings_df


class CategoryEncoder(BaseEstimator, TransformerMixin):
	"""
	自定义转换器， BaseEstimator可以获得get_params()和set_params()
	get_params()和set_params() https://qa.1r1g.com/sf/ask/3480174491/
	TransformerMixin可以获得fit_transformer的方法
	"""
	def __init__(self, min_cnt=5, word2idx=None, idx2word=None):
		super().__init__()
		self.min_cnt = min_cnt
		self.word2idx = word2idx if word2idx else dict()
		self.idx2word = idx2word if idx2word else dict()

	def fit(self, x, y=None):
		if not self.word2idx:
			# np.array()与np.asarray的区别 https://blog.csdn.net/xiaomifanhxx/article/details/82498176
			counter = Counter(np.asarray(x).ravel())
			selected_terms = sorted(list(filter(lambda x: counter[x] >= self.min_cnt, counter)))
			self.word2idx = dict(zip(selected_terms, range(1, len(selected_terms)+1)))
			self.word2idx['__PAD__'] = 0
			if '__UNKNOWN__' not in self.word2idx:
				self.word2idx['__UNKNOWN__'] = len(self.word2idx)
		if not self.idx2word:
			self.idx2word = {idx: word for idx, word in enumerate(self.word2idx)}
		return self

	def transform(self, x):
		transformed_x = list()
		for term in np.asarray(x).ravel():
			try:
				transformed_x.append(self.word2idx[term])
			except KeyError:
				transformed_x.append(self.word2idx['__UNKNOWN__'])
		return np.asarray(transformed_x, dtype=np.int64)

	def dimension(self):
		return len(self.word2idx)


class SequenceEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, sep=' ', min_cnt=5, max_len=None, word2idx=None, idx2word=None):
		super().__init__()
		self.sep = sep
		self.min_cnt = min_cnt
		self.max_len = max_len
		self.word2idx = word2idx if word2idx else dict()
		self.idx2word = idx2word if idx2word else dict()

	def fit(self, x, y=None):
		if not self.word2idx:
			counter = Counter()
			max_len = 0
			for sequence in np.asarray(x).ravel():
				words = sequence.split(self.sep)
				counter.update(words)
				max_len = max(max_len, len(words))
			if self.max_len is None:
				self.max_len = max_len
			selected_terms = sorted(list(filter(lambda x: counter[x] >= self.min_cnt, counter)))
			self.word2idx = dict(zip(selected_terms, range(1, len(selected_terms)+1)))
			self.word2idx['__PAD__'] = 0
			if '__UNKNOWN__' not in self.word2idx:
				self.word2idx['__UNKNOWN__'] = len(self.word2idx)
		if not self.idx2word:
			self.idx2word = {idx: word for idx, word in enumerate(self.word2idx)}
		if not self.max_len:
			max_len = 0
			for sequence in np.asarray(x).ravel():
				words = sequence.split(self.sep)
				max_len = max(max_len, len(words))
			self.max_len = max_len
		return self

	def transform(self, x):
		transformed_x = list()
		for sequence in np.asarray(x).ravel():
			words = list()
			for word in sequence.split(self.sep):
				try:
					words.append(self.word2idx[word])
				except KeyError:
					words.append(self.word2idx['__UNKNOWN__'])
			transformed_x.append(np.asarray(words[-self.max_len:], dtype=np.int64))
		return np.asarray(transformed_x, dtype=object)

	def dimension(self):
		return len(self.word2idx)

	def max_length(self):
		return self.max_len


class DinDataset(Dataset):
	def __init__(self, dfdata, num_features, cate_features, seq_features, encoders, label_col='label'):
		self.dfdata = dfdata
		self.num_features = num_features
		self.cate_features = cate_features
		self.seq_features = seq_features
		self.encoders = encoders
		self.label = label_col

	def __len__(self):
		return len(self.dfdata)

	@staticmethod
	def pad_sequence(sequence, max_len):
		"""
		@staticmethod 是类的静态函数，调用不用先实例类
		https://www.jianshu.com/p/65ae71173b7e
		"""
		padded_sequence = np.zeros(max_len, np.int64)
		padded_sequence[0:sequence.shape[0]] = sequence
		return padded_sequence

	def __getitem__(self, idx):
		record = OrderedDict()
		for col in self.num_features:
			record[col] = self.dfdata[col].iloc[idx].astype(np.float32)
		for col in self.cate_features:
			record[col] = self.dfdata[col].iloc[idx].astype(np.int64)
		for col in self.seq_features:
			sequence = self.dfdata[col].iloc[idx]
			max_len = self.encoders[col].max_length()
			record[col] = DinDataset.pad_sequence(sequence, max_len)
		if self.label is not None:
			record['label'] = self.dfdata[self.label].iloc[idx].astype(np.float32)
		return record

	def get_num_batches(self, batch_size):
		return np.ceil(len(self.dfdata) / batch_size)


def create_dataloader_din(batch_size, num_features, cate_features, seq_features, data_path='data/ml-1m/'):
	# 读取数据集
	if os.path.exists(data_path + 'train.csv'):
		dftrain = pd.read_csv(data_path + 'train.csv')
		dfvalid = pd.read_csv(data_path + 'test.csv')
	else:
		dftrain, dfvalid = data_preprocess()
	# 特征预处理
	for col in cate_features + seq_features:
		dftrain[col] = dftrain[col].astype(str)
		dfvalid[col] = dfvalid[col].astype(str)
	# 连续特征先填充缺失值在标准化
	num_pipeline = Pipeline(steps=[('impute', SimpleImputer()), ('quantile', QuantileTransformer())])
	encoders = {}
	for col in tqdm(num_features, desc='preprocess number features'):
		dftrain[col] = num_pipeline.fit_transform(dftrain[[col]]).astype(np.float32)
		dfvalid[col] = num_pipeline.transform(dfvalid[[col]]).astype(np.float32)
	for col in tqdm(cate_features, desc='preprocess cate features'):
		encoders[col] = CategoryEncoder(min_cnt=5)
		dftrain[col] = encoders[col].fit_transform(dftrain[col])
		dfvalid[col] = encoders[col].transform(dfvalid[col])
	for col in tqdm(seq_features, desc='preprocess sequence features'):
		encoders[col] = SequenceEncoder(sep='|', min_cnt=5)
		dftrain[col] = encoders[col].fit_transform(dftrain[col])
		dfvalid[col] = encoders[col].transform(dfvalid[col])
	cat_nums = {k: v.dimension() for k, v in encoders.items()}
	# 创建dataloader
	dataset_train = DinDataset(dftrain, num_features, cate_features, seq_features, encoders)
	dataset_valid = DinDataset(dfvalid, num_features, cate_features, seq_features, encoders)
	dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
	dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
	return cat_nums, dataloader_train, dataloader_valid


class AttentionGroup():
	def __init__(self, name, pairs, hidden_layers, activation='dice', att_dropout=0.0, gru_type='AUGRU', gru_dropout=0.0):
		self.name = name
		self.pairs = pairs
		self.hidden_layers = hidden_layers
		self.activation = activation
		self.att_dropout = att_dropout
		self.gru_type = gru_type
		self.gru_dropout = gru_dropout
		self.related_features = set()
		self.neg_features = set()
		for pair in self.pairs:
			self.related_features.add(pair['ad'])
			self.related_features.add(pair['pos_hist'])
			if 'neg_hist' in pair:
				self.related_features.add(pair['neg_hist'])
				self.neg_features.add(pair['neg_hist'])

	def is_attention_feature(self, feature):
		if feature in self.related_features:
			return True
		return False

	def is_neg_feature(self, feature):
		if feature in self.neg_features:
			return True
		return False

	@property
	# 属性装饰器 https://zhuanlan.zhihu.com/p/64487092
	def pairs_count(self):
		return len(self.pairs)


class Dice(nn.Module):
	"""
	根据输入的自适应损失函数，实行来源于sigmoid和prelu的结合
	"""
	def __init__(self, embed_size, dim=2, epsilon=1e-8):
		super(Dice, self).__init__()
		assert dim == 2 or dim == 3

		self.bn = nn.BatchNorm1d(embed_size, eps=epsilon)
		self.sigmoid = nn.Sigmoid()
		self.dim = dim

		if self.dim == 2:
			self.alpha = nn.Parameter(torch.zeros((embed_size, )))
		else:
			self.alpha = nn.Parameter(torch.zeros((embed_size, 1)))

	def forward(self, x):
		assert x.dim() == self.dim
		if self.dim == 2:
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1-x_p) * x + x_p * x
		else:
			x = torch.transpose(x, 1, 2)
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1-x_p) * x + x_p * x
			x = torch.transpose(x, 1, 2)
		return out


class MLP(nn.Module):
	def __init__(self, input_size, hidden_layers, dropout=0.0, batchnorm=True, activation='prelu'):
		super(MLP, self).__init__()
		self.mlp = nn.Sequential()
		hidden_layers.insert(0, input_size)
		for idx, item in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
			self.mlp.add_module(f'linear_{idx}', nn.Linear(item[0], item[1]))
			if batchnorm:
				self.mlp.add_module(f'batchnorm_{idx}', nn.BatchNorm1d(item[1]))
			if activation == 'dice':
				self.mlp.add_module(f'activation_{idx}', Dice(item[1], dim=2))
			else:
				self.mlp.add_module(f'activation_{idx}', nn.PReLU())
			if dropout:
				self.mlp.add_module(f'dropout_{idx}', nn.Dropout(dropout))

	def forward(self, x):
		return self.mlp(x)


class Attention(nn.Module):
	def __init__(self, input_size, hidden_layers, dropout=0.0, batchnorm=True, activation='prelu', return_scores=False):
		super(Attention, self).__init__()
		self.return_scores = return_scores
		self.mlp = MLP(
			input_size=input_size*4,
			hidden_layers=hidden_layers,
			dropout=dropout,
			batchnorm=batchnorm,
			activation=activation
		)
		self.fc = nn.Linear(hidden_layers[-1], 1)

	def forward(self, query, keys, keys_length):
		"""
		:param query: (batch_size, embed_size * num_pair)
		:param keys: (batch_size, max_len, embed_size * num_pair)
		:param keys_length: (batch_size,)
		:return:
		"""
		b, l, d = keys.size()
		# (batch_size, max_len, embed_size * num_pair)
		query = query.unsqueeze(dim=1).expand(-1, l, -1)
		# (batch_size, max_len, embed_size * num_pair * 4)
		din_all = torch.cat([query, keys, query-keys, query*keys], dim=-1)
		# view 和reshape的区别 https://zhuanlan.zhihu.com/p/555700619
		# (batch_size * max_len, embed_size * num_pair * 4)
		din_all = din_all.view(b * l, -1)
		outputs = self.mlp(din_all)
		# (batch_size, max_len)
		outputs = self.fc(outputs).view(b, l)
		outputs = outputs / (d**0.5)
		# (batch_size, max_len)
		mask = (torch.arange(l, device=keys_length.device).repeat(b, 1) < keys_length.view(-1, 1))
		outputs[~mask] = -np.inf
		# 不同于传统attention机制scores和1, sigmoid以后的数值是一种相关性
		outputs = torch.sigmoid(outputs)
		# (batch_size, 1, max_len)
		# (batch_size, max_len, embed_size * num_pair)
		# (batch_size, embed_size * num_pair)
		if not self.return_scores:
			outputs = torch.matmul(outputs.unsqueeze(dim=1), keys).squeeze()
		return outputs

