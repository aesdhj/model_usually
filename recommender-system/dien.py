import torch
from torch import nn
from collections import OrderedDict
from utils import AttentionGroup, Attention, MLP, create_dataloader_din, train_task
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F


class AuxiliaryNet(nn.Module):
	def __init__(self, input_size, hidden_layers, activation='sigmoid'):
		super(AuxiliaryNet, self).__init__()
		self.mlp = nn.Sequential()
		hidden_layers.insert(0, input_size)
		for idx, item in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
			self.mlp.add_module(f'linear_{idx}', nn.Linear(item[0], item[1]))
			self.mlp.add_module(f'activation_{idx}', nn.Sigmoid())
		self.mlp.add_module(f'final_layer', nn.Linear(hidden_layers[-1], 1))
		self.mlp.add_module(f'final_activation', nn.Sigmoid())

	def forward(self, x):
		return self.mlp(x)


class AttentionGRU(nn.Module):
	def __init__(self, input_size, hidden_size, bias=True):
		super(AttentionGRU, self).__init__()
		self.hidden_size = hidden_size
		self.w_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.w_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.b_r = nn.Parameter(torch.Tensor(hidden_size))
		self.w_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.b_h = nn.Parameter(torch.Tensor(hidden_size))

	def reset_parameters(self):
		stdv = 1.0 / (self.hidden_size)**0.5
		for weight in self.parameters():
			nn.init.uniform_(weight, -stdv, stdv)

	def forward(self, x, hx, att_score):
		"""
		:param x: (mini_step, embed_size * num_pair)
		:param hx: (mini_step, embed_size * num_pair)
		:param att_score: (mini_step,)
		:return:
		"""
		# (mini_step, embed_size * num_pair)
		r = torch.sigmoid(x@self.w_xr+hx@self.w_hr+self.b_r)
		# (mini_step, embed_size * num_pair)
		h_tilda = torch.tanh(x@self.w_xh+(r*hx)@self.w_hh+self.b_h)
		# (mini_step, 1)
		att_score = att_score.view(-1, 1)
		# att_score越大和h_tilda关系越大
		# (mini_step, embed_size * num_pair)
		hy = (1.0-att_score) * hx + att_score * h_tilda
		return hy


class AttentionUpdateGateGRU(nn.Module):
	def __init__(self, input_size, hidden_size, bias=True):
		super(AttentionUpdateGateGRU, self).__init__()
		self.hidden_size = hidden_size
		self.w_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.w_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.b_r = nn.Parameter(torch.Tensor(hidden_size))
		self.w_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.b_h = nn.Parameter(torch.Tensor(hidden_size))
		self.w_xz = nn.Parameter(torch.Tensor(input_size, hidden_size))
		self.w_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
		self.b_z = nn.Parameter(torch.Tensor(hidden_size))

	def reset_parameters(self):
		stdv = 1.0 / (self.hidden_size)**0.5
		for weight in self.parameters():
			nn.init.uniform_(weight, -stdv, stdv)

	def forward(self, x, hx, att_score):
		"""
		:param x: (mini_step, embed_size * num_pair)
		:param hx: (mini_step, embed_size * num_pair)
		:param att_score: (mini_step,)
		:return:
		"""
		# (mini_step, embed_size * num_pair)
		r = torch.sigmoid(x @ self.w_xr + hx @ self.w_hr + self.b_r)
		z = torch.sigmoid(x @ self.w_xz + hx @ self.w_hz + self.b_z)
		# (mini_step, embed_size * num_pair)
		h_tilda = torch.tanh(x@self.w_xh+(r*hx)@self.w_hh+self.b_h)
		# (mini_step, 1)
		att_score = att_score.view(-1, 1)
		z = att_score * z
		# att_score越大和h_tilda关系越大
		# (mini_step, embed_size * num_pair)
		hy = (1.0-z) * hx + z * h_tilda
		return hy


class DynamicGRU(nn.Module):
	def __init__(self, input_size, hidden_size, gru_type='AUGRU', bias=True):
		super(DynamicGRU, self).__init__()
		self.hidden_size = hidden_size
		if gru_type == 'AGRU':
			self.rnn = AttentionGRU(input_size, hidden_size)
		elif gru_type == 'AUGRU':
			self.rnn = AttentionUpdateGateGRU(input_size, hidden_size)

	def forward(self, x, att_scores, hx=None):
		"""
		:param x: (batch_size*keys_length, embed_size * num_pair)
		:param att_scores: (batch_size*keys_length,)
		:param hx: (batch_size*keys_length, embed_size * num_pair)
		:return:
		"""
		x, batch_sizes, sorted_indices, unsorted_indices = x
		att_scores, _, _, _ = att_scores
		max_batch_size = int(batch_sizes[0])
		if hx is None:
			hx = torch.zeros(max_batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
		outputs = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)

		begin = 0
		for batch in batch_sizes:
			new_hx = self.rnn(
				x[begin:begin+batch],
				hx[0:batch],
				att_scores[begin:begin+batch]
			)
			outputs[begin:begin+batch] = new_hx
			hx = new_hx
			begin += batch
		# (batch_size*keys_length, embed_size * num_pair)
		return nn.utils.rnn.PackedSequence(
			outputs, batch_sizes, sorted_indices, unsorted_indices
		)


class Interest(nn.Module):
	def __init__(self, input_size, gru_type='AUGRU',gru_dropout=0.0,
				att_hidden_layers=[16, 8], att_dropout=0.0,
				att_batchnorm=True, att_activation='prelu',
				use_negsampling=True):
		super(Interest, self).__init__()
		assert gru_type in ['GRU', 'AIGRU', 'AGRU', 'AUGRU']
		self.gru_type = gru_type
		self.use_negsampling = use_negsampling

		# 第一层gru
		self.interest_extractor = nn.GRU(
			input_size=input_size,
			hidden_size=input_size,
			batch_first=True,
			bidirectional=False
		)
		if self.use_negsampling:
			self.auxiliary_net = AuxiliaryNet(input_size * 2, hidden_layers=[64, 32])

		# 第二层gru
		if self.gru_type == 'GRU':
			self_attention = Attention(
				input_size=input_size, hidden_layers=att_hidden_layers, dropout=att_dropout,
				batchnorm=att_batchnorm, activation=att_activation, return_scores=False
			)
			self.interest_evolution = nn.GRU(
				input_size=input_size, hidden_size=input_size,
				batch_first=True, bidirectional=False
			)
		elif self.gru_type == 'AIGRU':
			self_attention = Attention(
				input_size=input_size, hidden_layers=att_hidden_layers, dropout=att_dropout,
				batchnorm=att_batchnorm, activation=att_activation, return_scores=True
			)
			self.interest_evolution = nn.GRU(
				input_size=input_size, hidden_size=input_size,
				batch_first=True, bidirectional=False
			)
		elif self.gru_type in ['AGRU', 'AUGRU']:
			self.attention = Attention(
				input_size=input_size, hidden_layers=att_hidden_layers, dropout=att_dropout,
				batchnorm=att_batchnorm, activation=att_activation, return_scores=True
			)
			self.interest_evolution = DynamicGRU(
				input_size=input_size,
				hidden_size=input_size,
				gru_type=gru_type
			)

	def cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
		"""
		:param states: interests, (batch_size, max_len-1, embed_size * num_pair)
		:param click_seq: keys, (batch_size, max_len-1, embed_size * num_pair)
		:param noclick_seq: neg_keys, (batch_size, max_len-1, embed_size * num_pair)
		:param keys_length: keys_length-1, (batch_size,)
		:return:
		"""
		batch_size, max_length, dim = states.size()
		# (batch_size, max_len)
		mask = (torch.arange(max_length, device=states.device).repeat(batch_size, 1) < keys_length.view(-1, 1)).float()
		# (batch_size, max_len-1, embed_size * num_pair * 2)
		click_input = torch.cat([states, click_seq], dim=-1)
		# (batch_size, max_len-1, embed_size * num_pair * 2)
		noclick_input = torch.cat([states, noclick_seq], dim=-1)
		# (batch_size*max_length, 1)
		click_p = self.auxiliary_net(click_input.view(batch_size*max_length, -1))
		# (batch_size*max_length, 1)
		click_p = click_p.view(batch_size, max_length)[mask > 0].view(-1, 1)
		click_target = torch.ones(click_p.size(), dtype=torch.float32, device=click_p.device)
		noclick_p = self.auxiliary_net(noclick_input.view(batch_size*max_length, -1))
		noclick_p = noclick_p.view(batch_size, max_length)[mask > 0].view(-1, 1)
		noclick_target = torch.zeros(noclick_p.size(), dtype=torch.float32, device=noclick_p.device)
		loss = F.binary_cross_entropy(
			torch.cat([click_p, noclick_p], dim=0),
			torch.cat([click_target, noclick_target], dim=0)
		)
		return loss

	def forward(self, query, keys, keys_length, neg_keys):
		"""
		:param query: (batch_size, embed_size * num_pair)
		:param keys: (batch_size, max_len, embed_size * num_pair)
		:param keys_length: (batch_size,)
		:param neg_keys: (batch_size, max_len, embed_size * num_pair)
		:return:
		"""
		batch_size, max_length, dim = keys.size()
		# pack_padded_sequence 默认的填充模式 post
		# https://blog.csdn.net/wangchaoxjtu/article/details/118023187
		# (batch_size*keys_length, embed_size * num_pair)
		packed_keys = pack_padded_sequence(
			keys,
			lengths=keys_length.squeeze().cpu(),
			batch_first=True,
			enforce_sorted=False
		)
		# 用GRU在每个时间点输出代表这个时间点的兴趣点
		# (batch_size*keys_length, embed_size * num_pair)
		packed_interests, _ = self.interest_extractor(packed_keys)

		aloss = None
		if self.gru_type != 'GRU' or self.use_negsampling:
			# (batch_size, max_len, embed_size * num_pair)
			interests, _ = pad_packed_sequence(
				packed_interests,
				batch_first=True,
				padding_value=0.0,
				total_length=max_length
			)
			if self.use_negsampling:
				# 每个时间点的输出(兴趣点)导致下一个时间点的商品和负采样的商品
				# 时间有错位
				aloss = self.cal_auxiliary_loss(
					interests[:, :-1, :],
					keys[:, 1:, :],
					neg_keys[:, 1:, :],
					keys_length-1
				)

		if self.gru_type == 'GRU':
			# 先用gru算出packed_interests每个时间点输出， 和query用attention提取特征
			# (batch_size*keys_length, embed_size * num_pair)
			packed_interests, _ = self.interest_evolution(packed_interests)
			# (batch_size, max_len, embed_size * num_pair)
			interests, _ = pad_packed_sequence(
				packed_interests,
				batch_first=True,
				padding_value=0.0,
				total_length=max_length
			)
			# (batch_size, embed_size * num_pair)
			outputs = self.attention(query, interests, keys_length)
		elif self.gru_type == 'AIGRU':
			# 先计算query和interests的attention_scores,
			# 用interests * scores作为gru的输入取最后一步为特征
			# 缺点：即使interests被scores=0替换掉，也会影响gru的输出
			# (batch_size, max_len)
			scores = self.attention(query, interests, keys_length)
			# (batch_size, max_len, embed_size * num_pair)
			interests = interests * scores.unsqueeze(dim=-1)
			# (batch_size*keys_length, embed_size * num_pair)
			packed_interests = pack_padded_sequence(
				interests,
				lengths=keys_length.squeeze().cpu(),
				batch_first=True,
				enforce_sorted=False
			)
			# (batch_size, embed_size * num_pair)
			_, outputs = self.interest_evolution(packed_interests)
		elif self.gru_type in ['AGRU', 'AUGRU']:
			# 常规gru的实现 https://zh-v2.d2l.ai/chapter_recurrent-modern/gru.html
			# ----------------- AGRU
			# 用att_score代替gru中的z更新门
			# 缺点 z.shape=(n, h), att_score(n, 1),失去了维度的信息
			# ----------------- AUGRU
			# 用att_score * z 来影响z，克服直接用att来代替z的缺点

			# (batch_size, max_len)
			scores = self.attention(query, interests, keys_length)
			# (batch_size*keys_length, embed_size * num_pair)
			packed_interests = pack_padded_sequence(
				interests,
				lengths=keys_length.squeeze().cpu(),
				batch_first=True,
				enforce_sorted=False
			)
			# (batch_size*keys_length,)
			packed_scores = pack_padded_sequence(
				scores,
				lengths=keys_length.squeeze().cpu(),
				batch_first=True,
				enforce_sorted=False
			)
			# (batch_size*keys_length, embed_size * num_pair)
			packed_outputs = self.interest_evolution(packed_interests, packed_scores)
			# (batch_size, max_len, embed_size * num_pair)
			outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
			mask = (torch.arange(max_length, device=keys_length.device).repeat(batch_size, 1) == (keys_length.view(-1, 1)-1))
			# 取最后一个非填充值时间点特征
			# (batch_size, embed_size * num_pair)
			outputs = outputs[mask]

		return outputs, aloss


""""
DIEN 细节详解 https://zhongqiang.blog.csdn.net/article/details/109532438
重点：
	1， 由两层循环神经网络构成，第一层是兴趣抽取层，第二层兴趣进化层
	2， 兴趣抽取层gru，每个时间点输出代表了每个时间点的兴趣，还可以用来决定下一时间点的点击或者不点击(auxiliary_loss)
	3， 兴趣进化层atten-gru，把attention_score魔改gru中的更新门z，以最后一个时间点的输出为特征,可以理解为对兴趣变化路劲的融合
"""
class DIEN(nn.Module):
	def __init__(self, num_features, cat_features, seq_features,
				cat_nums, embedding_size, attention_groups,
				mlp_hidden_layers, mlp_activation='prelu', mlp_dropout=0.0,
				use_negsampling=True, d_out=1):
		super(DIEN, self).__init__()
		self.num_features = num_features
		self.cat_features = cat_features
		self.seq_features = seq_features
		self.cat_nums = cat_nums
		self.embedding_size = embedding_size
		self.attention_groups = attention_groups
		self.d_out = d_out
		self.use_negsampling = use_negsampling

		# embedding
		self.embeddings = OrderedDict()
		for feature in self.cat_features + self.seq_features:
			self.embeddings[feature] = nn.Embedding(self.cat_nums[feature], self.embedding_size, padding_idx=0)
			# 不是Module类型, pytorch不会自动注册网络模块， 用add_module来实现
			# https://www.cnblogs.com/datasnail/p/14903643.html
			self.add_module(f'embedding:{feature}', self.embeddings[feature])

		# attention_grus
		self.attention_grus = OrderedDict()
		for group in self.attention_groups:
			self.attention_grus[group.name] = Interest(
				input_size=group.pairs_count*self.embedding_size,
				gru_type=group.gru_type,
				gru_dropout=group.gru_dropout,
				att_hidden_layers=group.hidden_layers,
				att_dropout=group.att_dropout,
				att_activation=group.activation,
				use_negsampling=self.use_negsampling
			)
			self.add_module(f'attention_grus:{group.name}', self.attention_grus[group.name])
		# mlp
		total_input_size = len(self.num_features)
		for feature in self.cat_features:
			total_input_size += self.embedding_size
		for feature in self.seq_features:
			if not self.is_neg_feature(feature):
				total_input_size += self.embedding_size
		self.mlp = MLP(
			total_input_size, mlp_hidden_layers,
			dropout=mlp_dropout,
			batchnorm=True,
			activation=mlp_activation
		)
		self.final_layer = nn.Sequential(
			nn.Linear(mlp_hidden_layers[-1], self.d_out),
			nn.Sigmoid() if self.d_out == 1 else nn.Softmax(dim=-1)
		)

	def is_attention_feature(self, feature):
		for group in self.attention_groups:
			if group.is_attention_feature(feature):
				return True
		return False

	def is_neg_feature(self, feature):
		for group in self.attention_groups:
			if group.is_neg_feature(feature):
				return True
		return False

	def forward(self, x):
		# 连续特征
		num_inputs = list()
		for feature in self.num_features:
			# (batch_size, 1)
			num_inputs.append(x[feature].view(-1, 1))
		# 分类特征
		embeddings = OrderedDict()
		for feature in self.cat_features:
			# (batch_size, embed_size)
			embeddings[feature] = self.embeddings[feature](x[feature])
		# 列表特征
		for feature in self.seq_features:
			if not self.is_attention_feature(feature):
				# (batch_size, embed_size) (seq_len方向max)
				embeddings[feature] = torch.max(self.embeddings[feature](x[feature]), dim=1)[0]
		# attention_gru特征
		auxiliary_loss = []
		for group in self.attention_groups:
			# (batch_size, embed_size * num_pair)
			query = torch.cat([embeddings[pair['ad']] for pair in group.pairs], dim=-1)
			# (batch_size, max_len, embed_size * num_pair)
			pos_hist = torch.cat(
				[self.embeddings[pair['pos_hist']](x[pair['pos_hist']]) for pair in group.pairs],
				dim=-1)
			# (batch_size, num_pair)
			keys_length = torch.cat(
				[torch.sum(x[pair['pos_hist']] > 0, dim=-1).view(-1, 1) for pair in group.pairs],
				dim=-1)
			# (batch_size,)
			keys_length = torch.min(keys_length, dim=-1)[0]
			neg_hist = None
			if self.use_negsampling:
				# (batch_size, max_len, embed_size * num_pair)
				neg_hist = torch.cat(
					[self.embeddings[pair['neg_hist']](x[pair['neg_hist']]) for pair in group.pairs],
					dim=-1)
			embeddings[group.name], tmp_loss = self.attention_grus[group.name](
				query, pos_hist, keys_length, neg_hist
			)
			if tmp_loss is not None:
				auxiliary_loss.append(tmp_loss)
		# 合并特征
		emb_concat = torch.cat(num_inputs + [emb for emb in embeddings.values()], dim=-1)
		final_layer_inputs = self.mlp(emb_concat)
		output = self.final_layer(final_layer_inputs)
		if self.d_out == 1:
			output = output.squeeze()

		auxiliary_loss_avg = None
		if len(auxiliary_loss) >= 1:
			auxiliary_loss_avg = torch.mean(torch.tensor(auxiliary_loss))

		return output, auxiliary_loss_avg


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size, lr, epochs = 128, 0.002, 10
	num_features = ['age']
	cate_features = ['gender', 'movieid', 'occupation', 'zipcode']
	seq_features = ['genres', 'hist_movieids', 'neg_hist_movieids']
	cat_nums, train_loader, test_loader = create_dataloader_din(batch_size, num_features, cate_features, seq_features)
	dien_attention_groups = [
		AttentionGroup(
			name='group1',
			pairs=[{'ad': 'movieid', 'pos_hist': 'hist_movieids', 'neg_hist': 'neg_hist_movieids'}],
			activation='dice',
			hidden_layers=[16, 8],
			att_dropout=0.1,
			gru_type='AUGRU',
			gru_dropout=0.0
		)
	]
	model = DIEN(
		num_features=num_features,
		cat_features=cate_features,
		seq_features=seq_features,
		cat_nums=cat_nums,
		embedding_size=16,
		attention_groups=dien_attention_groups,
		mlp_hidden_layers=[32, 16],
		mlp_activation='prelu',
		mlp_dropout=0.25,
		use_negsampling=True,
		d_out=1
	)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer)

