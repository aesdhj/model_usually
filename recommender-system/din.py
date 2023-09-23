from collections import OrderedDict
import torch
from torch import nn
from utils import train_task, create_dataloader_din, AttentionGroup, Attention, MLP


"""
DIN细节详解 https://zhongqiang.blog.csdn.net/article/details/109532346
重点：
	1,历史行为(keys)会影响商品广告的点击(query), 引出之间的相似性(类似注意力机制)作为特征
	2,这里的注意力机制计算的相关系数作为权重，求和不为1
	3,引入Dice损失函数，可以理解为关注输入数据分布(batchnorm)的Prelu
缺点:
	1,历史行为之间的依赖性没有充分表达，不同兴趣的变化之间概率的移转
"""
class DIN(nn.Module):
	def __init__(self, num_features, cat_features, seq_features,
				cat_nums, embedding_size, attention_groups,
				mlp_hidden_layers, mlp_activation='prelu', mlp_dropout=0.0,
				d_out=1):
		super(DIN, self).__init__()
		self.num_features = num_features
		self.cat_features = cat_features
		self.seq_features = seq_features
		self.cat_nums = cat_nums
		self.embedding_size = embedding_size
		self.attention_groups = attention_groups
		self.d_out = d_out

		# embedding
		self.embeddings = OrderedDict()
		for feature in self.cat_features + self.seq_features:
			self.embeddings[feature] = nn.Embedding(self.cat_nums[feature], self.embedding_size, padding_idx=0)
			# 不是Module类型, pytorch不会自动注册网络模块， 用add_module来实现
			# https://www.cnblogs.com/datasnail/p/14903643.html
			self.add_module(f'embedding:{feature}', self.embeddings[feature])

		# attention
		self.attentions = OrderedDict()
		for group in self.attention_groups:
			self.attentions[group.name] = Attention(
				input_size=group.pairs_count*self.embedding_size,
				hidden_layers=group.hidden_layers,
				dropout=group.att_dropout,
				activation=group.activation
			)
			self.add_module(f'attention:{group.name}', self.attentions[group.name])
		# mlp
		total_input_size = len(self.num_features) + len(self.cat_features + self.seq_features) * self.embedding_size
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
		# attention特征
		for group in self.attention_groups:
			# (batch_size, embed_size * num_pair)
			query = torch.cat([embeddings[pair['ad']] for pair in group.pairs], dim=-1)
			# (batch_size, max_len, embed_size * num_pair)
			keys = torch.cat(
				[self.embeddings[pair['pos_hist']](x[pair['pos_hist']]) for pair in group.pairs],
				dim=-1)
			# (batch_size, num_pair)
			key_length = torch.cat(
				[torch.sum(x[pair['pos_hist']] > 0, dim=-1).view(-1, 1) for pair in group.pairs],
				dim=-1)
			# (batch_size,)
			key_length = torch.min(key_length, dim=-1)[0]
			# (batch_size, embed_size * num_pair)
			embeddings[group.name] = self.attentions[group.name](query, keys, key_length)

		# 合并特征
		emb_concat = torch.cat(num_inputs + [emb for emb in embeddings.values()], dim=-1)
		final_layer_inputs = self.mlp(emb_concat)
		output = self.final_layer(final_layer_inputs)
		if self.d_out == 1:
			output = output.squeeze()
		return output


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size, lr, epochs = 128, 0.002, 10
	num_features = ['age']
	cate_features = ['gender', 'movieid', 'occupation', 'zipcode']
	seq_features = ['genres', 'hist_movieids', 'neg_hist_movieids']
	cat_nums, train_loader, test_loader = create_dataloader_din(batch_size, num_features, cate_features, seq_features)

	din_attention_groups = [
		AttentionGroup(
			name='group1',
			pairs=[{'ad': 'movieid', 'pos_hist': 'hist_movieids'}],
			activation='dice',
			hidden_layers=[16, 8],
			att_dropout=0.1
		)
	]
	model = DIN(
		num_features=num_features,
		cat_features=cate_features,
		seq_features=seq_features,
		cat_nums=cat_nums,
		embedding_size=16,
		attention_groups=din_attention_groups,
		mlp_hidden_layers=[32, 16],
		mlp_activation='prelu',
		mlp_dropout=0.25,
		d_out=1
	)
	model.to(device)
	loss = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
	train_task(epochs, model, train_loader, test_loader, device, loss, optimizer)

