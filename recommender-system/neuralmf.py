import torch
from torch import nn
from torch.nn import functional as F


"""
NeuralMF细节详解， https://zhongqiang.blog.csdn.net/article/details/108985457
重点:  
	1，内积或者相加(等价于concat)都可以带来非线性特征
	2，gmf是mf加上一个非线性映射，可以退化成mf
	3, mlp是user,item的非线性特征
"""
class NeuralMF(nn.Module):
	def __init__(self, num_users, num_items, hidden_units, embed_dim):
		super(NeuralMF, self).__init__()
		# mf和mlp embedding层不共享
		self.mf_embedding_user = nn.Embedding(num_users, embed_dim)
		self.mf_embedding_item = nn.Embedding(num_items, embed_dim)
		self.mlp_embedding_user = nn.Embedding(num_users, embed_dim)
		self.mlp_embedding_item = nn.Embedding(num_items, embed_dim)
		# mlp dnn
		self.dnn_linear = nn.ModuleList([
			nn.Linear(layers[0], layers[1]) for layers in zip(hidden_units[:-1], hidden_units[1:])
		])
		self.linear = nn.Linear(hidden_units[-1], embed_dim)
		# concat linear
		self.concat_linear = nn.Linear(2*embed_dim, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		"""
		:param x: batch_size,2(user, item)
		:return:
		"""
		# mf
		mf_user = self.mf_embedding_user(x[:, 0])
		mf_item = self.mf_embedding_item(x[:, 1])
		mf_vec = torch.mul(mf_user, mf_item)
		# mlp
		mlp_user = self.mlp_embedding_user(x[:, 0])
		mlp_item = self.mlp_embedding_item(x[:, 1])
		x = torch.cat([mlp_user, mlp_item], dim=-1)
		for linear in self.dnn_linear:
			x = linear(x)
			x = F.relu(x)
		mlp_vec = self.linear(x)
		# concat
		vec_concat = torch.cat([mf_vec, mlp_vec], dim=-1)
		output = self.concat_linear(vec_concat).squeeze()
		output = self.sigmoid(output)
		return output

