from torch import nn
from in_former import *
from reformer_pytorch import LSHSelfAttention


class ReformerLayer(nn.Module):
	def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, n_hashes=4):
		super(ReformerLayer, self).__init__()
		self.bucket_size = bucket_size
		self.attn = LSHSelfAttention(
			dim=d_model,
			heads=n_heads,
			bucket_size=bucket_size,
			n_hashes=n_hashes,
			causal=causal
		)

	def fit_length(self, queries):
		# ReFormer的实验中将块的长度为桶平均大小的两倍
		B, N, C = queries.shape
		if N % (self.bucket_size * 2) == 0:
			return queries
		else:
			fill_len = self.bucket_size * 2 - (N % (self.bucket_size * 2))
			return torch.cat(
				[queries, torch.zeros([B, fill_len, C]).to(queries.device)],
				dim=1
			)

	def forward(self, queries, keys, values, attn_mask):
		"""
		:param queries: batch_size, seq_len+pred_len, d_model
		:param keys:
		:param values:
		:param attn_mask:
		:return:
		"""
		B, N, C = queries.shape
		queries = self.attn(self.fit_length(queries))[:, :N, :]
		return queries, None


"""
reformer细节参考， https://zhuanlan.zhihu.com/p/357628257
重点：
	局部敏感性哈希的核心思想就是对Q(Q=K)进行相关性分组，组内进行atten_score计算
"""
class Reformer(nn.Module):
	def __init__(self, configs):
		super(Reformer, self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# embedding
		self.enc_embedding = DataEmbedding(
			configs.enc_in, configs.d_model, configs.embed,
			configs.freq, configs.dropout
		)
		# encoder
		self.encoder = Encoder(
			attn_layers=[
				EncoderLayer(
					attention=ReformerLayer(
						attention=None,
						d_model=configs.d_model,
						n_heads=configs.n_heads,
						d_keys=None,
						d_values=None,
						causal=False,
						bucket_size=configs.bucket_size,
						n_hashes=configs.n_hashes
					),
					d_model=configs.d_model,
					d_ff=configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation
				) for _ in range(configs.e_layers)
			],
			conv_layers=None,
			norm_layer=nn.LayerNorm(configs.d_model)
		)
		self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
	):
		# batch_size, seq_len+pred_len, d_model
		x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
		x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

		# encoder
		# batch_size, seq_len+pred_len, d_model
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
		enc_out = self.projection(enc_out)

		if self.output_attention:
			return enc_out[:, -self.pred_len:, :], attns
		else:
			return enc_out[:, -self.pred_len:, :]






