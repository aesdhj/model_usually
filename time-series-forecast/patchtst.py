from utils import *
import torch
from torch import nn
from in_former import *
from trans_former import *


class PatchEmbedding(nn.Module):
	def __init__(self, d_model, patch_len, stride, padding, dropout):
		super(PatchEmbedding, self).__init__()
		self.patch_len = patch_len
		self.stride = stride

		self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
		self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
		self.position_embedding = PositionEmbedding(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		"""
		:param x: batch_size, df, seq_len
		:return:
		"""
		n_vars = x.shape[1]
		# 窗口滑动的时候保留序列长度方向最后一个值
		# batch_size, df, seq_len+padding
		x = self.padding_patch_layer(x)
		# batch_size, df, patch_num, patch_len
		x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
		# batch_size*df, patch_num, patch_len
		x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
		x = self.value_embedding(x) + self.position_embedding(x)
		return self.dropout(x), n_vars


"""
PatchTST细节详解， https://zhuanlan.zhihu.com/p/611909521
重点：
	1，利用滑动窗口将原序列切分输入到transformer模型
	2， patch之间重叠的直接训练和不重叠的预训练
"""
class PatchTST(nn.Module):
	def __init__(self, configs):
		super(PatchTST, self).__init__()
		self.pred_len = configs.pred_len
		self.seq_len = configs.seq_len
		self.output_attention = configs.output_attention

		# patching
		self.patch_embedding = PatchEmbedding(
			configs.d_model, configs.patch_len, configs.stride,
			configs.padding, configs.dropout
		)
		# encoder
		self.encoder = Encoder(
			attn_layers=[
				EncoderLayer(
					attention=AttentionLayer(
						attention=FullAttention(
							mask_flag=False,
							factor=configs.factor,
							scale=None,
							attention_dropout=configs.dropout,
							output_attention=configs.output_attention
						),
						d_model=configs.d_model,
						n_heads=configs.n_heads
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
		# decoder(flatten, projection)
		self.projection_nf = configs.d_model * int((configs.seq_len-configs.patch_len)/configs.stride+2)

		self.flatten = nn.Flatten(start_dim=-2)
		self.projection = nn.Linear(self.projection_nf, configs.pred_len)
		self.projection_dropout = nn.Dropout(configs.dropout)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		"""
		x_enc, batch_size, seq_len, df
		"""
		# Normalization from Non-stationary Transformer
		# batch_size, seq_len, df
		means = x_enc.mean(dim=1, keepdim=True).detach()
		x_enc = x_enc - means
		stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
		x_enc = x_enc / stdev

		# 按滑动窗口的的方式(卷积的方式)在len_seq分段
		# patch之间有重叠的可以直接训练；没有重叠的可以输入进行预训练
		# batch_size, df, seq_len
		x_enc = x_enc.permute(0, 2, 1)
		# batch_size*df, patch_num, d_model  df
		enc_out, n_vars = self.patch_embedding(x_enc)

		# encoder
		# batch_size*df, patch_num, d_model
		enc_out, attns = self.encoder(enc_out)
		# batch_size, df, patch_num, d_model
		enc_out = torch.reshape(
			enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
		)
		# batch_size, df, d_model, patch_num
		enc_out = enc_out.permute(0, 1, 3, 2)

		# decoder(flatten, projection)
		# batch_size, df, d_model*patch_num
		dec_out = self.flatten(enc_out)
		# batch_size, df, pred_len
		dec_out = self.projection_dropout(self.projection(dec_out))
		# batch_size, pred_len, df
		dec_out = dec_out.permute(0, 2, 1)

		# De-Normalization from Non-stationary Transformer
		dec_out = dec_out * (
			stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
		)
		dec_out = dec_out + (
			means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
		)
		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]

