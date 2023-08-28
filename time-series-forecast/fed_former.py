import torch
from torch import nn
from utils import *
from auto_former import *


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
	modes = min(modes, seq_len//2)
	if mode_select_method == 'random':
		index = list(range(seq_len//2))
		np.random.shuffle(index)
		index = index[:modes]
	else:
		index = list(range(modes))
	index.sort()
	return index


class FourierBlock(nn.Module):
	def __init__(self, in_channels, out_channels, seq_len, n_heads, modes=64, mode_select_method='random'):
		super(FourierBlock, self).__init__()
		self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
		self.scale = 1 / (in_channels * out_channels)
		# n_heads, d_keys, d_keys, len_index
		self.weightsl = nn.Parameter(
			self.scale * torch.rand(
				n_heads, in_channels//n_heads, out_channels//n_heads, len(self.index),
				dtype=torch.cfloat)
		)

	def forward(self, queries, keys, values, mask):
		"""
		:param queries: batch_size, len_q, n_heads, d_keys
		:param keys:
		:param values:
		:param mask:
		:return:
		"""
		B, L, H, E = queries.shape
		# batch_size, n_heads, d_keys, len_q
		q_tr = queries.permute(0, 2, 3, 1)
		# 傅里叶变换成频域, batch_size, n_heads, d_keys, len_q//2 + 1
		q_ft = torch.fft.rfft(q_tr, dim=-1)
		# torch.cfloat 64位复数
		# batch_size, n_heads, d_keys, len_q//2 + 1
		out_ft = torch.zeros(
			[B, H, E, L//2+1],
			device=queries.device,
			dtype=torch.cfloat
		)
		for wi, i in enumerate(self.index):
			# batch_size, n_heads, d_keys
			# batch_size, n_heads, d_keys
			# n_heads, d_keys, d_keys
			# 在len_q上对频域进行取样，消除噪声
			out_ft[:, :, :, wi] = torch.einsum(
				'bhi,hio->bho', q_ft[:, :, :, i], self.weightsl[:, :, :, wi]
			)
		# 频域逆变换成时域, batch_size, n_heads, d_keys, len_q
		out = torch.fft.irfft(out_ft, n=q_tr.size(-1))
		# batch_size, len_q, n_heads, d_keys
		out = out.permute(0, 3, 1, 2)
		return out, None


class FourierCrossAttention(nn.Module):
	def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, n_heads, modes=64, mode_select_method='random', activation='tanh'):
		super(FourierCrossAttention, self).__init__()
		self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
		self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
		self.activation = activation
		self.scale = 1 / (in_channels * out_channels)
		self.weightsl = nn.Parameter(
			self.scale * torch.rand(
				n_heads, in_channels//n_heads, out_channels//n_heads, len(self.index_q),
				dtype=torch.cfloat)
		)
		self.in_channels = in_channels
		self.out_channels = out_channels

	def forward(self, queries, keys, values, mask):
		"""
		:param queries: batch_size, len_q, n_heads, d_keys
		:param keys: batch_size, len_kv, n_heads, d_keys
		:param values: batch_size, len_kv, n_heads, d_keys
		:return:
		"""
		B, L, H, E = queries.shape
		# batch_size, n_heads, d_keys, len_q
		q_tr = queries.permute(0, 2, 3, 1)
		# batch_size, n_heads, d_keys, len_kv
		k_tr = keys.permute(0, 2, 3, 1)
		v_tr = values.permute(0, 2, 3, 1)

		# 1, queries傅里叶变换以后抽样
		# batch_size, n_heads, d_keys, index_q
		q_out_ft = torch.zeros(
			[B, H, E, len(self.index_q)],
			device=queries.device,
			dtype=torch.cfloat
		)
		# batch_size, n_heads, d_keys, len_q//2+1
		q_ft = torch.fft.rfft(q_tr, dim=-1)
		for i, j in enumerate(self.index_q):
			q_out_ft[:, :, :, i] = q_ft[:, :, :, j]

		# 2, keys傅里叶变换以后抽样
		# batch_size, n_heads, d_keys, index_kv
		k_out_ft = torch.zeros(
			[B, H, E, len(self.index_kv)],
			device=keys.device,
			dtype=torch.cfloat
		)
		# batch_size, n_heads, d_keys, len_kv//2+1
		k_ft = torch.fft.rfft(k_tr, dim=-1)
		for i, j in enumerate(self.index_kv):
			k_out_ft[:, :, :, i] = k_ft[:, :, :, j]

		# 3, 计算cross_attention score
		# batch_size, n_heads, index_q, index_kv
		qk_cross_ft = torch.einsum('bhex,bhey->bhxy', q_out_ft, k_out_ft)
		if self.activation == 'tanh':
			qk_cross_ft = qk_cross_ft.tanh()
		elif self.activation == 'softmax':
			# abs(复数)=模
			qk_cross_ft = torch.softmax(abs(qk_cross_ft), dim=-1)
			qk_cross_ft = torch.complex(qk_cross_ft, torch.zeros_like(qk_cross_ft))
		else:
			raise Exception('wrong cross attention activation with {}'.format(self.activation))

		# 4, 根据atten_score计算输出
		# cross attention keys_values， 所以k_out_ft=v_out_ft
		# batch_size, n_heads, d_keys, index_q
		qkv_out = torch.einsum('bhxy,bhey->bhex', qk_cross_ft, k_out_ft)

		# 5, 在len_q上对频域进行取样，消除噪声
		# batch_size, n_heads, d_keys, index_q
		qkvw_out = torch.einsum('bhex,heox->bhox', qkv_out, self.weightsl)
		# batch_size, n_heads, d_keys, len_q//2 + 1
		out_ft = torch.zeros(
			[B, H, E, L//2+1],
			device=queries.device,
			dtype=torch.cfloat
		)
		for i, j in enumerate(self.index_q):
			out_ft[:, :, :, j] = qkvw_out[:, :, :, i]

		# 6, 频域逆变换成时域
		# batch_size, n_heads, d_keys, len_q
		out = torch.fft.irfft(out_ft/self.in_channels/self.out_channels, n=q_tr.size(-1))
		out = out.permute(0, 3, 1, 2)
		return out, None


"""
前置知识：
	傅里叶变换概念， https://zhuanlan.zhihu.com/p/19763358
	离散傅里叶变换实际计算， https://www.zhihu.com/question/21314374/answer/542909849
	小波变换， https://zhuanlan.zhihu.com/p/22450818
fedformer细节参考, https://zhuanlan.zhihu.com/p/528131016
重点:
	1,利用频域稀疏分布的特点随机采样去除噪声
	2，相对于傅里叶变换小波分析对时序特别非平稳时序有天然的优势
"""
class FEDformer(nn.Module):
	def __init__(self, configs):
		global decoder_self_att, encoder_self_att, decoder_cross_att
		super(FEDformer, self).__init__()
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# decomp
		kernel_size = configs.moving_avg
		if isinstance(kernel_size, list):
			self.decomp = series_decomp_multi(kernel_size)
		else:
			self.decomp = series_decomp(kernel_size)
		# embedding
		self.enc_embedding = DataEmbedding_wo_pos(
			configs.enc_in, configs.d_model, configs.embed,
			configs.freq, configs.dropout
		)
		self.dec_embedding = DataEmbedding_wo_pos(
			configs.dec_in, configs.d_model, configs.embed,
			configs.freq, configs.dropout
		)

		if configs.version == 'Wavelets':
			pass
		else:
			encoder_self_att = FourierBlock(
				in_channels=configs.d_model,
				out_channels=configs.d_model,
				seq_len=self.seq_len,
				n_heads=configs.n_heads,
				modes=configs.modes,
				mode_select_method=configs.mode_select)
			decoder_self_att = FourierBlock(
				in_channels=configs.d_model,
				out_channels=configs.d_model,
				seq_len=self.label_len + self.pred_len,
				n_heads=configs.n_heads,
				modes=configs.modes,
				mode_select_method=configs.mode_select)
			decoder_cross_att = FourierCrossAttention(
				in_channels=configs.d_model,
				out_channels=configs.d_model,
				seq_len_q=self.label_len + self.pred_len,
				seq_len_kv=self.seq_len,
				n_heads=configs.n_heads,
				modes=configs.modes,
				mode_select_method=configs.mode_select)
		# encoder
		self.encoder = Encoder(
			attn_layers=[EncoderLayer(
				attention=AutoCorrelationLayer(
					correlation=encoder_self_att,
					d_model=configs.d_model,
					n_heads=configs.n_heads
				),
				d_model=configs.d_model,
				d_ff=configs.d_ff,
				moving_avg=configs.moving_avg,
				dropout=configs.dropout,
				activation=configs.activation
			) for _ in range(configs.e_layers)],
			conv_layers=None,
			norm_layer=my_Layernorm(configs.d_model)
		)
		# decoder
		self.decoder = Decoder(
			layers=[DecoderLayer(
				self_attention=AutoCorrelationLayer(
					correlation=decoder_self_att,
					d_model=configs.d_model,
					n_heads=configs.n_heads
				),
				cross_attention=AutoCorrelationLayer(
					correlation=decoder_cross_att,
					d_model=configs.d_model,
					n_heads=configs.n_heads
				),
				d_model=configs.d_model,
				c_out=configs.c_out,
				d_ff=configs.d_ff,
				moving_avg=configs.moving_avg,
				dropout=configs.dropout,
				activation=configs.activation
			) for _ in range(configs.d_layers)],
			norm_layer=my_Layernorm(configs.d_model),
			projection=nn.Linear(configs.d_model, configs.c_out)
		)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		"""
		x_enc, batch_size, seq_len, df
		x_mark_enc, batch_size, seq_len, dt
		x_dec, batch_size, label_len+pred_len, df
		x_mark_dec, batch_size, label_len+pred_len,dt
		"""
		# enc
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		# batch_size, seq_len, d_model
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		# decomp init
		# batch_size, pred_len, d_f
		mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
		zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
		seasonal_init, trend_init = self.decomp(x_enc)
		# decoder input
		seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
		trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
		# dec
		dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
		seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)

		# final
		dec_out = trend_part + seasonal_part
		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]

