import torch
from torch import nn
import math
from torch.nn import functional as F
from utils import *


class DataEmbedding_wo_pos(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding_wo_pos, self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		if embed_type != 'timeF':
			# fixed,正余弦位置编码
			# learned,embedding
			self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
		else:
			# timef,线性映射
			self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		"""
		:param x: batch_size, seq_len, df
		:param x_mark: batch_size, seq_len, dt
		:return: (batch_size, seq_len, d_model)
		"""
		# (batch_size, seq_len, d_model) + (batch_size, seq_len, d_model)
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)


class Encoder(nn.Module):
	def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
		super(Encoder, self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
		self.norm = norm_layer

	def forward(self, x, attn_mask=None):
		attns = []
		if self.conv_layers is not None:
			for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
				x, attn = attn_layer(x, attn_mask=attn_mask)
				x = conv_layer(x)
				attns.append(attn)
			x, attn = self.attn_layers[-1](x)
			attns.append(attn)
		else:
			for attn_layer in self.attn_layers:
				x, attn = attn_layer(x, attn_mask=attn_mask)
				attns.append(attn)
		if self.norm is not None:
			x = self.norm(x)

		return x, attns


class EncoderLayer(nn.Module):
	def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation='relu'):
		super(EncoderLayer, self).__init__()
		d_ff = d_ff or d_model * 4
		self.attention = attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
		self.decomp1 = series_decomp(moving_avg)
		self.decomp2 = series_decomp(moving_avg)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == 'relu' else F.gelu

	def forward(self, x, attn_mask=None):
		# 1, auto-correlation
		new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
		# 2, residual, series-decomp
		x = x + self.dropout(new_x)
		x, _ = self.decomp1(x)
		y = x
		# 3, feed-forward
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		# 4, residual, series-decomp
		res, _ = self.decomp2(x + y)
		return res, attn


class AutoCorrelationLayer(nn.Module):
	def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
		super(AutoCorrelationLayer, self).__init__()
		d_keys = d_keys or (d_model // n_heads)
		d_values = d_values or (d_model // n_heads)
		self.inner_correlation = correlation
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.out_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask):
		"""
		:param queries: (batch_size, seq_len/label_len+pred_len, d_model)
		:param keys: (batch_size, seq_len/label_len+pred_len, d_model)
		:param values:(batch_size, seq_len/label_len+pred_len, d_model)
		:param attn_mask:
		:return:(batch_size, seq_len/label_len+pred_len, d_model)
		"""
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads

		queries = self.query_projection(queries).view(B, L, H, -1)
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)
		# (batch_size, seq_len/label_len+pred_len, n_heads, d_keys)
		out, attn = self.inner_correlation(queries, keys, values, attn_mask)
		out = out.view(B, L, -1)
		return self.out_projection(out), attn


class AutoCorrelation(nn.Module):
	def __init__(self, mask=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
		super(AutoCorrelation, self).__init__()
		self.factor = factor
		self.output_attention = output_attention

	def time_delay_agg_training(self, values, corr):
		"""
		:param values: (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:param corr: (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:return:
		"""
		_, head, channel, length = values.shape
		top_k = int(self.factor * math.log(length))
		# batch_size, seq_len/label_len+pred_len
		mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
		# topk
		index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
		# batch_size, topk
		weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
		tmp_corr = torch.softmax(weights, dim=-1)

		# Q、K不断重复，形成...QQQQQ...，...KKKKK...两个无穷序列,取seq_len数计算corr
		tmp_values = values
		delays_agg = torch.zeros_like(values).float()
		for i in range(top_k):
			pattern = torch.roll(tmp_values, shifts=-int(index[i]), dims=-1)
			delays_agg = delays_agg + pattern * (
				tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
			)
		return delays_agg

	def time_delay_agg_inference(self, values, corr):
		"""
		相当于time_delay_agg_training一个复杂版本，weights精确到batch_size的delay不同
		:param values:(batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:param corr:(batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:return:
		"""
		batch, head, channel, length = values.shape
		top_k = int(self.factor * math.log(length))
		# (batch_size, seq_len / label_len + pred_len)
		mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
		# batch_size, topk
		weights, delay = torch.topk(mean_value, top_k, dim=-1)
		tmp_corr = torch.softmax(weights, dim=-1)

		tmp_values = values.repeat(1, 1, 1, 2)
		delays_agg = torch.zeros_like(values).float()
		# (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
			batch, head, channel, 1
		).to(values.device)
		for i in range(top_k):
			# 每个batch的delay
			tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
				1, head, channel, length
			)
			# 每个batch的delay对应的序列
			pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
			delays_agg = delays_agg + pattern * (
				tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
			)
		return delays_agg

	def time_delay_agg_inference_v1(self, values, corr):
		"""

		:param values: (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:param corr: (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:return:
		"""
		batch, head, channel, length = values.shape
		top_k = int(self.factor * math.log(length))
		# batch_size, n_heads, seq_len
		mean_value = torch.mean(corr, dim=2)
		# batch_size, n_heads, topk
		weight, delay = torch.topk(mean_value, top_k, dim=-1)
		# batch_size, n_heads, topk
		tmp_corr = torch.softmax(weight, dim=-1)

		tmp_values = values.repeat(1, 1, 1, 2)
		delay_agg = torch.zeros_like(values).float()
		init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
			batch, head, channel, 1
		).to(values.device)
		for i in range(top_k):
			tmp_delay = init_index + delay[:, :, i].unsqueeze(2).unsqueeze(2).repeat(
				1, 1, channel, length
			)
			pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
			delay_agg = delay_agg + pattern * (tmp_corr[:, :, i].unsqueeze(2).unsqueeze(2).repeat(1, 1, channel, length))
		return delay_agg

	def time_delay_agg_inference_v2(self, values, corr):
		"""

		:param values: (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:param corr: (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		:return:
		"""
		batch, head, channel, length = values.shape
		top_k = int(self.factor * math.log(length))
		# batch_size, n_heads, d_keys, top_k
		weight, delay = torch.topk(corr, top_k, dim=-1)
		# batch_size, n_heads, d_keys, top_k
		tmp_corr = torch.softmax(weight, dim=-1)

		tmp_values = values.repeat(1, 1, 1, 2)
		delay_agg = torch.zeros_like(values).float()
		init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
			batch, head, channel, 1
		).to(values.device)
		for i in range(top_k):
			tmp_delay = init_index + delay[:, :, :, i].unsqueeze(3).repeat(
				1, 1, 1, length
			)
			pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
			delay_agg = delay_agg + pattern * (tmp_corr[:, :, :, i].unsqueeze(3).repeat(1, 1, 1, length))
		return delay_agg

	def forward(self, queries, keys, values, attn_mask):
		"""
		:param queries: (batch_size, seq_len/label_len+pred_len, n_heads, d_keys);
		:param keys: (batch_size, seq_len/label_len+pred_len, n_heads, d_keys);
		:param values: (batch_size, seq_len/label_len+pred_len, n_heads, d_keys);
		:param attn_mask:
		:return:
		"""
		B, L, H, E = queries.shape
		_, S, _, D = values.shape
		# 统一序列长度,以queries的为准
		if L > S:
			zeros = torch.zeros_like(queries[:, :(L-S), :, :]).float()
			keys = torch.cat([keys, zeros], dim=1)
			values = torch.cat([values, zeros], dim=1)
		else:
			keys = keys[:, :L, :, :]
			values = values[:, :L, :, :]

		# 利用傅里叶变换求出序列和滞后序列的相关性,降低时间复杂度
		# 傅里叶变换针对的无限序列,对于有限序列可以不断重复拼接序列,对应time_delay_agg函数中torch.roll操作
		# 傅里叶变换入门, https://zhuanlan.zhihu.com/p/19763358
		# 2.自相关机制, https://zhuanlan.zhihu.com/p/472624073
		q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
		k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
		res = q_fft * torch.conj(k_fft)
		# (batch_size, n_heads, d_keys, seq_len/label_len+pred_len)
		corr = torch.fft.irfft(res, dim=-1)

		# (batch_size, seq_len/label_len+pred_len, n_heads, d_keys)
		# view()、permute()和contiguous()函数详解, https://zhuanlan.zhihu.com/p/545769141
		if self.training:
			V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2).contiguous()
		else:
			V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2).contiguous()
		if self.output_attention:
			return V, corr.permute(0, 3, 1, 2)
		else:
			return V, None


class my_Layernorm(nn.Module):
	# cnn, mlp用bn, rnn用ln，https://www.zhihu.com/question/454292446/answer/1837760076
	def __init__(self, channels):
		super(my_Layernorm, self).__init__()
		self.layernorm = nn.LayerNorm(channels)

	def forward(self, x):
		"""
		:param x: batch_size, seq_len, d
		:return:
		"""
		x_hat = self.layernorm(x)
		# 和decompose一个道理，seq-mean就是周期性因素
		bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
		return x_hat - bias


class Decoder(nn.Module):
	def __init__(self, layers, norm_layer=None, projection=None):
		super(Decoder, self).__init__()
		self.layers = nn.ModuleList(layers)
		self.norm = norm_layer
		self.projection = projection

	def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
		for layer in self.layers:
			x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
			trend = trend + residual_trend
		if self.norm is not None:
			x = self.norm(x)
		if self.projection is not None:
			x = self.projection(x)
		return x, trend


class DecoderLayer(nn.Module):
	def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None, moving_avg=25, dropout=0.1, activation='relu'):
		super(DecoderLayer, self).__init__()
		self.self_attention = self_attention
		self.cross_attention = cross_attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
		self.decomp1 = series_decomp(moving_avg)
		self.decomp2 = series_decomp(moving_avg)
		self.decomp3 = series_decomp(moving_avg)
		self.dropout = nn.Dropout(dropout)
		self.projection = nn.Conv1d(
			in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
			padding_mode='circular', bias=False
		)
		self.activation = F.relu if activation == 'relu' else F.gelu

	def forward(self, x, cross, x_mask=None, cross_mask=None):
		"""
		:param x: batch_size, label_len+pred_len, d_model
		:param cross: batch_size, seq_len, d_model
		:param x_mask:
		:param cross_mask:
		:return:
		"""
		# 1, auto_correlation
		x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
		# 2, series_decomp
		x, trend1 = self.decomp1(x)
		# 3, residual, cross_correlation
		x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
		# 4, series_decomp
		x, trend2 = self.decomp2(x)
		y = x
		# 5, feed_forward
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		# 6, residual, series_decomp
		x, trend3 = self.decomp3(x+y)

		residual_trend = trend1 + trend2 + trend3
		# batch_size, label_len+pred_len, c_out
		residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
		return x, residual_trend


"""
autoformer细节详解参考 https://zhuanlan.zhihu.com/p/472624073
重点：
1， 把时间序列分解成周期性和趋势性
2， attention机制用序列的自相关代替，用到傅里叶变换计算相关系数
3， 继承informer的预测模式，不采用传统时间预测的step-to_step模型，一步到位，减少误差积累
4， 没有decoder self_attention中maksed
"""
class Autoformer(nn.Module):
	def __init__(self, configs):
		super(Autoformer, self).__init__()
		self.pred_len = configs.pred_len
		self.label_len = configs.label_len
		self.output_attention = configs.output_attention

		# decomp
		kernel_size = configs.moving_avg
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
		# encoder
		self.encoder = Encoder(
			attn_layers=[
				EncoderLayer(
					attention=AutoCorrelationLayer(
						correlation=AutoCorrelation(
							mask=False,
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
					moving_avg=configs.moving_avg,
					dropout=configs.dropout,
					activation=configs.activation
				) for _ in range(configs.e_layers)
			],
			conv_layers=None,
			norm_layer=my_Layernorm(configs.d_model)
		)
		# decoder
		self.decoder = Decoder(
			layers=[
				DecoderLayer(
					self_attention=AutoCorrelationLayer(
						correlation=AutoCorrelation(
							mask=True,
							factor=configs.factor,
							attention_dropout=configs.dropout,
							output_attention=False
						),
						d_model=configs.d_model,
						n_heads=configs.n_heads
					),
					cross_attention=AutoCorrelationLayer(
						correlation=AutoCorrelation(
							mask=False,
							factor=configs.factor,
							attention_dropout=configs.dropout,
							output_attention=False
						),
						d_model=configs.d_model,
						n_heads=configs.n_heads
					),
					d_model=configs.d_model,
					c_out=configs.c_out,
					d_ff=configs.d_ff,
					moving_avg=configs.moving_avg,
					dropout=configs.dropout,
					activation=configs.activation
				) for _ in range(configs.d_layers)
			],
			norm_layer=my_Layernorm(configs.d_model),
			projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
		)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
	):
		"""
		x_enc, batch_size, seq_len, df
		x_mark_enc, batch_size, seq_len, dt
		x_dec, batch_size, label_len+pred_len, df
		x_mark_dec, batch_size, label_len+pred_len,dt
		"""
		# encoder
		# batch_size, seq_len, d_model
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		# batch_size, seq_len, d_model
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		# decomp init
		# batch_size, pred_len, df
		mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
		zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
		# 分解周期性和趋势性, batch_size, seq_len, df
		seasonal_init, trend_init = self.decomp(x_enc)
		# decoder input
		# batch_size, label_len+pred_len, df
		seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
		trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
		# decoder
		# batch_size, label_len+pred_len, d_model
		dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
		seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)

		# final
		dec_out = trend_part + seasonal_part
		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]

